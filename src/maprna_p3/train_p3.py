"""VCPE-rna P2.3-B training: direct per-gene deviation head (no SE backbone).

Protocol (inherited from P2.1/P2.2, kept intact):
  true_fc      = pert_mean_hvg - dataset_ctrl_mean_hvg
  common_fc    = mean over TRAIN perturbations only (no leakage)
  true_dev     = true_fc - common_fc   <- the model predicts THIS directly
  pred_fc      = pred_dev + common_fc  (for full-fc metrics)

Eval reports full-fc metrics (mse_DE / pearson_delta / top50) AND dev metrics
(mse_dev / pearson_dev / top50_dev) PLUS a built-in conditioning ablation:
each epoch, pred_dev is recomputed with the target ESM2 vector zeroed and
shuffled; ablation_r = corr(real, zero). P2/P2.1/P2.2 all showed 1.000 (dead).
If conditioning is alive, ablation_r drops well below 1.

No SE / flash-attention / MAP repo / env vars required. Runs in minutes.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
for _sib in ("maprna_p1", "maprna_p2", "."):
    _p = _HERE.parent / _sib
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ds_knockdown import build_symbol2row, load_kd_datasets, make_hvg_list, resolve_pert_row  # noqa: E402
from model_dev import DeviationModel, load_esm_matrix  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dirs", nargs="+", required=True)
    p.add_argument("--esm_table", type=str, required=True)
    p.add_argument("--string_neighbors", type=str, default="",
                   help="string_neighbors.npy (network axis); empty = off")
    p.add_argument("--rna_encoder_ckpt", type=str, default="",
                   help="aligned RNA encoder (.pt, Stage A); empty = no seq axis")
    p.add_argument("--fasta", type=str, default="",
                   help="gene_transcripts.fa (required if rna_encoder_ckpt set)")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--n_hvg", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--test_frac", type=float, default=0.15)
    p.add_argument("--min_cells", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--d_model", type=int, default=256)
    return p.parse_args()


# ---------------- data ----------------

def build_dev_data(args):
    sym2row, esm_dim = build_symbol2row(args.esm_table)
    kd = load_kd_datasets(args.data_dirs, sym2row, min_cells=args.min_cells)
    hvg_rows = make_hvg_list(kd, n_hvg=args.n_hvg)

    # dataset-level control mean over HVG (deterministic ctrl baseline)
    ctrl_mean = np.zeros((len(kd["X_ctrl"]), len(hvg_rows)), dtype=np.float32)
    for di, (Xc, rows) in enumerate(zip(kd["X_ctrl"], kd["row_of_gene"])):
        cm = Xc.mean(axis=0)
        row2col = {int(r): c for c, r in enumerate(rows)}
        for hi, hr in enumerate(hvg_rows):
            c = row2col.get(int(hr))
            if c is not None:
                ctrl_mean[di, hi] = cm[c]
    # z-score across genes within each dataset (feature scale for the MLP)
    mu, sd = ctrl_mean.mean(axis=1, keepdims=True), ctrl_mean.std(axis=1, keepdims=True) + 1e-6
    ctrl_feat_all = (ctrl_mean - mu) / sd

    # perturbation items: all (ds, condition) with enough cells and a resolvable target
    items = []
    for di in range(len(kd["X_pert"])):
        cp = kd["pert_labels"][di]
        for cond in sorted(set(cp)):
            if cond.lower() == "ctrl":
                continue
            idx = np.where(cp == cond)[0]
            # after pseudobulk aggregation each condition has exactly 1 row;
            # the >=min_cells gate already ran on cell counts inside load_kd_datasets
            if len(idx) < 1:
                continue
            if resolve_pert_row(sym2row, cond) < 0:
                continue
            items.append((di, cond))
    rng = np.random.default_rng(args.seed)
    items = sorted(items)
    rng.shuffle(items)
    n_test = max(1, int(round(len(items) * args.test_frac)))
    test_items, train_items = sorted(items[:n_test]), sorted(items[n_test:])

    # targets: deterministic pert-mean over HVG
    def targets_of(split_items):
        T = np.zeros((len(split_items), len(hvg_rows)), dtype=np.float32)
        rows_p = np.zeros(len(split_items), dtype=np.int64)
        for k, (di, cond) in enumerate(split_items):
            cp = kd["pert_labels"][di]
            idx = np.where(cp == cond)[0]
            mean = kd["X_pert"][di][idx].mean(axis=0)
            row2col = {int(r): c for c, r in enumerate(kd["row_of_gene"][di])}
            for hi, hr in enumerate(hvg_rows):
                c = row2col.get(int(hr))
                if c is not None:
                    T[k, hi] = mean[c]
            rows_p[k] = resolve_pert_row(sym2row, cond)
        return T, rows_p

    T_tr, rows_tr = targets_of(train_items)
    T_te, rows_te = targets_of(test_items)

    # V2-1a: 靶基因自身 ctrl 表达（raw log1p，供 self-response 门控）。
    # 目标基因在所属数据集 panel 内 -> 取该列 ctrl 均值；panel 外 -> 0.0
    #（此时目标一般也不在 HVG 响应 panel，is_tgt 行不存在，门控不起作用，安全）。
    def pert_expr_of(split_items):
        E = np.zeros(len(split_items), dtype=np.float32)
        for k, (di, cond) in enumerate(split_items):
            r = resolve_pert_row(sym2row, cond)
            if r < 0:
                continue
            row2col = {int(rr): c for c, rr in enumerate(kd["row_of_gene"][di])}
            c = row2col.get(int(r))
            if c is not None:
                E[k] = float(kd["X_ctrl"][di].mean(axis=0)[c])
        return E

    pert_expr_tr = pert_expr_of(train_items)
    pert_expr_te = pert_expr_of(test_items)
    print(f"[v2-1a] target ctrl expr: train median {np.median(pert_expr_tr):.3f} "
          f"| test median {np.median(pert_expr_te):.3f} "
          f"| near-zero(<0.1) train {int((pert_expr_tr < 0.1).sum())}/{len(pert_expr_tr)}",
          flush=True)

    # full fc and residual targets (train-only common core)
    fc_tr = T_tr - ctrl_mean[np.array([di for di, _ in train_items])]
    fc_te = T_te - ctrl_mean[np.array([di for di, _ in test_items])]
    common_fc = fc_tr.mean(axis=0)
    dev_tr, dev_te = fc_tr - common_fc, fc_te - common_fc
    print(f"[P2.3-B] common fc (train-only): std={common_fc.std():.4f} "
          f"max|.|={np.abs(common_fc).max():.4f}", flush=True)

    return dict(sym2row=sym2row, esm_dim=esm_dim, hvg_rows=hvg_rows,
                ctrl_feat_all=ctrl_feat_all, train_items=train_items,
                test_items=test_items, rows_tr=rows_tr, rows_te=rows_te,
                pert_expr_tr=pert_expr_tr, pert_expr_te=pert_expr_te,
                fc_tr=fc_tr, fc_te=fc_te, dev_tr=dev_tr, dev_te=dev_te,
                common_fc=common_fc, n_ds=len(kd["X_ctrl"]))


def make_rna_emb_lookup(args, data):
    """Precompute RNA-encoder embeddings for all perturbations (train+test)."""
    if not args.rna_encoder_ckpt:
        return None
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "maprna_p2"))
    from rna_encoder import RNAEncoder, encode_seq, load_fasta_symbol_seqs, lookup_seq
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = RNAEncoder(d_model=256, max_len=600).to(dev).eval()
    state = torch.load(args.rna_encoder_ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "rna_encoder" in state:
        state = state["rna_encoder"]        # Stage-A ckpt is a wrapper dict
    enc.load_state_dict(state)
    symbol_map, ensembl_map = load_fasta_symbol_seqs(args.fasta)
    all_items = data["train_items"] + data["test_items"]
    embs = np.zeros((len(all_items), 256), dtype=np.float32)
    with torch.no_grad():
        for k, (di, cond) in enumerate(all_items):
            gene = cond.split("+")[0]
            seq = lookup_seq(gene, symbol_map, ensembl_map)
            if seq:
                ids, m = encode_seq(seq, 600)
            else:
                ids, m = [6] + [0] * 599, [1] + [0] * 599  # CLS + PAD fallback
            tok = torch.tensor([ids], device=dev)
            msk = torch.tensor([m], device=dev).bool()
            embs[k] = enc(tok, msk)[0].float().cpu().numpy()
    print(f"[P2.3-B] rna embeddings: {len(all_items)} perts", flush=True)
    return torch.from_numpy(embs)


# ---------------- eval ----------------

@torch.no_grad()
def evaluate(model, data, split, device, esm_override_mode=None):
    rows = data["rows_te"] if split == "test" else data["rows_tr"]
    fc = data["fc_te"] if split == "test" else data["fc_tr"]
    dev = data["dev_te"] if split == "test" else data["dev_tr"]
    items = data["test_items"] if split == "test" else data["train_items"]
    ds_idx = torch.tensor([di for di, _ in items], dtype=torch.long)
    pr = torch.tensor(rows, dtype=torch.long)
    cf = torch.from_numpy(data["ctrl_feat_all"][ds_idx.numpy()])
    pe = torch.from_numpy(data["pert_expr_te"] if split == "test" else data["pert_expr_tr"])
    rna = None
    if model.proj_r is not None:
        rna = data["rna_embs"][len(data["train_items"]):] if split == "test" \
            else data["rna_embs"][:len(data["train_items"])]

    def _predict_chunks(pr, ds_idx, cf, rna, pe, ov=None, chunk=128):
        outs = []
        for lo in range(0, pr.shape[0], chunk):
            hi = lo + chunk
            kw = {"pert_esm_override": ov[lo:hi]} if ov is not None else {}
            outs.append(model(pr[lo:hi], ds_idx[lo:hi], cf[lo:hi],
                              rna_emb=rna[lo:hi] if rna is not None else None,
                              pert_ctrl_expr=pe[lo:hi], **kw).cpu())
        return torch.cat(outs).numpy()

    pred_dev = _predict_chunks(pr, ds_idx, cf, rna, pe)
    if esm_override_mode == "zero":
        ov = torch.zeros_like(model.esm_table[pr])
        pred_dev = _predict_chunks(pr, ds_idx, cf, rna, pe, ov=ov)
    elif esm_override_mode == "shuffle":
        g = torch.Generator().manual_seed(123)
        perm = torch.randperm(pr.shape[0], generator=g)
        ov = model.esm_table[pr[perm]]
        pred_dev = _predict_chunks(pr, ds_idx, cf, rna, pe, ov=ov)

    pred_fc = pred_dev + data["common_fc"]
    out = {}
    mse = np.mean((pred_fc - fc) ** 2, axis=1)
    out["mse_DE"] = float(mse.mean())
    out["mse_DE_baseline_ctrl"] = float(np.mean((fc) ** 2))
    prs, tops = [], []
    for b in range(len(fc)):
        t, p = fc[b], pred_fc[b]
        prs.append(float(np.corrcoef(t, p)[0, 1]) if t.std() > 1e-6 and p.std() > 1e-6 else 0.0)
        k = 50
        tops.append(len(set(np.argsort(-np.abs(t))[:k]) & set(np.argsort(-np.abs(p))[:k])) / k)
    out["pearson_delta"], out["top50_deg_overlap"] = float(np.mean(prs)), float(np.mean(tops))
    # dev metrics
    mse_d = np.mean((pred_dev - dev) ** 2, axis=1)
    out["mse_dev"] = float(mse_d.mean())
    prd, topd = [], []
    for b in range(len(dev)):
        t, p = dev[b], pred_dev[b]
        prd.append(float(np.corrcoef(t, p)[0, 1]) if t.std() > 1e-6 and p.std() > 1e-6 else 0.0)
        k = 50
        topd.append(len(set(np.argsort(-np.abs(t))[:k]) & set(np.argsort(-np.abs(p))[:k])) / k)
    out["pearson_dev"], out["top50_dev"] = float(np.mean(prd)), float(np.mean(topd))
    return out


# ---------------- main ----------------

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = build_dev_data(args)
    esm = load_esm_matrix(args.esm_table)

    rna_enc = None
    if args.rna_encoder_ckpt:
        from rna_encoder import RNAEncoder
        rna_enc = RNAEncoder(d_model=256, max_len=600)
        state = torch.load(args.rna_encoder_ckpt, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "rna_encoder" in state:
            state = state["rna_encoder"]    # Stage-A ckpt is a wrapper dict
        rna_enc.load_state_dict(state)
        data["rna_embs"] = make_rna_emb_lookup(args, data)

    model = DeviationModel(esm, data["hvg_rows"], n_ds=data["n_ds"],
                           d_model=args.d_model, rna_encoder=rna_enc).to(device)
    if args.string_neighbors and os.path.exists(args.string_neighbors):
        model.set_neighbor_table(np.load(args.string_neighbors), device)
        cov = int((model.neighbor_table >= 0).any(dim=1).sum())
        print(f"[P2.3-B] neighbor table: {tuple(model.neighbor_table.shape)} "
              f"(coverage {cov}/{model.neighbor_table.shape[0]})", flush=True)
    else:
        print("[P2.3-B] network axis OFF", flush=True)

    n_par = sum(p.numel() for p in model.parameters())
    print(f"params total={n_par/1e6:.1f}M (all trainable)", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    dev_tr_t = torch.from_numpy(data["dev_tr"]).float().to(device)
    ds_tr = torch.tensor([di for di, _ in data["train_items"]], dtype=torch.long)
    rows_tr = torch.tensor(data["rows_tr"], dtype=torch.long)
    cf_tr = torch.from_numpy(data["ctrl_feat_all"][ds_tr.numpy()]).float().to(device)
    pe_tr = torch.from_numpy(data["pert_expr_tr"]).float().to(device)
    rna_tr = (data["rna_embs"][:len(data["train_items"])].to(device)
              if model.proj_r is not None else None)

    log_path = os.path.join(args.out_dir, "train_log.jsonl")
    best_pd = -1.0
    n = len(rows_tr)
    for epoch in range(args.epochs):
        model.train()
        t0, run = time.time(), []
        perm = torch.randperm(n)
        for lo in range(0, n, args.batch_size):
            idx = perm[lo:lo + args.batch_size]
            pred = model(rows_tr[idx], ds_tr[idx], cf_tr[idx],
                         rna_emb=rna_tr[idx] if rna_tr is not None else None,
                         pert_ctrl_expr=pe_tr[idx])
            loss = nn.functional.mse_loss(pred, dev_tr_t[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run.append(float(loss.item()))
        sched.step()
        msg = dict(epoch=epoch + 1, steps=(n + args.batch_size - 1) // args.batch_size,
                   train_loss=float(np.mean(run)), secs=round(time.time() - t0, 1))
        model.eval()
        with torch.no_grad():
            ev = evaluate(model, data, "test", device)
        # ablation r: corr(real pred_dev, zero-pert pred_dev) over flattened test
        with torch.no_grad():
            rows_t = torch.tensor(data["rows_te"], dtype=torch.long)
            ds_t = torch.tensor([di for di, _ in data["test_items"]], dtype=torch.long)
            cf_t = torch.from_numpy(data["ctrl_feat_all"][ds_t.numpy()])
            pe_t = torch.from_numpy(data["pert_expr_te"])
            rna_t = (data["rna_embs"][len(data["train_items"]):]
                     if model.proj_r is not None else None)
            ov_all = torch.zeros_like(model.esm_table[rows_t])

            def _abl_chunks(chunk=128):
                reals, zeros = [], []
                for lo in range(0, rows_t.shape[0], chunk):
                    hi = lo + chunk
                    kw = dict(rna_emb=rna_t[lo:hi] if rna_t is not None else None,
                              pert_esm_override=ov_all[lo:hi])
                    reals.append(model(rows_t[lo:hi], ds_t[lo:hi], cf_t[lo:hi],
                                       pert_ctrl_expr=pe_t[lo:hi]).cpu())
                    zeros.append(model(rows_t[lo:hi], ds_t[lo:hi], cf_t[lo:hi],
                                       pert_ctrl_expr=pe_t[lo:hi], **kw).cpu())
                return torch.cat(reals).numpy().ravel(), torch.cat(zeros).numpy().ravel()

            p_real, p_zero = _abl_chunks()
        ab_r = float(np.corrcoef(p_real, p_zero)[0, 1]) if p_real.std() > 1e-9 else 1.0
        msg["eval"] = ev
        msg["ablation_r_real_vs_zeropert"] = round(ab_r, 4)
        print(json.dumps(msg), flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(msg) + "\n")
        if ev["pearson_dev"] > best_pd:
            best_pd = ev["pearson_dev"]
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch + 1,
                        "metrics": ev, "hvg_rows": data["hvg_rows"],
                        "common_fc": data["common_fc"]},
                       os.path.join(args.out_dir, "ckpt_p3_best_dev.pt"))
            print(f"💾 saved best-dev (pearson_dev={best_pd:.4f})", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
