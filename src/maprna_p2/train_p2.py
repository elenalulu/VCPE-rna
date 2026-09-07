"""VCPE-rna P2 training: dual-axis (mechanism + RNA sequence) fine-tune.

Starts from a P1 checkpoint (default: ckpt_best_cosine.pt), attaches the
Stage-A-aligned RNA encoder, and fine-tunes with dual-axis conditioning.
Engineering deltas vs P1 (per P1_result.md): lr 5e-5 base (was 1e-4), AMP off
by default (kills the ep5/ep10 spikes), dual-track ckpt saving.

Run (GPU box):
  cd $BASE/MAP-KG-main/MAP && export MAP_KG_ENCODER_CKPT=... 
  python $BASE/src/train_p2.py \
    --data_dirs $BASE/data/{adamson,norman,replogle_rpe1_essential}/perturb_processed.h5ad \
    --esm_table $BASE/weights/Homo_sapiens...ESM2.pt \
    --se_ckpt $BASE/weights/se600m.safetensors --se_config configs/se600m.yaml \
    --map_repo $BASE/MAP-KG-main/MAP --p1_ckpt $BASE/p1_full/ckpt_best_cosine.pt \
    --rna_encoder_ckpt $BASE/p2_align/rna_encoder_aligned.pt \
    --fasta $BASE/data/gene_transcripts.fa \
    --out_dir $BASE/p2_out --epochs 20 --batch_size 2
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dirs", type=str, nargs="+", required=True)
    p.add_argument("--esm_table", type=str, required=True)
    p.add_argument("--se_ckpt", type=str, required=True)
    p.add_argument("--se_config", type=str, required=True)
    p.add_argument("--map_repo", type=str, required=True)
    p.add_argument("--p1_ckpt", type=str, required=True,
                   help="P1 checkpoint to start from (ckpt_best_cosine.pt recommended)")
    p.add_argument("--rna_encoder_ckpt", type=str, required=True,
                   help="Stage-A aligned encoder (rna_encoder_aligned.pt)")
    p.add_argument("--fasta", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="./p2_out")
    p.add_argument("--set_size", type=int, default=8)
    p.add_argument("--sent_len", type=int, default=2048)
    p.add_argument("--n_hvg", type=int, default=2000)
    p.add_argument("--rna_max_len", type=int, default=600)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=5e-5, help="base lr (down from P1's 1e-4: spike fix)")
    p.add_argument("--lr_new", type=float, default=1e-4, help="lr for rna branch (new modules)")
    p.add_argument("--num_warmup_steps", type=int, default=200)
    p.add_argument("--hvg_loss_weight", type=float, default=1.0,
                   help="P2.1: raised 0.1→1.0, deviation loss is now the main loss")
    p.add_argument("--pert_token_scale", type=float, default=5.0,
                   help="P2.1: scale pert token so conditioning survives the residual stream")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--test_frac", type=float, default=0.15)
    p.add_argument("--min_cells", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--amp", action="store_true", help="OFF by default: P1 spikes were AMP-related")
    p.add_argument("--eval_every_epochs", type=int, default=1)
    p.add_argument("--stop_after_epoch", type=int, default=0,
                   help="early stop aid: 0 = run all epochs")
    p.add_argument("--string_neighbors", type=str, default="",
                   help="P2.2: string_neighbors.npy (network axis); empty = off")
    return p.parse_args()


def split_perturbations(items, test_frac, seed):
    rng = np.random.default_rng(seed)
    by_ds = {}
    for it in items:
        by_ds.setdefault(it[0], []).append(it)
    train, test = [], []
    for di, lst in sorted(by_ds.items()):
        lst = sorted(lst)
        idx = np.random.default_rng(seed).permutation(len(lst))
        n_test = max(1, int(round(len(lst) * test_frac)))
        test.extend([lst[i] for i in idx[:n_test]])
        train.extend([lst[i] for i in idx[n_test:]])
    return train, test


def precompute_pert_emb_targets(model, kd, items, device):
    """SE embedding of each perturbation's perturbed bulk (frozen SE)."""
    model.eval()
    out = {}
    with torch.no_grad():
        for di, cond in items:
            rows = kd["row_of_gene"][di]
            cp = kd["pert_labels"][di]
            sel = np.where(cp == cond)[0]
            Xb = kd["X_pert"][di][sel].mean(axis=0)
            m = rows >= 0
            gid, ex = rows[m], Xb[m]
            order = np.argsort(-ex)[: 2047]
            ids = np.concatenate([[0], gid[order]]).astype(np.int32)[None, :]
            exs = np.concatenate([[0.0], ex[order]]).astype(np.float32)[None, :]
            src = model.se.pe_embedding(torch.from_numpy(ids).long().to(device))
            src = torch.nn.functional.normalize(src, dim=2)
            cls = model.se.cls_token.expand(src.size(0), 1, -1)
            src = torch.cat([cls, src[:, 1:, :]], dim=1)
            if model.se.dataset_token is not None:
                src = torch.cat((src, model.se.dataset_token.expand(src.size(0), 1, -1)), dim=1)
            with torch.no_grad():
                _, emb, _ = model.se(src=src,
                                     counts=torch.from_numpy(exs).float().to(device),
                                     dataset_nums=None, profile=False)
            out[(di, cond)] = emb.reshape(emb.shape[0], -1)[0].float().cpu().numpy()
    return out


@torch.no_grad()
def evaluate(model, loader, device, hvg_rows, kd, common_fc):
    model.eval()
    mse_l, pear_l, top_l, cos_l, bm_c, bm_p = [], [], [], [], [], []
    mse_dev_l, dev_pear, dev_top = [], [], []
    for batch in loader:
        ctrl_gene = batch["control_gene_ids"].to(device)
        ctrl_expr = batch["control_expressions"].to(device)
        pert_row = batch["pert_row"].to(device)
        rna_tok = batch["rna_tokens"].to(device)
        rna_mask = batch["rna_mask"].to(device)
        true_hvg = batch["pert_hvg"].numpy()
        with torch.autocast("cuda", enabled=False):
            pred_embs, pred_hvgs = model(ctrl_gene, ctrl_expr, pert_row, rna_tok, rna_mask)
        pred_hvg = pred_hvgs.float().mean(dim=1).cpu().numpy()

        ctrl_profiles = [kd["X_ctrl"][di].mean(axis=0) for di in batch["ds"]]
        ctrl_hvg = np.stack([
            np.array([ctrl_profiles[b][np.where(kd["row_of_gene"][batch["ds"][b]] == int(hr))[0][0]]
                      if (kd["row_of_gene"][batch["ds"][b]] == int(hr)).any() else 0.0
                      for hr in hvg_rows]) for b in range(len(batch["ds"]))])

        true_fc = true_hvg - ctrl_hvg
        pred_fc = pred_hvg - ctrl_hvg
        true_dev = true_fc - common_fc
        # P2.2 fix: model target = (true_hvg - common_fc) -> pred_fc IS the residual
        # estimate already. Subtracting common_fc again (the old line) shared the
        # -(ctrl_hvg+common) term between pred_dev and true_dev and inflated
        # pearson_dev/top50_dev (epoch-1 read 0.925/0.704 while conditioning dead).
        pred_dev = pred_fc
        mse_l.extend(np.mean((pred_fc - true_fc) ** 2, axis=1))
        mse_dev_l.extend(np.mean((pred_dev - true_dev) ** 2, axis=1))
        bm_c.extend(np.mean((-true_fc) ** 2, axis=1))
        for b in range(len(true_fc)):
            t, p = true_fc[b], pred_fc[b]
            pear_l.append(float(np.corrcoef(t, p)[0, 1]) if t.std() > 1e-6 and p.std() > 1e-6 else 0.0)
            k = min(50, len(t))
            top_l.append(len(set(np.argsort(-np.abs(t))[:k]) & set(np.argsort(-np.abs(p))[:k])) / k)
            # P2.1 dev metrics: 去共享成分后的判别力（不被共性响应灌水）
            td, pd_ = true_dev[b], pred_dev[b]
            dev_pear.append(float(np.corrcoef(td, pd_)[0, 1])
                            if td.std() > 1e-6 and pd_.std() > 1e-6 else 0.0)
            dev_top.append(len(set(np.argsort(-np.abs(td))[:k]) & set(np.argsort(-np.abs(pd_))[:k])) / k)
        tgt = batch["pert_emb"].numpy()
        pe = pred_embs.float().mean(dim=1).cpu().numpy()
        cos_l.extend(np.sum(pe * tgt, axis=1) /
                     (np.linalg.norm(pe, axis=1) * np.linalg.norm(tgt, axis=1) + 1e-8))
    return dict(mse_DE=float(np.mean(mse_l)), pearson_delta=float(np.mean(pear_l)),
                top50_deg_overlap=float(np.mean(top_l)), fm_cosine=float(np.mean(cos_l)),
                mse_dev=float(np.mean(mse_dev_l)),
                pearson_dev=float(np.mean(dev_pear)),
                top50_dev=float(np.mean(dev_top)),
                n_eval=len(mse_l))


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    print(f"Device: {device}", flush=True)

    sys.path.insert(0, args.map_repo)
    from omegaconf import OmegaConf  # noqa: E402
    from model_kd2 import MAPmodelKD2  # noqa: E402
    from ds_knockdown import build_symbol2row, load_kd_datasets, make_hvg_list  # noqa: E402
    from ds_knockdown2 import KD2Dataset, collate_kd2  # noqa: E402
    from rna_encoder import load_fasta_symbol_seqs  # noqa: E402

    sym2row, _ = build_symbol2row(args.esm_table)
    kd = load_kd_datasets(args.data_dirs, sym2row, min_cells=args.min_cells)
    hvg_rows = make_hvg_list(kd, n_hvg=args.n_hvg)
    train_items, test_items = split_perturbations(kd["pert_index"], args.test_frac, args.seed)
    print(f"perts: train={len(train_items)} test={len(test_items)}", flush=True)

    # ---- P2.1: common response core（仅从 train split 估计，防泄漏）----
    def true_hvg_of(di, cond):
        rows = kd["row_of_gene"][di]
        sel = np.where(kd["pert_labels"][di] == cond)[0]
        mf = kd["X_pert"][di][sel].mean(axis=0)
        return np.array([mf[np.where(rows == int(hr))[0][0]]
                         if (rows == int(hr)).any() else 0.0 for hr in hvg_rows])

    ctrl_hvg_ds = []
    for di in range(len(kd["X_ctrl"])):
        rows = kd["row_of_gene"][di]
        cm = kd["X_ctrl"][di].mean(axis=0)
        ctrl_hvg_ds.append(np.array([cm[np.where(rows == int(hr))[0][0]]
                                     if (rows == int(hr)).any() else 0.0
                                     for hr in hvg_rows]))
    ctrl_hvg_global = np.mean(np.stack(ctrl_hvg_ds), axis=0)
    common_fc = np.mean(np.stack([true_hvg_of(di, c) - ctrl_hvg_global
                                  for di, c in train_items]), axis=0)
    print(f"[P2.2 protocol] common fc (train-only): std={common_fc.std():.4f} "
          f"max|.|={np.abs(common_fc).max():.4f} — targets become fc minus this core",
          flush=True)
    common_fc_t = torch.tensor(common_fc, dtype=torch.float32)

    symbol_map, ensembl_map = load_fasta_symbol_seqs(args.fasta)
    from ds_knockdown import KnockdownDataset  # noqa: E402
    base_train = KnockdownDataset(kd, train_items, sym2row, set_size=args.set_size,
                                  L=args.sent_len, hvg_rows=hvg_rows, is_train=True, seed=args.seed)
    base_test = KnockdownDataset(kd, test_items, sym2row, set_size=args.set_size,
                                 L=args.sent_len, hvg_rows=hvg_rows, is_train=False, seed=args.seed)
    ds_train = KD2Dataset(base_train, symbol_map, ensembl_map, rna_max_len=args.rna_max_len)
    ds_test = KD2Dataset(base_test, symbol_map, ensembl_map, rna_max_len=args.rna_max_len)

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_kd2, drop_last=True)
    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collate_kd2)

    cfg = OmegaConf.load(args.se_config)
    rna_state = torch.load(args.rna_encoder_ckpt, map_location="cpu", weights_only=False)["rna_encoder"]
    model = MAPmodelKD2(se_ckpt=args.se_ckpt, se_cfg=cfg, esm_table_path=args.esm_table,
                        rna_encoder_state=rna_state, hvg_info=None, freeze_se=True)
    # load P1 ckpt (rna/net modules are new -> strict=False, assert nothing unexpected)
    p1 = torch.load(args.p1_ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(p1["model_state_dict"], strict=False)
    rna_new = [k for k in missing if k.startswith("rna_") or k.startswith("net_proj")]
    assert not unexpected, f"unexpected P1 keys: {unexpected[:5]}"
    assert len(missing) == len(rna_new), f"unexpected missing: {missing[:5]}"
    print(f"loaded P1 ckpt (epoch {p1.get('epoch')}): rna modules fresh, rest restored", flush=True)
    model = model.to(device)
    model.pert_token_scale = args.pert_token_scale
    print(f"pert_token_scale = {args.pert_token_scale}", flush=True)
    # P2.2: STRING 网络轴
    if args.string_neighbors and os.path.exists(args.string_neighbors):
        model.set_neighbor_table(np.load(args.string_neighbors), device)
        n_cov = int((model.neighbor_table >= 0).any(dim=1).sum())
        print(f"[P2.2] neighbor table loaded: {tuple(model.neighbor_table.shape)} "
              f"(network axis ON, coverage {n_cov}/{model.neighbor_table.shape[0]})", flush=True)
    else:
        print("[P2.2] neighbor table NOT provided — network axis OFF", flush=True)

    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if name.startswith("pert_model.") and "kg_smiles_encoder" not in name:
            p.requires_grad = True
        if name.startswith("rna_encoder.") or name.startswith("rna_projector."):
            p.requires_grad = True
        if name.startswith("net_proj."):
            p.requires_grad = True
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params total={total/1e6:.1f}M trainable={trainable/1e6:.1f}M", flush=True)

    new_params = [p for n, p in model.named_parameters()
                  if p.requires_grad and ("rna_" in n or "kd_projector" in n
                                          or "net_proj" in n)]
    base_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and "rna_" not in n and "kd_projector" not in n
                   and "net_proj" not in n]
    optimizer = torch.optim.AdamW([
        dict(params=base_params, lr=args.lr),
        dict(params=new_params, lr=args.lr_new),
    ], weight_decay=0.01)
    total_steps = args.epochs * (len(train_loader) // max(1, 1))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps), eta_min=args.lr * 0.05)

    loss_emb, loss_hvg = nn.MSELoss(), nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    print("precomputing SE target embeddings...", flush=True)
    emb_targets = precompute_pert_emb_targets(model, kd, kd["pert_index"], device)
    ds_train.emb_targets = emb_targets
    ds_test.emb_targets = emb_targets

    def wrap_loader(loader, dataset):
        for batch in loader:
            batch["pert_emb"] = torch.stack([
                torch.from_numpy(dataset.emb_targets[(batch["ds"][i], batch["condition"][i])])
                for i in range(len(batch["ds"]))])
            yield batch

    log_path = os.path.join(args.out_dir, "train_log.jsonl")
    best_mse = best_cos = None
    ref = ("P2.1 targets = fc deviation from train-only common core; "
           "P1 ref mse 0.0215 / P2 ref cosine 0.9710 (both shared-response-inflated)")
    common_fc_dev = common_fc_t.to(device)
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        t0, run = time.time(), []
        for batch in train_loader:
            ctrl_gene = batch["control_gene_ids"].to(device, non_blocking=True)
            ctrl_expr = batch["control_expressions"].to(device, non_blocking=True)
            pert_row = batch["pert_row"].to(device)
            rna_tok = batch["rna_tokens"].to(device)
            rna_mask = batch["rna_mask"].to(device)
            # P2.1: target = full fc minus common core → 模型必须用 pert token 才能降 loss
            true_hvg = batch["pert_hvg"].to(device) - common_fc_dev
            tgt_emb = torch.stack([
                torch.from_numpy(emb_targets[(batch["ds"][i], batch["condition"][i])])
                for i in range(len(batch["ds"]))]).to(device)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
                pred_embs, pred_hvgs = model(ctrl_gene, ctrl_expr, pert_row, rna_tok, rna_mask)
                e_loss = loss_emb(pred_embs.float().mean(dim=1), tgt_emb)
                h_loss = loss_hvg(pred_hvgs.float().mean(dim=1), true_hvg)
                loss = e_loss + args.hvg_loss_weight * h_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                           args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            sched.step()
            global_step += 1
            run.append(float(loss.item()))

        msg = dict(epoch=epoch + 1, step=global_step,
                   train_loss=float(np.mean(run)) if run else None,
                   secs=round(time.time() - t0, 1))
        print(json.dumps(msg), flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(msg) + "\n")

        if (epoch + 1) % args.eval_every_epochs == 0 or epoch == args.epochs - 1:
            res = evaluate(model, wrap_loader(test_loader, ds_test), device,
                           hvg_rows, kd, common_fc)
            res.update(epoch=epoch + 1, step=global_step)
            print("[EVAL]", json.dumps(res), f"({ref})", flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps({"eval": res}) + "\n")
            ck = {"model_state_dict": model.state_dict(), "epoch": epoch + 1,
                  "metrics": res, "hvg_rows": hvg_rows.tolist()}
            if best_mse is None or res["mse_DE"] < best_mse:
                best_mse = res["mse_DE"]
                torch.save(ck, os.path.join(args.out_dir, "ckpt_best_mse.pt"))
                print(f"💾 saved best-mse (mse_DE={best_mse:.4f})", flush=True)
            if best_cos is None or res["fm_cosine"] > best_cos:
                best_cos = res["fm_cosine"]
                torch.save(ck, os.path.join(args.out_dir, "ckpt_best_cosine.pt"))
                print(f"💾 saved best-cosine (fm_cosine={best_cos:.4f})", flush=True)

        if args.stop_after_epoch and (epoch + 1) >= args.stop_after_epoch:
            print(f"stop_after_epoch={args.stop_after_epoch} reached", flush=True)
            break

    torch.save({"model_state_dict": model.state_dict(), "epoch": args.epochs,
                "hvg_rows": hvg_rows.tolist()},
               os.path.join(args.out_dir, "ckpt_last.pt"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
