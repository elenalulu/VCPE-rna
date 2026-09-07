"""VCPE-rna cache export v2: batch inference with the P2.3-B deviation head.

No SE / MAP repo / env vars required. Loads ckpt_p3_best_dev.pt, runs the
per-gene deviation head over ALL ESM2-table genes (19,790) in the K562
(adamson) control context, and writes a cache aligned to the AIDO schema:

  genes[sym] = {top_up, top_down}  ranked by the RESIDUAL (common core
               subtracted, AIDO methodology) so per-perturbation lists are
               gene-specific; values are residual fc.
  common_response_core = the shared component (train-only common_fc).
  metadata: source / model / cell_type / license (MIT|Apache, no GenBio).

Runtime: ~5-10 min on GPU (h5ad load + RNA embeddings + tiny-head inference).
"""
import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
for _sib in ("maprna_p1", "maprna_p2"):
    _p = _HERE.parent / _sib
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ds_knockdown import build_symbol2row, load_kd_datasets  # noqa: E402
from model_dev import DeviationModel, load_esm_matrix  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adamson_h5ad", type=str, required=True,
                   help="adamson perturb_processed.h5ad (K562 control context)")
    p.add_argument("--ctrl_h5ad", type=str, default=None,
                   help="control-context h5ad; default = adamson_h5ad")
    p.add_argument("--context_name", type=str, default="k562a",
                   help="k562a / k562g / jurkat / hepg2")
    p.add_argument("--esm_table", type=str, required=True)
    p.add_argument("--string_neighbors", type=str, default="")
    p.add_argument("--rna_encoder_ckpt", type=str, default="")
    p.add_argument("--fasta", type=str, default="")
    p.add_argument("--ckpt", type=str, required=True, help="ckpt_p3_best_dev.pt")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--top_k", type=int, default=24)
    p.add_argument("--n_sig_threshold", type=float, default=0.05,
                   help="|residual fc| above this counts as a significant response gene")
    p.add_argument("--batch_genes", type=int, default=256)
    return p.parse_args()


def batched_rna_embeddings(enc, symbol_map, ensembl_map, row2sym, device, batch=64):
    """RNA-encoder embedding for every ESM2 table row; CLS-only fallback if no seq."""
    from rna_encoder import encode_seq
    V = len(row2sym)
    embs = np.zeros((V, 256), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for lo in range(0, V, batch):
            hi = min(lo + batch, V)
            toks, msks = [], []
            for r in range(lo, hi):
                sym = row2sym.get(r)
                seq = None
                if sym:
                    seq = symbol_map.get(sym) or ensembl_map.get(sym) \
                        or symbol_map.get(sym + "1")
                if seq:
                    ids, m = encode_seq(seq, 600)
                else:
                    ids, m = [6] + [0] * 599, [1] + [0] * 599  # CLS + PAD
                toks.append(ids)
                msks.append(m)
            tok = torch.tensor(toks, device=device)
            msk = torch.tensor(msks, device=device).bool()
            e = enc(tok, msk).float().cpu().numpy()
            embs[lo:hi] = e
            if (hi // batch) % 50 == 0:
                print(f"[rna] {hi}/{V} ({time.time()-t0:.0f}s)", flush=True)
    return embs


def assemble_cache(pred_dev, common_fc, row2sym, hvg_rows, args, extra_meta):
    """AIDO-schema assembly: rank by RESIDUAL (common core subtracted)."""
    genes = {}
    n_sig_all, rmax_all = [], []
    core_up_idx = np.argsort(-common_fc)[: args.top_k]
    core_down_idx = np.argsort(common_fc)[: args.top_k]
    core = {"note": ("shared response component (train-only common core), "
                     "subtracted before per-gene ranking"),
            "top_up": [[row2sym.get(int(hvg_rows[i]), f"row{int(hvg_rows[i])}"),
                        round(float(common_fc[i]), 6)] for i in core_up_idx],
            "top_down": [[row2sym.get(int(hvg_rows[i]), f"row{int(hvg_rows[i])}"),
                          round(float(common_fc[i]), 6)] for i in core_down_idx]}
    for r in range(pred_dev.shape[0]):
        sym = row2sym.get(r)
        if not sym:
            continue
        dev = pred_dev[r]
        up_idx = np.argsort(-dev)[: args.top_k]
        down_idx = np.argsort(dev)[: args.top_k]
        n_sig = int((np.abs(dev) >= args.n_sig_threshold).sum())
        rmax = float(np.percentile(np.abs(dev), 99))
        n_sig_all.append(n_sig)
        rmax_all.append(rmax)
        genes[sym] = {
            "top_up": [[row2sym.get(int(hvg_rows[i]), f"row{int(hvg_rows[i])}"),
                        round(float(dev[i]), 6)] for i in up_idx],
            "top_down": [[row2sym.get(int(hvg_rows[i]), f"row{int(hvg_rows[i])}"),
                          round(float(dev[i]), 6)] for i in down_idx],
            "n_sig": n_sig,
            "robust_max": round(rmax, 6),
        }
    cache = {
        "genes": genes,
        "supported_genes": sorted(genes.keys()),
        "n_genes_modeled": len(genes),
        "common_response_core": core,
        "source": "VCPE-rna P2.3-B (per-gene deviation head, residual targets, "
                  "train-only common core; adamson+norman+replogle Perturb-seq)",
        "model": "VCPE-rna deviation head 5.7M (ESM2 mechanism axis + STRING "
                 "network axis + RNA sequence axis), MIT/Apache licensed",
        "cell_type": f"{args.context_name} control context; residual = full fc minus common core",
        "license": "MIT/Apache (no third-party non-commercial components)",
        "hvg_space": "2000-HVG residual fc",
    }
    cache.update(extra_meta)
    print(f"[stats] n_sig: mean {np.mean(n_sig_all):.1f} median {np.median(n_sig_all):.0f} | "
          f"robust_max: mean {np.mean(rmax_all):.4f} max {np.max(rmax_all):.4f}", flush=True)
    return cache


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    hvg_rows = np.asarray(ck["hvg_rows"], dtype=np.int64)
    common_fc = np.asarray(ck["common_fc"], dtype=np.float32)
    print(f"loaded ckpt (epoch {ck['epoch']}, pearson_dev={ck['metrics']['pearson_dev']:.4f})", flush=True)

    sym2row, _ = build_symbol2row(args.esm_table)
    row2sym = {v: k for k, v in sym2row.items()}
    esm = load_esm_matrix(args.esm_table)

    rna_enc = None
    if args.rna_encoder_ckpt:
        from rna_encoder import RNAEncoder, load_fasta_symbol_seqs
        rna_enc = RNAEncoder(d_model=256, max_len=600).to(device).eval()
        state = torch.load(args.rna_encoder_ckpt, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "rna_encoder" in state:
            state = state["rna_encoder"]
        rna_enc.load_state_dict(state)
        symbol_map, ensembl_map = load_fasta_symbol_seqs(args.fasta)
        print("[rna] precomputing embeddings for all table rows...", flush=True)
        rna_embs = batched_rna_embeddings(rna_enc, symbol_map, ensembl_map,
                                          row2sym, device)
        rna_embs_t = torch.from_numpy(rna_embs)
    else:
        rna_embs_t = None

    # n_ds 从 ckpt 的 ds_emb 权重形状自动推导（v3 训练 6 数据集、GSE293987 并入后 7，
    # 写死会在 load_state_dict 时 shape mismatch）
    n_ds = int(ck["model_state_dict"]["ds_emb.weight"].shape[0])
    print(f"[ckpt] n_ds={n_ds} (derived from ds_emb.weight)", flush=True)
    model = DeviationModel(esm, hvg_rows, n_ds=n_ds, rna_encoder=rna_enc).to(device).eval()
    model.load_state_dict(ck["model_state_dict"])
    if args.string_neighbors and os.path.exists(args.string_neighbors):
        model.set_neighbor_table(np.load(args.string_neighbors), device)
        cov = int((model.neighbor_table >= 0).any(dim=1).sum())
        print(f"[P2.3-B] neighbor table: {tuple(model.neighbor_table.shape)} "
              f"(coverage {cov}/{model.neighbor_table.shape[0]})", flush=True)

    # K562 (adamson) control context: dataset-level ctrl mean over HVG, z-scored
    # with the SAME per-dataset stats used in training
    ctrl_h5ad = args.ctrl_h5ad or args.adamson_h5ad
    kd = load_kd_datasets([ctrl_h5ad], sym2row, min_cells=3)
    rows0 = kd["row_of_gene"][0]
    cm = kd["X_ctrl"][0].mean(axis=0)
    ctrl_raw = np.zeros(len(hvg_rows), dtype=np.float32)
    row2col = {int(r): c for c, r in enumerate(rows0)}
    for hi, hr in enumerate(hvg_rows):
        c = row2col.get(int(hr))
        if c is not None:
            ctrl_raw[hi] = cm[c]
    mu, sd = ctrl_raw.mean(), ctrl_raw.std() + 1e-6
    ctrl_feat = torch.from_numpy(((ctrl_raw - mu) / sd)[None, :].repeat(1, 0)).float()
    print(f"[ctx] {args.context_name} ctrl features ready (hvg {len(hvg_rows)})", flush=True)

    V = esm.shape[0]
    pred_dev = np.zeros((V, len(hvg_rows)), dtype=np.float32)
    ds0 = torch.zeros(args.batch_genes, dtype=torch.long)
    cf = ctrl_feat.to(device)
    # v2-1a: 每个候选靶基因的 ctrl 表达（raw log1p，self-response 门控）。
    # 靶基因在语境 panel 内 -> 该列 ctrl 均值；panel 外 -> 0.0（近零，硬压，
    # 与 v2-0a 后处理的外部 HPA 谱系门控口径一致）。
    pert_expr_all = np.zeros(V, dtype=np.float32)
    for vi in range(V):
        c = row2col.get(int(vi))
        if c is not None:
            pert_expr_all[vi] = float(cm[c])
    pe = torch.from_numpy(pert_expr_all).to(device)
    t0 = time.time()
    with torch.no_grad():
        for lo in range(0, V, args.batch_genes):
            hi = min(lo + args.batch_genes, V)
            n = hi - lo
            pr = torch.arange(lo, hi, device=device)
            dsi = ds0[:n].to(device)
            cf_b = cf.expand(n, -1)
            rna = rna_embs_t[lo:hi].to(device) if rna_embs_t is not None else None
            pred_dev[lo:hi] = model(pr, dsi, cf_b, rna_emb=rna,
                                    pert_ctrl_expr=pe[lo:hi]).float().cpu().numpy()
            if (hi // args.batch_genes) % 20 == 0:
                print(f"[infer] {hi}/{V} genes ({time.time()-t0:.0f}s)", flush=True)

    amp = np.abs(pred_dev).max(axis=1)
    print(f"[check] residual amplitude: median max|dev| {np.median(amp):.4f} "
          f"(AIDO 同口径残差参照 ~0.4；修复前塌缩版 ~0.003)", flush=True)

    extra = {"export_protocol": "residual ranking (common core subtracted), "
                                "values = residual fc",
             "ckpt": os.path.basename(args.ckpt),
             "ckpt_epoch": ck["epoch"]}
    cache = assemble_cache(pred_dev, common_fc, row2sym, hvg_rows, args, extra)
    with gzip.open(args.out, "wt") as f:
        json.dump(cache, f)
    mb = os.path.getsize(args.out) / 1e6
    print(f"✅ wrote {args.out} ({mb:.1f} MB): {len(cache['genes'])} genes, "
          f"top_k={args.top_k}, residual ranking", flush=True)


if __name__ == "__main__":
    main()
