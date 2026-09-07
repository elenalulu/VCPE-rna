#!/usr/bin/env python
"""ASO 域增益考卷：GSE293987 ACTN1（A431，从未训练）holdout 上对比两个 ckpt。

考卷设计（防循环论证）：两个 ckpt 都不用 GSE293987 训练——
  ckpt1 = p3_v21（6 CRISPR 数据集 + 门控，纯 CRISPR 教材）
  ckpt2 = p3_v21c（6 CRISPR + GSE183535 MYC/HeLa，ASO 域教材 ×2 条件）
考题 = 预测 ACTN1 敲低响应，真值 = GSE293987 实测（ACTN1 pseudobulk - ctrl，
NTASO 匹配对照）。预测协议两 ckpt 完全一致：
  pert=ACTN1, ds=donor(hepg2, idx=5, 上皮谱系最近), rna_emb=ACTN1 转录本编码,
  pe=hepg2 ctrl 的 ACTN1 表达（标量门控）, is_nb=STRING。
指标（在 ckpt hvg panel ∩ GSE293987 可测基因上）：
  spearman(pred_fc, measured_fc) | top50|measured| 方向一致率 | ACTN1 self pred vs 实测。

用法（GPU）：
  python eval_aso_domain.py \
    --ckpt1 $BASE/p3_v21/ckpt_p3_best_dev.pt --label1 crispr_only \
    --ckpt2 $BASE/p3_v21c/ckpt_p3_best_dev.pt --label2 aso_trained \
    --data_dirs <6 CRISPR dirs，与 p3_v21 训练一致> \
    --esm_table ... --string_neighbors ... --rna_encoder_ckpt ... --fasta ... \
    --gse $BASE/data/gse293987/gse293987_proc.h5ad
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
_HERE = os.path.abspath(HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
for _sib in ("maprna_p1", "maprna_p2", "maprna_p3"):
    _p = os.path.abspath(os.path.join(HERE, "..", _sib))
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from ds_knockdown import load_kd_datasets, build_symbol2row, resolve_pert_row  # noqa: E402
from model_dev import DeviationModel, load_esm_matrix  # noqa: E402


def load_ckpt(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    n_ds = int(ck["model_state_dict"]["ds_emb.weight"].shape[0])
    return ck, n_ds


def predict_actn1(ck, n_ds, data_dirs, donor_idx, args, sym2row, sym_by_row,
                  esm, device, tag):
    hvg_rows = np.asarray(ck["hvg_rows"], dtype=np.int64)
    common_fc = np.asarray(ck["common_fc"], dtype=np.float32)

    # donor(hepg2) ctrl over ckpt hvg panel（dataset 内 z-score，同 train_p3）
    kd = load_kd_datasets(data_dirs, sym2row, min_cells=args.min_cells)
    di = donor_idx
    r2c = {int(r): c for c, r in enumerate(kd["row_of_gene"][di])}
    cm = np.asarray(kd["X_ctrl"][di].mean(axis=0), dtype=np.float32)
    ctrl_vec = np.array([cm[r2c.get(hr, -1)] if r2c.get(hr, -1) >= 0 else 0.0
                         for hr in hvg_rows], dtype=np.float32)
    cf = ((ctrl_vec - ctrl_vec.mean()) / (ctrl_vec.std() + 1e-6))[None, :]

    # rna_encoder 必须挂回模型：ckpt 的 rna_encoder.* 是随训练微调过的权重，
    # 先以 Stage-A 初始化占位，load_state_dict 后即被 ckpt 权重覆盖
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "maprna_p2")))
    from rna_encoder import RNAEncoder, encode_seq, load_fasta_symbol_seqs, lookup_seq
    rna_enc = RNAEncoder(d_model=256, max_len=600).to(device).eval()
    state = torch.load(args.rna_encoder_ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "rna_encoder" in state:
        state = state["rna_encoder"]
    rna_enc.load_state_dict(state)

    model = DeviationModel(esm, hvg_rows, n_ds=n_ds, rna_encoder=rna_enc).to(device).eval()
    if args.string_neighbors and os.path.exists(args.string_neighbors):
        model.set_neighbor_table(np.load(args.string_neighbors), device)
    model.load_state_dict(ck["model_state_dict"])

    # ACTN1 rna emb：用 ckpt 内 fine-tuned 的 encoder（不是 Stage-A）
    symbol_map, ensembl_map = load_fasta_symbol_seqs(args.fasta)
    seq = lookup_seq("ACTN1", symbol_map, ensembl_map)
    if seq:
        ids, m = encode_seq(seq, 600)
    else:
        ids, m = [6] + [0] * 599, [1] + [0] * 599

    row = resolve_pert_row(sym2row, "ACTN1")
    # pe：ACTN1 在 donor 语境的 ctrl 表达
    hepg2_sym2col = {}
    for i, r in enumerate([int(x) for x in kd["row_of_gene"][di]]):
        s = sym_by_row.get(r)
        if s is not None:
            hepg2_sym2col[s] = i
    col = hepg2_sym2col.get(sym_by_row.get(int(row)))
    pe = float(cm[col]) if col is not None else 0.0
    print(f"[{tag}] n_ds={n_ds} | ACTN1 donor(hepg2) expr={pe:.3f} | "
          f"rna_emb={'seq' if seq else 'CLS-fallback'}", flush=True)

    with torch.no_grad():
        pr = torch.tensor([row], dtype=torch.long).to(device)
        dsi = torch.tensor([donor_idx], dtype=torch.long).to(device)
        rna_e = model.rna_encoder(torch.tensor([ids], device=device),
                                  torch.tensor([m], device=device).bool())[0].float().unsqueeze(0)  # [1,256]
        dev = model(pr, dsi, torch.from_numpy(cf).float().to(device),
                    pert_ctrl_expr=torch.tensor([pe], dtype=torch.float32).to(device),
                    rna_emb=rna_e)
    pred_fc = dev.cpu().numpy()[0] + common_fc
    return pred_fc, hvg_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt1", required=True)
    ap.add_argument("--label1", default="model1")
    ap.add_argument("--ckpt2", default="")
    ap.add_argument("--label2", default="model2")
    ap.add_argument("--data_dirs", nargs="+", required=True)
    ap.add_argument("--esm_table", required=True)
    ap.add_argument("--string_neighbors", default="")
    ap.add_argument("--rna_encoder_ckpt", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--gse", required=True, help="gse293987_proc.h5ad")
    ap.add_argument("--donor_idx", type=int, default=5, help="hepg2 在 data_dirs 中的下标")
    ap.add_argument("--min_cells", type=int, default=3)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sym2row, _ = build_symbol2row(args.esm_table)
    sym_by_row = {int(r): s for s, r in sym2row.items()}
    esm = load_esm_matrix(args.esm_table)

    # ---- 真值：GSE293987 ACTN1 measured fc（log1pTPM）----
    import anndata as ad
    a = ad.read_h5ad(args.gse, backed="r")
    gse_genes = set(map(str, a.var_names))
    obs = a.obs
    actn_idx = np.where(obs["condition"].values == "ACTN1")[0]
    ctl_idx = np.where(obs["control"].values == 1)[0]
    a.file.close()
    a = ad.read_h5ad(args.gse)
    X = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
    gse_var = list(map(str, a.var_names))
    gse_i = {g: i for i, g in enumerate(gse_var)}
    measured = X[actn_idx].mean(axis=0) - X[ctl_idx].mean(axis=0)
    row = resolve_pert_row(sym2row, "ACTN1")
    actn1_sym = sym_by_row.get(int(row))
    m_self = measured[gse_i[actn1_sym]] if actn1_sym in gse_i else float("nan")
    print(f"[truth] ACTN1 self measured fc = {m_self:+.3f} | "
          f"measured genes {len(gse_var)}", flush=True)

    out = {}
    for path, label in [(args.ckpt1, args.label1), (args.ckpt2, args.label2)]:
        if not path:
            continue
        ck, n_ds = load_ckpt(path, device)
        pred_fc, hvg_rows = predict_actn1(ck, n_ds, args.data_dirs, args.donor_idx,
                                          args, sym2row, sym_by_row, esm,
                                          device, label)
        # 对齐到 GSE293987 可测基因
        syms = [sym_by_row.get(int(r)) for r in hvg_rows]
        pred_m, meas_m = [], []
        for hi, s in enumerate(syms):
            if s in gse_i:
                pred_m.append(pred_fc[hi])
                meas_m.append(measured[gse_i[s]])
        pred_m, meas_m = np.array(pred_m), np.array(meas_m)
        from scipy.stats import spearmanr
        rho = spearmanr(pred_m, meas_m).statistic
        top = np.argsort(-np.abs(meas_m))[:50]
        agree = float(np.mean(np.sign(pred_m[top]) == np.sign(meas_m[top])))
        i_self = syms.index(actn1_sym) if actn1_sym in syms else None
        p_self = pred_fc[i_self] if i_self is not None else float("nan")
        out[label] = dict(n_genes=int(len(pred_m)), spearman=float(rho),
                          top50_direction_consistency=agree,
                          actn1_self_pred=float(p_self),
                          actn1_self_measured=float(m_self))
        print(f"\n[{label}] n={len(pred_m)} | spearman {rho:+.4f} | "
              f"top50 方向一致率 {agree:.0%} | ACTN1 self pred {p_self:+.3f} "
              f"vs measured {m_self:+.3f}", flush=True)

    if len(out) == 2:
        l1, l2 = args.label1, args.label2
        d_rho = out[l2]["spearman"] - out[l1]["spearman"]
        d_agr = out[l2]["top50_direction_consistency"] - out[l1]["top50_direction_consistency"]
        verdict = ("ASO 域训练有效" if (d_rho > 0.02 or d_agr > 0.05)
                   else "无增益" if abs(d_rho) <= 0.02 else "负增益")
        out["verdict"] = dict(delta_spearman=d_rho, delta_top50_consistency=d_agr,
                              conclusion=verdict)
        print(f"\n[verdict] Δspearman {d_rho:+.4f} | Δtop50 一致率 {d_agr:+.0%} "
              f"-> {verdict}", flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(args.ckpt1) or ".",
                                     "aso_domain_exam.json"), "w"),
              indent=2, ensure_ascii=False)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
