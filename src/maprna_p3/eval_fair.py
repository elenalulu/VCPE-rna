"""Fair-comparison evaluation for the expanded response head.

Loads ckpt_p3_best_dev.pt (ep9 of the expanded run), rebuilds the same
deterministic split, and reports pearson_dev stratified by source dataset,
plus the 'legacy-domain' subset (test perts from adamson/norman/replogle)
as the distribution-level fair comparison against the old 0.3079 baseline.

Run on GPU host:  python $BASE/src/eval_fair.py --ckpt $BASE/p3_expanded/ckpt_p3_best_dev.pt ...
(same data args as train_p3)
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_p3 import build_dev_data, make_rna_emb_lookup  # noqa: E402
from model_dev import DeviationModel  # noqa: E402

DS_NAMES = ["adamson", "norman", "replogle_ess", "gwps", "jurkat", "hepg2"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dirs", nargs="+", required=True)
    p.add_argument("--esm_table", required=True)
    p.add_argument("--string_neighbors", default="")
    p.add_argument("--rna_encoder_ckpt", default="")
    p.add_argument("--fasta", default="")
    p.add_argument("--n_hvg", type=int, default=2000)
    p.add_argument("--test_frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min_cells", type=int, default=3)
    p.add_argument("--batch", type=int, default=128)
    args = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = build_dev_data(args)
    if args.rna_encoder_ckpt:
        data["rna_embs"] = make_rna_emb_lookup(args, data)
        print(f"[eval] rna embeddings: {len(data['rna_embs'])} perts", flush=True)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    # 一致性守卫：重建的 HVG panel / common_fc 必须与 ckpt 训练时一致，
    # 否则第一层权重与基因对不上（表现为全域 pearson_dev 崩到 ~0.08 的假回退）。
    # 规则：eval 的 --data_dirs 必须与训练完全相同（顺序一致）。
    if "hvg_rows" in ck:
        if not np.array_equal(np.asarray(ck["hvg_rows"], dtype=np.int64),
                              np.asarray(data["hvg_rows"], dtype=np.int64)):
            raise RuntimeError(
                "重建的 HVG panel 与 ckpt 不一致！eval 的 --data_dirs 必须与该 ckpt "
                "训练时完全相同（含 GSE293987 等 7 数据集 ckpt 就要给 7 个目录，顺序一致）。"
                f"ckpt panel n={len(ck['hvg_rows'])} vs 重建 n={len(data['hvg_rows'])}")
    if "common_fc" in ck:
        cf_ck = np.asarray(ck["common_fc"], dtype=np.float32)
        if not np.allclose(cf_ck, data["common_fc"], atol=1e-5):
            print("[warn] common_fc 与 ckpt 不一致，已用 ckpt 值覆盖残差目标", flush=True)
            data["common_fc"] = cf_ck
            data["dev_tr"] = data["fc_tr"] - cf_ck
            data["dev_te"] = data["fc_te"] - cf_ck
    print(f"loaded ckpt: epoch {ck['epoch']} metrics={ {k: round(v, 4) for k, v in ck['metrics'].items() if isinstance(v, float)} }",
          flush=True)

    # rebuild model exactly as train_p3 does (incl. RNA encoder branch)
    rna_enc = None
    if args.rna_encoder_ckpt:
        from rna_encoder import RNAEncoder
        rna_enc = RNAEncoder(d_model=256, max_len=600)
        state = torch.load(args.rna_encoder_ckpt, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "rna_encoder" in state:
            state = state["rna_encoder"]    # Stage-A ckpt is a wrapper dict
        rna_enc.load_state_dict(state)
    tab = torch.load(args.esm_table, map_location="cpu", weights_only=False)
    esm_matrix = torch.stack(list(tab.values())).float()
    # n_ds 从 ckpt 自动推导（eval data_dirs 可以不含全部训练数据集，
    # 如 GSE293987 并入后 7 数据集 ckpt 配 6 目录 eval）
    n_ds_ckpt = int(ck["model_state_dict"]["ds_emb.weight"].shape[0])
    print(f"[ckpt] n_ds={n_ds_ckpt} (derived; eval data n_ds={data['n_ds']})", flush=True)
    model = DeviationModel(esm_matrix, data["hvg_rows"], n_ds=n_ds_ckpt,
                           d_model=256, rna_encoder=rna_enc).to(dev).eval()
    if args.string_neighbors and os.path.exists(args.string_neighbors):
        model.set_neighbor_table(np.load(args.string_neighbors), dev)
    model.load_state_dict(ck["model_state_dict"])

    items = data["test_items"]
    ds_idx = torch.tensor([di for di, _ in items], dtype=torch.long)
    pr = torch.tensor(data["rows_te"], dtype=torch.long)
    cf = torch.from_numpy(data["ctrl_feat_all"][ds_idx.numpy()])
    rna = data["rna_embs"][len(data["train_items"]):] if model.proj_r is not None else None
    true_dev = data["dev_te"]

    # chunked inference
    outs = []
    with torch.no_grad():
        for lo in range(0, pr.shape[0], args.batch):
            hi = min(lo + args.batch, pr.shape[0])
            kw = {"rna_emb": rna[lo:hi].to(dev)} if rna is not None else {}
            outs.append(model(pr[lo:hi].to(dev), ds_idx[lo:hi].to(dev),
                              cf[lo:hi].to(dev), **kw).cpu().numpy())
    pred_dev = np.concatenate(outs)

    def pearson(a, b):
        if a.std() < 1e-9 or b.std() < 1e-9:
            return float("nan")
        return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])

    # overall
    print(f"\nOVERALL: n={len(items)} pearson_dev={pearson(pred_dev, true_dev):.4f}", flush=True)

    # stratify by source dataset
    print("\n--- stratified by source dataset (test split) ---", flush=True)
    for d in range(data["n_ds"]):
        sel = np.array([di == d for di, _ in items])
        if sel.sum() == 0:
            continue
        name = DS_NAMES[d] if d < len(DS_NAMES) else f"ds{d}"
        print(f"{name:14} n={sel.sum():5d}  pearson_dev={pearson(pred_dev[sel], true_dev[sel]):.4f}",
              flush=True)

    # legacy-domain subset (old three datasets)
    legacy = np.array([di in (0, 1, 2) for di, _ in items])
    print(f"\nLEGACY-DOMAIN (adamson/norman/replogle_ess test perts): "
          f"n={legacy.sum()} pearson_dev={pearson(pred_dev[legacy], true_dev[legacy]):.4f}", flush=True)
    print("(fair-comparison caveat: distribution-level vs old 0.3079 baseline — "
          "same source domains, not the identical 276-pert set)", flush=True)

    # gene-level signal: which domains does the model generalize to?
    new_domains = np.array([di in (3, 4, 5) for di, _ in items])
    print(f"NEW-DOMAIN (gwps/jurkat/hepg2 test perts): n={new_domains.sum()} "
          f"pearson_dev={pearson(pred_dev[new_domains], true_dev[new_domains]):.4f}", flush=True)


if __name__ == "__main__":
    main()
