"""VCPE-rna cache exporter — batch inference over all supported genes.

Produces `vcpe_cache.json.gz` with the SAME schema as the platform's
`aido_vc_cache.json.gz` (per-gene top_up/top_down [[gene, fc]...x24] +
n_sig + robust_max, common_response_core, supported_genes, metadata), so the
platform's virtual_cell.py can switch caches with ZERO code/dependency changes
(the platform itself never runs the model — everything is precomputed here).

Run ONCE on the GPU machine (3090, ~40-60 min):
  cd $BASE/MAP-KG-main/MAP
  python $BASE/src/export_vcpe_cache.py \
    --data_dirs $BASE/data/adamson/perturb_processed.h5ad \
                $BASE/data/norman/perturb_processed.h5ad \
                $BASE/data/replogle_rpe1_essential/perturb_processed.h5ad \
    --esm_table $BASE/weights/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt \
    --se_ckpt $BASE/weights/se600m.safetensors --se_config configs/se600m.yaml \
    --map_repo $BASE/MAP-KG-main/MAP \
    --p2_ckpt $BASE/p2_out/ckpt_best_cosine.pt \
    --rna_encoder_ckpt $BASE/p2_align/rna_encoder_aligned.pt \
    --fasta $BASE/data/gene_transcripts.fa \
    --out $BASE/vcpe_cache.json.gz
"""
import os
import sys
import json
import gzip
import argparse
from datetime import date

import numpy as np
import torch

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
    p.add_argument("--p2_ckpt", type=str, required=True,
                   help="P2 checkpoint (ckpt_best_cosine.pt recommended)")
    p.add_argument("--rna_encoder_ckpt", type=str, required=True)
    p.add_argument("--fasta", type=str, required=True)
    p.add_argument("--out", type=str, default="./vcpe_cache.json.gz")
    p.add_argument("--sent_len", type=int, default=2048)
    p.add_argument("--n_hvg", type=int, default=2000)
    p.add_argument("--rna_max_len", type=int, default=600)
    p.add_argument("--ctrl_set_size", type=int, default=4)
    p.add_argument("--batch_genes", type=int, default=32)
    p.add_argument("--top_k", type=int, default=24, help="entries per top_up/top_down list (AIDO uses 24)")
    p.add_argument("--n_sig_threshold", type=float, default=0.05,
                   help="|fc| above this counts toward n_sig")
    p.add_argument("--min_cells", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="smoke test: only first N genes")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    print(f"Device: {device}", flush=True)

    sys.path.insert(0, args.map_repo)
    from omegaconf import OmegaConf  # noqa: E402
    from model_kd2 import MAPmodelKD2  # noqa: E402
    from ds_knockdown import build_symbol2row, load_kd_datasets, make_hvg_list  # noqa: E402
    from rna_encoder import encode_seq, load_fasta_symbol_seqs, lookup_seq, RNAVocab  # noqa: E402

    # ---- data (ctrl reference + HVG selection) ----
    sym2row, _ = build_symbol2row(args.esm_table)
    esm_tab = torch.load(args.esm_table, map_location="cpu", weights_only=False)
    keys = list(esm_tab.keys())                      # row i <-> keys[i]
    row2sym = {i: k for i, k in enumerate(keys)}
    supported = keys                                 # every table row is a conditionable gene
    kd = load_kd_datasets(args.data_dirs, sym2row, min_cells=args.min_cells)
    hvg_rows = make_hvg_list(kd, n_hvg=args.n_hvg)
    hvg_syms = [row2sym[int(r)] for r in hvg_rows]
    print(f"HVG genes: {len(hvg_syms)}", flush=True)

    # ctrl baseline in HVG space (per-dataset ctrl mean at HVG rows, then averaged)
    ctrl_hvg_ds = []
    for di in range(len(kd["X_ctrl"])):
        rows = kd["row_of_gene"][di]
        cm = kd["X_ctrl"][di].mean(axis=0)
        vec = np.array([cm[np.where(rows == int(hr))[0][0]]
                        if (rows == int(hr)).any() else 0.0 for hr in hvg_rows])
        ctrl_hvg_ds.append(vec)
    ctrl_hvg = np.mean(np.stack(ctrl_hvg_ds), axis=0)

    # ---- control sentences: REAL control cells (training-distribution input) ----
    # P1/P2 训练与评估的控制句都是单个真实控制细胞的 top-expressed 谱；全局均值谱是
    # 分布外输入，会让预测退化成均值回归（2026-09-02 抽检实测：fc 幅值塌缩 50 倍、
    # 不同扰动响应雷同）。因此固定用 K562（adamson，训练分布内）真实控制细胞。
    rng_ctrl = np.random.default_rng(args.seed + 1)
    di0 = 0  # adamson = K562
    rows0 = kd["row_of_gene"][di0]
    pool0 = kd["X_ctrl"][di0]
    picks = rng_ctrl.choice(pool0.shape[0], args.ctrl_set_size, replace=False)
    sent_ids, sent_exs = [], []
    for pi in picks:
        x = pool0[pi]
        m = rows0 >= 0
        gid, ex = rows0[m], x[m]
        order = np.argsort(-ex)[: args.sent_len - 1]
        sent_ids.append(np.concatenate([[0], gid[order]]).astype(np.int32))
        sent_exs.append(np.concatenate([[0.0], ex[order]]).astype(np.float32))
    S = args.ctrl_set_size
    ctrl_gene = torch.from_numpy(np.stack(sent_ids)).unsqueeze(0)          # [1, S, L]
    ctrl_expr = torch.from_numpy(np.stack(sent_exs)).unsqueeze(0).float()  # [1, S, L]
    print(f"control sentences: {S} real K562 ctrl cells "
          f"(picks={picks.tolist()}, training-distribution input)", flush=True)

    # ---- model (P2 dual-axis) ----
    cfg = OmegaConf.load(args.se_config)
    rna_state = torch.load(args.rna_encoder_ckpt, map_location="cpu", weights_only=False)["rna_encoder"]
    model = MAPmodelKD2(se_ckpt=args.se_ckpt, se_cfg=cfg, esm_table_path=args.esm_table,
                        rna_encoder_state=rna_state, hvg_info=None, freeze_se=True)
    p2 = torch.load(args.p2_ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(p2["model_state_dict"], strict=False)
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    assert all(k.startswith("rna_") for k in missing), f"unexpected missing: {missing[:5]}"
    model = model.to(device).eval()
    print(f"loaded P2 ckpt (epoch {p2.get('epoch')}): "
          f"metrics={p2.get('metrics')}", flush=True)

    # ---- RNA tokens per supported gene (CLS fallback when transcript missing) ----
    symbol_map, ensembl_map = load_fasta_symbol_seqs(args.fasta)
    n_seq = 0
    tok_mat = np.zeros((len(supported), args.rna_max_len), dtype=np.int64)
    mask_mat = np.zeros((len(supported), args.rna_max_len), dtype=np.int64)
    for i, sym in enumerate(supported):
        seq = lookup_seq(sym, symbol_map, ensembl_map)
        if seq:
            ids, m = encode_seq(seq, args.rna_max_len)
            tok_mat[i], mask_mat[i] = ids, m
            n_seq += 1
        else:
            tok_mat[i, 0], mask_mat[i, 0] = RNAVocab.CLS, 1
    print(f"transcript coverage: {n_seq}/{len(supported)} "
          f"({len(supported) - n_seq} CLS-only, mechanism axis only)", flush=True)

    if args.limit:
        supported = supported[: args.limit]
        tok_mat, mask_mat = tok_mat[: args.limit], mask_mat[: args.limit]
        print(f"[smoke] limit={args.limit}", flush=True)

    # ---- batched inference ----
    N = len(supported)
    fc_mat = np.zeros((N, len(hvg_rows)), dtype=np.float32)
    n_batches = (N + args.batch_genes - 1) // args.batch_genes
    for bi in range(n_batches):
        lo, hi = bi * args.batch_genes, min((bi + 1) * args.batch_genes, N)
        pert_rows = torch.tensor([sym2row[s] for s in supported[lo:hi]], dtype=torch.long)
        tok = torch.from_numpy(tok_mat[lo:hi]).to(device)
        msk = torch.from_numpy(mask_mat[lo:hi]).bool().to(device)
        pr = pert_rows.to(device)
        # one global control sentence shared by every gene in the batch:
        # expand ctrl batch dim to match the pert batch dim (B == Bp is a
        # model invariant — P1/P2 eval always paired them 1:1)
        bg = hi - lo
        cg = ctrl_gene.expand(bg, -1, -1).to(device)
        ce = ctrl_expr.expand(bg, -1, -1).to(device)
        with torch.no_grad():
            _, pred_hvgs = model(cg, ce, pr, tok, msk)
        pred_hvg = pred_hvgs.float().mean(dim=1).cpu().numpy()          # [B, n_hvg]
        fc_mat[lo:hi] = pred_hvg - ctrl_hvg
        if (bi + 1) % 20 == 0 or bi == n_batches - 1:
            print(f"[infer] {hi}/{N} genes", flush=True)

    # ---- common response core (shared across ALL perturbations) ----
    core_fc = fc_mat.mean(axis=0)
    fc_sub = fc_mat - core_fc                                           # subtract shared response
    core_top_up = [[hvg_syms[j], round(float(core_fc[j]), 4)]
                   for j in np.argsort(-core_fc)[: args.top_k]]
    core_top_down = [[hvg_syms[j], round(float(core_fc[j]), 4)]
                     for j in np.argsort(core_fc)[: args.top_k]]

    # ---- assemble cache (AIDO schema) ----
    genes = {}
    for i, sym in enumerate(supported):
        fc = fc_sub[i]
        order_up = np.argsort(-fc)[: args.top_k]
        order_dn = np.argsort(fc)[: args.top_k]
        top_up = [[hvg_syms[j], round(float(fc[j]), 4)] for j in order_up if fc[j] > 0]
        top_down = [[hvg_syms[j], round(float(fc[j]), 4)] for j in order_dn if fc[j] < 0]
        genes[sym] = {
            "top_up": top_up,
            "top_down": top_down,
            "n_sig": int((np.abs(fc) > args.n_sig_threshold).sum()),
            "robust_max": round(float(np.abs(fc).max()), 4),
        }

    cache = {
        "genes": genes,
        "supported_genes": sorted(supported),
        "n_genes_modeled": len(supported),
        "contrastive": False,
        "common_response_core": {
            "note": ("generic response shared by ALL modeled perturbations "
                     "(mean fc over genes, subtracted from per-gene top lists "
                     "before ranking; methodology mirrors the AIDO cache)"),
            "top_up": core_top_up,
            "top_down": core_top_down,
        },
        "source": ("VCPE-rna (MAP MIT/Apache base + self-trained dual-axis conditioning: "
                   "mechanism axis ESM2+KG, sequence axis self-trained RNA encoder)"),
        "model": ("VCPE-rna P2 dual-axis predictor "
                  f"(ckpt_best_cosine epoch {p2.get('epoch')}; P2 eval fm_cosine 0.9710 on held-out)"),
        "cell_type": "joint (K562 adamson+norman, RPE1 replogle) - predicted, dual-axis",
        "license": "MIT/Apache (no GenBio dependency)",
        "exported": str(date.today()),
        "response_space": f"HVG-{len(hvg_rows)} (top-variance genes, ESM2-table rows)",
        "transcript_coverage": f"{n_seq}/{len(supported)} with RNA transcripts",
    }
    with gzip.open(args.out, "wt") as f:
        json.dump(cache, f)
    sz = os.path.getsize(args.out) / 1e6
    print(f"✅ wrote {args.out} ({sz:.1f} MB): {len(genes)} genes, "
          f"top_k={args.top_k}, common core subtracted", flush=True)
    print("Platform swap: replace aido_vc_cache.json.gz with this file "
          "+ update attribution strings (see P3_接入分析.md).", flush=True)


if __name__ == "__main__":
    main()
