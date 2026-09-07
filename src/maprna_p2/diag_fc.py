"""VCPE-rna 诊断：预测区分度塌缩定位（只读模型与数据，不改任何状态）。

对一个 P2 ckpt，用三类前向对比同一批扰动目标的 pred_fc：
  A. eval-style   —— 复刻 P2 评估路径：S=8 个该扰动所属数据集的真实控制细胞
  B. export-style —— 复刻导出路径：S=1 个 K562 真实控制细胞
  C. train-style   —— 训练集内的扰动（模型见过的），eval-style 前向

每组输出：std(pred_fc) / max|pred_fc| / 对 true_fc 的 pearson。
判读：
  A 有区分度 + B 塌缩  → 输入构造差异（S 数量/控制句来源）是根因
  A、B 都塌缩          → 模型对未见扰动只输出共性响应，P2 指标被灌水，
                          需要重审评估协议（补"去共享成分后的判别力"指标）
  C 有区分度 + A 塌缩  → 模型只记住了训练扰动，未见扰动泛化失败

Run (GPU box):
  cd $BASE/MAP-KG-main/MAP
  python $BASE/src/l2/diag_fc.py \
    --data_dirs ... --esm_table ... --se_ckpt ... --se_config configs/se600m.yaml \
    --map_repo ... --p2_ckpt $BASE/p2_out/ckpt_best_cosine.pt \
    --rna_encoder_ckpt $BASE/p2_align/rna_encoder_aligned.pt \
    --fasta $BASE/data/gene_transcripts.fa
"""
import os
import sys
import argparse

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "maprna_p2"))


def build_sentence(rows, expr, sent_len, rng, n_cells=1, pool=None):
    """n_cells=1: single (rows,expr); n_cells>1: sample n_cells real cells from pool."""
    sents = []
    for _ in range(n_cells):
        if pool is not None:
            x = pool[rng.choice(pool.shape[0])]
        else:
            x = expr
        m = rows >= 0
        gid, ex = rows[m], x[m]
        order = np.argsort(-ex)[: sent_len - 1]
        sents.append((np.concatenate([[0], gid[order]]).astype(np.int32),
                      np.concatenate([[0.0], ex[order]]).astype(np.float32)))
    ids = np.stack([s[0] for s in sents])
    exs = np.stack([s[1] for s in sents])
    return (torch.from_numpy(ids).unsqueeze(0),
            torch.from_numpy(exs).unsqueeze(0).float())


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dirs", type=str, nargs="+", required=True)
    p.add_argument("--esm_table", type=str, required=True)
    p.add_argument("--se_ckpt", type=str, required=True)
    p.add_argument("--se_config", type=str, required=True)
    p.add_argument("--map_repo", type=str, required=True)
    p.add_argument("--p2_ckpt", type=str, required=True)
    p.add_argument("--rna_encoder_ckpt", type=str, required=True)
    p.add_argument("--fasta", type=str, required=True)
    p.add_argument("--n_test", type=int, default=8, help="test perts to probe")
    p.add_argument("--n_train", type=int, default=4, help="train perts to probe")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--string_neighbors", type=str, default="",
                   help="P2.2: string_neighbors.npy (network axis); empty = off")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sys.path.insert(0, args.map_repo)
    from omegaconf import OmegaConf  # noqa: E402
    from model_kd2 import MAPmodelKD2  # noqa: E402
    from ds_knockdown import (build_symbol2row, load_kd_datasets,  # noqa: E402
                              make_hvg_list, KnockdownDataset)
    from ds_knockdown2 import KD2Dataset, collate_kd2  # noqa: E402
    from rna_encoder import load_fasta_symbol_seqs  # noqa: E402
    from torch.utils.data import DataLoader  # noqa: E402

    sym2row, _ = build_symbol2row(args.esm_table)
    kd = load_kd_datasets(args.data_dirs, sym2row, min_cells=3)
    hvg_rows = make_hvg_list(kd, n_hvg=2000)
    hvg_syms = None

    # P2 split 复现（seed 0）
    rng = np.random.default_rng(0)
    by_ds = {}
    for it in kd["pert_index"]:
        by_ds.setdefault(it[0], []).append(it)
    train_items, test_items = [], []
    for di, lst in sorted(by_ds.items()):
        lst = sorted(lst)
        idx = np.random.default_rng(0).permutation(len(lst))
        n_test = max(1, int(round(len(lst) * 0.15)))
        test_items.extend([lst[i] for i in idx[:n_test]])
        train_items.extend([lst[i] for i in idx[n_test:]])

    # 挑 K562 系（adamson/norman）的 test 扰动 + 若干 train 扰动
    k562_test = [it for it in test_items if it[0] in (0, 1)][: args.n_test]
    k562_train = [it for it in train_items if it[0] in (0, 1)][: args.n_train]
    probes = [("TEST", it) for it in k562_test] + [("TRAIN", it) for it in k562_train]
    print(f"probes: {len(probes)} (test {len(k562_test)} + train {len(k562_train)})", flush=True)

    symbol_map, ensembl_map = load_fasta_symbol_seqs(args.fasta)
    cfg = OmegaConf.load(args.se_config)
    rna_state = torch.load(args.rna_encoder_ckpt, map_location="cpu", weights_only=False)["rna_encoder"]
    model = MAPmodelKD2(se_ckpt=args.se_ckpt, se_cfg=cfg, esm_table_path=args.esm_table,
                        rna_encoder_state=rna_state, hvg_info=None, freeze_se=True)
    p2 = torch.load(args.p2_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(p2["model_state_dict"], strict=False)
    model = model.to(device).eval()
    if args.string_neighbors and os.path.exists(args.string_neighbors):
        model.set_neighbor_table(np.load(args.string_neighbors), device)
        n_cov = int((model.neighbor_table >= 0).any(dim=1).sum())
        print(f"[P2.2] neighbor table loaded: {tuple(model.neighbor_table.shape)} "
              f"(coverage {n_cov}/{model.neighbor_table.shape[0]})", flush=True)
    else:
        print("[P2.2] neighbor table NOT provided - network axis OFF", flush=True)

    # per-dataset ctrl 均值（HVG 空间）——true_fc 与 pred_fc 都相对它
    def ctrl_hvg_of(di):
        rows = kd["row_of_gene"][di]
        cm = kd["X_ctrl"][di].mean(axis=0)
        return np.array([cm[np.where(rows == int(hr))[0][0]]
                         if (rows == int(hr)).any() else 0.0 for hr in hvg_rows])

    print()
    print(f"{'split':6} {'ds':3} {'condition':18} {'style':16} "
          f"{'std(fc)':>8} {'max|fc|':>8} {'pearson':>8} {'r(real,zero)':>12}", flush=True)

    def pert_vec_of(gene_sym):
        row = sym2row.get(gene_sym, sym2row.get(gene_sym + "1", -1))
        if row < 0:
            return None
        from rna_encoder import encode_seq, lookup_seq  # noqa: E402
        seq = lookup_seq(gene_sym, symbol_map, ensembl_map)
        if seq:
            ids, msk = encode_seq(seq, 600)
        else:
            ids, msk = [6] + [0] * 599, [1] + [0] * 599  # CLS + PAD
        tok = torch.tensor([ids], device=device)
        msk = torch.tensor([msk], device=device).bool()
        pr = torch.tensor([row], device=device)
        return model.compute_pert_vec(pr, tok, msk)  # [1, dim_emb]（含网络轴）

    probe_pvs = {}
    for split, (di, cond) in probes:
        gene = cond.replace("+ctrl", "").split("+")[0]
        pv = pert_vec_of(gene)
        if pv is None:
            continue
        probe_pvs[(split, di, cond)] = pv
    all_pvs = torch.cat(list(probe_pvs.values()), dim=0)  # [n_probe, dim_emb]

    for split, (di, cond) in probes:
        gene = cond.replace("+ctrl", "").split("+")[0]
        pert_row = sym2row.get(gene, sym2row.get(gene + "1", -1))
        if pert_row < 0:
            continue
        rows = kd["row_of_gene"][di]
        cp = kd["pert_labels"][di]
        sel = np.where(cp == cond)[0]
        true_hvg = np.array([
            kd["X_pert"][di][sel].mean(axis=0)[np.where(rows == int(hr))[0][0]]
            if (rows == int(hr)).any() else 0.0 for hr in hvg_rows])
        ch = ctrl_hvg_of(di)
        true_fc = true_hvg - ch

        pv_real = probe_pvs[(split, di, cond)]
        pv_zero = torch.zeros_like(pv_real)
        # swap: 取另一个 probe 的 pert_vec
        swap_idx = (list(probe_pvs.keys()).index((split, di, cond)) + 1) % len(probe_pvs)
        pv_swap = all_pvs[swap_idx: swap_idx + 1]

        # 控制句：eval-style = 本数据集 8 个真实控制细胞；export-style = 单个 K562 真细胞
        rng_c = np.random.default_rng(args.seed + 7)
        sent_ids, sent_exs = [], []
        for _ in range(8):
            x = kd["X_ctrl"][di][rng_c.choice(kd["X_ctrl"][di].shape[0])]
            m = rows >= 0
            gid, ex = rows[m], x[m]
            order = np.argsort(-ex)[: 2047]
            sent_ids.append(np.concatenate([[0], gid[order]]).astype(np.int32))
            sent_exs.append(np.concatenate([[0.0], ex[order]]).astype(np.float32))
        cg = torch.from_numpy(np.stack(sent_ids)).unsqueeze(0).to(device)
        ce = torch.from_numpy(np.stack(sent_exs)).unsqueeze(0).float().to(device)

        from rna_encoder import encode_seq, lookup_seq  # noqa: E402
        seq = lookup_seq(gene, symbol_map, ensembl_map)
        if seq:
            ids, msk = encode_seq(seq, 600)
        else:
            ids, msk = [6] + [0] * 599, [1] + [0] * 599  # CLS + PAD
        tok = torch.tensor([ids], device=device)
        msk_t = torch.tensor([msk], device=device).bool()

        pr_real = None
        for style, pv in [("eval(S=8) real", None), ("export(S=1) real", pv_real),
                          ("export zeroPert", pv_zero), ("export swapPert", pv_swap)]:
            with torch.no_grad():
                if pv is None:  # eval-style：常规前向（模型内部算 pert token）
                    _, pred_hvgs = model(cg, ce,
                                         torch.tensor([pert_row], device=device),
                                         tok, msk_t)
                else:           # ablation：pert_vec_override
                    _, pred_hvgs = model(cg, ce,
                                         torch.tensor([pert_row], device=device),
                                         tok, msk_t, pert_vec_override=pv)
            pred_hvg = pred_hvgs.float().mean(dim=1).cpu().numpy()[0]
            pred_fc = pred_hvg - ch
            pr = (float(np.corrcoef(true_fc, pred_fc)[0, 1])
                  if true_fc.std() > 1e-6 and pred_fc.std() > 1e-6 else 0.0)
            if style.startswith("export(S=1)"):
                pr_real = pred_fc.copy()
            print(f"{split:6} {di:3} {cond:18} {style:16} "
                  f"{pred_fc.std():8.4f} {np.abs(pred_fc).max():8.4f} {pr:8.3f}",
                  flush=True)
        # 关键判据：real 输出与 zeroPert 输出的相关（≈1 → pert token 被架空）
        pz = (float(np.corrcoef(pr_real, pred_fc)[0, 1])
              if pr_real is not None and pr_real.std() > 1e-6 and pred_fc.std() > 1e-6
              else float("nan"))
        print(f"{'':6} {'':3} {'(real vs zeroPert r)':18} {'':16} "
              f"{'':8} {'':8} {'':8} {pz:12.3f}", flush=True)
        print(f"{'':6} {'':3} {'(true_fc)':18} {'':16} "
              f"{true_fc.std():8.4f} {np.abs(true_fc).max():8.4f} {'1.000':>8}",
              flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
