#!/usr/bin/env python
"""同裁判对比：XGBoost v5 (design pipeline) vs efficacy_v1 (transformer)。

协议：aso_atlas_clean.parquet + build_split(seed=0, n_holdout_genes=15) 复现
efficacy_v1 的 patent-group/gene-holdout 划分（ckpt metrics n=21989 互证）；
XGBoost 在同 train 上重训（同超参、化学 8 维由 sugar/backbone 串等价重建、
cell/gene one-hot 与剂量 median 只来自 train），两模型在同一 val_group /
val_gene 上评 PCC/Spearman/enrich_top5。公平性口径：
- 评测行限制在 XGBoost 原生序列域（16-20nt）内的共同子集；
- y 统一 = inhibition_clipped/100。
"""
import os
import sys
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RNA_ROBOT = os.environ.get("RNA_ROBOT_HOME", "")  # optional: rna_robot platform root
if RNA_ROBOT and os.path.isdir(RNA_ROBOT):
    sys.path.insert(0, RNA_ROBOT)

import train_efficacy as te
from types import SimpleNamespace
te.args = SimpleNamespace(n_holdout_genes=15)  # build_split 依赖

from model_efficacy import EfficacyHead
from sklearn.model_selection import KFold  # noqa: F401 (对齐 xgb 训练环境)

CKPT = os.path.abspath(os.path.join(HERE, "..", "..", "results", "efficacy_v1",
                                    "ckpt_efficacy_v1.pt"))
ESM_TABLE = te.ESM_TABLE

# ---------- 数据与 split ----------
df = pd.read_parquet(str(te.DATA / "aso_atlas_clean.parquet"))
print(f"rows={len(df):,}", flush=True)
train_df, val_group, val_gene, holdout_genes = te.build_split(df, 0)
print(f"train={len(train_df):,} val_group={len(val_group):,} "
      f"val_gene={len(val_gene):,} holdout_genes={len(holdout_genes)}", flush=True)

def seq_len_mask(d):
    return d["aso_sequence_5_to_3"].astype(str).str.len().between(16, 20)

vg = val_group[seq_len_mask(val_group)]
vgene = val_gene[seq_len_mask(val_gene)]
print(f"common rows (16-20nt): val_group={len(vg):,} val_gene={len(vgene):,}", flush=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device}", flush=True)

# ---------- transformer (efficacy_v1, 现役 ckpt) ----------
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
tab = torch.load(ESM_TABLE, map_location="cpu", weights_only=False)
symbols = list(tab.keys())
esm = torch.stack([v.float() for v in tab.values()])
gene2row = {s: i for i, s in enumerate(symbols)}
gene2row["UNKNOWN"] = 0            # te.evaluate 的 default 急切求值需要键存在（cov 过滤后不会命中）
n_cell = ck["model_state_dict"]["cell_emb.weight"].shape[0]
ck["cell2id"].setdefault("UNKNOWN", n_cell - 1)  # 末行未知回退保留位
model = EfficacyHead(esm, n_cell_lines=n_cell)
model.load_state_dict(ck["model_state_dict"])
model.to(device).eval()
print("transformer loaded", flush=True)

def cov(d):
    ok = d["target_gene"].isin(gene2row) & d["cell_line"].isin(ck["cell2id"])
    print(f"  coverage filter: {len(d)} -> {int(ok.sum())} (drop {int((~ok).sum())})", flush=True)
    return d[ok]
val_group, val_gene = cov(val_group), cov(val_gene)
vg, vgene = cov(vg), cov(vgene)
m_t_group = te.evaluate(model, val_group, gene2row, ck["cell2id"], device, ck["dose_mu"])
m_t_gene = te.evaluate(model, val_gene, gene2row, ck["cell2id"], device, ck["dose_mu"])
m_t_group_c = te.evaluate(model, vg, gene2row, ck["cell2id"], device, ck["dose_mu"])
m_t_gene_c = te.evaluate(model, vgene, gene2row, ck["cell2id"], device, ck["dose_mu"])
print(f"[transformer] val_group {m_t_group} | val_gene {m_t_gene}", flush=True)
print(f"[transformer] common(16-20) group {m_t_group_c} | gene {m_t_gene_c}", flush=True)

# ---------- XGBoost（同 train 重训） ----------
from utils.aso_features import extract_aso_features_full  # 39 序列维 + 占位 8 化学维

def chem8(sugar_seq, bb_seq, L):
    """从 parquet sugar/backbone 串等价重建 extract_chemistry_features 的 8 维。"""
    su = str(sugar_seq).split(",")[:L]
    bbs = str(bb_seq).split(",")[:L]
    mod_pos = {i + 1: tok for i, tok in enumerate(su) if tok != "DNA"}
    w5 = sum(1 for p in range(1, 6) if p in mod_pos)
    w3 = sum(1 for p in range(L - 4, L + 1) if p in mod_pos)
    n_bb_mod = sum(1 for t in bbs if t != "PO")
    ps_frac = n_bb_mod / max(1, L)
    has_moe = int(any(v == "MOE" for v in mod_pos.values()))
    has_lna = int(any(v == "LNA" for v in mod_pos.values()))
    has_cet = int(any(v == "cEt" for v in mod_pos.values()))
    w5_list = [mod_pos[p] for p in range(1, w5 + 1) if p in mod_pos]
    w5_homog = 1.0 if (w5_list and len(set(w5_list)) <= 1) else 0.0
    return np.array([w5, w3, n_bb_mod, ps_frac, has_moe, has_lna, has_cet, w5_homog],
                    dtype=np.float32)

from collections import Counter
tr = train_df[seq_len_mask(train_df)]
seqs = tr["aso_sequence_5_to_3"].astype(str).values
sugars = tr["sugar_seq"].values
bbs = tr["backbone_seq"].values
y_tr = (tr["inhibition_clipped"].values / 100.0).astype(np.float32)
cells = tr["cell_line"].astype(str).values
genes = tr["target_gene"].astype(str).values
dose = tr["dosage_log10"].values.astype(float)
dose_med = float(np.median(dose[dose == dose]))

cell_counts = Counter(cells)
cell_vocab = {c: i for i, (c, _) in enumerate(
    [(c, n) for c, n in cell_counts.most_common() if n >= 100])}
gene_counts = Counter(genes)
gene_vocab = {g: i for i, (g, _) in enumerate(
    [(g, n) for g, n in gene_counts.most_common() if n >= 100])}
print(f"xgb vocab: cells {len(cell_vocab)} genes {len(gene_vocab)} | "
      f"val_gene 覆盖外的 test 基因见下", flush=True)

def xgb_matrix(d):
    n = len(d)
    total = 47 + len(cell_vocab) + 1 + len(gene_vocab) + 1 + 2
    X = np.zeros((n, total), dtype=np.float32)
    y = (d["inhibition_clipped"].values / 100.0).astype(np.float32)
    sl = d["cell_line"].astype(str).values
    gl = d["target_gene"].astype(str).values
    dl = d["dosage_log10"].values.astype(float)
    for i, (row, su, bb_s) in enumerate(zip(d.itertuples(), d["sugar_seq"].values,
                                            d["backbone_seq"].values)):
        seq = str(row.aso_sequence_5_to_3)
        f = extract_aso_features_full(seq.upper().replace("T", "U"), None)
        f[39:47] = chem8(su, bb_s, len(seq))
        X[i, :47] = f
        X[i, 47 + cell_vocab.get(sl[i], len(cell_vocab))] = 1.0
        X[i, 47 + len(cell_vocab) + 1 + gene_vocab.get(gl[i], len(gene_vocab))] = 1.0
        dd = dl[i] if dl[i] == dl[i] else dose_med
        X[i, -2] = dd / 5.0
        X[i, -1] = dd / 5.0
    return X, y

print("building xgb features...", flush=True)
X_tr, y_tr2 = xgb_matrix(tr)
import xgboost as xgb
xgb_model = xgb.XGBRegressor(
    n_estimators=300, max_depth=8, learning_rate=0.06,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1, tree_method="hist")
xgb_model.fit(X_tr, y_tr2)
print("xgb trained", flush=True)

from scipy.stats import spearmanr, pearsonr
def metrics(y, p):
    if len(y) < 10 or np.std(p) < 1e-9:
        return dict(spearman=None, pearson=None, enrich=None, n=len(y))
    rho = float(spearmanr(y, p).statistic)
    r = float(pearsonr(y, p).statistic)
    top_true = set(np.argsort(-y)[: max(1, len(y) // 20)])
    top_pred = set(np.argsort(-p)[: max(1, len(p) // 20)])
    return dict(spearman=round(rho, 4), pearson=round(r, 4),
                enrich_top5=round(len(top_true & top_pred) / max(1, len(top_true)) * 5, 3),
                n=len(y))

for name, d in [("val_group", vg), ("val_gene", vgene)]:
    X_ev, y_ev = xgb_matrix(d)
    p = xgb_model.predict(X_ev)
    m = metrics(y_ev, p)
    print(f"[xgboost] {name} {m}", flush=True)

# 新靶点子集：test 中不在 xgb gene_vocab 的行（transformer ESM 轴 vs one-hot 回退）
novel = d["target_gene"].astype(str).isin(set(gene_vocab)) == False
print("done", flush=True)
