#!/usr/bin/env python
"""GSE289964 摄入：in vivo 小鼠肝 Scarb1 gapmer ASO（3 修饰 × 3 时间点，48 样本）。

数据：本地 AmpliSeq bcmatrix（48 样本 × 23,930 小鼠基因，counts）+ GEO series_matrix
设计表（在线拉取，已验证）：
  PBS 12（对照）| LX-A1656 / LX-A2003 / LX-A3928（均打 Scarb1）× 24h/72h/168h × 4 reps。

口径：
  condition = "Scarb1@{修饰}{时间}"（12 个扰动条件，每条件 4 reps）
  control = 1 仅 PBS（12 样本）
  X = CP10K + log1p（counts 直接归一化）
  基因：小鼠 symbol -> 人 ortholog（MGI 首字母大写约定），与 ESM 表取交集，
        映射不上的丢弃并报告映射率。

Sanity：Scarb1 在各 ASO 组 vs PBS 应显著下调（gapmer 直接敲低）。

用法：python ingest_gse289964.py [--list-only]
"""
import argparse
import gzip
import json
import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "..", "data", "aso_tx_validation"))
BCM_FP = os.path.join(DATA, "GSE289964_AmpliSeq_bcmatrix.tsv.gz")
ESM_FP = os.path.abspath(os.path.join(HERE, "..", "..", "data",
                                      "drive_weights",
                                      "Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt"))
OUT_FP = os.path.join(DATA, "gse289964_proc.h5ad")
SERIES_URL = ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE289nnn/"
              "GSE289964/matrix/GSE289964_series_matrix.txt.gz")


def fetch_design():
    with urllib.request.urlopen(SERIES_URL, timeout=60) as r:
        data = gzip.decompress(r.read()).decode("utf-8", errors="replace")
    titles = None
    for line in data.split("\n"):
        if line.startswith("!Sample_title"):
            titles = [p.strip('"') for p in line.split("\t")[1:]]
            break
    assert titles and len(titles) == 48, f"样本数异常: {len(titles) if titles else 0}"
    samples = []
    for t in titles:
        tl = t.lower()
        mod = next((m for m in ("a1656", "a2003", "a3928") if m in tl), None)
        day = next((d for d in ("24h", "72h", "168h") if d in tl), None)
        if "pbs" in tl:
            samples.append(dict(title=t, mod="PBS", day=day or ""))
        else:
            assert mod and day, f"无法解析样本: {t}"
            samples.append(dict(title=t, mod=mod, day=day))
    return samples


def mouse_to_human(syms, esm_syms):
    """MGI 命名约定：小鼠首字母大写、其余全大写 = 人同源基因（绝大多数 1:1）。"""
    mapping, unmapped = {}, 0
    for s in syms:
        h = s.upper()          # Scarb1 -> SCARB1, Mup3 -> MUP3
        if h in esm_syms:
            mapping[s] = h
        else:
            unmapped += 1
    return mapping, unmapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-only", action="store_true")
    args = ap.parse_args()

    samples = fetch_design()
    from collections import Counter
    cnt = Counter(f"{s['mod']}@{s['day']}" if s["mod"] != "PBS" else "PBS"
                  for s in samples)
    print("[design]", dict(cnt), flush=True)
    if args.list_only:
        return

    # ---- 矩阵 ----
    df = pd.read_csv(BCM_FP, sep="\t", index_col=0)
    df.columns = [c.strip() for c in df.columns]
    print(f"[data] {df.shape} | 基因 head: {list(df.index[:3])}", flush=True)

    # ---- ESM 符号集 + 小鼠->人映射 ----
    import torch
    tab = torch.load(ESM_FP, map_location="cpu", weights_only=False)
    esm_syms = set(tab.keys())
    mapping, unmapped = mouse_to_human(list(df.index), esm_syms)
    print(f"[map] 小鼠基因 {len(df.index)} -> 人同源命中 {len(mapping)} "
          f"(未映射 {unmapped})", flush=True)

    # ---- 聚合到人 symbol（同名多转录/多鼠基因求和）----
    genes = sorted(set(mapping.values()))
    gi = {g: i for i, g in enumerate(genes)}
    R = np.zeros((df.shape[1], len(genes)), dtype=np.float32)
    col_of_gene = {}
    for mouse_sym, human_sym in mapping.items():
        col_of_gene.setdefault(human_sym, []).append(mouse_sym)
    sub = df.loc[[m for m in mapping]]
    for h_sym, m_syms in col_of_gene.items():
        R[:, gi[h_sym]] = sub.loc[m_syms].sum(axis=0).values
    # CP10K + log1p
    tot = R.sum(axis=1, keepdims=True) + 1e-6
    X = np.log1p(R / tot * 1e4)
    print(f"[agg] 人基因: {len(genes)} | 样本: {R.shape[0]}", flush=True)

    # ---- obs ----
    obs_rows, order = [], []
    cond_list, ctrl_list = [], []
    for j, s in enumerate(samples):
        sid = s["title"].replace(" ", "_")
        if s["mod"] == "PBS":
            cond, ctrl, praw = "ctrl", 1, "ctrl"
        else:
            # resolve_pert_row 用整串 condition 查 ESM 符号表（人源大写），
            # 修饰/时间点信息保留在 perturbation_raw；9 条件合流为同一靶点
            # （模型无修饰轴，语义正确，与 GSE183535 condition=MYC 同口径）
            cond, ctrl, praw = "SCARB1", 0, \
                f"SCARB1@{s['mod']}{s['day']}"
        cond_list.append(cond); ctrl_list.append(ctrl)
        obs_rows.append(dict(
            sample_id=sid, condition=cond, control=ctrl, perturbation_raw=praw,
            cell_line="MouseLiver_invivo", perturbation_type="ASO",
            tissue_type="liver_invivo", mod=s["mod"], timepoint=s["day"]))
        order.append(sid)
    obs = pd.DataFrame(obs_rows).set_index("sample_id").loc[order]

    import anndata as ad
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes))
    adata.write_h5ad(OUT_FP)
    print(f"[done] {OUT_FP} {adata.shape} | 扰动条件 "
          f"{len(set(c for c in cond_list if c != 'ctrl'))} | ctrl "
          f"{sum(1 for c in ctrl_list if c == 1)}", flush=True)

    # ---- sanity：Scarb1 敲低方向（各修饰 24h vs PBS）----
    import torch as _t
    sym2row_sanity = None
    if "SCARB1" in gi:
        i = gi["SCARB1"]
        pbs_m = X[[j for j, s in enumerate(samples) if s["mod"] == "PBS"], i].mean()
        for mod in ("a1656", "a2003", "a3928"):
            m = X[[j for j, s in enumerate(samples)
                   if s["mod"] == mod and s["day"] == "24h"], i].mean()
            fc = np.log2((np.expm1(m) + 1e-6) / (np.expm1(pbs_m) + 1e-6))
            print(f"[sanity] SCARB1 {mod} 24h vs PBS: log2FC {fc:+.2f}（gapmer 应显著负）",
                  flush=True)
    else:
        print("[sanity] SCARB1 不在映射结果中——检查映射！", flush=True)


if __name__ == "__main__":
    main()
