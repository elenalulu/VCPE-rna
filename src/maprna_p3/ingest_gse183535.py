#!/usr/bin/env python
"""V2-1c 扩展：GSE183535（HeLa, MYC gapmer ASO）摄入为 ASO 域训练集。

数据：21 样本 TPM 矩阵（RefSeq transcript 级）：Vehicle/NTASO/MYCASO3 × 4h/18h ×3
+ Untreated ×3。TPM 矩阵已在本地 data/aso_tx_validation/。

口径（selectivity 对照，同 GSE293987 决策）：
  condition = "MYC"       <- MYCASO3 18h（3 样本；4h 转录组几乎未响应，排除）
  condition = "ctrl"      <- NTASO 4h+18h（6 样本，转染匹配对照）
  condition = "ref_excluded" <- Vehicle/Untreated/MYCASO3 4h（不进 ctrl 也不进扰动，
                              load 后 resolve 失败被安全跳过）
  X = log1p(TPM)；基因 = RefSeq accession 经 mygene /v3/query 映射到 symbol 后按基因求和。

用法：
  python ingest_gse183535.py              # 全量（映射有本地缓存断点续传）
  python ingest_gse183535.py --list-only  # 只列设计表
"""
import argparse
import gzip
import json
import os
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "..", "data", "aso_tx_validation"))
TPM_FP = os.path.join(DATA, "GSE183535_MYCASO_RNAseq_TPM_table.csv.gz")
CACHE_FP = os.path.join(DATA, "refseq2symbol_full.json")
MIN_TPM = 0.5


def map_accessions(accs):
    """RefSeq accession -> symbol，多线程 GET /v3/query（缓存断点续传）。"""
    m = json.load(open(CACHE_FP)) if os.path.exists(CACHE_FP) else {}
    todo = [a for a in accs if a not in m]
    print(f"[map] cache {len(m)} | todo {len(todo)}", flush=True)

    def one(acc):
        url = ("https://mygene.info/v3/query?" +
               urllib.parse.urlencode({"q": acc, "scopes": "accession",
                                       "fields": "symbol", "species": "human"}))
        for _ in range(3):
            try:
                with urllib.request.urlopen(url, timeout=30) as r:
                    hits = json.loads(r.read()).get("hits", [])
                for h in hits:
                    sym = h.get("symbol")
                    if sym:
                        return acc, sym
                return acc, None
            except Exception:
                continue
        return acc, None

    done = 0
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = [ex.submit(one, a) for a in todo]
        for f in as_completed(futs):
            acc, sym = f.result()
            m[acc] = sym
            done += 1
            if done % 500 == 0:
                json.dump(m, open(CACHE_FP, "w"))
                print(f"[map] {done}/{len(todo)}", flush=True)
    json.dump(m, open(CACHE_FP, "w"))
    mapped = {a: s for a, s in m.items() if s}
    print(f"[map] total mapped {len(mapped)}/{len(m)}", flush=True)
    return mapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-only", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(TPM_FP, index_col=0)
    df.columns = [c.strip() for c in df.columns]
    print(f"[data] {df.shape} | accessions head: {list(df.index[:3])}", flush=True)

    # 设计表
    def cls(col):
        if col.startswith("MYCASO3"):
            return "MYC" if "18h" in col else "MYC4h"
        if col.startswith("NTASO"):
            return "NTASO"
        if col.startswith("Vehicle"):
            return "Vehicle"
        return "Untreated"
    design = pd.DataFrame({"sample": df.columns,
                           "class": [cls(c) for c in df.columns]})
    print(design.to_string(index=False))
    if args.list_only:
        return

    # ---- 过滤 + 映射 ----
    keep = df.index[df.max(axis=1) >= MIN_TPM]
    print(f"[filter] max TPM >= {MIN_TPM}: {len(keep)}/{len(df)} transcripts", flush=True)
    acc2sym = map_accessions(list(keep))

    # ---- 聚合到 symbol（TPM 求和）----
    sym_of_col = {}
    gene_set = set()
    for acc in keep:
        s = acc2sym.get(acc)
        if s:
            gene_set.add(s)
    genes = sorted(gene_set)
    gi = {g: i for i, g in enumerate(genes)}
    acc_gene = {acc: acc2sym.get(acc) for acc in keep}
    n = df.shape[1]
    R = np.zeros((n, len(genes)), dtype=np.float32)
    sub = df.loc[keep]
    for j, acc in enumerate(sub.index):
        g = acc_gene.get(acc)
        if g is None:
            continue
        R[:, gi[g]] += sub.iloc[j].values.astype(np.float32)
    X = np.log1p(R)
    print(f"[agg] genes: {len(genes)} | samples: {n}", flush=True)

    # ---- obs ----
    samples = list(df.columns)
    cond, ctrl = [], []
    for s in samples:
        c = cls(s)
        if c == "MYC":
            cond.append("MYC"); ctrl.append(0)
        elif c == "NTASO":
            cond.append("ctrl"); ctrl.append(1)
        else:
            cond.append("ref_excluded"); ctrl.append(0)
    obs = pd.DataFrame({
        "sample_id": samples, "class": [cls(s) for s in samples],
        "condition": cond, "control": ctrl,
        "perturbation_raw": np.where(np.array(ctrl) == 1, "ctrl",
                             np.where(np.array(cond) == "MYC", "MYC", "ref_excluded")),
        "cell_line": "HeLa", "perturbation_type": "ASO",
        "tissue_type": "epithelial",
    }, index=samples)

    import anndata as ad
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes))
    out_fp = os.path.join(DATA, "gse183535_proc.h5ad")
    adata.write_h5ad(out_fp)
    print(f"[done] {out_fp} {adata.shape} | MYC 扰动 {cond.count('MYC')} | "
          f"ctrl {ctrl.count(1)} | ref_excluded {cond.count('ref_excluded')}", flush=True)

    # ---- sanity：MYC log2fc（MYCASO18h vs NTASO18h），期望 ≈ -0.91 ----
    gi_myc = gi.get("MYC")
    if gi_myc is not None:
        m18 = X[samples.index("MYCASO3 18h REP 1 - TPM"), gi_myc]
        n18 = X[samples.index("NTASO 18h REP 1 - TPM"), gi_myc]
        fc = np.log2((np.expm1(m18) + 1e-6) / (np.expm1(n18) + 1e-6))
        print(f"[sanity] MYC log1pTPM: MYCASO18h {m18:.3f} vs NTASO18h {n18:.3f} "
              f"-> log2fc {fc:+.2f}（期望 ≈ -0.91）", flush=True)


if __name__ == "__main__":
    main()
