"""CRISPR→ASO migration test: does the CRISPR-trained response head transfer to ASO measured response?

Data: GSE183535 (HeLa, MYC gapmer ASO), TPM matrix, 4 conditions x 2 timepoints x 3 reps.
Prediction: cache v3 (VCPE-rna, CRISPR-trained) MYC entry — top_up/top_down residual lists.
Verdict metrics:
  1. MYC self-knockdown (anchor check)
  2. Direction consistency of predicted top_up/top_down vs measured log2FC
  3. Spearman over the predicted top-48 gene set
  4. NTASO-vs-Vehicle response vs cache common_response_core (is the shared core the ASO transfection stress?)

Run locally (CPU): python eval_aso_migration.py
"""
import gzip
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
TPM = os.path.join(REPO_ROOT, "data", "aso_tx_validation",
                   "GSE183535_MYCASO_RNAseq_TPM_table.csv.gz")
CACHE = os.path.join(REPO_ROOT, "vcpe_cache_v4_k562a.json.gz")  # v4: p3_v21b 七数据集门控版
MAP_CACHE = os.path.join(HERE, "refseq2symbol_cache.json")
MYC_REFSEQ = "NM_002467"


def symbol2refseq(symbols):
    """HGNC symbol -> RefSeq RNA accession via mygene GET API (small n, cached)."""
    cache_f = os.path.join(HERE, "symbol2refseq_cache.json")
    m = json.load(open(cache_f)) if os.path.exists(cache_f) else {}
    import requests
    for i, sym in enumerate(symbols):
        if sym in m:
            continue
        try:
            r = requests.get("https://mygene.info/v3/query",
                             params={"q": sym, "scopes": "symbol", "fields": "refseq.rna",
                                     "species": "human"}, timeout=30)
            hits = r.json().get("hits", [])
            accs = None
            for h in hits:
                rr = h.get("refseq", {})
                accs = rr.get("rna") if isinstance(rr, dict) else rr
                if accs:
                    break
            if isinstance(accs, str):
                accs = [accs]
            # strip version suffixes
            m[sym] = [a.split(".")[0] for a in (accs or [])]
        except Exception as e:
            print(f"[map] {sym} ERR {str(e)[:60]}", flush=True)
            m[sym] = []
        if (i + 1) % 20 == 0:
            print(f"[map] {i+1}/{len(symbols)}", flush=True)
    json.dump(m, open(cache_f, "w"))
    return m


def group_mean(df, prefix):
    cols = [c for c in df.columns if c.startswith(prefix)]
    return df[cols].mean(axis=1)


def main():
    df = pd.read_csv(TPM, index_col=0)
    print(f"matrix: {df.shape}", flush=True)

    # --- 1. anchor: MYC self-knockdown ---
    myc = df.loc[MYC_REFSEQ]
    groups = {}
    for c in df.columns:
        groups.setdefault(c.rsplit(" REP", 1)[0], []).append(myc[c])
    gm = {g: float(np.mean(v)) for g, v in groups.items()}
    kd_18h = np.log2((gm["MYCASO3 18h"] + 1) / (gm["NTASO 18h"] + 1))
    kd_4h = np.log2((gm["MYCASO3 4h"] + 1) / (gm["NTASO 3 4h".replace(" 3 4h", " 4h")] + 1))
    print(f"\n[anchor] MYC TPM: NTASO18h {gm['NTASO 18h']:.1f} -> MYCASO18h {gm['MYCASO3 18h']:.1f} "
          f"(log2FC {kd_18h:+.2f}); 4h {kd_4h:+.2f}", flush=True)

    # --- measured log2FC: ASO vs NTASO (clean control), per timepoint ---
    fc18 = np.log2((group_mean(df, "MYCASO3 18h") + 1) / (group_mean(df, "NTASO 18h") + 1))
    fc4 = np.log2((group_mean(df, "MYCASO3 4h") + 1) / (group_mean(df, "NTASO 4h") + 1))
    nta_stress = np.log2((group_mean(df, "NTASO 18h") + 1) / (group_mean(df, "Vehicle 18h") + 1))

    # --- cache prediction ---
    cache = json.load(gzip.open(CACHE, "rt"))
    myc_pred = cache["genes"].get("MYC")
    if myc_pred is None:
        print("MYC not in cache — abort", flush=True)
        return
    core = cache["common_response_core"]

    # --- symbol mapping (reverse: only genes the cache needs) ---
    need = set(x[0] for x in myc_pred["top_up"]) | set(x[0] for x in myc_pred["top_down"]) \
         | set(x[0] for x in core["top_up"])
    sym2rows = symbol2refseq(sorted(need))
    n_hit = sum(1 for s2 in sym2rows.values() if any(r in df.index for r in s2))
    print(f"[map] {n_hit}/{len(need)} cache genes -> RefSeq rows in matrix", flush=True)


    def rows_of(sym):
        rs = sym2rows.get(sym, [])
        return [r for r in rs if r in df.index]

    # --- 2/3. direction consistency + spearman on predicted top-48 ---
    for tp, fc in [("18h", fc18), ("4h", fc4)]:
        print(f"\n=== timepoint {tp} (ASO vs NTASO) ===", flush=True)
        for arm, lst in [("top_up", myc_pred["top_up"]), ("top_down", myc_pred["top_down"])]:
            cons, fcs = [], []
            for sym, pred_fc in lst:
                rs = rows_of(sym)
                if not rs:
                    continue
                mfc = float(np.mean(fc.loc[rs]))
                # for top_up: predicted up -> measured fc > 0 is consistent
                cons.append(1 if (pred_fc > 0 and mfc > 0) or (pred_fc < 0 and mfc < 0) else 0)
                fcs.append((sym, pred_fc, mfc))
            n = len(fcs)
            dir_rate = float(np.mean(cons)) if cons else float("nan")
            rho = spearmanr([x[1] for x in fcs], [x[2] for x in fcs]).statistic if n > 3 else float("nan")
            print(f"  {arm}: n={n}/{len(lst)} mapped | direction consistent {dir_rate*100:.0f}% | "
                  f"spearman(pred, measured) {rho:.3f}", flush=True)
            worst = sorted(fcs, key=lambda x: -abs(x[1]))[:3]
            for sym, pf, mf in worst:
                print(f"      {sym:10} pred {pf:+.3f} | measured {mf:+.3f}", flush=True)

    # --- 4. common core vs NTASO transfection stress ---
    core_map = {x[0]: x[1] for x in core["top_up"]}
    rows_core, core_vals = [], []
    for s_sym, cv in core_map.items():
        rs = rows_of(s_sym)
        if rs:
            rows_core.append(rs[0])
            core_vals.append(cv)
    if len(rows_core) > 5:
        v = nta_stress.loc[rows_core].values
        c = np.array(core_vals)
        if v.std() > 1e-9 and c.std() > 1e-9:
            rho = float(np.corrcoef(v, c)[0, 1])
        else:
            rho = float("nan")
        print(f"\n[common-core vs NTASO stress] n={len(rows_core)} pearson {rho:.3f} "
              f"(high = 共性核≈ASO 转染应激, 扣除逻辑自洽)", flush=True)

    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
