"""L2/P2.2 Phase A: precompute STRING network neighborhoods for all ESM2 genes.

Output: string_neighbors.npy  [V, K] int64 — for each ESM2-table row (symbol),
the ESM2-row indices of its top-K STRING neighbors (combined_score >= threshold,
human only), -1 padded. This matrix feeds the response model's network-axis
conditioning (pert token = proj(own ESM2 ⊕ mean(neighbor ESM2))).

Inputs:
  --string-db     bio_literature bundles/string/string.db (edges table:
                  protein1/protein2 '9606.ENSP...', combined_score 0-1000)
  --protein-info  9606.protein.info.v12.0.txt.gz (2.0 MB, auto-download if missing;
                  maps preferred_name(symbol) <-> protein_external_id '9606.ENSP...')
  --esm-table     ESM2 embedding table (defines the symbol list & row order)

Run (local machine, CPU, ~10-20 min — streams the 20M-edge table once):
  python build_string_neighbors.py \
    --string-db <bio_literature>/bundles/string/string.db \
    --esm-table <drive_weights>/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt \
    --out ./data/l2/string_neighbors.npy
"""
import argparse
import gzip
import io
import json
import os
import sqlite3
import urllib.request

import numpy as np

PROTEIN_INFO_URL = "https://stringdb-downloads.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz"


def ensure_protein_info(path):
    if os.path.exists(path):
        return path
    print(f"downloading protein.info -> {path}", flush=True)
    req = urllib.request.Request(PROTEIN_INFO_URL, headers={"User-Agent": "vcpe-rna"})
    with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f:
        f.write(r.read())
    return path


def load_protein_info(path):
    """-> (sym2ensp, ensp2sym)。兼容 v11.5（protein_external_id）与 v12.0（#string_protein_id）。"""
    sym2ensp, ensp2sym = {}, {}
    with gzip.open(path, "rt") as f:
        header = f.readline().rstrip("\n").lstrip("#").split("\t")
        i_ext = next((i for i, h in enumerate(header)
                      if "external" in h or "string_protein" in h), 0)
        i_pref = next((i for i, h in enumerate(header) if "preferred" in h),
                      1 if i_ext == 0 else 0)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(i_ext, i_pref):
                continue
            ensp = parts[i_ext].split(".")[-1] if "." in parts[i_ext] else parts[i_ext]
            sym = parts[i_pref]
            if sym and ensp:
                sym2ensp[sym] = ensp
                ensp2sym.setdefault(ensp, sym)
    return sym2ensp, ensp2sym


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--string-db", type=str, required=True)
    p.add_argument("--esm-table", type=str, required=True)
    p.add_argument("--protein-info", type=str, default="./9606.protein.info.v12.0.txt.gz")
    p.add_argument("--out", type=str, default="./string_neighbors.npy")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--min-score", type=int, default=700, help="STRING combined_score threshold (700=high conf)")
    args = p.parse_args()

    import torch
    esm_tab = torch.load(args.esm_table, map_location="cpu", weights_only=False)
    keys = list(esm_tab.keys())                      # row i <-> keys[i]
    sym2row = {k: i for i, k in enumerate(keys)}
    print(f"esm table: {len(keys)} symbols", flush=True)

    path = ensure_protein_info(args.protein_info)
    sym2ensp, ensp2sym = load_protein_info(path)
    print(f"protein.info: {len(sym2ensp)} symbols", flush=True)

    # our table symbols that exist in STRING
    our2ensp = {s: sym2ensp[s] for s in keys if s in sym2ensp}
    ensp2our = {e: s for s, e in our2ensp.items()}
    print(f"table symbols mapped to STRING: {len(our2ensp)}/{len(keys)}", flush=True)

    # one streaming pass over the 20M-edge table; keep edges among our genes
    import heapq
    adj = {s: [] for s in our2ensp}                  # sym -> list[(score, neighbor_sym)]
    con = sqlite3.connect(args.string_db)
    cur = con.cursor()
    cur.execute("SELECT protein1, protein2, combined_score FROM edges WHERE species='9606'")
    scanned = kept = 0
    for p1, p2, score in cur:
        scanned += 1
        if score < args.min_score:
            continue
        s1 = ensp2our.get(p1.split(".")[-1])
        if s1 is None:
            continue
        s2 = ensp2our.get(p2.split(".")[-1])
        if s2 is None or s2 == s1:
            continue
        adj[s1].append((score, s2))
        adj[s2].append((score, s1))
        kept += 1
        if scanned % 5_000_000 == 0:
            print(f"  scanned {scanned/1e6:.0f}M edges, kept {kept}", flush=True)
    con.close()
    print(f"edges scanned={scanned/1e6:.1f}M kept(high-conf, both-mapped)={kept}", flush=True)

    # top-K per gene -> ESM2 row indices
    V = len(keys)
    table = np.full((V, args.top_k), -1, dtype=np.int64)
    n_with = 0
    for s, lst in adj.items():
        if s not in sym2row:
            continue
        lst.sort(reverse=True)
        rows = [sym2row[nb] for sc, nb in lst[: args.top_k] if nb in sym2row]
        table[sym2row[s], :len(rows)] = rows
        if rows:
            n_with += 1
    np.save(args.out, table)
    cov = int((table[:, 0] >= 0).sum())
    stats = {"V": V, "genes_with_neighbors": n_with, "coverage": round(cov / V, 4),
             "top_k": args.top_k, "min_score": args.min_score,
             "avg_neighbors": round(float((table >= 0).sum()) / max(V, 1), 1)}
    with open(args.out + ".stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"✅ {args.out}: {json.dumps(stats)}", flush=True)


if __name__ == "__main__":
    main()
