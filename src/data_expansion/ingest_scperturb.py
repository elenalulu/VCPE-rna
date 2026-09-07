"""Ingest scPerturb h5ad files into GEARS-compatible perturb_processed.h5ad.

Transformations per dataset:
  1. X: raw/size-normalized counts -> CP10K + log1p (aligns with GEARS
     perturb_processed space used by the existing adamson/norman/replogle).
  2. obs.condition <- obs.perturbation, with 'control' rows -> 'ctrl';
     obs.control 0/1 column added (GEARS convention).
  3. var_names kept as gene symbols (scPerturb ships symbols; ESM2 table
     is symbol-keyed -> zero-cost mapping).
  4. Cell subsampling to keep the processed file light (P2.3-B only needs
     ctrl cells for sentence sampling + per-perturbation means for targets):
       - ctrl cells: random downsample to --max_ctrl
       - perturbed cells: keep up to --cells_per_pert per perturbation
         (>=3 rows survive the min_cells=3 gate; the mean of 16 cells is a
         fair estimator for the dev target)

Output: <out_dir>/<name>_proc.h5ad  (drop-in for --data_dirs)
"""
import argparse
import os
import re

import anndata as ad
import numpy as np
import scipy.sparse as sp


def process_file(path, out_dir, max_ctrl=3000, cells_per_pert=16, seed=0):
    name = re.sub(r"\.h5ad$", "", os.path.basename(path))
    out_path = os.path.join(out_dir, f"{name}_proc.h5ad")
    if os.path.exists(out_path):
        print(f"skip {name} (exists)", flush=True)
        return
    a = ad.read_h5ad(path, backed="r")
    n, G = a.shape
    pert = np.array([str(p) for p in a.obs["perturbation"]])
    is_ctrl = np.array([p.lower() in ("control", "ctrl", "non-targeting",
                                      "ntc", "safe-targeting") for p in pert])
    rng = np.random.default_rng(seed)

    ctrl_idx = np.where(is_ctrl)[0]
    if len(ctrl_idx) > max_ctrl:
        ctrl_idx = rng.choice(ctrl_idx, max_ctrl, replace=False)

    # target rows: up to cells_per_pert per perturbation label
    pert_pick = {}
    for lbl in np.unique(pert[~is_ctrl]):
        idx = np.where(pert == lbl)[0]
        if len(idx) < 3:
            continue  # min_cells gate
        if len(idx) > cells_per_pert:
            idx = rng.choice(idx, cells_per_pert, replace=False)
        pert_pick[lbl] = idx
    keep = np.sort(np.concatenate([ctrl_idx] + list(pert_pick.values())))
    print(f"[{name}] cells {n} -> {len(keep)} (ctrl {len(ctrl_idx)}, "
          f"perts {len(pert_pick)}, genes {G})", flush=True)

    # stream X in chunks, build the condensed matrix
    row_pos = -np.ones(n, dtype=np.int64)
    row_pos[keep] = np.arange(len(keep))
    X_out = np.zeros((len(keep), G), dtype=np.float32)
    CH = 50_000
    for lo in range(0, n, CH):
        hi = min(lo + CH, n)
        blk = a.X[lo:hi]
        blk = blk.toarray() if sp.issparse(blk) else np.asarray(blk)
        sel = row_pos[lo:hi] >= 0
        if not sel.any():
            continue
        rows = blk[sel].astype(np.float32)
        # CP10K + log1p on the selected rows only (their own library sizes)
        lib = rows.sum(axis=1, keepdims=True)
        lib[lib == 0] = 1.0
        rows = np.log1p(rows / lib * 1e4)
        X_out[row_pos[lo:hi][sel]] = rows
    print(f"[{name}] X condensed + CP10K-log1p done", flush=True)

    pert_out = pert[keep].astype(object)
    cond = np.where(is_ctrl[keep], "ctrl", pert_out).astype(str)
    ctrl_flag = is_ctrl[keep].astype(int)
    obs = ad.AnnData(
        X=sp.csr_matrix(X_out),
        obs={"condition": cond, "control": ctrl_flag,
             "perturbation_raw": pert_out},
        var=a.var[[]].copy(),  # symbols only
    )
    obs.obs["control"] = obs.obs["control"].astype(int)
    obs.write_h5ad(out_path, compression="gzip")
    print(f"✅ {out_path} ({os.path.getsize(out_path)/1e6:.0f} MB)", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--files", nargs="+", required=True)
    p.add_argument("--out_dir", default="data/scperturb_proc")
    p.add_argument("--max_ctrl", type=int, default=3000)
    p.add_argument("--cells_per_pert", type=int, default=16)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for f in args.files:
        process_file(f, args.out_dir, args.max_ctrl, args.cells_per_pert)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
