"""P1 dataset: knockdown-conditioned Perturb-seq for MAP-rna.

Reads GEARS-processed h5ad files (perturb_processed.h5ad from
adamson / norman / replogle_rpe1_essential — same sources as our
16-training vertical_slice data), and yields:

  control_gene_ids   [S, L]   int32, rows of the 19790-row ESM2 gene embedding
                              table (position 0 = placeholder, sliced off by
                              MAPmodel.forward)
  control_expressions[S, L]   float32
  pert_row           int      row index of the knocked-down gene in the same
                              ESM2 table (the conditioning signal)
  pert_hvg           [2000]   float32, mean expression of perturbed cells on
                              the fixed HVG list (training target)
  pert_emb_target    [2048]   SE embedding of the perturbed bulk (training
                              target, precomputed once by train_p1.py)
"""
import json
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import anndata as ad
except ImportError:  # pragma: no cover
    ad = None


def build_symbol2row(esm_table_path):
    """gene symbol -> row index of the ESM2 embedding table.

    MAPmodel stacks dict.values() in insertion order, so row i <-> key i.
    """
    tab = torch.load(esm_table_path, map_location="cpu", weights_only=False)
    if not isinstance(tab, dict):
        raise TypeError(f"expected dict gene_symbol->embedding, got {type(tab)}")
    symbols = list(tab.keys())
    sym2row = {s: i for i, s in enumerate(symbols)}
    first = next(iter(tab.values()))
    dim = int(first.shape[-1])
    return sym2row, dim


def resolve_pert_row(sym2row, cond):
    """Condition gene for a perturbation label.

    'GENE+ctrl' -> GENE. Combos 'A+B' condition on the FIRST gene (P1 scope:
    single-target conditioning; multi-token combos come in P2). Fallback: try
    '<gene>1' — HGNC 2020 renamed many aminoacyl-tRNA synthetases etc.
    (AARS -> AARS1) and the cellxgene-derived ESM2 table uses the new names.
    Returns -1 if unresolvable (sample will be skipped).
    """
    gene = cond.split("+")[0]
    if gene.lower() == "ctrl":
        return -1
    r = sym2row.get(gene, -1)
    if r < 0:
        r = sym2row.get(gene + "1", -1)
    return r


def load_kd_datasets(data_dirs, sym2row, min_cells=3):
    """Load several GEARS h5ad files into a unified perturbation table.

    Returns dict with:
      X_ctrl        list of np.ndarray [n_ctrl, n_genes] per dataset
      X_pert        list of np.ndarray [n_pert_cells, n_genes] per dataset
      pert_labels   list of np.ndarray [n_pert_cells] condition strings
      pert_index    list of (ds_idx, condition) unique perturbations
      gene_names    list of np.ndarray [n_genes] per dataset
      row_of_gene   list of np.ndarray [n_genes] int (ESM2 table row, -1 if absent)
    """
    X_ctrl, X_pert, pert_labels, pert_index, gene_names, row_of_gene = [], [], [], [], [], []
    for di, d in enumerate(data_dirs):
        a = ad.read_h5ad(d)
        # GEARS h5ad: var_names are Ensembl IDs; real symbols live in var['gene_name']
        if "gene_name" in a.var.columns:
            genes = np.array([str(g) for g in a.var["gene_name"]])
        else:
            genes = np.array([str(g) for g in a.var_names])
        rows = np.array([sym2row.get(g, -1) for g in genes], dtype=np.int64)
        gene_names.append(genes)
        row_of_gene.append(rows)

        conds = np.array([str(c) for c in a.obs["condition"]])
        X = a.X  # keep sparse: dense conversion happens per-chunk only

        # control cells: prefer the explicit 0/1 'control' obs column (GEARS convention);
        # fall back to case-insensitive condition match
        if "control" in a.obs.columns:
            ctrl_mask = np.asarray(a.obs["control"]).astype(int) == 1
        else:
            ctrl_mask = np.array([c.lower() == "ctrl" for c in conds])

        # ctrl pool: downsample to 3000 (memory: 15GB host; 3000 cells give ample
        # sentence-sampling diversity for the dev head)
        ctrl_idx = np.where(ctrl_mask)[0]
        if len(ctrl_idx) > 3000:
            ctrl_idx = np.random.default_rng(0).choice(ctrl_idx, 3000, replace=False)
        Xc = X[ctrl_idx]
        Xc = (Xc.toarray() if hasattr(Xc, "toarray") else np.asarray(Xc))
        X_ctrl.append(np.asarray(Xc, dtype=np.float32))

        pert_idx_all = np.where(~ctrl_mask)[0]
        cp_all = conds[pert_idx_all]
        keep_cond = np.array([c.lower() != "ctrl" for c in cp_all])
        pert_idx_all, cp_all = pert_idx_all[keep_cond], cp_all[keep_cond]

        # pseudobulk aggregation, streamed: one mean row per condition.
        # Never densifies more than CHUNK rows at once (OOM-safe on 15GB hosts).
        uniq, inverse, counts = np.unique(cp_all, return_inverse=True,
                                          return_counts=True)
        bulk = np.zeros((len(uniq), X.shape[1]), dtype=np.float64)
        CHUNK = 50_000
        for lo in range(0, len(pert_idx_all), CHUNK):
            pos = pert_idx_all[lo:lo + CHUNK]
            blk = X[pos]
            blk = blk.toarray() if hasattr(blk, "toarray") else np.asarray(blk)
            np.add.at(bulk, inverse[lo:lo + CHUNK], blk.astype(np.float64))
        bulk /= counts[:, None]
        kept = [(di, u) for u, c in zip(uniq, counts) if c >= min_cells]
        X_pert.append(bulk.astype(np.float32))
        pert_labels.append(uniq.astype(str))
        pert_index.extend(kept)
        n_mapped = sum(1 for _, u in kept if resolve_pert_row(sym2row, u) >= 0)
        print(f"[ds {di}] {d}: cells={X.shape[0]} genes={X.shape[1]} "
              f"ctrl={int(ctrl_mask.sum())} perts_kept={len(kept)}/{len(uniq)} "
              f"targets_mapped={n_mapped}/{len(kept)}", flush=True)
    # drop perturbations whose condition gene cannot be resolved in the ESM2 table
    before = len(pert_index)
    pert_index = [it for it in pert_index if resolve_pert_row(sym2row, it[1]) >= 0]
    print(f"[kd] perturbations mapped: {len(pert_index)}/{before} "
          f"({before - len(pert_index)} skipped, unresolvable target symbol)", flush=True)
    return dict(X_ctrl=X_ctrl, X_pert=X_pert, pert_labels=pert_labels,
                pert_index=pert_index, gene_names=gene_names, row_of_gene=row_of_gene)


def make_hvg_list(kd, n_hvg=2000):
    """Top-variance genes (on the ESM2-mapped universe) across all datasets."""
    var_per_ds = []
    for X, rows in zip(kd["X_pert"], kd["row_of_gene"]):
        m = rows >= 0
        if m.sum() == 0:
            continue
        v = X[:, m].var(axis=0)
        var_per_ds.append((rows[m], v))
    # aggregate variance by table row (mean over datasets where gene present)
    agg = {}
    for rows, v in var_per_ds:
        for r, x in zip(rows, v):
            agg.setdefault(int(r), []).append(float(x))
    agg = {r: float(np.mean(vs)) for r, vs in agg.items()}
    ranked = sorted(agg.items(), key=lambda kv: -kv[1])[:n_hvg]
    hvg_rows = np.array([r for r, _ in ranked], dtype=np.int64)
    return hvg_rows  # [n_hvg] ESM2 table rows, fixed order


class KnockdownDataset(Dataset):
    """One sample = one perturbation.

    set_size control cells are re-sampled every __getitem__ call (is_train).
    """

    def __init__(self, kd, pert_items, sym2row, set_size=8, L=2048,
                 hvg_rows=None, is_train=True, seed=0):
        self.kd = kd
        self.items = list(pert_items)
        self.sym2row = sym2row
        self.set_size = set_size
        self.L = L
        self.hvg_rows = hvg_rows
        self.is_train = is_train
        self.rng = np.random.default_rng(seed)
        # per-dataset control row pools
        self._ctrl_pool = [np.arange(X.shape[0]) for X in kd["X_ctrl"]]
        # group perturbed cells by (ds, condition)
        self._pert_rows = {}
        for di, cond in self.items:
            cp = kd["pert_labels"][di]
            idx = np.where(cp == cond)[0]
            self._pert_rows[(di, cond)] = idx

    def __len__(self):
        return len(self.items)

    def _ctrl_sentence(self, di):
        pool = self._ctrl_pool[di]
        k = min(self.set_size, len(pool))
        chosen = self.rng.choice(pool, k, replace=False) if self.is_train else pool[:k]
        Xc = self.kd["X_ctrl"][di][chosen]            # [k, n_genes]
        rows = self.kd["row_of_gene"][di]             # [n_genes]
        S, out_ids, out_expr = Xc.shape[0], [], []
        for s in range(S):
            x = Xc[s]
            m = rows >= 0
            gid, ex = rows[m], x[m]
            order = np.argsort(-ex)[: self.L - 1]     # top-expressed genes first
            ids = gid[order]
            exs = ex[order].astype(np.float32)
            pad = self.L - 1 - len(ids)
            if pad > 0:
                ids = np.concatenate([ids, np.zeros(pad, dtype=np.int64)])
                exs = np.concatenate([exs, np.zeros(pad, dtype=np.float32)])
            out_ids.append(np.concatenate([[0], ids]))     # pos 0 = placeholder
            out_expr.append(np.concatenate([[0.0], exs]))
        return (torch.from_numpy(np.stack(out_ids).astype(np.int32)),
                torch.from_numpy(np.stack(out_expr)).float())

    def _pert_target(self, di, cond):
        idx = self._pert_rows[(di, cond)]
        Xp = self.kd["X_pert"][di][idx]               # [n, n_genes]
        mean = Xp.mean(axis=0)
        rows = self.kd["row_of_gene"][di]
        target = np.zeros(len(self.hvg_rows), dtype=np.float32)
        row2col = {int(r): c for c, r in enumerate(rows)}
        for hi, hr in enumerate(self.hvg_rows):
            c = row2col.get(int(hr))
            if c is not None:
                target[hi] = mean[c]
        pert_row = resolve_pert_row(self.sym2row, cond)  # first gene; +1 alias fallback
        return target, pert_row, cond

    def __getitem__(self, i):
        di, cond = self.items[i]
        ids, expr = self._ctrl_sentence(di)
        hvg_t, pert_row, cond_s = self._pert_target(di, cond)
        return dict(control_gene_ids=ids, control_expressions=expr,
                    pert_row=int(pert_row), pert_hvg=torch.from_numpy(hvg_t),
                    condition=cond_s, ds=di)


def collate_kd(batch):
    ctrl_gene = torch.stack([b["control_gene_ids"] for b in batch])          # [B,S,L]
    ctrl_expr = torch.stack([b["control_expressions"] for b in batch])       # [B,S,L]
    pert_row = torch.tensor([b["pert_row"] for b in batch], dtype=torch.long)
    pert_hvg = torch.stack([b["pert_hvg"] for b in batch])                   # [B,2000]
    conds = [b["condition"] for b in batch]
    dss = [b["ds"] for b in batch]
    return dict(control_gene_ids=ctrl_gene, control_expressions=ctrl_expr,
                pert_row=pert_row, pert_hvg=pert_hvg, condition=conds, ds=dss)
