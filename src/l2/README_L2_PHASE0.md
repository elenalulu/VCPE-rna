# L2 Phase 0 Runbook (data preparation, all local)

Prerequisites: the three scripts in `src/l2/`; output directory suggestion `./data/l2/`
(disk budget ~50 GB).

## Run order

```bash
cd <repo>/src/l2

# 1. Download (10.92 GB, resumable; --list-only previews the file list)
python download_rnacentral.py --out-dir ../../data/l2 --list-only
python download_rnacentral.py --out-dir ../../data/l2

# 2. Filter + exact dedup (20-500 nt, N <= 10%, T->U normalization; streaming, ~1 GB RAM)
python filter_dedup.py --fasta ../../data/l2/rnacentral_active.fasta.gz \
  --out ../../data/l2/l2_filtered.fasta

# 3. Token store (flat uint16 + offsets, no padding; train/val 99.5/0.5)
python tokenize_store.py --fasta ../../data/l2/l2_filtered.fasta --out-dir ../../data/l2
```

## Artifacts

| File | Purpose |
|---|---|
| `rnacentral_active.fasta.gz` | raw corpus (10.92 GB, deletable after filtering) |
| `l2_filtered.fasta` | filtered + deduplicated training corpus (~5–15 M rows) |
| `l2_tokens.u16` + `l2_offsets.npy` | token store (read directly by L2-1 pretraining) |
| `l2_train_idx.npy` / `l2_val_idx.npy` | row-index splits |
| `*.stats.json` / `tokenize_stats.json` | per-stage statistics (for diagnostics) |
