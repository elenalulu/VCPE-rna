"""Train the ASO efficacy head v1 on ASO Atlas (190,927 rows).

Three evaluation slices (all reported every epoch on val):
  patent-group val : rows from held-out patent tables (leakage-controlled, ~official folds)
  gene-holdout     : rows of ~15 genes entirely unseen (tests ESM2-mechanism generalization)
Metrics: Spearman (primary, comparable to OligoAI paper 0.419), Pearson,
top-5% enrichment factor (5.72x random = paper's experimental target bar).

Runs on CPU in ~15-30 min (small model); GPU-optional via CUDA_VISIBLE_DEVICES.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr, pearsonr

from model_efficacy import EfficacyHead, BASES, SUGARS, BACKBONES, MAX_LEN

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "data" / "aso_atlas"
ESM_TABLE = HERE.parent.parent / "data" / "drive_weights" / \
    "Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=str(DATA / "aso_atlas_clean.parquet"))
    p.add_argument("--esm_table", type=str, default=str(ESM_TABLE))
    p.add_argument("--out_dir", type=str, default=str(HERE.parent.parent / "results" / "efficacy_v1"))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n_holdout_genes", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def encode_batch(seqs, sugar_seqs, bb_seqs):
    B = len(seqs)
    base = np.zeros((B, MAX_LEN), dtype=np.int64)
    sug = np.zeros((B, MAX_LEN), dtype=np.int64)
    bb = np.zeros((B, MAX_LEN), dtype=np.int64)
    pad = np.ones((B, MAX_LEN), dtype=bool)
    for i, (s, su, bb_s) in enumerate(zip(seqs, sugar_seqs, bb_seqs)):
        L = min(len(s), MAX_LEN)
        for j in range(L):
            base[i, j] = BASES.get(s[j], 4)
            sug[i, j] = SUGARS.get(su.split(",")[j], 0)
            bb[i, j] = BACKBONES.get(bb_s.split(",")[j], 0)
            pad[i, j] = False
    return (torch.from_numpy(base), torch.from_numpy(sug), torch.from_numpy(bb),
            torch.from_numpy(pad))


def build_split(df, seed):
    """patent-group split + gene-holdout (rows of held-out genes go to test)."""
    rng = np.random.default_rng(seed)
    customs = df["custom_id"].astype(str).unique()
    rng.shuffle(customs)
    n_val = max(1, int(len(customs) * 0.15))
    val_customs = set(customs[:n_val])

    big_genes = df["target_gene"].value_counts()
    big_genes = big_genes[big_genes >= 300].index.tolist()
    holdout_genes = set(rng.choice(sorted(big_genes), size=args.n_holdout_genes,
                                   replace=False))

    is_group_val = df["custom_id"].astype(str).isin(val_customs)
    is_gene_hold = df["target_gene"].isin(holdout_genes)
    train = df[~is_group_val & ~is_gene_hold]
    val_group = df[is_group_val & ~is_gene_hold]
    val_gene = df[is_gene_hold]
    return train, val_group, val_gene, holdout_genes


def evaluate(model, df_part, gene2row, cell2id, device, dose_mu):
    model.eval()
    preds, trues = [], []
    B = 1024
    rows = list(df_part.itertuples())
    with torch.no_grad():
        for lo in range(0, len(rows), B):
            chunk = rows[lo:lo + B]
            base, sug, bb, pad = encode_batch(
                [r.aso_sequence_5_to_3 for r in chunk],
                [r.sugar_seq for r in chunk], [r.backbone_seq for r in chunk])
            gene_rows = torch.tensor([gene2row.get(r.target_gene, gene2row["UNKNOWN"])
                                      for r in chunk], dtype=torch.long)
            cell_ids = torch.tensor([cell2id.get(r.cell_line, cell2id["UNKNOWN"])
                                     for r in chunk], dtype=torch.long)
            dl = np.array([r.dosage_log10 if r.dosage_log10 == r.dosage_log10 else dose_mu
                           for r in chunk], dtype=np.float32)
            dm = np.array([0.0 if r.dosage_log10 == r.dosage_log10 else 1.0
                           for r in chunk], dtype=np.float32)
            out = model(base.to(device), sug.to(device), bb.to(device), pad.to(device),
                        gene_rows.to(device), cell_ids.to(device),
                        torch.from_numpy(dl).to(device), torch.from_numpy(dm).to(device))
            preds.extend(out.cpu().numpy().tolist())
            trues.extend([r.inhibition_clipped for r in chunk])
    preds, trues = np.array(preds), np.array(trues)
    rho = spearmanr(trues, preds).statistic
    r = pearsonr(trues, preds).statistic
    top_true = set(np.argsort(-trues)[: max(1, len(trues) // 20)])
    top_pred = set(np.argsort(-preds)[: max(1, len(preds) // 20)])
    enrich = len(top_true & top_pred) / max(1, len(top_true)) * 5  # 1.0 = random
    return dict(spearman=float(rho), pearson=float(r), enrich_top5=float(enrich),
                n=len(trues))


def main():
    global args
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_parquet(args.data)
    print(f"data: {df.shape}", flush=True)

    # ESM2 table (frozen mechanism features)
    tab = torch.load(args.esm_table, map_location="cpu", weights_only=False)
    symbols = list(tab.keys()) if isinstance(tab, dict) else None
    if isinstance(tab, dict):
        esm = torch.stack([v.float() for v in tab.values()])
        gene2row = {s: i for i, s in enumerate(symbols)}
    else:
        esm = tab.float()
        gene2row = {}
    gene2row.setdefault("UNKNOWN", 0)

    cell_lines = sorted(df["cell_line"].unique())
    cell2id = {c: i for i, c in enumerate(cell_lines)}
    dose_mu = float(df["dosage_log10"].dropna().mean())

    train, val_group, val_gene, holdout_genes = build_split(df, args.seed)
    print(f"train={len(train)} patent-val={len(val_group)} "
          f"gene-holdout={len(val_gene)} ({len(holdout_genes)} genes: "
          f"{sorted(holdout_genes)[:6]}...)", flush=True)

    model = EfficacyHead(esm, n_cell_lines=len(cell_lines) + 1).to(device)
    print(f"params trainable={sum(p.numel() for p in model.parameters())/1e6:.2f}M "
          f"(esm table frozen buffer)", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    trues_tr = torch.tensor(train["inhibition_clipped"].values, dtype=torch.float32)
    dose_mu_t = torch.tensor(dose_mu)

    # pre-encode train tensors once
    base, sug, bb, pad = encode_batch(train["aso_sequence_5_to_3"].tolist(),
                                      train["sugar_seq"].tolist(),
                                      train["backbone_seq"].tolist())
    gene_rows = torch.tensor([gene2row.get(g, gene2row["UNKNOWN"])
                              for g in train["target_gene"]], dtype=torch.long)
    cell_ids = torch.tensor([cell2id.get(c, cell2id["UNKNOWN"])
                             for c in train["cell_line"]], dtype=torch.long)
    dl = train["dosage_log10"].fillna(dose_mu).values.astype(np.float32)
    dm = train["dosage_log10"].isna().values.astype(np.float32)
    dl_t = torch.from_numpy(dl)
    dm_t = torch.from_numpy(dm)

    log_path = Path(args.out_dir) / "train_log.jsonl"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    best = -1.0
    n = len(train)
    for epoch in range(args.epochs):
        model.train()
        t0, run = time.time(), []
        perm = torch.randperm(n)
        for lo in range(0, n, args.batch_size):
            idx = perm[lo:lo + args.batch_size]
            out = model(base[idx], sug[idx], bb[idx], pad[idx],
                        gene_rows[idx], cell_ids[idx], dl_t[idx], dm_t[idx])
            loss = nn.functional.huber_loss(out, trues_tr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run.append(float(loss.item()))
        sched.step()
        model.eval()
        with torch.no_grad():
            ev_g = evaluate(model, val_group, gene2row, cell2id, device, dose_mu)
        msg = dict(epoch=epoch + 1, train_loss=float(np.mean(run)),
                   secs=round(time.time() - t0, 1), val_patent_group=ev_g)
        print(json.dumps(msg), flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(msg) + "\n")
        if ev_g["spearman"] > best:
            best = ev_g["spearman"]
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch + 1,
                        "metrics": ev_g, "cell2id": cell2id, "gene2row": gene2row,
                        "dose_mu": dose_mu},
                       Path(args.out_dir) / "ckpt_efficacy_v1.pt")

    # final: gene-holdout report with best ckpt
    ck = torch.load(Path(args.out_dir) / "ckpt_efficacy_v1.pt", weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    with torch.no_grad():
        ev_gene = evaluate(model, val_gene, gene2row, cell2id, device, dose_mu)
    print(f"[GENE-HOLDOUT] {json.dumps(ev_gene)} (unseen genes — mechanism-axis "
          f"generalization test)", flush=True)
    with open(log_path, "a") as f:
        f.write(json.dumps({"gene_holdout": ev_gene, "holdout_genes":
                            sorted(holdout_genes)}) + "\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
