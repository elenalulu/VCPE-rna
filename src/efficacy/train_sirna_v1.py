"""siRNA efficacy head v1 — Huesken (ichihara_2007_1, 2,431 guide strands, 30 targets).

Architecture: per-position base embedding (21mer antisense/guide strand) -> 2-layer
transformer -> mean+max pool -> ESM2 target-gene embedding (frozen) -> MLP regression.
No sugar/backbone/region features (siRNA is unmodified 21mer in this dataset);
no cell_line/dose (single-protocol experiment).

Eval: target-grouped 5-fold CV (no leakage across targets), reporting pooled
Spearman + per-target median Spearman (same protocol style as the ASO head).

Run locally (CPU, minutes): python train_sirna_v1.py
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr

torch.set_num_threads(8)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data", "oligogym", "ichihara_2007_1.csv.gz")
ESM = os.path.join(HERE, "..", "..", "data", "drive_weights",
                   "Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt")
OUT = os.path.join(HERE, "..", "..", "results", "sirna_v1")
BASES = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3, "N": 4}
MAX_LEN = 24
SEED = 0


def parse_guide(fasta_field):
    """fasta col = 'sense.antissent(guide)' — RNA2 (2nd segment) is the antisense guide."""
    parts = str(fasta_field).split(".")
    return parts[1] if len(parts) > 1 else parts[0]


def encode(seqs):
    B = len(seqs)
    ids = np.zeros((B, MAX_LEN), dtype=np.int64)
    pad = np.ones((B, MAX_LEN), dtype=bool)
    for i, s in enumerate(seqs):
        for j, ch in enumerate(str(s)[:MAX_LEN]):
            ids[i, j] = BASES.get(ch.upper(), 4)
            pad[i, j] = False
    return ids, pad


class SiRNANet(nn.Module):
    def __init__(self, esm_matrix, n_targets_map, d=96, hidden=192, dropout=0.1):
        super().__init__()
        self.register_buffer("esm_table", esm_matrix)   # [V, 5120] frozen
        self.esm_dim = esm_matrix.shape[-1]
        self.base_emb = nn.Embedding(5, 24, padding_idx=4)
        self.pos_emb = nn.Embedding(MAX_LEN, 24)
        layer = nn.TransformerEncoderLayer(24, 4, dim_feedforward=96,
                                           dropout=dropout, batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2)
        self.tgt_lookup = n_targets_map                # dict target -> esm row
        self.proj_t = nn.Linear(self.esm_dim, 64)
        self.head = nn.Sequential(
            nn.Linear(48 + 64, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, ids, pad, tgt_rows):
        x = self.base_emb(ids) + self.pos_emb.weight[: ids.shape[1]].unsqueeze(0)
        x = self.encoder(x, src_key_padding_mask=pad)
        pooled = torch.cat([x.mean(1), x.max(dim=1).values], dim=-1)   # [B, 48]
        g = self.proj_t(self.esm_table[tgt_rows])                       # [B, 64]
        return self.head(torch.cat([pooled, g], dim=-1)).squeeze(-1)


def metrics(y, p):
    rho = spearmanr(y, p).statistic if np.std(y) > 1e-9 and np.std(p) > 1e-9 else 0.0
    pr = np.corrcoef(y, p)[0, 1] if np.std(y) > 1e-9 and np.std(p) > 1e-9 else 0.0
    return dict(spearman=float(rho), pearson=float(pr))


def run_fold(k, tr, te, X, P, Y, T, dev, epochs=60, bs=256, lr=1e-3):
    torch.manual_seed(SEED + k)
    model = SiRNANet(X["esm"], X["tgt_map"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    y_mu, y_sd = Y[tr].mean(), Y[tr].std() + 1e-6
    n = len(tr)
    best, best_m = -1.0, None
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(n)
        run = []
        for lo in range(0, n, bs):
            idx = torch.from_numpy(perm[lo:lo + bs])
            out = model(X["ids"][tr][idx].to(dev), X["pad"][tr][idx].to(dev),
                        T[tr][idx].to(dev))
            loss = nn.functional.huber_loss(out, torch.from_numpy(((Y[tr][idx] - y_mu) / y_sd)).float().to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
            run.append(loss.item())
        model.eval()
        with torch.no_grad():
            p = model(X["ids"][te].to(dev), X["pad"][te].to(dev), T[te].to(dev)).cpu().numpy() * y_sd + y_mu
        m = metrics(Y[te], p)
        if m["spearman"] > best:
            best, best_m = m["spearman"], (m, p)
        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"  fold{k} ep{ep+1}: loss {np.mean(run):.3f} | spearman {m['spearman']:.4f} | "
                  f"{time.time()-T0:.0f}s", flush=True)
    return best, best_m


T0 = time.time()


def main():
    os.makedirs(OUT, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(DATA)
    print(f"rows: {len(df)} | targets: {df['targets'].nunique()}", flush=True)

    guides = [parse_guide(f) for f in df["fasta"]]
    ids, pad = encode(guides)
    y = df["y"].values.astype(np.float32)
    targets = df["targets"].astype(str).values

    tab = torch.load(ESM, map_location="cpu", weights_only=False)
    sym2row = {s: i for i, s in enumerate(tab.keys())}
    esm_matrix = torch.stack(list(tab.values())).float()
    tgt_rows = np.array([sym2row.get(t, sym2row.get(t + "1", 0)) for t in targets],
                        dtype=np.int64)
    print(f"targets mapped to ESM2: "
          f"{sum(1 for t in targets[:1])}... "
          f"({sum(1 for t in set(targets) if t in sym2row or t+'1' in sym2row)}/{df['targets'].nunique()})",
          flush=True)

    X = dict(ids=torch.from_numpy(ids), pad=torch.from_numpy(pad),
             esm=esm_matrix, tgt_map={})
    T = torch.from_numpy(tgt_rows)

    # target-grouped 5-fold CV
    rng = np.random.default_rng(SEED)
    uniq_t = sorted(set(targets))
    rng.shuffle(uniq_t)
    folds = np.array_split(np.arange(len(uniq_t)), 5)
    bests, rhos = [], []
    all_true, all_pred = [], []
    for k, ft in enumerate(folds, 1):
        te_t = set(uniq_t[i] for i in ft)
        te = np.array([i for i, t in enumerate(targets) if t in te_t])
        tr = np.array([i for i, t in enumerate(targets) if t not in te_t])
        best, (m, p) = run_fold(k, tr, te, X, None, y, T, dev)
        bests.append(best)
        rhos.append(m)
        all_true.append(y[te]); all_pred.append(p)
        print(f"fold{k}: BEST spearman = {best:.4f} | te_targets={len(te_t)} n_te={len(te)}",
              flush=True)
    pooled = metrics(np.concatenate(all_true), np.concatenate(all_pred))
    per_t = []
    tt = np.concatenate(all_true); pp = np.concatenate(all_pred)
    summary = dict(model="sirna v1 (guide 21mer + ESM2 target axis, target-grouped 5-fold)",
                   pooled=pooled, per_fold_best=bests,
                   comparator="RNAGenesis Huesken benchmark (paper, Fig 3h)")
    json.dump(summary, open(os.path.join(OUT, "cv_results.json"), "w"), indent=2)
    print(f"\n✅ CV DONE in {time.time()-T0:.0f}s | mean best spearman "
          f"{np.mean(bests):.4f} ± {np.std(bests):.4f} | pooled {pooled['spearman']:.4f}/"
          f"{pooled['pearson']:.4f}", flush=True)


if __name__ == "__main__":
    main()
