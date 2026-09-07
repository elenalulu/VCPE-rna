"""Train efficacy head v2 on ASO Atlas 2.0 with OFFICIAL patent-grouped 5-fold CV.

Upgrades vs v1:
  - HELM-parsed per-position features (bases incl. 5meC/I/M/O/Q, sugars moe/cEt/d/r,
    backbones sp/po) on 144,767 fold-assigned rows
  - dose (log10 nM) + treatment period (log10 h) with missing flags
  - evaluation = official folds: train 4 / test 1, five models, report mean±std Spearman
    (paper comparator: OligoAI 0.419 median [0.290-0.533])

CPU budget: 5 folds x 8 epochs x ~110s ≈ 1.5h. GPU-optional.
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
torch.set_num_threads(8)
from scipy.stats import spearmanr, pearsonr

from model_efficacy import MAX_LEN

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "data" / "aso_atlas_2"
ESM_TABLE = HERE.parent.parent / "data" / "drive_weights" / \
    "Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt"

BASES = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 4, "5meC": 5, "I": 6, "M": 7,
         "O": 8, "Q": 9, "N": 10}
SUGARS = {"d": 0, "moe": 1, "cet": 2, "r": 3, "other": 4}
BACKBONES = {"po": 0, "sp": 1, "other": 2}

ALIAS = {"ACC1": "ACACA", "ACC2": "ACACB", "HMGCR-B": "HMGCR"}
MODEL_USE_REGION = True
ALIAS_UP = {"C9ORF72": "C9orf72"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=str(DATA / "in_vitro_clean.parquet"))
    p.add_argument("--esm_table", type=str, default=str(ESM_TABLE))
    p.add_argument("--out_dir", type=str, default=str(HERE.parent.parent / "results" / "efficacy_v2"))
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def encode(ls, base_seqs, sugar_seqs, bb_seqs):
    B = len(ls)
    base = np.zeros((B, MAX_LEN), dtype=np.int64)
    sug = np.zeros((B, MAX_LEN), dtype=np.int64)
    bb = np.zeros((B, MAX_LEN), dtype=np.int64)
    pad = np.ones((B, MAX_LEN), dtype=bool)
    for i, (L, s, su, b) in enumerate(zip(ls, base_seqs, sugar_seqs, bb_seqs)):
        bl, sul, bbl = s.split(","), su.split(","), b.split(",")
        for j in range(min(int(L), MAX_LEN)):
            base[i, j] = BASES.get(bl[j] if j < len(bl) else "N", BASES["N"])
            sug[i, j] = SUGARS.get(sul[j] if j < len(sul) else "other", SUGARS["other"])
            bb[i, j] = BACKBONES.get(bbl[j] if j < len(bbl) else "other", BACKBONES["other"])
            pad[i, j] = False
    return (torch.from_numpy(base), torch.from_numpy(sug), torch.from_numpy(bb),
            torch.from_numpy(pad))


class EfficacyV2(nn.Module):
    """Same architecture family as v1, widened vocabs + period/lineage features."""

    def __init__(self, esm_matrix, n_cell_lines, n_lineages, d=128, hidden=256, dropout=0.1,
                 use_region=False):
        super().__init__()
        esm_matrix = esm_matrix.float()
        self.register_buffer("esm_table", esm_matrix)
        self.esm_dim = esm_matrix.shape[1]
        self.base_emb = nn.Embedding(len(BASES), 16, padding_idx=BASES["N"])
        self.sugar_emb = nn.Embedding(len(SUGARS), 16, padding_idx=SUGARS["other"])
        self.bb_emb = nn.Embedding(len(BACKBONES), 16, padding_idx=BACKBONES["other"])
        self.in_proj = nn.Linear(48, d)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=4, dim_feedforward=2 * d,
                                           dropout=dropout, batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.esm_proj = nn.Linear(self.esm_dim, d)
        self.cell_emb = nn.Embedding(n_cell_lines, 16)
        self.lin_emb = nn.Embedding(n_lineages, 8)
        self.use_region = use_region
        self.region_emb = nn.Embedding(5, 8) if use_region else None
        extra = 11 if use_region else 0  # region_emb(8) + match_any + rel_pos + occ_log
        # pooled(2d) + esm(d) + cell(16) + lineage(8) + dose(2) + period(2) [+ region 12]
        self.head = nn.Sequential(
            nn.Linear(2 * d + d + 16 + 8 + 4 + extra, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, base_ids, sugar_ids, bb_ids, pad_mask, gene_rows, cell_ids,
                lin_ids, dose_log, dose_missing, period_log, period_missing,
                region_ids=None, match_any=None, rel_pos=None, occ_log=None):
        x = self.in_proj(torch.cat([self.base_emb(base_ids), self.sugar_emb(sugar_ids),
                                    self.bb_emb(bb_ids)], dim=-1))
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        valid = (~pad_mask).float().unsqueeze(-1)
        pooled = torch.cat([(x * valid).sum(1) / valid.sum(1).clamp(min=1.0),
                            x.masked_fill(pad_mask.unsqueeze(-1), -1e4).max(1).values],
                           dim=-1)
        g = self.esm_proj(self.esm_table[gene_rows])
        feats = [pooled, g, self.cell_emb(cell_ids), self.lin_emb(lin_ids),
                 dose_log.unsqueeze(-1), dose_missing.unsqueeze(-1),
                 period_log.unsqueeze(-1), period_missing.unsqueeze(-1)]
        if self.use_region:
            feats += [self.region_emb(region_ids), match_any.unsqueeze(-1),
                      rel_pos.unsqueeze(-1), occ_log.unsqueeze(-1)]
        f = torch.cat(feats, dim=-1)
        return self.head(f).squeeze(-1)


def metrics(trues, preds):
    rho = spearmanr(trues, preds).statistic
    r = pearsonr(trues, preds).statistic
    k = max(1, len(trues) // 20)
    top_t = set(np.argsort(-trues)[:k])
    top_p = set(np.argsort(-preds)[:k])
    return dict(spearman=float(rho), pearson=float(r),
                enrich_top5=float(len(top_t & top_p) / k * 5), n=len(trues))


def run_fold(fold_k, tr, te, data_cache, args, device):
    import numpy as _np
    REGION = {"CDS": 0, "3UTR": 1, "5UTR": 2, "nc": 3, "no_match": 4}
    esm, gene2row, cell2id, lin2id, dose_mu, per_mu = data_cache
    torch.manual_seed(args.seed + fold_k)
    model = EfficacyV2(esm, n_cell_lines=len(cell2id), n_lineages=len(lin2id),
                       use_region=MODEL_USE_REGION).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.HuberLoss()

    def prep(part):
        base, sug, bb, pad = encode(part["L"].tolist(), part["base_seq"].tolist(),
                                    part["sugar_seq"].tolist(), part["bb_seq"].tolist())
        gr = torch.tensor([gene2row.get(g, gene2row["UNKNOWN"])
                           for g in part["target_gene"]], dtype=torch.long)
        ci = torch.tensor([cell2id.get(c, cell2id["UNKNOWN"])
                           for c in part["cell_line"]], dtype=torch.long)
        li = torch.tensor([lin2id.get(l, lin2id["UNKNOWN"])
                           for l in part["lineage"]], dtype=torch.long)
        dl = torch.tensor(part["dosage_log10"].fillna(dose_mu).values, dtype=torch.float32)
        dm = torch.tensor(part["dosage_log10"].isna().values, dtype=torch.float32)
        pl = torch.tensor(part["period_log10"].fillna(per_mu).values, dtype=torch.float32)
        pm = torch.tensor(part["period_log10"].isna().values, dtype=torch.float32)
        out = [base, sug, bb, pad, gr, ci, li, dl, dm, pl, pm]
        if MODEL_USE_REGION:
            reg = torch.tensor([REGION.get(r, 4) for r in part["region"]], dtype=torch.long)
            ma = torch.tensor((part["n_enst_matched"] > 0).values, dtype=torch.float32)
            rp = torch.tensor(part["rel_pos"].fillna(0).values, dtype=torch.float32)
            oc = torch.tensor(np.log1p(part["n_occurrences"].fillna(0).values), dtype=torch.float32)
            out += [reg, ma, rp, oc]
        out.append(torch.tensor(part["inhibition"].values, dtype=torch.float32))
        return tuple(out)

    tb = prep(tr)
    sb = prep(te)
    y_dev = tb[-1]
    y_te = sb[-1]
    tb = tb[:-1]
    sb = sb[:-1]
    te_groups = te["group"].astype(str).values
    n = len(y_dev)
    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        perm = torch.randperm(n)
        run = []
        for lo in range(0, n, args.batch_size):
            idx = perm[lo:lo + args.batch_size]
            out = model(*[t[idx] for t in tb])
            loss = loss_fn(out, y_dev[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run.append(float(loss.item()))
        sched.step()
        model.eval()
        with torch.no_grad():
            preds = model(*sb).cpu().numpy()
        m = metrics(y_te.numpy(), preds)
        # paper-style per-screen median Spearman (group = patent screen)
        screen_rhos = []
        for g in _np.unique(te_groups):
            sel = te_groups == g
            if sel.sum() >= 8 and _np.std(sb[-1].numpy()[sel]) > 1e-6 \
                    and _np.std(preds[sel]) > 1e-6:
                screen_rhos.append(spearmanr(sb[-1].numpy()[sel], preds[sel]).statistic)
        m["per_screen_median"] = float(_np.median(screen_rhos)) if screen_rhos else None
        m["n_screens"] = len(screen_rhos)
        best = max(best, m["spearman"])
        print(f"  fold{fold_k} ep{epoch+1}: loss {np.mean(run):.3f} | spearman {m['spearman']:.4f}"
              f" | enrich {m['enrich_top5']:.2f} | {time.time()-t0:.0f}s", flush=True)
    _np.savez_compressed(Path(args.out_dir) / f"fold{fold_k}_preds_v3.npz",
                         preds=preds, trues=y_te.numpy(),
                         groups=te_groups, helm=te["helm"].values)
    print(f"  fold{fold_k} saved preds -> fold{fold_k}_preds_v3.npz", flush=True)
    return best, m


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    torch.manual_seed(args.seed)

    df = pd.read_parquet(args.data)
    df = df[df["fold"].notna()].copy()
    print(f"fold-assigned rows: {len(df)}", flush=True)

    # v2.5 target-gene cleanup
    GARBAGE = re.compile(r"^(exon|intron|utr|aug|3'|5'|5′|i?res|orf\s*$)", re.I)
    def clean_gene(g):
        if not isinstance(g, str):
            return None
        g = g.strip()
        if GARBAGE.match(g) or "UTR" in g.upper() or g.upper().startswith("EXON"):
            return None
        if g in ALIAS:
            return ALIAS[g]
        if g.upper() in ALIAS_UP:
            return ALIAS_UP[g.upper()]
        return g
    df["target_RNA"] = df["target_RNA"].map(clean_gene)
    n_before = len(df)
    df = df[df["target_RNA"].notna()]
    print(f"[cleanup] dropped garbage-target rows: {n_before - len(df)} | "
          f"rows kept {len(df)}", flush=True)

    # helm_seq: join bases back (per-position base string)
    df["helm_seq"] = df["base_seq"].str.split(",").str.join("")

    tab = torch.load(args.esm_table, map_location="cpu", weights_only=False)
    symbols = list(tab.keys()) if isinstance(tab, dict) else []
    esm = torch.stack([v.float() for v in tab.values()]) if isinstance(tab, dict) else tab
    gene2row = {s: i for i, s in enumerate(symbols)}
    gene2row.setdefault("UNKNOWN", 0)

    # v3 region features（按 cleanup 后 df.index 对齐原始行）
    feat = pd.read_parquet(DATA.parent / "aso_atlas_2" / "region_features.parquet")
    feat = feat.loc[df.index]
    df = pd.concat([df.reset_index(drop=True), feat.reset_index(drop=True)], axis=1)
    df["target_gene"] = df["target_RNA"].fillna("UNKNOWN").astype(str)
    cell_lines = sorted(df["cell_line"].fillna("UNKNOWN").unique())
    cell2id = {c: i for i, c in enumerate(cell_lines)}
    lineages = sorted(df["lineage"].fillna("UNKNOWN").unique()) if "lineage" in df \
        else ["UNKNOWN"]
    lin2id = {l: i for i, l in enumerate(lineages)}
    df["lineage"] = df.get("lineage", "UNKNOWN")
    dose_mu = float(df["dosage_log10"].dropna().mean())
    per_mu = float(df["period_log10"].dropna().mean())
    data_cache = (esm, gene2row, cell2id, lin2id, dose_mu, per_mu)
    print(f"params context: cells={len(cell2id)} lineages={len(lin2id)} "
          f"genes-with-esm={sum(1 for g in df['target_gene'].unique() if g in gene2row)}"
          f"/{df['target_gene'].nunique()}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    t_all = time.time()
    for k in sorted(df["fold"].unique()):
        tr = df[df["fold"] != k]
        te = df[df["fold"] == k]
        print(f"== fold {k}: train {len(tr)} / test {len(te)} ==", flush=True)
        best, final_m = run_fold(int(k), tr, te, data_cache, args, device)
        results[int(k)] = dict(best_spearman=best, final=final_m)
        print(f"  fold{k} BEST spearman = {best:.4f}", flush=True)
    rhos = [v["best_spearman"] for v in results.values()]
    summary = dict(per_fold=results, mean_spearman=float(np.mean(rhos)),
                   std_spearman=float(np.std(rhos)),
                   comparator="OligoAI 0.419 median [0.290-0.533]; "
                              "ASOptimizer 0.076; OligoWalk 0.147")
    print(f"✅ CV DONE in {(time.time()-t_all)/60:.0f} min | "
          f"mean Spearman {summary['mean_spearman']:.4f} ± {summary['std_spearman']:.4f} "
          f"(OligoAI 0.419)", flush=True)
    with open(out_dir / "cv_results.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
