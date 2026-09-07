"""siRNA efficacy head v1 — EXTERNAL held-out eval (Takayuki Ichihara 2007 set).

训练：Huesken (ichihara_2007_1, 2,431 guide strands, 30 targets) 全量，架构/超参
与 train_sirna_v1.py 完全一致（21mer guide transformer + 冻结 ESM2 靶轴, 60ep/bs256/lr1e-3）。

外部验证：ichihara_2007_2（OligoGym 过滤版 Takayuki Ichihara 集, 419 条,
12 靶点 EGFR/TP53/GAPDH/... 与训练集零靶点重叠——真正的 external held-out。
RNAGenesis 论文用原始 702 条版本，此处为 OligoGym CC-BY 同源子集）。

指标：pooled Spearman/Pearson + per-target Spearman（排名指标，跨实验批次稳健）。

Run locally (CPU, minutes): python eval_sirna_external.py
"""
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

torch.set_num_threads(8)
HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(HERE, "..", "..", "data", "oligogym", "ichihara_2007_1.csv.gz")
TEST_CSV = os.path.join(HERE, "..", "..", "data", "oligogym", "ichihara_2007_2.csv.gz")
ESM = os.path.join(HERE, "..", "..", "data", "drive_weights",
                   "Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt")
OUT = os.path.join(HERE, "..", "..", "results", "sirna_v1")
BASES = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3, "N": 4}
MAX_LEN = 24
SEED = 0
EPOCHS, BS, LR = 60, 256, 1e-3


def parse_guide(fasta_field):
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


class SiRNANet(torch.nn.Module):
    def __init__(self, esm_matrix, d=96, hidden=192, dropout=0.1):
        super().__init__()
        self.register_buffer("esm_table", esm_matrix)   # [V, 5120] frozen
        self.esm_dim = esm_matrix.shape[-1]
        self.base_emb = torch.nn.Embedding(5, 24, padding_idx=4)
        self.pos_emb = torch.nn.Embedding(MAX_LEN, 24)
        layer = torch.nn.TransformerEncoderLayer(24, 4, dim_feedforward=96,
                                                dropout=dropout, batch_first=True,
                                                norm_first=True)
        self.encoder = torch.nn.TransformerEncoder(layer, 2)
        self.proj_t = torch.nn.Linear(self.esm_dim, 64)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(48 + 64, hidden), torch.nn.GELU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, 1))

    def forward(self, ids, pad, tgt_rows):
        x = self.base_emb(ids) + self.pos_emb.weight[: ids.shape[1]].unsqueeze(0)
        x = self.encoder(x, src_key_padding_mask=pad)
        pooled = torch.cat([x.mean(1), x.max(dim=1).values], dim=-1)
        g = self.proj_t(self.esm_table[tgt_rows])
        return self.head(torch.cat([pooled, g], dim=-1)).squeeze(-1)


def metrics(y, p):
    rho = spearmanr(y, p).statistic if np.std(y) > 1e-9 and np.std(p) > 1e-9 else 0.0
    pr = np.corrcoef(y, p)[0, 1] if np.std(y) > 1e-9 and np.std(p) > 1e-9 else 0.0
    return dict(spearman=float(rho), pearson=float(pr))


def main():
    os.makedirs(OUT, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T0 = time.time()

    # ---- 数据 ----
    tr_df = pd.read_csv(TRAIN_CSV)
    te_df = pd.read_csv(TEST_CSV)
    n0 = len(te_df)
    te_df = te_df[te_df["targets"].astype(str).str.len() > 0].copy()  # 去空靶点行
    print(f"train: {len(tr_df)} rows / {tr_df['targets'].nunique()} targets | "
          f"test: {len(te_df)}/{n0} rows (dropped {n0-len(te_df)} empty-target) / "
          f"{te_df['targets'].nunique()} targets", flush=True)

    tab = torch.load(ESM, map_location="cpu", weights_only=False)
    sym2row = {s: i for i, s in enumerate(tab.keys())}
    esm_matrix = torch.stack(list(tab.values())).float()

    def build(df):
        guides = [parse_guide(f) for f in df["fasta"]]
        ids, pad = encode(guides)
        y = df["y"].values.astype(np.float32)
        targets = df["targets"].astype(str).values
        rows = np.array([sym2row.get(t, sym2row.get(t + "1", 0)) for t in targets],
                        dtype=np.int64)
        return dict(ids=torch.from_numpy(ids), pad=torch.from_numpy(pad),
                    y=y, targets=targets, rows=torch.from_numpy(rows))

    TR, TE = build(tr_df), build(te_df)
    covered = [t for t in set(TE["targets"]) if t in sym2row or t + "1" in sym2row]
    print(f"test targets mapped to ESM2: {len(covered)}/{len(set(TE['targets']))} "
          f"-> {sorted(covered)}", flush=True)

    # ---- 训练（全量 Huesken，同 v1 超参）----
    torch.manual_seed(SEED)
    model = SiRNANet(esm_matrix).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    y_mu, y_sd = TR["y"].mean(), TR["y"].std() + 1e-6
    n = len(TR["y"])
    tr_ids, tr_pad, tr_rows = TR["ids"].to(dev), TR["pad"].to(dev), TR["rows"].to(dev)
    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(n)
        for lo in range(0, n, BS):
            idx = torch.from_numpy(perm[lo:lo + BS])
            out = model(tr_ids[idx], tr_pad[idx], tr_rows[idx])
            yb = torch.from_numpy((TR["y"][perm[lo:lo + BS]] - y_mu) / y_sd).float().to(dev)
            loss = torch.nn.functional.huber_loss(out, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 20 == 0 or ep == 0:
            model.eval()
            with torch.no_grad():
                p_tr = model(tr_ids, tr_pad, tr_rows).cpu().numpy() * y_sd + y_mu
            print(f"ep{ep+1}: train spearman {metrics(TR['y'], p_tr)['spearman']:.4f} | "
                  f"{time.time()-T0:.0f}s", flush=True)

    torch.save(model.state_dict(), os.path.join(OUT, "sirna_v1_full_train.pt"))

    # ---- 外部评估 ----
    model.eval()
    with torch.no_grad():
        p_te = model(TE["ids"].to(dev), TE["pad"].to(dev),
                     TE["rows"].to(dev)).cpu().numpy() * y_sd + y_mu
    pooled = metrics(TE["y"], p_te)
    per_t = {}
    for t in sorted(set(TE["targets"])):
        m = np.array([i for i, x in enumerate(TE["targets"]) if x == t])
        per_t[t] = metrics(TE["y"][m], p_te[m])["spearman"]
    med_t = float(np.median([v for v in per_t.values() if np.isfinite(v)]))

    summary = {
        "model": "sirna v1 (Huesken full-train) -> external Takayuki/Ichihara_2007_2",
        "train": {"rows": int(n), "targets": int(tr_df["targets"].nunique())},
        "test": {"rows": int(len(TE["y"])), "dropped_empty_target": int(n0 - len(TE["y"])),
                 "targets": int(len(set(TE["targets"]))), "esm2_covered": len(covered)},
        "pooled": pooled,
        "per_target_spearman": per_t,
        "per_target_median": med_t,
        "internal_cv_reference": 0.6387,
    }
    json.dump(summary, open(os.path.join(OUT, "external_ichihara2.json"), "w"),
              indent=2, ensure_ascii=False)
    print(f"\n✅ EXTERNAL DONE {time.time()-T0:.0f}s | pooled spearman "
          f"{pooled['spearman']:.4f} / pearson {pooled['pearson']:.4f} | "
          f"per-target median {med_t:.4f}", flush=True)
    for t, v in per_t.items():
        print(f"   {t:12s} {v:+.3f}")


if __name__ == "__main__":
    main()
