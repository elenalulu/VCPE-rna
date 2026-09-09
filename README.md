<div align="center">

# VCPE-rna

**Virtual-Cell Perturbation-response Engine — RNA branch**

A license-clean (MIT / Apache-2.0 all the way down) **knockdown perturbation-response prediction engine**.
Given a knockdown perturbation (CRISPRi / siRNA / ASO target gene), it predicts the transcriptome-wide
response direction and magnitude — a commercially usable replacement for Non-Commercial weights.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)]()
[![Single GPU](https://img.shields.io/badge/Hardware-1%20x%2024GB%20GPU%20or%20CPU-orange.svg)]()

</div>

---

## Highlights

- **License-clean by construction** — architecture upstream is [MAP](https://github.com/MAGIC-AI4Med/MAP) (MIT) + MAP-KG (Apache-2.0); the RNA sequence encoder is self-trained; AIDO.RNA (Non-Commercial) is never used in any deliverable. That constraint is the reason this project exists.
- **A documented negative result turned win** — the P2 "pass" was retracted by our own diagnostics: conditioning was short-circuited by the shared-response core. The per-gene deviation head (P2.3-B) revived conditioning in **80 seconds** of training. Full evidence chain included.
- **Data scaling done cheaply** — 1,843 → 16,276 perturbations (×8.8) via scPerturb ingestion; legacy-domain pearson_dev **0.31 → 0.71** with a 5.7M-parameter head.
- **Zero-dependency platform integration** — the engine ships as a precomputed cache (`vcpe_cache.json.gz`) with a schema isomorphic to the platform's existing AIDO cache: swapping the file swaps the engine.

## Key Results

| Stage | Headline numbers | Verdict |
|---|---|---|
| **P1** — MAP base + knockdown conditioning | mse_DE **0.0215** (32% better than ctrl baseline) · fm_cosine 0.9604 | G2 conditional pass |
| **P2** — dual-axis (ESM2+KG ⊕ self-trained RNA encoder) | fm_cosine 0.9710 — **retracted by erratum**: conditioning short-circuited by shared response | ⚠️ see erratum |
| **P2.3-B** — GEARS-style per-gene deviation head | **Conditioning revived**: ablation_r 1.000 → 0.277 · pearson_dev **0.3079** · **80 s** training | G2''' pass |
| **Data expansion** (+6 scPerturb datasets) | perturbations 1,843 → **16,276** (×8.8) · legacy-domain pearson_dev **0.31 → 0.71** · overall 0.37 | pass |
| **ASO efficacy head v3** (ASO Atlas, patent-split CV) | pooled Spearman **0.283** · per-screen median 0.261 (GENCODE region features +0.034) | comparable to OligoAI baseline |
| **siRNA efficacy head v1** (Huesken 2,431 guides) | CV Spearman **0.607** · external Ichihara_2007_2 **0.588** | matches RNAGenesis benchmark |

> **Scientific honesty note.** P2's fm_cosine 0.9710 was initially judged a gate pass, then retracted:
> three-way diagnostics (`diag_fc.py`) showed the model had learned the dataset-level **shared
> knockdown signature** (K562 core response), with conditioning bypassed (`ablation_r = 1.000`).
> All four conventional metrics (mse / pearson / cosine / top50) can be inflated by the shared
> component. This is the single most important methodological lesson of the project — see
> [the P2 erratum](docs/reports/P2_result.md) and [P2.3-B report](docs/reports/P2.3-B_result.md).

## Parallel Tracks

| Module | Path | What it does |
|---|---|---|
| L2 RNA encoder | `src/l2/` | RNAcentral 45M seqs → 20–500 nt filter + dedup → MLM pretraining → InfoNCE alignment into ESM2 embedding space |
| Efficacy heads | `src/efficacy/` | ASO sequence + chemistry → knockdown depth (ASO Atlas, 188k gapmers); siRNA guide → efficiency (Huesken benchmark) |
| Data expansion | `src/data_expansion/`, `src/maprna_p3/ingest_*.py` | scPerturb (Zenodo) download → pseudobulk processing |

## Evidence Chain

| Report | Contents |
|---|---|
| [docs/PLAN.md](docs/PLAN.md) | Project plan v0.3: decisions D1–D3, staged go/no-go gates, benchmark protocol, license checklist |
| [docs/reports/p1_result.md](docs/reports/p1_result.md) | P1 gate report: 13-epoch trajectory, AMP spike diagnosis |
| [docs/reports/p2_result.md](docs/reports/p2_result.md) | P2 verdict + **erratum**: complete evidence chain of the shared-response shortcut |
| [docs/reports/p2_3b_result.md](docs/reports/p2_3b_result.md) | Why conditioning came back to life (per-gene direct head vs additive token) |
| [docs/reports/data_expansion_report.md](docs/reports/data_expansion_report.md) | ×8.8 expansion: per-domain comparison table and domain-gradient interpretation |
| [docs/reports/p3_platform_integration.md](docs/reports/p3_platform_integration.md) | Platform integration design: offline cache mode, zero new dependencies |
| [docs/reports/aso_sirna_data_landscape.md](docs/reports/aso_sirna_data_landscape.md) | Public ASO/siRNA data landscape (ASO Atlas and friends) |
| [docs/reports/l2_pretraining_log.md](docs/reports/l2_pretraining_log.md) | L2 encoder pretraining log |
| [docs/virtual_liver_p0_report.md](docs/virtual_liver_p0_report.md) | Virtual-liver P0: organ-level exam design, donor-context repair (P0.5), G0 adjudication |

## Repository Layout

```
src/
  maprna_p1/        P1 training (MAP base + knockdown conditioning; needs patched MAP repo + SE weights)
  maprna_p2/        P2 dual-axis training + RNA sequence encoder (InfoNCE alignment) + diag_fc diagnostics
  maprna_p3/        P2.3-B per-gene deviation head (standalone, no SE/MAP deps) + data ingestion + fair eval
  l2/               RNAcentral download → filter/dedup → tokenize → MLM pretrain → align
  efficacy/         ASO/siRNA efficacy heads (ASO Atlas / OligoGym / Huesken) + predict service
  data_expansion/   scPerturb Zenodo download & ingestion
  export_vcpe_cache.py   platform cache export (AIDO-cache-isomorphic schema)
docs/               PLAN, full evidence-chain reports
results/            train logs, CV metrics, eval artifacts (large ckpts not in git — see below)
data/               data acquisition guide (data itself is not redistributed)
```

## Quickstart

The P3 response head has **no dependency** on the SE backbone, the MAP repo, or flash-attention —
it reproduces in minutes on a single GPU or CPU:

```bash
pip install -r requirements.txt

# 1) Prepare data (see data/README.md):
#    - Perturb-seq h5ad in GEARS format (adamson / norman / replogle_rpe1_essential)
#    - ESM2 gene embedding table Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt

# 2) Train the P3 per-gene deviation head (60 epochs ≈ 80 s on GPU; CPU works too)
python src/maprna_p3/train_p3.py \
  --data_dirs data/adamson/perturb_processed.h5ad \
              data/norman/perturb_processed.h5ad \
              data/replogle_rpe1_essential/perturb_processed.h5ad \
  --esm_table data/drive_weights/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt \
  --out_dir ./p3_out --epochs 60

# 3) Full chain (P1/P2 need the MAP base + SE weights, 1 × 24G GPU)
#    Command templates live in each script's docstring and in docs/PLAN.md
```

**Pretrained artifacts.** `results/p3_v21e/` ships the aligned RNA sequence encoder
(`rna_enc_from_ckpt.pt`, 9 MB) and an example full-gene vector cache. Full checkpoints
(P3 response head 428 MB, efficacy heads ~410 MB, …) are planned for GitHub Releases /
HuggingFace and are intentionally kept out of git.

## License & Attribution

- Code in this repository: **MIT License** (see [LICENSE](LICENSE))
- Architecture upstream: [MAP](https://github.com/MAGIC-AI4Med/MAP) (MIT) + MAP-KG (Apache-2.0); the P2.3-B deviation head is an independent implementation
- Per-dataset licenses and acquisition paths: [data/README.md](data/README.md)
- ⚠️ **ASO Atlas is derived from USPTO patents.** This repository ships training/eval code only, never the data; any commercial use requires independent legal review
- AIDO.RNA-650M (Non-Commercial) was used for internal ablation only and appears in no deliverable — which is precisely why this project exists

## Methodology Notes (selected)

1. **Evaluation must include shared-component-free discrimination metrics** (`pearson_dev` / `top50_dev` + zeroPert ablation) — otherwise any gate can be fooled by the shared response (the P2 lesson).
2. **Additive tokens get structurally drowned** in a large backbone's residual stream — in data-limited regimes, a per-gene direct residual head is the right architecture (the P2.3-B conclusion).
3. **Re-calibrate LR after data scaling**: ×8.8 data + lr 1e-3 collapses into the zero-residual basin; lr 3e-4 trains 40 epochs with zero instability.
