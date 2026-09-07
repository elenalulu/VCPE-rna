# Data Acquisition Guide

VCPE-rna does **not redistribute any dataset** (size and licensing reasons). This file lists, module by
module, what each component needs, where to get it, and under which license. Download scripts:
`src/l2/download_rnacentral.py`, `src/data_expansion/download_scperturb.py`, `src/efficacy/download_oligogym.py`.

## 1. Main chain: Perturb-seq (GEARS-format h5ad)

| Dataset | Used for | Source | License |
|---|---|---|---|
| adamson / norman / replogle_rpe1_essential `perturb_processed.h5ad` | P1 / P2 / P3 training | [GEARS](https://github.com/snap-stanford/GEARS) data release (~0.5–2.7 GB each) | see GEARS repo |
| ReplogleWeissman2022 K562 gwps, NadigOConner2024 hepg2/jurkat, et al. (6 datasets) | data-expansion track | [scPerturb, Zenodo record 13350497](https://zenodo.org/records/13350497) (`src/data_expansion/download_scperturb.py`, supports `--all`) | CC BY 4.0 |
| ESM2 gene embedding table `Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt` (392 MB) | conditioning lookup | MAP released weights (see below) | presumed per MAP repo (MIT); open an issue to confirm before commercial use |
| `gene_transcripts.fa` | RNA sequence axis | extracted from GENCODE v44 transcript FASTA | GENCODE license |

## 2. MAP base weights (only P1/P2 need them; the P2.3-B head does not)

From the MAP / MAP-KG releases ([MAGIC-AI4Med/MAP](https://github.com/MAGIC-AI4Med/MAP), MIT;
HF `RainGate/MAP-KG`, Apache-2.0):

| File | Size |
|---|---|
| `epoch_3.pt` (perturbation-predictor base) | 4.2 GB |
| `se600m.safetensors` (SE cell encoder) | 2.4 GB |
| `Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt` | 392 MB |
| `mapkg_encoder_v3.pt` (KG encoder) | 1.1 GB |

P1/P2 training additionally needs the patched MAP repo (10 engineering patches by this team; patch
notes in `docs/gpu_pack/`).

## 3. Efficacy heads (`src/efficacy/`)

| Dataset | Used for | Source | License | ⚠️ |
|---|---|---|---|---|
| ASO Atlas v1/v2 (188k gapmers, chemistry + potency) | ASO efficacy head | HF `barneyhill/aso-atlas` / `aso-atlas-2` (OligoAI paper companion) | see HF dataset page | **derived from USPTO patents**; independent legal review required before any commercial use; not included in this repo |
| OligoGym, 12 datasets (ichihara / alharbi / huesken, etc.) | siRNA/ASO external eval | HF `CollageBio/oligo-datasets` (direct download via `download_oligogym.py`) | CC BY 4.0 | — |
| GSE183535 (MYC ASO RNA-seq), GSE289964 (in vivo whole liver), GSE293987 | domain-transfer eval | GEO | see each series | — |

## 4. L2 RNA encoder pretraining (`src/l2/`)

| Dataset | Source | License |
|---|---|---|
| RNAcentral active fasta (~11 GB, 45M seqs) | [RNAcentral FTP](https://ftp.ebi.ac.uk/pub/databases/RNAcentral/current_release/) (`download_rnacentral.py`, resumable) | CC0 |
| STRING v12 human edges | [string-db.org](https://string-db.org) | CC BY 4.0 |

## 5. Miscellaneous

| Dataset | Used for | Source |
|---|---|---|
| GTEx v10 gene median TPM | primary-hepatocyte donor context (`p05_donor_context.py`) | [gtexportal.org](https://www.gtexportal.org) |
| MAP-KG knowledge-graph CSVs | KG alignment | HF `RainGate/MAP-KG` (Apache-2.0) |

## Conventions

- Default data root: `data/` inside the repository (git-ignored); some scripts honor `VCPE_DATA_DIR`
- Why not committed: ~35 GB total + several licenses that do not allow redistribution
