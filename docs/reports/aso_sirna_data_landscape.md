# ASO / siRNA Data Landscape (2026-09-03, three-track search with evidence)

> Purpose: map the public ASO/siRNA data space for VCPE-rna's "sequence-modality expansion"
> (Tier 2/3) and the data-expansion track.
> Conclusion up front: **one heavyweight finding on the ASO side (ASO Atlas, 188 k chemically
> modified potency records); siRNA efficiency-prediction data is mature (classic benchmarks all
> available); both connect into our existing chained architecture.**

## 1. ASO data (three classes, different uses)

### A1. Sequence/chemistry → potency (the ticket to Tier 2/3) ⭐⭐⭐

| Resource | Scale | Contents | Source |
|---|---|---|---|
| **ASO Atlas** ⭐ | **188,521 gapmer ASOs**, 334 target genes, 417 USPTO patents (2001–2025) | ASO sequence + **chemistry annotations** (52.6% 5-10-5 2'-MOE gapmer, 30.4% 3-10-3 cEt, PS backbone patterns) + target gene + qRT-PCR knockdown potency; multiple cell lines (A-431 44,855 / HepG2 23,428 / SH-SY5Y 18,192); full potency spread (median 45%, 17.4% ineffective <10%) | OligoAI paper (bioRxiv 2025.10.29.685292), companion ML method published |
| **ASOptimizer database** | **187,090 records**, 67 target mRNAs | literature+patent full set: sequence, chemistry type/position, **dose (nM), treatment duration (h), cell line, transfection conditions**, qRT-PCR inhibition | Molecular Therapy NA 2024 (ASOptimizer paper), platform open-sourced |
| **RNATx-Bench ASO task** | 34,773 validated ASOs | 4 clinically relevant targets (SOD1/K-RAS/APOL1/SNHG14), sequence + inhibition-regression labels | RNAGenesis paper (bioRxiv 2024.12.30) benchmark data |

**Use**: train "ASO sequence + chemistry → target knockdown depth" — the direct optimization target
of ASO design (the ASO counterpart of "siRNA sequence efficiency is priority"). **Note**: this is
single-target qPCR potency, not transcriptome response — it feeds the "efficiency modulator",
complementary (not alternative) to VCPE's response prediction.

**Compliance note**: ASO Atlas / ASOptimizer data derive from **patent documents** (USPTO/Lens,
public). The patent texts themselves are public, but **the redistribution and commercial-use boundary
of data mined from them requires a dedicated legal review** — the single compliance checkpoint of
this track.

### A2. ASO transcriptome response (Tier-1 validation / future fine-tuning) ⭐⭐

| GEO accession | Contents | Value |
|---|---|---|
| **GSE293987** (Ionis, public 2025-11) | **CR-DGE concentration-gradient workflow**: A431 cells, ACTN1 gapmer ASO + non-targeting controls, **concentration-response** 3'Tag-Seq, 53 samples; Nucleic Acid Ther 2025 | **dose dimension + Ionis official methodology** — the gold-standard protocol for ASO transcriptome response; validates concentration dependence of our response prediction |
| **GSE289964** (2025-03) | **in vivo**: mouse liver, 3 chemistry gapmer ASOs (Scarb1) + PBS, **day 1/3/7 time course**, 48 samples | rare in vivo + multi-chemistry + time dimension |
| **PRJNA787253** (2021) | Huh-7 cells, LNA gapmer **off-target assessment**, microarray, n=4/group; conclusion: off-target degree correlates with complementarity | **direct data for off-target modeling** — ASO sequence similarity → off-target gene sets; plugs into our off-target pipeline |
| **GSE183535** (2021) | HeLa, MYC gapmer ASO, 10 nM, 4 h/18 h, 21 samples | small and clean time course |

**Scale judgment**: transcriptome-level ASO data is small in total (single-digit GSEs, hundreds of
samples) — enough for a **validation set** (testing CRISPR→ASO transfer bias) and later small-scale
fine-tuning, not for main training.

### A3. ASO off-target sequence rules
- PRJNA787253's quantitative "complementarity → off-target" finding + CR-DGE's selectivity spectrum →
  directly strengthen the scoring rules of our off-target search (seed-complementarity weighting).

## 2. siRNA data

### B1. Efficiency prediction (sequence → silencing efficacy) — mature benchmarks ⭐⭐

| Dataset | Scale | Note |
|---|---|---|
| **Huesken 2005** | 2,361 records | the classic benchmark (24-mers, 34 mouse/human genes); the training set of nearly every siRNA efficiency model |
| **Takayuki** | 702 records | second RNAGenesis benchmark set |
| **shRNA efficacy set** | 2,076 records | RNAGenesis shRNA task (continuous labels) |
| siRNA design-rule literature | — | the 2004–2012 rules era + subsequent ML (incl. off-target variables), well documented |

**Use**: the siRNA efficiency module — with these two benchmarks plus ASO Atlas's methodology, work
can start immediately. RNAGenesis has the eval protocol (AUROC/AUPRC) fixed, which makes
benchmarking straightforward.

### B2. RNAi transcriptome response (starter data for Tier-2 fine-tuning) ⭐⭐

| Resource | Scale | Key point |
|---|---|---|
| **LINCS GSE106127** (CMap RNAi assessment) | **13,000+ shRNAs / 9 cell lines** (L1000) + 373 sgRNAs / 6 cell lines | Levels 4/5 available; **CGS (consensus gene signatures, Level 6)**: merging same-gene shRNAs ≈ 4,370 usable profiles |
| **GSE92742 / GSE70138** (LINCS Phase 1/2) | 5,806 genetic perturbations (shRNA KD + OE) | 978 landmark genes + ~10 k BING genes inferable |

**⚠️ The most important honest warning (from the CMap official analysis)**: shRNA
**miRNA-like seed off-target effects are "stronger and more pervasive than commonly appreciated"** —
shRNAs sharing a seed are more similar to each other than different shRNAs against the same gene.
The same holds for siRNA (same mechanism). Implications:
1. "Responses" in shRNA/siRNA data are mixed with substantial seed-driven off-target signal — using
   them directly as an on-target response training set introduces systematic bias
2. Good news: our CRISPR data (X-Atlas/PerturbDB) has **no seed problem** — CRISPR transcriptome
   responses are cleaner, which strengthens the "CRISPR learns responses + ASO Atlas learns potency"
   division of labor
3. When using LINCS CGS as Tier 2, filter seed matches with our existing off-target pipeline first

## 3. The puzzle: how ASO/siRNA data plugs into the VCPE architecture

```
                    ┌─ CRISPR Perturb-seq (X-Atlas/PerturbDB, expanding)
                    │   → response half: target knockdown → whole-transcriptome response
                    │     (pearson_dev 0.31 validated, 0.71 after expansion)
                    │
VCPE-rna engine ────┼─ ASO Atlas 188k / RNATx-Bench 34.8k
                    │   → potency half: ASO sequence + chemistry → target knockdown depth
                    │     (dose/time adjustable)  ⚠️ patent-derived, legal review before
                    │        commercial use
                    │
                    └─ 4 GEO ASO transcriptome GSEs
                        → validation set: CRISPR→ASO mechanism-transfer bias (RNase H1 vs DNA level)
                        → off-target rule strengthening (complementarity → off-targets)
```

**Prediction shape (full Tier-2 chain)**: user provides an ASO sequence → the off-target pipeline
yields the target spectrum → the response engine yields per-target transcriptome response → the
ASO-Atlas-trained potency head applies **knockdown-depth modulation** (incl. dose) → final
synthesized prediction. Every piece's data source is now identified and public.

## 4. Action list (suggested priority)

| # | Action | Cost | Output |
|---|---|---|---|
| 1 | Download ASO Atlas (OligoAI companion) + RNATx-Bench, inspect formats | half day | efficacy-module data in place |
| 2 | ASO efficacy head v1 (sequence + chemistry → inhibition; RNAGenesis protocol as reference) | 1–2 days | the siRNA/ASO sequence-efficiency priority landed |
| 3 | Ingest the 4 ASO transcriptome GSEs → ASO validation set | 1 day | first CRISPR→ASO transfer-bias measurement |
| 4 | Legal review: commercial boundary of patent-derived data (ASO Atlas/ASOptimizer) | external/self | compliance conclusion |
| 5 | Evaluate LINCS CGS 4,370 (after seed filtering) as the Tier-2 fine-tuning pool | 1 day | decision on Tier-2 main training-set composition |

## Appendix: search provenance

- Search date: 2026-09-03, three tracks (ASO transcriptome / siRNA LINCS / potency & chemistry
  databases)
- Key sources: bioRxiv 2025.10.29.685292 (OligoAI/ASO Atlas), Mol Ther NA 2024 (ASOptimizer),
  bioRxiv 2024.12.30 (RNAGenesis/RNATx-Bench), NCBI GEO (GSE293987/GSE289964/GSE183535/GSE106127/
  GSE92742/GSE70138), PRJNA787253
