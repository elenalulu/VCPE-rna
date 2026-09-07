# VCPE-rna Project Plan (Virtual-Cell Perturbation-response Engine)

> VCPE = **V**irtual-**C**ell **P**erturbation-response **E**ngine.
> Positioning: the **perturbation-response engine layer** of a virtual-cell stack (cell-population /
> pseudo-bulk, transcriptome modality) — not a full virtual-cell simulator.
> Branches: `VCPE-rna` (this repo, RNA-drug / oligonucleotide knockdown) → `VCPE-sm` (small molecules)
> → `VCPE-prot` (protein targeting).
> Every named entity is written out in full on first use, to avoid confusing VCPE with a
> "virtual-cell platform".

## 1. Positioning (one sentence)

Port MAP's "knowledge-graph contrastive alignment + conditioned perturbation prediction" architecture
to the oligonucleotide / knockdown modality: replace the small-molecule SMILES encoder with an RNA
sequence encoder, and produce a **license-clean (MIT/Apache end to end) perturbation-response engine**
that can go head-to-head with AIDO.RNA-Pert and progressively replace it, while keeping an interface
for zero-shot small molecules later.

## 2. Background & Asset Inventory

### What we already had (zero download cost)
- Training data: Norman / Adamson / Replogle joint Perturb-seq (16-training data layout,
  genes = 37,062)
- Production baseline (head-to-head benchmark, frozen 17-result): mse_DE 0.5539 · vs ctrl +27.0% ·
  vs mean +11.3% · FM cosine 0.984
- GPU-side agreement: training/inference only, no serving; deliverable = cache file (combo_cache model)
- Downstream service is restarted manually by the operator

### MAP side (external assets)
- Code `github.com/MAGIC-AI4Med/MAP`: MIT
- MAP-KG (HF `RainGate/MAP-KG`): Apache-2.0, 98.5 MB / 4 CSVs
- MAP released weights ("Drive" weights): `epoch_3.pt` 3.94 GB (perturbation predictor) +
  `Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt` 391.9 MB (**gene symbol → embedding table,
  the key file for knockdown conditioning**) + `mapkg_encoder_v3.pt` 1.05 GB; no explicit license
  (presumed per repo MIT; open an issue to confirm before commercial use)
- **Tahoe-100M (429 GB) not needed**: v1 knockdown line uses only the Perturb-seq data we already
  have — the biggest saving from the scope decision

## 3. The Three Decisions (made up-front)

| # | Decision | Options | Chosen |
|---|---|---|---|
| D1 | v1 scope | A. knockdown line only B. + small molecules (needs 4 TB Tahoe) | **A** (small molecules at Phase 4+) |
| D2 | Oligonucleotide encoder | A. self-trained small encoder (20–25 nt, own IP, zero license risk) B. MIT-licensed RNA FM C. AIDO.RNA-650M (**Non-Commercial, internal ablation only, never in deliverables**) | **A primary, B as control; C internal ablation only** |
| D3 | Base acquisition | A. fine-tune from released `epoch_3.pt` (fast) B. full retrain from MIT code (cleanest IP) | **A first, then B**: P1 uses A for fast parity; production retrain at P3 |

## 4. Staged Plan (each stage has a go/no-go gate)

### P0 — Survey & calibration (CPU, 1–2 days)
1. Clone the MAP repo; read `MAP/model/model.py`, `pert.py`, `train_nips.py` and confirm the
   perturbation conditioning interface (drug embedding) can be swapped for target-gene embedding
   (direct ESM2 table lookup)
2. Run MAP's `demo.py` (CPU forward pass) to validate the environment
3. Freeze the benchmark protocol (see §5)
4. Download three asset groups: MAP-KG (98.5 MB) + Drive weights (5.4 GB) + Perturb-seq h5ad
5. **License screening**: shortlist 2–3 MIT/Apache RNA FMs as controls; open an issue on the MAP repo
   to confirm Drive-weight licensing

**G1**: demo.py runs + conditioning swap confirmed feasible. If not → fallback discussion
(e.g., re-implement the architecture directly).

### P1 — Baseline head-to-head (1 GPU)
1. Replace the conditioning head: perturbation condition = target-gene ESM2 + KG embedding
   (+ modality flag), replacing the drug SMILES embedding
2. Retrain on Norman/Adamson/Replogle (same 16-training split for comparability)
3. Eval per epoch: mse_DE, vs ctrl/mean, FM cosine, UPR/ISG checkpoints
4. Output: P1 result table (MAP-rna vs frozen production baseline, line by line)

**G2**: mse_DE ≤ 0.56 and FM cosine ≥ 0.97 (parity tolerance vs production baseline) → proceed to P2.
If > 0.60 → stop and diagnose (embedding dim / alignment quality / training budget).

### P2 — Oligonucleotide-encoder integration (GPU ×2–3)
1. Sequence encoder: primary = self-trained small encoder (own IP); control = MIT-licensed RNA FM;
   AIDO.RNA-650M internal ablation only (Non-Commercial, never in deliverables) + efficacy-prior
   features (Yip 2022 / Hagedorn calibrated prior)
2. Contrastive alignment: oligo sequence embeddings ↔ target-gene embeddings (MAP's contrastive
   strategy; positives = paired design→target)
3. Unseen-gene Tier-2 head-to-head: MAP-rna (ESM2+KG+RNA dual-axis generalization) vs existing
   AIDO.RNA Tier-2 (pure-sequence generalization), same eval set (incl. annotation-poor genes +
   out-of-coverage edge cases)
4. Output: P2 result + unseen-gene generalization table

**G3**: unseen-gene generalization (direction accuracy / pearson) ≥ existing Tier-2 → proceed to P3.
Otherwise MAP-rna is repositioned as a small-molecule-only engine and the knockdown line stays on
AIDO.RNA-Pert.

### P3 — Production integration (CPU only)
1. GPU batch run to produce the platform cache (same flow as the existing AIDO cache: script → GPU
   batch → ship back and swap)
2. Add a third-engine route in the platform's virtual-cell engine (`engine: aidornapert | vcpe`);
   Tier-2 hit logic unchanged
3. Frontend: engine badge + methodology copy (MIT/Apache attribution; GenBio attribution only on
   paths still using AIDO.RNA)
4. Re-run a batch of combo baselines with the new engine
5. Acceptance: live queries (incl. out-of-coverage edge cases) + service restart + hard refresh

**G4 (commercial-delivery gate)**: component audit — every component in a deliverable must be
MIT/Apache/CC0/own-work (checklist in §8); **zero occurrence of AIDO.RNA weights**. If Drive weights
still lack explicit authorization by then, the commercial version must go through D3-B retraining.

## 5. Benchmark Protocol (frozen at P0)

- Eval set: existing crispr bucket (same source and split as the frozen 17-result)
- Metrics: mse_DE / Δ vs ctrl / Δ vs mean / FM cosine / UPR & ISG checkpoints / unseen-gene
  direction accuracy
- Controls: frozen production 17-result (mse_DE 0.5539, etc.) + two trivial baselines
  (perturb-mean / ctrl)
- Reporting: decision-matrix tables + line-item numbers; conclusions stated up-front

## 6. Risk Register

| Risk | Level | Mitigation |
|---|---|---|
| MAP code is research-grade (hardcoded paths, incomplete data) | Medium | P0 read-through + smoke before proceeding; fork/re-implement if needed |
| `epoch_3.pt` ~1B params is tight on a 24 GB GPU | Medium | AMP + frozen base, train conditioning head only |
| ASO has no direct Perturb-seq supervision (CRISPRi is a proxy) | Medium | Efficacy prior as auxiliary feature; keep external wording as "prediction", no overclaiming |
| Drive weights have no explicit license | Low | Commercial path via D3-B retraining, MIT all the way |
| MAP-KG "release soon" data incomplete | Low | Knockdown line doesn't need drug triplets, only the gene-embedding file |

## 7. Directory & Artifact Conventions

```
repo/
├── PLAN.md          # this file
├── third_party/MAP/ # upstream repo (MIT, read-only)
├── data/            # MAP-KG, ESM2 gene embeddings, Drive weights
├── src/             # our modifications (conditioning heads, encoders, training scripts)
├── gpu_pack/        # job packages shipped to the GPU machine
└── results/         # P1/P2 head-to-head results, cache artifacts
```

- GPU jobs follow a closed loop: package locally → run remotely → paste traceback → root-cause fix →
  repackage
- All result files carry a manifest (data split, ckpt, git commit hash)

## 8. Commercial License Checklist (per-deliverable audit)

| Component | Source | License | OK for commercial deliverable? |
|---|---|---|---|
| MAP architecture code | third_party/MAP | MIT | ✅ keep copyright + license notice |
| MAP-KG data | HF RainGate/MAP-KG | Apache-2.0 | ✅ keep notice |
| ESM2 gene embeddings | Meta ESM2 | MIT | ✅ |
| Drive weights (epoch_3.pt etc.) | MAP authors | none explicit | ⚠️ experiments only; commercial path = D3-B retrain (or obtain via issue) |
| Training data Norman/Adamson/Replogle | GEO | public | ✅ cite source |
| Efficacy priors (Yip/Hagedorn) | self-implemented from literature | own work | ✅ |
| Oligonucleotide encoder | self-trained (D2-A) | own work | ✅ |
| AIDO.RNA-650M weights | GenBio | **Non-Commercial** | ❌ **internal research ablation only, never in deliverables** |
| Frozen 17-result benchmark numbers | ours | — | ✅ (benchmark numbers only, no license contamination) |
| AIDO.RNA-Pert production line (coexisting engine) | ours + GenBio base | Non-Commercial | ❌ commercial use needs GenBio authorization (**existing debt**; MAP-rna's success is the cure) |

> Note: "AIDO engineering assets" referenced in planning fall into two classes — **conventions /
> formats / benchmark numbers** (16-training split, combo_cache flow, frozen baselines) carry no
> license obligations and remain in use; **AIDO.RNA model weights** are license-encumbered and were
> removed from all default paths as of plan v0.2.

## 9. Extension Tracks (P4, optional, parallel to P2)

### 9.1 Own RNA FM ladder ("fully self-developed" milestones, triggered as needed)

| Level | Scope | Scale | Single-GPU cost | Trigger |
|---|---|---|---|---|
| L1 | small oligo encoder (already in v1, see D2-A) | 5–20 M params | hours–1 day | default (v1) |
| L2 | own 100 M RNA FM | RNAcentral ~45 M seqs ≈ 13 B tokens, 6×10¹⁸ FLOPs | 1–3 days | P2 shows sequence prior insufficient |
| L3 | own 650 M RNA FM (AIDO.RNA scale) | same corpus, 4×10¹⁹ FLOPs | 1–2 weeks exclusive | knockdown line fully independent of GenBio |

- Corpus synergy: RNAcentral v26 ~45 M seqs shares one download with the "30 M sequence ingestion"
  project (download locally → ship to GPU box)
- Fine-tuning from an open RNA FM: **must pass the P0 license screen first** (academic models are
  often NC/unlabeled); if it fails, fall back to L2 from scratch (1–3 days is acceptable)

### 9.2 MAP-KG-RNA: adding RNA targets to the knowledge graph (originality contribution)

MAP-KG only has drug–protein-target relations; mechanism embeddings for RNA-targeting small
molecules are missing. Two-layer fix:

- **L1 light extension (data-only)**: data = R-BIND / R-BIND-2.0 (RNA-binding small-molecule
  libraries) + PDB RNA–ligand complexes (hundreds; also relieves the local RNA-ligand library
  shortage for TAR screening) + approved ribosome-targeting antibiotics
  (aminoglycosides/macrolides/tetracyclines) + splicing-modulator literature entries. Add
  `(drug, binds RNA motif / modulates splicing, RNA entity)` triples to the MAP-KG CSV schema and
  re-run KG contrastive pretraining (mapkg_encoder ~1 GB, hours on one GPU). Eval = does risdiplam +
  ribosome-antibiotic zero-shot improve.
- **L2 deep extension (differentiation)**: RNA-entity embeddings = sequence + **RNA structures
  predicted by our own platform** (RhoFold+/trRosettaRNA/ViennaRNA), aligned into the unified
  embedding space — an original increment beyond the MAP paper.
- Honest risk: RNA-targeting small molecules ~10³ vs protein-targeting drugs ~10⁵ — two orders of
  magnitude apart → enhancement, not replacement.

## 10. P0 Execution Record

### 2026-08-30 G1 pre-check (code read-through complete)

**Architecture confirmed** (MAP/model/model.py 159 lines + pert.py 138 lines, read in full):
- Single-cell base = `StateEmbeddingModel` (config se600m, token_dim 5120, d_model 2048, 16 layers,
  frozen) = Drive's `epoch_3.pt`
- Gene tokens = ESM2 embedding lookup (`pe_embedding`, 5120-dim = ESM2-3B hidden, i.e. the
  gene_symbol_to_embedding_ESM2.pt file)
- **Perturbation condition = a single token**: `[CLS | Drug | Genes]` (pert.py:95); drug passes
  through frozen KGSmilesEncoder_v3 → [B, 1024]
- **Swap to knockdown = zero architecture change**: `encode_drug(smiles)` →
  `encode_perturbation(target_gene)` via the ESM2 table → project 1024 → perturbation token;
  modality = add modality embedding; combo knockdowns = multiple perturbation tokens
  `[CLS | P1 | P2 | Genes]` (**natively supported, stronger than our current mean-pool**)
- Output = latent embedding + LatentToGeneDecoder → 2000-HVG counts

**Research-grade defects found (must fix before P1)**:
1. `model.py:8` imports `model.pert_ca` — file does not exist → ImportError
2. `model.py:123` `pert_model_hvg_ids_already_set` not initialized in `__init__` → AttributeError on
   first forward
3. `pert.py:41` KG-encoder ckpt path hardcoded `path/to/...`
4. `configs/se600m.yaml` has multiple `/large_storage/`, `/path/to/` hardcodes
5. **HVG space: 2000 genes vs our 37,062 panel** — eval needs alignment (intersect or widen decoder;
   decided at P1)

**G1 status**: pre-check passed (conditioning swap confirmed); gate formally closed once demo.py runs.

### 2026-08-30 Defect fixes complete (git: pristine d696646 → fix bc77f90)

| # | Defect | Fix | Status |
|---|---|---|---|
| 1 | `model.py:8` imports non-existent `model.pert_ca` | remove the import | ✅ |
| 2 | `pert_model_hvg_ids_already_set` / `hvg_ids` uninitialized | init in `__init__` + `hvg_ids is not None` guard in forward | ✅ |
| 3 | `pert.py` KG-encoder ckpt hardcoded `path/to/...` | parametrize `kg_encoder_ckpt` + env `MAP_KG_ENCODER_CKPT`, explicit error when missing | ✅ |
| 4 | `se600m.yaml` hardcoded `/path/to/`, `/large_storage/` | env overrides in code (`MAP_ALL_EMBEDDINGS` / `MAP_DS_EMB_MAPPING`), yaml untouched, minimal diff | ✅ |
| 5 | HVG 2000 vs 37,062 panel | design decision, not a code defect — **deferred to P1** | ⏳ P1 |

Verified: `py_compile` on model.py / pert.py ✅; upstream repo git-initialized with pristine + fix
commits for reviewable diff.
demo.py prerequisites: env `MAP_KG_ENCODER_CKPT`, `MAP_ALL_EMBEDDINGS` (two files under weights/).

### 2026-08-30 demo.py runs — G1 closed ✅

**Additional fixes (#6–#10, see git log)**:

| # | Issue | Fix |
|---|---|---|
| 6 | demo passes `smile_encoder="PrimeKG_ver1"` but pert.py only accepts MAP-KG | demo changed to "MAP-KG" |
| 7 | MolSTM vocab hardcoded `/mnt/petrelfs/.../bart_vocab.txt` | download vocab from chao1224/MoleculeSTM into mega_molbart/, default path alongside + env `MAP_BART_VOCAB` |
| 8 | MolSTM ckpt `molecule_model.pth` hardcoded and missing | redundant load — extract `smiles_encoder._model.*` (176 keys, strict pass) from the full KG ckpt (`MAP_KG_ENCODER_CKPT`) |
| 9 | `epoch_3.pt` uses old architecture key names (smiles_encoder/drug_projector), mismatching the new KGSmilesEncoder wrapper | key remap + drop non-persistent rotary buffers; strict load passes |
| 10 | RoPE 128 vs 84: config-style LlamaRotaryEmbedding uses config.head_dim (84) while attention actually uses head_dim 128 | `LlamaBidirectionalModel` rebuilds model-level rotary with explicit `dim=hidden//heads` |

**Runtime env** (isolated from the system Python): demo venv with system-site-packages +
lightning/torchmetrics/transformers 4.44.2/tokenizers 0.19.1.
**Smoke input**: synthetic CVCL_00223 control bulk (note: the upstream memmap reads **raw binary**
written via `tofile`, not np.save).
**Result**: pred_embs [1,4,2048] + pred_hvg [1,4,2000], CPU 2.5 min.
**Conclusion: G1 passed (demo runs + conditioning interface confirmed swappable to knockdown) → P1.**

### 2026-08-30 P1 training scripts complete (src/maprna_p1/, py_compile ✅)

| File | Contents |
|---|---|
| `ds_knockdown.py` | GEARS h5ad (adamson/norman/replogle perturb_processed.h5ad) → perturbation-level samples: control-bulk sentences [S, 2048] (ESM2-table row ids, top-expression order) + target-gene row id (sym2row) + HVG-2000 target; perturbation-level 85/15 holdout (seed 0, stratified by dataset) |
| `model_kd.py` | `MAPmodelKD`: PerturbationEncoderKD with kd_projector (ESM2 5120→1024), perturbation token into `[CLS\|Pert\|Genes]`; SE frozen, KG-SMILES encoder idle; **zero upstream architecture surgery** |
| `train_p1.py` | single GPU: AdamW dual lr (base 1e-4 / kd 5e-4), bf16 AMP, grad clip; per-epoch eval of mse_DE / pearson-delta / top50-DEG overlap / FM cosine + ctrl & perturb-mean baselines; best ckpt saved by mse_DE |
| `gpu_pack/README_P1.md` | copy manifest + env + full run command + G2 reading notes (protocol-space differences vs the 17-result protocol; read relative relations first) |

**Data fact**: the real vertical-slice data = GEARS perturb_processed.h5ad (the perturbseq/ dir is
synthetic GENE_000 data — do not use).
**Protocol note**: P1 mse_DE (HVG-2000 space, perturbation-level holdout) and the 17-result
(vertical-slice protocol) are not directly comparable in absolute terms; G2 reads relative-to-baseline
first, protocol alignment handled in the P1 result report.

---

## P1 Result (2026-08-31, GPU 3090, 13 epochs)

**G2 verdict: conditional pass → P2.** Full report: [reports/p1_result.md](reports/p1_result.md).

| Metric | VCPE-rna best | ctrl baseline | Verdict |
|---|---|---|---|
| mse_DE ↓ | **0.0215** (ep9) | 0.0316 | ✅ 32% better |
| **fm_cosine** ↑ | **0.9604** (ep7) | G2 gate 0.97 | ⚠️ 1% short |
| pearson_delta ↑ | 0.5661 (ep13) | — | ✅ |
| top50_deg_overlap ↑ | 0.4071 (ep13) | — | ✅ |

- Deliverables: `ckpt_best_mse.pt` (ep9) + `ckpt_best_cosine.pt` (ep7); dual-track saving works
- Training instability: two AMP spikes (ep5/ep10) — engineering item; P2 lowers LR / disables AMP
- **P2 hypothesis**: the cosine ceiling = conditioning information content (currently single-axis
  ESM2+KG); add an RNA sequence encoder for dual-axis generalization, target ≥ 0.97
- Space note: mse_DE is an HVG-2000 number, not directly comparable to the 17-result's 0.5539;
  fm_cosine shares the SE-embedding space, 0.9604 vs production 0.984 directionally comparable
  (2.4% gap)

---

## P2 Result (2026-09-01, GPU 3090, 20 epochs) — G2' pass ✅ (later retracted, see P2.3-B)

Full report: [reports/p2_result.md](reports/p2_result.md).

| Metric | P1 | **P2** | ctrl baseline | G2' |
|---|---|---|---|---|
| fm_cosine ↑ | 0.9604 | **0.9710** (ep18) | — | ≥0.97 ✅ |
| mse_DE ↓ | 0.0215 | **0.0204** | 0.0316 (35.4% better) | ✅ |
| pearson_delta ↑ | 0.5661 | **0.5949** | — | ✅ |
| top50_deg_overlap ↑ | 0.4071 | **0.4176** | — | ✅ |

- Dual-axis hypothesis validated: the RNA sequence axis pushed cosine past 0.97 (first at ep12,
  peak 0.9710 at ep18) with all other metrics improving in sync
- P1 engineering fixes held: zero spikes under fp32 + lower LR; dual-track ckpt delivered at ep18
- Gap to production narrowed: cosine 0.9710 vs AIDO.RNA-Pert 0.984 (1.3%, was 2.4% at P1)
- ⚠️ **Retracted 2026-09-02**: the pass was an artifact of shared-response inflation — see the
  P2.3-B entry below and the erratum section of [reports/p2_result.md](reports/p2_result.md)

---

## L2 Encoder Expansion Kickoff (2026-09-02)

RNA sequence encoder 2.5 M → ~100 M self-trained RNA FM (MLM, encoder-only, RNAcentral corpus).
Three gates: L2-a pretraining healthy → L2-b alignment beats L1 (5.2%) → **L2-c fm_cosine ≥ 0.975**
(the only gate that counts). GPU budget 3–4 days, 5–6 days end-to-end. L1 remains in service
(pluggable architecture, zero rollback cost). Order: Phase 0 data scripts → L2-1 pretrain →
L2-2 align → L2-3 fine-tune verdict.

---

## P2.3-B Result (2026-09-02, GPU 3090, 60 epochs / 80 seconds) — **conditioning revived**

Full report: [reports/p2_3b_result.md](reports/p2_3b_result.md). Architecture verdict validated:
with the same data and the same residual target, three generations of additive-token-into-SE-backbone
all failed (r = 1.000), while the per-gene direct head revived conditioning within one epoch
(r → 0.28 plateau).

| Metric | P2.3-B | Reference | Verdict |
|---|---|---|---|
| ablation_r | 0.277 | P2-line 1.000 | ✅ conditioning alive |
| **pearson_dev (G2''')** | **0.3079** | gate ≥ 0.30 | ✅ pass (barely) |
| top50_dev | 0.3245 | — | ✅ |
| mse_DE | 0.0188 | baseline 0.0377 (50% better) | ✅ |

- 5.7 M params, fully trainable, no SE dependency; 80 s training (×440 iteration speed)
- Ceiling judgment: data volume (1,843 perturbations), not the model — expansion = LINCS / more
  Perturb-seq
- Next: cache export v2 (minutes) → platform integration; L2-1 resumed (sequence-axis gains now have
  a measurable landing spot: pearson_dev)

---

## Data-Expansion Acceptance (2026-09-04, scPerturb ingestion → response-head retest) — **pass ✅**

Training set 1,843 → 16,276 perts (×8.8; gwps 9,726 + jurkat 2,352 + hepg2 2,355, ESM2 mapping
98.5%+). Stratified eval (eval_fair.py) of the lr 3e-4 (collapse-fixed) ep28 ckpt:

| Domain | lr 1e-3 ep9 (collapsed) | **lr 3e-4 ep28 (delivered)** | old P2.3-B baseline |
|---|---|---|---|
| **LEGACY** (legacy three-dataset test) | 0.6305 | **0.7126** | 0.3079 |
| **NEW** (gwps/jurkat/hepg2) | 0.2233 | **0.2763** | — (first measured) |
| OVERALL | 0.3141 | **0.3701** | — |
| gwps / hepg2 / norman | 0.166 / 0.323 / 0.159 | **0.188 / 0.423 / 0.509** | — |

- Same-distribution-domain pearson_dev 0.31 → **0.71 (×2.3)**; combo-perturbation domain norman
  0.16 → 0.51
- Domain gradient (replogle_ess 0.72 ≫ gwps 0.19) is biologically consistent: essential genes have
  strong, predictable responses; non-essential knockdowns are weak-signal by nature
- Engineering deposits: streaming pseudobulk loading (16 GB+ dataset on a 15 GB machine, 5 GB peak),
  chunked evaluation, lr 3e-4 collapse cure
- Caveat: distribution-level fair comparison (the exact legacy 276-pert test set is not reproducible
  after pseudobulk aggregation changes); delivered ckpt = `p3_expanded_lr3e4/ckpt_p3_best_dev.pt` (ep28)
