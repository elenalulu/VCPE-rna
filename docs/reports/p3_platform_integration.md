# VCPE-rna → Platform Integration Analysis (P3 pre-work; analysis only, no code changes)

- Date: 2026-09-01
- Scope: the response-prediction path of the ASO/siRNA views (MODZ / AIDO.RNA-Pert) → the VCPE-rna
  engine
- Conclusion up front: **the recommended integration mode is "offline cache export"** — VCPE-rna runs
  batch inference once and exports a schema isomorphic to the existing AIDO cache; the platform side
  only swaps the cache file + attribution copy, with **zero new dependencies** (the platform never
  installs torch).

## 1. Current data flow (who produces what, who consumes what)

```
ASO sequence → aso_offtarget_transcriptome.py (align vs gene_transcripts.fa for off-targets)
        → top off-target genes
        → virtual_cell.py tier routing:
            Tier-1  measured genes → LINCS L1000 MODZ (kd-scaled approximation, HepG2)   [unchanged]
            Tier-2  unmeasured    → AIDO.RNA-Pert precomputed cache                      [swap point]
        → frontend immune_sim.js (MODZ bars / ASO calibrated top-gene swap — frame already exists)
```

Key touch points:

| File | Role |
|---|---|
| `immune_simulation/outputs/aido_vc_cache.json.gz` | Tier-2 precomputed cache: **55,336 genes → {top_up, top_down} direction lists** |
| `immune_simulation/engine/virtual_cell.py` | tier routing + AIDO cache loading + Tier-2 prediction + MODZ + caveat (`is_aido` flag) |
| `rna_robot/utils/aso_offtarget_transcriptome.py` | off-target search + "Powered by GenBio AI" copy |
| `rna_robot/static/immune_sim.js` | MODZ display + ASO top-gene swap logic (**swap frame already exists**) |
| `immune_simulation/build_lincs_hep_cache.py` | Tier-1 LINCS cache build (untouched) |

## 2. What changes in the data (by category)

| # | Category | Change | Note |
|---|---|---|---|
| 1 | **Model artifact** | `aido_vc_cache.json.gz` → `vcpe_cache.json.gz` (1-for-1 swap) | generator moves from the AIDO export script to VCPE inference export; weights = `ckpt_best_cosine.pt` (871 M) + `rna_encoder_aligned.pt` (2.5 M) |
| 2 | Cache schema | **stays isomorphic**: per-gene `{top_up, top_down}` direction lists | VCPE exports ranked predictions from HVG-2000; downstream consumers unchanged |
| 3 | **Tier-2 coverage** ⚠️ | 55,336 → ~19,790 (ESM2-table ceiling, + HGNC `+1` aliases) | **honest note**: the set of genes addressable as knockdown targets shrinks; mitigation: the response output (2000 HVG) is unrestricted, and the Ensembl/`+1` fallback in `lookup_seq` is already in place |
| 4 | **Attribution copy** ×3 | "Powered by GenBio AI" → VCPE wording | `virtual_cell.py` (×2) and `aso_offtarget_transcriptome.py` (×1) — **this is the core motivation of the integration; without the copy change the swap is pointless**; `is_aido` flag becomes `is_vcpe` |
| 5 | Cell-type label | cache `cell_type` field | AIDO = single cell type (K562-trained); VCPE = three-dataset joint (K562×2 + RPE1); label and caveat updated |
| 6 | Frontend | caveat copy + MODZ → top-response-gene swap wording | swap frame exists (`immune_sim.js`), small change |

**Unchanged**: tier routing logic, Tier-1 LINCS measured ground truth, off-target search, the 8000
service interface, the frontend swap frame.

## 3. Quality trade-off (honestly labeled)

| Dimension | AIDO.RNA-Pert (incumbent) | VCPE-rna (after integration) |
|---|---|---|
| fm_cosine (embedding space) | 0.984 | 0.9710 (**−1.3%**) |
| Tier-2 target coverage | 55,336 | ~19,790 |
| **License** | Non-Commercial (no commercial use) | **MIT/Apache all the way** |
| Training data | single cell type | three-dataset joint (incl. RPE1) |

## 4. The two architecture options

| Option | Approach | Platform-side dependency | Verdict |
|---|---|---|---|
| **1. Offline cache export** | write `export_vcpe_cache.py` (AIDO-schema-isomorphic), batch-infer once in the VCPE env | **zero new dependencies** (swap the gz file + copy) | ✅ P3 choice |
| 2. Online inference | platform deploys the 871 M model + GPU + MAP code | full torch stack + GPU | later, as needed |

## 5. Implementation order (executed after approval)

1. `export_vcpe_cache.py`: load P2 ckpt → batch inference over `supported_genes` → export
   `vcpe_cache.json.gz` in the AIDO schema
2. `virtual_cell.py`: cache path/loading logic + `is_aido`→`is_vcpe` + copy ×2
3. `aso_offtarget_transcriptome.py`: copy ×1
4. `immune_sim.js`: caveat copy
5. Live acceptance: real ASO queries, compare Tier-2 output before/after, confirm schema
   compatibility + visible coverage difference
