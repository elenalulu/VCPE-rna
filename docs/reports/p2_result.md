# VCPE-rna P2 Training Result (G2' Verdict Report)

- Date: 2026-09-01
- Setup: P1 cosine-best ckpt start + dual-axis conditioning (mechanism axis ESM2+KG ⊕ sequence axis
  self-trained RNA encoder), lr 5e-5 (base) / 1e-4 (rna branch), AMP off, fp32, 20 epochs
  (1360 s/epoch, 3090)
- Data / eval: identical protocol to P1 (1,843 perturbations, 276 held-out, HVG-2000 + SE-embedding
  space)

## 1. Verdict: **G2' pass ✅** — *retracted on 2026-09-02, see §4*

| Metric | P1 best | **P2 best** | Δ | ctrl baseline | G2' gate | Verdict |
|---|---|---|---|---|---|---|
| **fm_cosine** ↑ | 0.9604 | **0.9710** (ep18) | +0.0106 | — | **≥0.97** | ✅ |
| mse_DE ↓ | 0.0215 | **0.0204** (ep18) | −5.1% | 0.0316 (**35.4%** better) | ≥30% better | ✅ |
| pearson_delta ↑ | 0.5661 | **0.5949** (ep15) | +5.1% | — | — | ✅ |
| top50_deg_overlap ↑ | 0.4071 | **0.4176** (ep8) | +2.6% | — | — | ✅ |

**Dual-axis hypothesis appeared validated**: adding the RNA sequence axis pushed cosine from the
0.9604 plateau (P1) past the 0.97 line (≥0.97 continuously from ep12, peak 0.9710 at ep18), with
all other metrics improving in sync — the sequence information looked like a genuine increment,
not a trade-off.

## 2. 20-epoch trajectory highlights

- Cosine climbs monotonically without instability: 0.9617 → 0.9687 (ep8) → **0.9703 (ep12, first
  gate pass)** → 0.9710 (ep18)
- mse slowly descends: 0.0211 → 0.0204; P1's two AMP spikes **completely disappear** under fp32 +
  lower LR
- ep14–20 plateau (cosine 0.970–0.971, mse 0.0204–0.0205), marginal returns converged

Deliverables: `ckpt_best_cosine.pt` (ep18, 0.9710), `ckpt_best_mse.pt` (ep18, 0.0204).

## 3. Directional comparison with the production baseline (AIDO.RNA-Pert)

| | fm_cosine | License |
|---|---|---|
| AIDO.RNA-Pert (production, 17-result) | 0.984 | Non-Commercial |
| **VCPE-rna P2** | **0.9710** | **MIT/Apache all the way** |

The gap narrowed from 2.4% (P1) to **1.3%** (same SE-embedding-space cosine, directionally
comparable). Candidate sources of the remaining gap: self-trained RNA encoder scale (2.5 M vs
AIDO.RNA 650 M), sequence coverage (~20 k genes), and conditioning fusion (additive vs attention).

## 4. Erratum (2026-09-02): **the G2' "pass" is retracted**

Three-way diagnostics (`diag_fc.py`: eval-style / export-style / TRAIN control) + cache
common-core amplitude audit — the evidence chain:

| Evidence | Value | Meaning |
|---|---|---|
| pred_fc amplitude (diag) | std 0.113, max 1.07 | the model outputs large-amplitude responses (not numeric collapse) |
| cache common-core amplitude | HBG2 2.23 / HBG1 2.09 / EEF2 1.91 | **the shared response was fully learned** (K562 erythroid signature — plausible) |
| **cache residual (after subtracting the common core)** | **max 0.0033** (AIDO same-metric 0.419) | **gene-specific discriminative power ≈ 0** |
| eval-style ≈ export-style | bit-identical | input construction (sentence counts / control-source) is innocent |
| TRAIN ≈ TEST | no difference | **even perturbations directly optimized in training have no residual** → not a generalization problem |

**Corrected conclusions**:
1. The P2 model learned a high-quality **shared response** (the transcriptome signature shared by
   K562 knockdowns) — real but one-sided
2. **Conditioning was bypassed (shortcut)**: the training loss is dominated by the shared component,
   so the cheapest solution is to ignore the perturbation token; all four metrics (mse/pearson/
   cosine/top50) then measure shared-response quality only, and the G2' "pass" does not hold
3. Root cause: the training target = full fc, and the shared core dominates the loss → ignoring
   conditioning is the minimum-cost solution
4. Fix: **P2.1 protocol revision** (target becomes the residual after subtracting the common core +
   common head + conditioning reinforcement + metric redesign); L2 pretraining is unaffected
   (capacity gains only pay off after the protocol fix)
5. Cache stays frozen; not shipped to the platform

**Lesson recorded**: mse/pearson/cosine/top50 can all be inflated by the shared component — a
perturbation-response model's evaluation **must** include "shared-component-free discrimination"
metrics (pearson_dev / top50_dev), otherwise any G-gate can be fooled by the shared response.
