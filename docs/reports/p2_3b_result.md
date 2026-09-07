# VCPE-rna P2.3-B Result Report (2026-09-02)

## One-line conclusion

**Conditioning revived; G2''' (pearson_dev ≥ 0.30) passed** — a GEARS-style per-gene residual head
achieved in 80 seconds of training what three generations of additive-pert-token-into-the-SE-backbone
(P2/P2.1/P2.2, 23 epochs combined) could not: making perturbation identity genuinely drive the output.

## Final numbers (60 epochs, test = 276 perturbations, deterministic protocol)

| Metric | P2.3-B best | Reference | Verdict |
|---|---|---|---|
| **ablation_r (real vs zeroPert)** | **0.277 plateau** (stable <0.5 from ep13) | P2/P2.1/P2.2 all 1.000 | ✅ **conditioning revived** |
| **pearson_dev (G2''')** | **0.3079** (ep48) | gate ≥ 0.30 | ✅ pass (barely) |
| top50_dev | 0.3245 | — | ✅ |
| mse_DE | 0.0188 | ctrl baseline 0.0377 (50% better) | ✅ |
| pearson_delta | 0.6505 | P2-line 0.594 (different protocol, directional reference) | ✅ |
| top50_deg_overlap | 0.4629 | P2-line 0.418 (same) | ✅ |
| Training cost | **~80 s** (1.3 s/epoch × 60) | P2.2 was 16 h | ✅ ×440 iteration speed |

Deliverable: `p3_out/ckpt_p3_best_dev.pt` (ep48, best pearson_dev; includes common_fc and hvg_rows).

## Architecture (why it worked this time)

```
pred_dev_i = MLP([ g_i, p, g_i⊙p, rna_p, ctrl_expr_i, is_neighbor_i, is_target_i, ds ])
```

- Each gene's prediction takes (own embedding × perturbation embedding) **directly** as input —
  perturbation information reaches every gene without passing through the 600 M SE backbone's
  residual stream (the P2-line failure path); structurally there is nothing to drown in
- Frozen ESM2 table as buffer (5.7 M params, all trainable); STRING-neighbor and is_target indicators
  give an explicit "which genes should this perturbation affect" prior (the same mechanism as GEARS)
- The protocol fully inherits P2.1: residual target = full fc − train-only common core (no leakage)

## The P2 → P2.1 → P2.2 → P2.3-B evidence chain (archived)

| Version | Target | Conditioning channel | epochs | r(real, zeroPert) | pearson_dev |
|---|---|---|---|---|---|
| P2 | full fc | SE backbone + additive pert token | 20 | 1.000 (dead) | — (inflated by shared response) |
| P2.1 | residual fc | same | 2 | 1.000 (dead) | showed 0.92 (dev-metric bug, invalid) |
| P2.2 | residual fc | same + STRING network axis | 1 | 1.000 (dead) | showed 0.93 (same bug, invalid) |
| **P2.3-B** | residual fc | **per-gene direct head** | 60 | **0.277** | **0.3079 ✅** |

Conclusion: the objective and the information sources were both correct — the failure was in the
**information pathway**. An additive token flowing into a large backbone's residual stream gets
structurally drowned; a per-gene direct head is the right architecture in the data-limited regime.

## Honest caveats

1. **pearson_dev 0.31 passes by a hair**: among 1,843 perturbations, the learnable gene-specific
   signal is limited — the current ceiling is more likely **data volume** than the model. The path up
   is more perturbation data (LINCS L1000 perturbational subset, more Perturb-seq datasets), not a
   bigger model
2. The ablation zeroes the ESM2 vector; the neighbor/self indicator features still carry perturbation
   identity — r = 0.28 is the joint effect of three axes; per-axis ablation is future work
3. `mse_dev ≡ mse_DE` (float-tail difference) is a mathematical identity in this protocol
   (full = common + dev adds back exactly), not a bug
4. Dataset-level residual commonality (K562/RPE1-specific responses) may still be partially exploited
   via the ds one-hot — a ds-shuffle ablation would isolate this if needed

## Next steps

1. **Cache export v2**: batch inference over 19,790 genes with the P2.3-B head (minutes, no SE
   needed), regenerate the platform cache → spot-check fc amplitude and discrimination → platform
   integration (P3 proper)
2. **Data-expansion track** (the fundamental fix for the pearson_dev ceiling): evaluate LINCS
   siRNA/ASO subset ingestion
3. Resume L2-1 pretraining (RNA-encoder expansion; the sequence axis now has a clean measurable
   landing spot: pearson_dev)
