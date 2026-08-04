# Rebuttal tables

Generated from `ranking_seed_stability.py`, `ranking_bootstrap_stability.py`, and
`attention_trades_beta_analysis.py` (this directory). All three metrics sets
now include `common` (common-corruption) accuracy alongside clean/L∞/L2/L1.

## Table 1 — Ranking stability across independent training seeds

For each of the 3 newly-trained, independent seeds, the sum-score (accuracy
summed across 6 RobustGenBench datasets × 5 metrics: clean, L∞, L2, L1,
common corruptions) is computed separately for the pre-assigned
Gold/Silver/Bronze configuration in each size tier; the table reports the
mean ± sample standard deviation of these 3 per-seed sum-scores. In all
three tiers the ranking reproduces the original Gold > Silver > Bronze
order.

| Tier | Gold | Silver | Bronze | Order |
|---|---|---|---|---|
| Base | 19.342 ± 0.098 | 17.758 ± 0.315 | 17.500 ± 0.160 | Gold > Silver > Bronze ✓ |
| Small | 16.287 ± 0.150 | 15.808 ± 0.051 | 14.777 ± 0.029 | Gold > Silver > Bronze ✓ |
| Tiny | 15.831 ± 0.070 | 14.519 ± 0.109 | 12.632 ± 0.095 | Gold > Silver > Bronze ✓ |

## Table 2 — Ranking stability under test-set resampling

Nonparametric bootstrap (1,000 resamples with replacement, seed 1
checkpoints, training seed held fixed) applied independently to each medal's
per-observation predictions, isolating finite test-set sampling noise from
training-seed variance. "% order held" is the fraction of bootstrap
iterations in which the resampled sum-score reproduces Gold > Silver >
Bronze exactly; the 95% CI is the percentile interval of each medal's
bootstrapped total score.

| Tier | % order held | Gold (95% CI) | Silver (95% CI) | Bronze (95% CI) |
|---|---|---|---|---|
| Base | 60.3% | 19.25 [18.91, 19.56] | 17.41 [16.99, 17.82] | 17.35 [16.95, 17.73] |
| Small | 100.0% | 16.29 [16.09, 16.49] | 15.79 [15.59, 15.97] | 14.81 [14.64, 15.00] |
| Tiny | 100.0% | 15.89 [15.60, 16.18] | 14.40 [14.21, 14.60] | 12.54 [12.36, 12.71] |

**Note:** for Base, Gold ranks first in 100% of bootstrap iterations; only
the Silver/Bronze order is unstable (Gold > Bronze > Silver in the remaining
39.7% of resamples), consistent with the small, seed-sensitive gap between
those two configurations observed under training-seed resampling as well
(Table 1).

## Table 3 — TRADES vs. Classic AT for attention-based architectures, augmented with TRADES (β=6)

Sum-score (clean, L∞, L2, L1, common corruptions — all 5 metrics present for
every condition — summed, not averaged, over all 6 RobustGenBench datasets;
higher is better), reported per backbone rather than as a pool-dependent
Borda score. All three loss conditions are evaluated on the *same matched
set* of 5 "fully attention" backbones (the only 5 of the 8 attention-type
backbones in the original pool that were retrained at β=6) over the *same
matched set* of n=30 (dataset, metric) values per cell, so columns are
directly comparable.

| Backbone | TRADES (β=1) | Classic AT | TRADES (β=6) | Best |
|---|---|---|---|---|
| deit_small_patch16_224.fb_in1k | 6.8581 | **8.4049** | 5.3882 | Classic AT |
| eva02_base_patch14_224.mim_in22k | **17.5995** | 13.1634 | 13.9792 | TRADES(β1) |
| swin_tiny_patch4_window7_224.ms_in1k | 9.9935 | **10.3060** | 9.0655 | Classic AT |
| vit_base_patch16_224.augreg_in1k | **12.4978** | 11.3069 | 10.9271 | TRADES(β1) |
| vit_small_patch16_224.augreg_in1k | **9.1447** | 2.7429 | 4.1469 | TRADES(β1) |
| **Total (matched, n=5 backbones)** | **56.0937** | 45.9240 | 43.5068 | TRADES(β1) |

**Note:** TRADES (β=6) is never the best-performing loss for any of the 5
backbones (0/5 wins) — it underperforms both TRADES (β=1) and Classic AT
uniformly, so the mean is not an average-only artifact. TRADES (β=1) and
Classic AT are closer, but not equivalent, and the direction is
backbone-dependent: TRADES (β=1) wins for 3/5 backbones (eva02_base,
vit_base, vit_small), while Classic AT wins for 2/5 (deit_small, swin_tiny)
— evidence against a single homogeneous "TRADES ≈ Classic AT for attention
architectures" story at the individual-backbone level, even though it may
hold approximately on average.

**Limitation:** the β=6 runs reuse the hyperparameters (learning rates,
weight decay) selected by the β=1 HPO search rather than an independent
search at β=6, consistent with this benchmark's fixed-HPO-per-configuration
evaluation protocol throughout. Since TRADES loss is CE + β·KL(robust,
natural), a 6× larger β substantially changes the loss landscape, so part of
the β=6 underperformance could reflect a hyperparameter mismatch rather than
a pure effect of β alone. We report this as a directional finding, not a
fully isolated causal claim about β.
