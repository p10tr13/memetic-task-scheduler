# Results

Full numerical experiments for the memetic scheduler. Reproduce with:

```
python -m src.experiments.run_experiments   # writes results/campaign/{runs,convergence,baselines}.csv
python -m src.experiments.analyze           # writes summary.csv, h1_wilcoxon.csv, h2_wilcoxon.csv
python -m src.experiments.plots             # writes results/campaign/plots/*.png
```

## Setup

- **Configurations (4):** `ox`, `pmx` (pure EA) and `ox_cpm`, `pmx_cpm` (memetic =
  EA + schedule-aware CPM local search). The CPM analysis uses the critical path of
  the *decoded schedule* (precedence and processor bottlenecks).
- **Seeds:** 30 independent runs per configuration (10 on the two largest Kasahara
  instances, `robot_control` and `random_300`, for tractability).
- **Quality measure:** best `C_max` per run. Tables report the **median** over seeds.
- **Reference:** exact MILP optimum for small instances (`|T| ≤ 30`, proven optimal
  via CBC), otherwise the lower bound `LB = max(⌈total work / M⌉, critical path)`.
- **Significance:** paired Wilcoxon signed-rank test (same seeds), Holm-corrected
  within each hypothesis family, α = 0.05.
- **Plots:** boxplots and convergence curves per instance in
  `results/campaign/plots/` (gitignored).

> **Budget caveat (H1).** The memetic configurations perform extra makespan
> evaluations inside the CPM local search, so configurations are compared at equal
> **generations**, not at an equal evaluation budget. The CPM advantage below is
> therefore partly a larger effective search effort per generation.

## Instances

| instance | source | nodes (+2 dummy) | edges | density `\|E\|/\|T\|²` | M | HLFET | MCP | MILP opt | LB |
|---|---|---|---|---|---|---|---|---|---|
| small_15 | synthetic | 17 | 31 | 0.107 | 2 | 53 | 53 | **52** | 52 |
| small_22 | synthetic | 24 | 48 | 0.083 | 3 | 47 | 47 | **44** | 44 |
| h2_sparse_a | synthetic | 42 | 78 | 0.044 | 4 | 55 | 55 | — | 52 |
| h2_sparse_b | synthetic | 42 | 80 | 0.045 | 4 | 57 | 57 | — | 54 |
| h2_dense_a | synthetic | 42 | 277 | 0.157 | 4 | 108 | 108 | — | 108 |
| h2_dense_b | synthetic | 42 | 280 | 0.159 | 4 | 96 | 96 | — | 96 |
| example | Kasahara/synthetic | 47 | 86 | 0.039 | 3 | 72 | 72 | — | 69 |
| random_50 | Kasahara | 52 | 119 | 0.044 | 4 | 77 | 77 | — | 66 |
| robot_control | Kasahara | 90 | 135 | 0.017 | 4 | 694 | 694 | — | 621 |
| random_300 | Kasahara | 302 | 7265 | 0.080 | 4 | 505 | 505 | — | 448 |

HLFET and MCP coincide on every instance here. The memetic EA (`pmx_cpm`) matches or
beats both list heuristics on every instance (e.g. `random_50` 66 vs 77, `random_300`
median 479 vs 505).

## Median C_max by configuration

Best configuration per instance in **bold**.

| instance | ref (kind) | ox | pmx | ox_cpm | pmx_cpm |
|---|---|---|---|---|---|
| small_15 | 52 (milp) | **52** | **52** | **52** | **52** |
| small_22 | 44 (milp) | 47.5 | 46.0 | **44.0** | **44.0** |
| h2_sparse_a | 52 (lb) | 55.0 | 53.0 | **52.0** | **52.0** |
| h2_sparse_b | 54 (lb) | 57.0 | 55.0 | **54.0** | **54.0** |
| h2_dense_a | 108 (lb) | **108** | **108** | **108** | **108** |
| h2_dense_b | 96 (lb) | **96** | **96** | **96** | **96** |
| example | 69 (lb) | 74.0 | 70.0 | 70.0 | **69.0** |
| random_50 | 66 (lb) | 73.0 | 68.0 | 66.0 | **66.0** |
| robot_control | 621 (lb) | 814.5 | 742.5 | 737.5 | **694.0** |
| random_300 | 448 (lb) | 532.5 | 510.5 | 519.0 | **479.0** |

Gap of the best configuration (`pmx_cpm`) to the reference: 0% on the small and dense
instances (reaches the optimum / bound), 1.9% (`h2_sparse_b`), 6.9% (`random_300`),
11.8% (`robot_control`). The dense H2 instances are effectively forced (every config
reaches the bound), so they carry no operator/CPM signal.

## H1 — does CPM local search lower C_max?

Paired Wilcoxon, CPM-on vs CPM-off, per crossover per instance (Holm-corrected).

| instance | crossover | CPM off | CPM on | Δ median | Holm p | significant |
|---|---|---|---|---|---|---|
| small_22 | ox | 47.5 | 44.0 | 3.5 | 0.000 | ✅ |
| small_22 | pmx | 46.0 | 44.0 | 2.0 | 0.000 | ✅ |
| h2_sparse_a | ox | 55.0 | 52.0 | 3.0 | 0.000 | ✅ |
| h2_sparse_a | pmx | 53.0 | 52.0 | 1.0 | 0.000 | ✅ |
| h2_sparse_b | ox | 57.0 | 54.0 | 3.0 | 0.000 | ✅ |
| h2_sparse_b | pmx | 55.0 | 54.0 | 1.0 | 0.000 | ✅ |
| example | ox | 74.0 | 70.0 | 4.0 | 0.000 | ✅ |
| example | pmx | 70.0 | 69.0 | 1.0 | 0.000 | ✅ |
| random_50 | ox | 73.0 | 66.0 | 7.0 | 0.000 | ✅ |
| random_50 | pmx | 68.0 | 66.0 | 2.0 | 0.000 | ✅ |
| robot_control | ox | 814.5 | 737.5 | 77.0 | 0.020 | ✅ |
| robot_control | pmx | 742.5 | 694.0 | 48.5 | 0.020 | ✅ |
| random_300 | ox | 532.5 | 519.0 | 13.5 | 0.020 | ✅ |
| random_300 | pmx | 510.5 | 479.0 | 31.5 | 0.020 | ✅ |

(The 6 omitted tests — `small_15` and both `h2_dense` — are ties at the optimum/bound
for every configuration, so there is no headroom for CPM to act.)

**Verdict: H1 supported.** On every instance with headroom, CPM local search
significantly lowers the median `C_max` for both crossovers (14/14 significant tests,
all favouring CPM-on). The effect grows with instance difficulty (e.g. `robot_control`
−77 for OX, −48.5 for PMX). Subject to the equal-generations budget caveat above.

## H2 — does OX dominate PMX on sparse graphs?

Paired Wilcoxon, OX vs PMX (pure EA), per instance (Holm-corrected). Sparse =
`|E|/|T|² < 0.1`.

| instance | class | ox | pmx | better | Holm p | significant |
|---|---|---|---|---|---|---|
| small_22 | sparse | 47.5 | 46.0 | pmx | 0.001 | ✅ |
| h2_sparse_a | sparse | 55.0 | 53.0 | pmx | 0.000 | ✅ |
| h2_sparse_b | sparse | 57.0 | 55.0 | pmx | 0.000 | ✅ |
| example | sparse | 74.0 | 70.0 | pmx | 0.000 | ✅ |
| random_50 | sparse | 73.0 | 68.0 | pmx | 0.000 | ✅ |
| robot_control | sparse | 814.5 | 742.5 | pmx | 0.010 | ✅ |
| random_300 | sparse | 532.5 | 510.5 | pmx | 0.010 | ✅ |
| small_15 | dense | 52.0 | 52.0 | tie | 1.000 | — |
| h2_dense_a | dense | 108.0 | 108.0 | tie | 1.000 | — |
| h2_dense_b | dense | 96.0 | 96.0 | tie | 1.000 | — |

**Verdict: H2 not supported — the opposite holds.** On every sparse instance PMX
significantly beats OX (0/7 sparse tests favour OX); on the dense instances every
configuration reaches the bound, so OX and PMX tie. PMX is the stronger crossover
across the board for this problem and decoder. The likely reason: the ready-list
repair already enforces precedence, so OX's order-preservation advantage is muted,
while PMX's position-based mixing explores the priority space more effectively.

## Summary

- **PMX + CPM (`pmx_cpm`) is the best configuration on every instance.**
- **H1 holds:** schedule-aware CPM significantly improves `C_max` wherever there is
  headroom (equal-generations budget).
- **H2 fails:** PMX dominates OX, including on sparse graphs.
- The memetic EA beats the HLFET/MCP list heuristics on every instance; remaining
  gaps to the lower bound are 7–12% on the two largest Kasahara graphs.
