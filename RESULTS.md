# Preliminary results

Strictly preliminary: no baselines, MILP optimality gaps, or significance tests
yet. Lower bound `LB = max(load, critical path)`, where `load = ceil(total work / M)`
and `critical path` is the longest weighted path through the DAG; `gap = (best - LB) / LB`.

Only the synthetic-instance numbers are reproducible with
`python -m src.experiments.run_partial` (writes `results/`). The Kasahara-instance
numbers come from exploratory runs with the per-instance budgets noted below.

Wall-clock times are for the full 4-config sweep on the test machine and reflect the
differing budgets, not just instance size (robot_control is slowest because it ran the
largest budget). CPM configurations dominate the runtime.

## Instances

| instance | source | nodes (+2 dummy) | edges | density |
|---|---|---|---|---|
| [example](data/sample/example.stg) | synthetic, generated | 47 | 86 | 0.039 |
| [random_50](data/sample/random_50.stg) | Kasahara STG, rnc50/rand0000 | 52 | 119 | 0.044 |
| [random_300](data/sample/random_300.stg) | Kasahara STG, rnc300/rand0000 | 302 | 7265 | 0.080 |
| [robot_control](data/sample/robot_control.stg) | Kasahara STG, robot control | 90 | 135 | 0.017 |

## example — synthetic, 45 tasks (M=3, 10 seeds, pop 40, 60 gen; ~7 s)

Optimal makespan is 69 (= load bound; critical path 54), so PMX configs reach the
optimum here (0% gap).

Best makespan over 10 seeds (lower is better):

| config | best | mean | median |
|---|---|---|---|
| ox | 73 | 74.2 | 74.0 |
| pmx | 69 | 69.8 | 70.0 |
| ox_cpm | 70 | 72.3 | 72.5 |
| pmx_cpm | 69 | 69.9 | 70.0 |

Mean best makespan per generation:

| gen | ox | pmx | ox_cpm | pmx_cpm |
|---|---|---|---|---|
| 0 | 74.4 | 74.4 | 74.4 | 74.4 |
| 10 | 74.2 | 70.7 | 72.3 | 70.3 |
| 30 | 74.2 | 70.0 | 72.3 | 70.0 |
| 60 | 74.2 | 69.8 | 72.3 | 69.9 |

- OX improves little on this instance (74.4 → 74.2), while PMX is clearly stronger
  and reaches the optimum (69).
- CPM local search helps the weaker OX (74.2 → 72.3 via ox_cpm) but gives no gain
  over PMX (69.9 vs 69.8). Directionally supports H1.
- The optimum equals a trivial load bound, so this instance is too easy to
  differentiate the configurations strongly.

## random_50 — Kasahara random graph (M=4, 5 seeds, pop 40, 60 gen; ~9 s)

LB = max(load 66, critical path 55) = 66.

| config | best | mean | gap vs LB |
|---|---|---|---|
| ox | 71 | 73 | 8% |
| pmx | 66 | 68 | 0% |
| ox_cpm | 70 | 71 | 6% |
| pmx_cpm | 67 | 68 | 2% |

- PMX reaches the optimum (66); CPM does not help PMX here (pmx_cpm 67).

## robot_control — Kasahara application graph (M=4, 5 seeds, pop 50, 100 gen; ~7.3 min)

LB = max(load 621, critical path 569) = 621.

| config | best | mean | gap vs LB |
|---|---|---|---|
| ox | 784 | 807 | 26% |
| pmx | 729 | 750 | 17% |
| ox_cpm | 769 | 787 | 24% |
| pmx_cpm | 717 | 731 | 16% |

- Large gaps (16–26%): a genuinely hard instance with plenty of headroom.
- CPM helps both operators (ox 784→769, pmx 729→717). Supports H1.

## random_300 — Kasahara random graph (M=4, 3 seeds, pop 30, 40 gen; ~45 s)

LB = max(load 423, critical path 448) = 448.

| config | best | mean | gap vs LB |
|---|---|---|---|
| ox | 530 | 533 | 18% |
| pmx | 511 | 513 | 14% |
| ox_cpm | 527 | 531 | 18% |
| pmx_cpm | 496 | 500 | 11% |

- Large gaps (11–18%) confirm the instance is hard.
- pmx_cpm clearly best; CPM helps most on PMX here.

## Cross-instance summary

- The PMX configurations are best on every instance.
- PMX beats OX on all four instances — no support yet for H2 (OX > PMX on sparse graphs).
- CPM helps OX on every instance and PMX on the larger ones — mixed support for H1.
- The larger instances sit 11–26% above the lower bound (room to improve); the
  smaller ones are at or near optimum.
