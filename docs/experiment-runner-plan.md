# Experiment runner — implementation plan

Branch: `experiments` (integrates `milp` + `fix/schedule-aware-cpm`).

## Goal

Produce the evidence required by the milestone document: ≥30-seed runs across a
set of instances, Wilcoxon significance tests, baselines (HLFET / MCP) and the
MILP optimality gap, plus boxplots and convergence curves — and from those, a
verdict on the two hypotheses:

- **H1** — CPM local search significantly lowers the median `C_max` versus a pure EA.
- **H2** — OX significantly dominates PMX on sparse graphs (`|E|/|T|² < 0.1`).

> Why now: the current [RESULTS.md](../RESULTS.md) numbers were produced with the
> old *graph-level* CPM. CPM is now schedule-aware, so every CPM-on configuration
> behaves differently and the preliminary H1 numbers are stale — they must be
> regenerated.

## Chosen options

- **H1 budget fairness: (A) equal generations + documented caveat.** The memetic
  algorithm performs extra makespan evaluations inside the CPM local search, so at
  equal generations it uses more objective evaluations than the pure EA. We compare
  at equal generations and **document this caveat** in RESULTS. (Option B —
  counting evaluations against a fixed budget — is a possible follow-up; it needs a
  small evaluation-counter change in the engine and is out of scope for v1.)
- **Scope: full** — Wilcoxon stats + MILP gap + plots.

## Design principle: split *run* from *analyze*

Two phases with CSV in between, so analysis and plots can be re-run without
repeating the (expensive) EA runs, and the raw data is preserved for the report.

```
run_experiments.py  ->  results/campaign/runs.csv          one row per instance×config×seed
                        results/campaign/convergence.csv    mean curve per instance×config×gen
                        results/campaign/baselines.csv       HLFET, MCP, MILP optimum+gap per instance
analyze.py          ->  results/campaign/h1_wilcoxon.csv, h2_wilcoxon.csv, summary.csv
plots.py            ->  results/campaign/plots/*.png         boxplots + convergence per instance
```

Outputs live under `results/campaign/` so they never collide with the existing
`run_partial.py` outputs (`results/summary.csv`, `results/convergence.csv`), which
stay as the quick-checkpoint reproducer.

## New files (additive — nothing existing is refactored)

1. **`src/experiments/instances.py`** — the instance set as code. Loads the four
   bundled Kasahara `.stg` files and generates fixed-seed synthetic instances
   (small `|T| ≤ 30` for the MILP gap; matched sparse/dense pairs for H2). Returns
   a list of entries: `name`, `ProblemInstance`, `processors`, per-instance
   `EvolutionConfig` (population/generations), `seed_count`, and metadata
   (nodes, edges, density).
2. **`src/experiments/run_experiments.py`** — the campaign runner (matrix below).
   CLI: `--quick` (tiny subset/seeds/budget for smoke-testing), `--seeds N`
   (override), `--output-dir`.
3. **`src/experiments/analyze.py`** — reads `runs.csv`, computes paired Wilcoxon
   tests and summary tables.
4. **`src/experiments/plots.py`** — boxplots and convergence curves (matplotlib).
5. **Light unit tests** for the pure helpers only (lower-bound / gap math,
   instance determinism, the Wilcoxon wrapper) — **not** full runs, to keep CI fast.
6. **`requirements.txt`** += `scipy`, `matplotlib`.

`run_partial.py` is left untouched (or gets a one-line pointer to the full runner).

## Run matrix

- **Configs:** `{OX, PMX} × {CPM-on (memetic), CPM-off (pure EA)}` = 4, reusing the
  existing `EvolutionStrategies` wiring (pure EA = `local_search_strategy=None`).
- **Baselines per instance:** HLFET, MCP; MILP optimum for `|T| ≤ 30`
  (time-limited via `solve_milp(time_limit_seconds=...)`).
- **Seeds:** fixed `range(N)`, default `N = 30`, **reused across configs** so all
  comparisons are *paired*. Heavy Kasahara instances (`random_300`,
  `robot_control`) use a smaller `seed_count` to keep wall-clock reasonable; this
  is recorded per instance and noted in RESULTS.
- **Per-run metrics:** best `C_max` and the convergence curve.
- **Per-instance metrics:** gap vs MILP (small instances) or vs the lower bound
  `LB = max(⌈total_work / M⌉, critical_path)` (larger instances).

## Analysis

- **H1:** paired Wilcoxon (same seeds), CPM-on vs CPM-off, per crossover per
  instance → p-value + median Δ + direction.
- **H2:** paired Wilcoxon, OX vs PMX, grouped by density (`|E|/|T|² < 0.1` sparse
  vs dense).
- **Holm correction** across the multiple tests (reported).
- `scipy.stats.wilcoxon`; results written as CSV plus a markdown block for RESULTS.

## Plots

Per instance: a boxplot of best `C_max` across the four configs, and a mean
convergence curve per config. `matplotlib`, saved under `results/campaign/plots/`.

## Reproducibility

Fixed seed lists for both the EA runs and synthetic-instance generation; the seed
count and `EvolutionConfig` per instance are recorded in `runs.csv`. Library
versions documented in `requirements.txt`. The full sweep is run offline, never in
CI.

## Runtime caveat

30 seeds × 4 configs is heavy — `robot_control` was ~7 min for 5 CPM seeds, so the
big Kasahara instances use reduced `seed_count`. Seeds, instances, and budget are
CLI-configurable; `--quick` runs a tiny version for verification.

## Commit breakdown (small, each green)

1. deps + `instances.py` (+ determinism tests)
2. `run_experiments.py` → raw CSVs (verified with `--quick`)
3. `analyze.py` (Wilcoxon + tables)
4. `plots.py`
5. regenerate RESULTS.md (with the schedule-aware CPM) + update STATUS / README

Each commit keeps `black`, `pylint` (`src/`), and `pytest` green. Commits are
created locally and **not pushed** until reviewed.
