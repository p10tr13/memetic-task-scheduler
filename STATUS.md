# Status — WAE: memetic-task-scheduler

As of: **2026-06-05** (Fri)

Project: memetic algorithm (EA + CPM local search) for multiprocessor task scheduling with precedence constraints, minimizing makespan (C_max). Authors: Wojciech Jasiński, Piotr Iśtok. Documentation: [docs/preliminary/main.tex](docs/preliminary/main.tex) (matches [../pro/milestone-1/main.tex](../pro/milestone-1/main.tex), the preliminary doc).

> ✅ **Full experiment campaign done.** Schedule-aware CPM and the exact MILP solver are **merged into `main`** (#9, #10). The experiment runner is built and the full statistical campaign has run; [RESULTS.md](RESULTS.md) is regenerated with the corrected CPM. **H1 supported, H2 rejected** (see below). Remaining gap to the milestone protocol: hyperparameter ablations.

## Overall state

| Area | State |
|------|-------|
| Algorithm engine (method from the preliminary doc) | 🟢 Complete, unit-tested (80 tests), runs end-to-end. Ready-list repair + **schedule-aware CPM** |
| Baselines (HLFET, MCP) + exact MILP | 🟢 Implemented + tested; wired into the campaign runner and `main.py` |
| Experiment runner (matrix, ≥30 seeds, Wilcoxon, MILP gap, plots) | 🟢 Built, unit-tested, and run end-to-end |
| RESULTS.md (regenerated with corrected CPM) | 🟢 Done — full campaign, H1/H2 verdicts, gap tables |
| Hyperparameter ablations (pop, p_c, p_m, tournament) | 🔴 Not done |

## Findings (full campaign — see [RESULTS.md](RESULTS.md))

- **H1 supported.** Schedule-aware CPM significantly lowers median C_max wherever there is headroom: 14/14 significant paired-Wilcoxon tests (Holm-corrected) favour CPM-on, for both crossovers; effect grows with instance size. Subject to the equal-generations budget caveat.
- **H2 rejected (opposite holds).** PMX significantly beats OX on every sparse instance; dense instances tie. `pmx_cpm` is the best configuration on every instance.
- Memetic EA beats the HLFET/MCP heuristics everywhere; residual gap to the lower bound is 7–12% on the two largest Kasahara graphs.
- Campaign runtime ≈ 39 min single-threaded on an Apple M4.

## ✅ Done (implemented + tested)

- **Domain model** — [src/core/models.py](src/core/models.py)
- **Kasahara STG parser** — [src/data/kasahara_parser.py](src/data/kasahara_parser.py)
- **List-scheduling decoder** — now also exposes the full schedule via `compute_schedule` (start/finish times, per-processor predecessor) — [src/core/decoder.py](src/core/decoder.py)
- **Schedule-aware CPM analysis** — critical path of the *decoded schedule* (precedence **and** resource bottlenecks) — [src/core/cpm_utils.py](src/core/cpm_utils.py)
- **CPM local search** (Lamarckian "pull-left", schedule-critical tasks recomputed per attempt) — [src/solvers/evolutionary/local_search.py](src/solvers/evolutionary/local_search.py)
- **Evolutionary algorithm (DEAP):** tournament selection, swap mutation, OX + PMX, topological init, ready-list repair, hall of fame, convergence curve — [algorithm.py](src/solvers/evolutionary/algorithm.py), [crossovers.py](src/solvers/evolutionary/crossovers.py), [strategies.py](src/solvers/evolutionary/strategies.py), [deap_setup.py](src/solvers/evolutionary/deap_setup.py)
- **Baseline heuristics HLFET + MCP** — [src/solvers/heuristics.py](src/solvers/heuristics.py)
- **Exact MILP solver** (time-indexed P|prec|C_max, CBC via PuLP; proven-optimal makespan for small instances) — [src/solvers/milp.py](src/solvers/milp.py)
- **Synthetic DAG generator + STG export** — [src/data/synthetic_generator.py](src/data/synthetic_generator.py)
- **Experiment runner (new this session):**
  - benchmark instance set as code — [src/experiments/instances.py](src/experiments/instances.py)
  - campaign runner (OX/PMX × CPM-on/off, ≥30 seeds, HLFET/MCP + MILP gap + lower bound) — [src/experiments/run_experiments.py](src/experiments/run_experiments.py)
  - statistical analysis (summary + paired Wilcoxon with Holm correction for H1/H2) — [src/experiments/analyze.py](src/experiments/analyze.py)
  - plots (boxplots + convergence curves) — [src/experiments/plots.py](src/experiments/plots.py)
- **Quick-checkpoint runner** (kept) — [src/experiments/run_partial.py](src/experiments/run_partial.py)
- **Bundled instances** — [data/sample/](data/sample/): `small.stg`, `example.stg`, `random_50.stg`, `random_300.stg`, `robot_control.stg`
- **CI** (black + pylint + pytest) — [.github/workflows/ci.yml](.github/workflows/ci.yml)

80 tests pass on Python 3.10+; `black --check src/ tests/` and `pylint --disable=C0114 src/` are clean (10.00/10). Verified against the rebased `main` by running the suite + the `--quick` campaign in a Python 3.11 conda env.

## ✅ Resolved (now on `main`)

- **CPM is schedule-aware (was the #1 risk).** `identify_critical_tasks` decodes the permutation and traces the critical chain of the *actual schedule*, following the precedence- or processor-binding predecessor at each step. Captures resource bottlenecks and varies per individual, matching the doc ("bottlenecks in the schedule"). Merged as #9.
- **Exact MILP + optimality-gap demo** — `solve_milp`, `small.stg`, `main.py` gap printout. Merged as #10.
- **`scipy` + `matplotlib`** added to [requirements.txt](requirements.txt) (`pulp` already present) — on the `experiments` branch.
- Earlier: ready-list repair (feasibility), baselines, synthetic generator, data bundled.

## ⚠️ Known issues / risks (still open)

- **No elitism.** `population = offspring` each generation; the best is only kept in `HallOfFame`, never reinjected ([algorithm.py:96](src/solvers/evolutionary/algorithm.py#L96)). Reported best can't regress, but the population can lose it.
- **Constant penalty → no feasibility gradient** ([deap_setup.py:151](src/solvers/evolutionary/deap_setup.py#L151)). Effectively dead code now that repair makes all offspring feasible — drop it or document it as a safety net.
- **`ProblemInstance(frozen=True)` with an `np.ndarray` field** — `__eq__`/`__hash__` trap ([models.py](src/core/models.py)); consider `eq=False`.

## ❌ To do

- **Merge the `experiments` branch into `main`** (open a PR) — it is the last unmerged piece.
- **Hyperparameter ablations** — pop size, p_c, p_m, tournament size (doc lists these; not swept).
- **H1 evaluation-budget comparison (optional)** — the campaign compares at *equal generations*; a fixed-evaluation-budget comparison would need an evaluation counter in the engine (follow-up, documented as a caveat in RESULTS.md).

## Notes

- Branch topology: `milp` (#10) and `fix/schedule-aware-cpm` (#9) are now **merged into `main`** (origin/main `b6f606f`). **`experiments`** is rebased onto the fresh `main` and is just `main` + the 6 experiment-runner commits (local only, not pushed). Backup of the pre-rebase branch kept at `experiments-backup`.
- `results/` is **gitignored** — campaign CSVs and `results/campaign/plots/*.png` are not committed; RESULTS.md (tracked) carries the numbers. (The earlier STATUS note that "results CSVs are committed" was wrong.)
- **STATUS.md stays untracked** (per request); not committed or pushed.
- Requires Python 3.10+ (PEP 604 `X | None` unions). System Python is 3.9 — use a 3.10+ venv/conda; CI runs 3.11.
- Reproduce the campaign: `python -m src.experiments.run_experiments` then `analyze` then `plots` (use `--quick` for a fast smoke test).
