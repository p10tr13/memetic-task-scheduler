# memetic-task-scheduler

Memetic algorithm (evolutionary algorithm + CPM local search) for multiprocessor
task scheduling with precedence constraints, minimizing makespan (C_max).

## Layout

- `src/core/` — domain model, list-scheduling decoder, critical-path (CPM) analysis
- `src/data/` — parser for Kasahara STG instance files
- `src/solvers/evolutionary/` — DEAP-based EA: crossovers (OX, PMX), strategies, CPM local search
- `src/solvers/heuristics.py` — HLFET and MCP list-scheduling baselines
- `src/solvers/milp.py` — exact time-indexed MILP (CBC via PuLP) for small instances, used to report optimality gaps
- `src/main.py` — runs HLFET, MCP, the memetic EA, and the MILP on a small bundled instance and prints each optimality gap
- `src/experiments/` — experiment campaign: `instances.py` (benchmark set), `run_experiments.py` (OX/PMX × CPM-on/off matrix, baselines, MILP gap), `analyze.py` (Wilcoxon for H1/H2), `plots.py` (boxplots, convergence); `run_partial.py` is the older quick-checkpoint runner
- `data/` — benchmark task graphs (see `data/README.md`): `kasahara/` (10 real STG graphs spanning size/density), `synthetic/` (10 generated graphs, `make data`), `example.stg` (tiny MILP-demo graph), and `optima.csv` (published makespan references for the Kasahara graphs)
- `tests/` — unit tests
- `docs/preliminary/` — preliminary documentation

## Setup

Requires Python 3.10+ (uses PEP 604 `X | None` type unions).

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run HLFET, MCP, the memetic EA, and the exact MILP on the small bundled instance
and print each makespan with its gap to the MILP optimum:

```
python -m src.main
```

Run the full experiment campaign, then the analysis and plots (all write to
`results/campaign/`, gitignored):

```
python -m src.experiments.run_experiments   # add --quick for a fast smoke test
python -m src.experiments.analyze
python -m src.experiments.plots
```

Run the tests:

```
pytest
```

## Results

See [RESULTS.md](RESULTS.md) for the full campaign: optimality/lower-bound gaps,
the HLFET/MCP baselines, and the Wilcoxon tests for H1 (CPM lowers C_max) and H2
(OX vs PMX on sparse graphs). Reproduce with the commands above.
