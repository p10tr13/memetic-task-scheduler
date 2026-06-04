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
- `src/experiments/run_partial.py` — OX-vs-PMX and CPM-on/off comparison runner
- `data/sample/` — bundled STG instances for local runs (incl. `small.stg`, small enough to solve to optimality)
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

Run the partial-results comparison (writes CSVs to `results/`, gitignored):

```
python -m src.experiments.run_partial
```

Run the tests:

```
pytest
```

## Preliminary results

See [RESULTS.md](RESULTS.md). Reproduce the synthetic-instance run with
`python -m src.experiments.run_partial` (writes `results/`).
