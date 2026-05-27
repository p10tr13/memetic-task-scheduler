# memetic-task-scheduler

Memetic algorithm (evolutionary algorithm + CPM local search) for multiprocessor
task scheduling with precedence constraints, minimizing makespan (C_max).

## Layout

- `src/core/` — domain model, list-scheduling decoder, critical-path (CPM) analysis
- `src/data/` — parser for Kasahara STG instance files
- `src/solvers/evolutionary/` — DEAP-based EA: crossovers (OX, PMX), strategies, CPM local search
- `src/main.py` — runs the scheduler on the bundled sample instance
- `src/experiments/run_partial.py` — OX-vs-PMX and CPM-on/off comparison runner
- `data/sample/` — small example STG instance for local runs
- `tests/` — unit tests
- `docs/preliminary/` — preliminary documentation

## Setup

Requires Python 3.11+.

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run on the bundled sample instance:

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
