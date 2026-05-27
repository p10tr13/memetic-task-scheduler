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

## Preliminary results

Strictly preliminary: a single dense 45-task instance
([data/sample/example.stg](data/sample/example.stg)), 3 groups, 10 seeds,
population 40, 60 generations. No baselines or significance tests yet.

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

Observations:

- OX improves little on this instance (74.4 → 74.2), while PMX is clearly stronger,
  converging to ~69.8.
- CPM local search helps the weaker OX (74.2 → 72.3 via ox_cpm) but gives no gain
  over PMX (69.9 vs 69.8). Directionally supports H1.
- PMX beats OX here. This instance is dense, while H2 concerns sparse graphs
  (|E|/|T|² < 0.1), so this is consistent with H2 rather than a counterexample.

Reproduce with `python -m src.experiments.run_partial` (writes `results/`).
