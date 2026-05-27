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
| ox | 73 | 74.4 | 74.0 |
| pmx | 70 | 70.4 | 70.0 |
| ox_cpm | 71 | 73.2 | 73.0 |
| pmx_cpm | 69 | 70.0 | 70.0 |

Mean best makespan per generation:

| gen | ox | pmx | ox_cpm | pmx_cpm |
|---|---|---|---|---|
| 0 | 74.4 | 74.4 | 74.4 | 74.4 |
| 10 | 74.4 | 71.2 | 73.2 | 70.3 |
| 30 | 74.4 | 70.6 | 73.2 | 70.0 |
| 60 | 74.4 | 70.4 | 73.2 | 70.0 |

Observations:

- OX never improves on this instance (flat at the initial 74.4). Precedence-blind
  crossover produces offspring that violate the precedence order, which the decoder
  penalizes — so no offspring beats the initial topological population.
- CPM local search helps, most visibly for OX (74.4 → 73.2); smaller gain for
  PMX (70.4 → 70.0). Directionally supports H1.
- PMX beats OX here. This instance is dense, while H2 concerns sparse graphs
  (|E|/|T|² < 0.1), so this is consistent with H2 rather than a counterexample.

Reproduce with `python -m src.experiments.run_partial` (writes `results/`).
