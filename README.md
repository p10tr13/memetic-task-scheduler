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
- `src/experiments/` — experiment runners: `instances.py` (benchmark graph registry), `run_single.py` (one instance × one EA config → a result cell), `run_campaign.py` (the OX/PMX × CPM-on/off matrix over all instances), `run_baselines.py` (HLFET/MCP/MILP/lower-bound per instance), `analyze.py` (Wilcoxon for H1/H2), `plots.py` (boxplots, convergence), `layout.py` (shared campaign layout)
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

Run the full experiment campaign, then the analysis and plots. A campaign writes
to a `--name`d (or timestamped) directory under `results/` (gitignored); each
`<instance>__<method>` cell and per-instance baseline is its own file, so a rerun
only refreshes the cells it touches:

```
python -m src.experiments.run_campaign --name run1   # add --quick for a fast smoke test
python -m src.experiments.analyze --campaign results/run1
python -m src.experiments.plots   --campaign results/run1
```

Run a single (instance, method) configuration directly:

```
python -m src.experiments.run_single --instance syn_t15_p010 --crossover ox --cpm --name run1
```

Run the tests:

```
pytest
```

## Hyperparameters

EA search parameters are **global defaults**, not per-instance — a benchmark
instance is a fixed problem, while the search budget is an experiment knob. They
live in `src/experiments/run_single.py` and are overridable on the command line
(`--population`, `--generations`, `--crossover-prob`, `--mutation-prob`, `--seeds`):

| parameter | default | basis |
|---|---|---|
| crossover probability | 0.8 | classic 0.6 (De Jong & Spears 1990) – 0.9 (Grefenstette 1986); DAG-scheduling GAs 0.65–0.95 |
| mutation probability | 0.3 | swap mutation applied *per individual* (not per-bit); permutation GAs 0.1–0.5 |
| population | 50 | De Jong & Spears (50); common range 30–100 |
| generations | 100 | fixed budget (seed-averaged convergence curves need a fixed length) |
| independent seeds | 15 | runtime-driven; below the usual 30-run convention — a documented limit on statistical power |

References: De Jong & Spears (1990); Grefenstette (1986), *Optimization of control
parameters for GAs*; Omara & Arafa (2010), *Genetic algorithms for task scheduling
problem*, JPDC 70(1); Hou, Ansari & Ren (1994), IEEE TPDS; Tobita & Kasahara
(2002), Standard Task Graph set.

## Results

See [RESULTS.md](RESULTS.md) for the full campaign: optimality/lower-bound gaps,
the HLFET/MCP baselines, and the Wilcoxon tests for H1 (CPM lowers C_max) and H2
(OX vs PMX on sparse graphs). Reproduce with the commands above.
