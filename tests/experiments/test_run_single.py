"""Tests for the single-run strategy builder and cell writer."""

import csv

from src.experiments import layout, run_single
from src.solvers.evolutionary.crossovers import ox_crossover, pmx_crossover


def test_timing_payload_records_wall_and_cpu_seconds() -> None:
    args = run_single._parse_arguments(["--instance", "syn", "--crossover", "ox"])
    run_single._resolve_defaults(args)
    cell_run = run_single._CellRun(
        per_seed_wall_seconds=[1.0, 3.0],
        per_seed_cpu_seconds=[0.8, 2.0],
        started_utc="2026-01-01T00:00:00+00:00",
    )

    class _Entry:
        name = "syn"

    timing = run_single._timing_payload(_Entry(), args, cell_run)

    assert timing["schema_version"] == 2
    assert timing["per_seed_seconds"] == [1.0, 3.0]
    assert timing["total_seconds"] == 4.0
    assert timing["mean_seconds"] == 2.0
    assert timing["per_seed_cpu_seconds"] == [0.8, 2.0]
    assert timing["total_cpu_seconds"] == 2.8
    assert timing["mean_cpu_seconds"] == 1.4


def test_timing_payload_handles_no_seeds() -> None:
    args = run_single._parse_arguments(["--instance", "syn", "--crossover", "ox"])
    run_single._resolve_defaults(args)

    class _Entry:
        name = "syn"

    timing = run_single._timing_payload(_Entry(), args, run_single._CellRun())

    assert timing["total_seconds"] == 0.0
    assert timing["mean_seconds"] == 0.0
    assert timing["total_cpu_seconds"] == 0.0
    assert timing["mean_cpu_seconds"] == 0.0


def test_build_strategies_enables_cpm_local_search() -> None:
    strategies = run_single.build_strategies("ox", True)

    assert strategies.local_search_strategy is not None
    assert strategies.crossover_strategy is ox_crossover


def test_build_strategies_pure_ea_has_no_local_search() -> None:
    strategies = run_single.build_strategies("pmx", False)

    assert strategies.local_search_strategy is None
    assert strategies.crossover_strategy is pmx_crossover


def test_build_strategies_threads_local_search_attempts() -> None:
    strategies = run_single.build_strategies("ox", True, local_search_attempts=7)

    assert strategies.local_search_strategy.keywords["max_attempts"] == 7


def test_resolve_defaults_equal_budget_sets_evaluation_budget() -> None:
    args = run_single._parse_arguments(
        ["--instance", "x", "--crossover", "ox", "--cpm", "--equal-budget"]
    )

    run_single._resolve_defaults(args)

    assert args.max_evaluations == run_single.EQUAL_BUDGET_EVALS
    assert args.generations == run_single.EQUAL_BUDGET_GENERATION_CAP


def test_resolve_defaults_applies_params_json(tmp_path) -> None:
    params_path = tmp_path / "best_params.json"
    layout.write_json(
        params_path,
        {
            "params": {
                "population_size": 33,
                "crossover_probability": 0.71,
                "mutation_probability": 0.22,
                "tournament_size": 5,
                "cpm_max_attempts": 42,
            }
        },
    )
    args = run_single._parse_arguments(
        ["--instance", "x", "--crossover", "ox", "--params-json", str(params_path)]
    )

    run_single._resolve_defaults(args)

    assert args.population == 33
    assert args.crossover_prob == 0.71
    assert args.mutation_prob == 0.22
    assert args.tournament_size == 5
    assert args.local_search_attempts == 42


def test_write_cell_writes_evaluation_convergence(tmp_path) -> None:
    cell = tmp_path / "runs" / "syn__ox"
    eval_rows = [
        {
            "instance": "syn",
            "config": "ox",
            "seed": 0,
            "evaluations": 50,
            "best_makespan": 12,
        }
    ]

    run_single.write_cell(
        cell, {"label": "ox"}, [], [], {"total_seconds": 0.0}, eval_rows
    )

    with (cell / "convergence_evals.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["evaluations"] == "50"


def test_resolve_defaults_uses_global_constants() -> None:
    args = run_single._parse_arguments(["--instance", "x", "--crossover", "ox"])

    run_single._resolve_defaults(args)

    assert args.population == run_single.DEFAULT_POPULATION
    assert args.generations == run_single.DEFAULT_GENERATIONS
    assert args.seeds == run_single.DEFAULT_SEEDS
    assert args.label == "ox"


def test_resolve_defaults_quick_uses_smoke_budget() -> None:
    args = run_single._parse_arguments(
        ["--instance", "x", "--crossover", "pmx", "--cpm", "--quick"]
    )

    run_single._resolve_defaults(args)

    assert args.population == run_single.QUICK_POPULATION
    assert args.seeds == run_single.QUICK_SEEDS
    assert args.label == "pmx_cpm"


def test_write_cell_round_trips(tmp_path) -> None:
    cell = tmp_path / "runs" / "syn__ox"
    config_payload = {"instance": "syn", "label": "ox", "seeds": 2}
    run_rows = [
        {
            "instance": "syn",
            "config": "ox",
            "crossover": "ox",
            "use_cpm": False,
            "seed": 0,
            "best_makespan": 10,
            "processors": 2,
            "nodes": 5,
            "edges": 4,
            "density": 0.16,
        }
    ]
    convergence_rows = [
        {
            "instance": "syn",
            "config": "ox",
            "seed": 0,
            "generation": 0,
            "best_makespan": 12,
        }
    ]
    timing_payload = {"instance": "syn", "label": "ox", "total_seconds": 0.5}

    run_single.write_cell(
        cell, config_payload, run_rows, convergence_rows, timing_payload
    )

    assert layout.read_json(cell / "config.json")["label"] == "ox"
    assert layout.read_json(cell / "timing.json")["total_seconds"] == 0.5
    with (cell / "runs.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["best_makespan"] == "10"
