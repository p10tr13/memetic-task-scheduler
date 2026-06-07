"""Tests for the single-run strategy builder and cell writer."""

import csv

from src.experiments import layout, run_single
from src.solvers.evolutionary.crossovers import ox_crossover, pmx_crossover


def test_build_strategies_enables_cpm_local_search() -> None:
    strategies = run_single.build_strategies("ox", True)

    assert strategies.local_search_strategy is not None
    assert strategies.crossover_strategy is ox_crossover


def test_build_strategies_pure_ea_has_no_local_search() -> None:
    strategies = run_single.build_strategies("pmx", False)

    assert strategies.local_search_strategy is None
    assert strategies.crossover_strategy is pmx_crossover


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
