"""Declarative experiment campaign: a config list driving the single runner.

The campaign is just the four methods below run on every registry instance. EA
hyperparameters are NOT set here — :mod:`src.experiments.run_single` fills them
from its documented global defaults — so the campaign passes no budget flags.
Baselines are computed once per instance first, then each (instance, method)
cell is run as a separate ``run_single`` subprocess::

    python -m src.experiments.run_campaign --quick --name smoke

Each cell writes its own directory, so a crash or partial rerun only re-runs the
missing cells; nothing shares a single clobberable file.
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

from src.experiments import layout, run_single
from src.experiments.instances import list_instances

REPOSITORY_ROOT = layout.REPOSITORY_ROOT
CAMPAIGN_SCHEMA_VERSION = 2

# The whole campaign: these four methods x every instance. Each runs at the
# global default budget (pure EA = no CPM local search; memetic = CPM on).
METHODS = (
    {"crossover": "ox", "cpm": False},  # pure EA,  order crossover
    {"crossover": "pmx", "cpm": False},  # pure EA,  PMX crossover
    {"crossover": "ox", "cpm": True},  # memetic (CPM), order crossover
    {"crossover": "pmx", "cpm": True},  # memetic (CPM), PMX crossover
)

# run_single flags forwarded verbatim from the campaign so a budget/tuning sweep
# can be configured once at the campaign level. ``store_true`` flags are emitted
# only when set; valued flags only when not None (None => run_single's default).
_PASSTHROUGH_STORE_TRUE = ("equal_budget",)
_PASSTHROUGH_VALUED = {
    "population": "--population",
    "generations": "--generations",
    "crossover_prob": "--crossover-prob",
    "mutation_prob": "--mutation-prob",
    "tournament_size": "--tournament-size",
    "local_search_attempts": "--local-search-attempts",
    "max_evaluations": "--max-evaluations",
    "params_json": "--params-json",
}


def main(argv: list[str] | None = None) -> None:
    """Runs the campaign by subprocess-driving the single runner per work unit."""
    args = _parse_arguments(argv)
    campaign = layout.resolve_campaign_dir(args.campaign, args.name)
    instances = args.instances or list_instances(quick=args.quick)
    seeds = _campaign_seeds(args)
    unit = f"{len(seeds)} seeds" if args.per_seed else "method"
    print(
        f"Campaign: {len(instances)} instances x {len(METHODS)} methods "
        f"({unit} per cell) -> {campaign}"
    )

    started = datetime.now(timezone.utc)
    wall_start = time.perf_counter()

    if not args.no_baselines:
        _baselines(args, campaign, instances)
    baselines_seconds = round(time.perf_counter() - wall_start, 4)

    jobs = _build_jobs(args, campaign, instances, seeds)
    cells_start = time.perf_counter()
    failures = _run_jobs(args, jobs)
    cells_seconds = round(time.perf_counter() - cells_start, 4)

    if args.per_seed:
        _merge_subcells(campaign, instances, seeds)

    _write_campaign_timing(
        campaign,
        started,
        {
            "schema_version": CAMPAIGN_SCHEMA_VERSION,
            "started_utc": started.isoformat(),
            "jobs": args.jobs,
            "per_seed": args.per_seed,
            "cells_run": len(jobs),
            "wall_seconds": round(time.perf_counter() - wall_start, 4),
            "baselines_wall_seconds": baselines_seconds,
            "cells_wall_seconds": cells_seconds,
        },
    )
    _report(campaign, failures)


def _campaign_seeds(args: argparse.Namespace) -> list[int]:
    """Returns the seed values the campaign covers."""
    count = args.seeds
    if count is None:
        count = run_single.QUICK_SEEDS if args.quick else run_single.DEFAULT_SEEDS
    return list(range(args.seed_start, args.seed_start + count))


def _build_jobs(
    args: argparse.Namespace,
    campaign,
    instances: list[str],
    seeds: list[int],
) -> list[tuple[list[str], str]]:
    """Builds the (command, label) work units, skipping any already on disk.

    In ``--per-seed`` mode each unit is one (instance, method, seed) sub-cell, so
    the campaign parallelises 1200-way (20x4x15) instead of 80-way; otherwise each
    unit is one (instance, method) cell that runs all seeds in-process.
    """
    jobs: list[tuple[list[str], str]] = []
    for instance in instances:
        for method in METHODS:
            label = layout.cell_label(method["crossover"], method["cpm"])
            seed_chunks = seeds if args.per_seed else [None]
            for seed in seed_chunks:
                if args.skip_existing_cells and _unit_exists(
                    campaign, instance, label, seed
                ):
                    continue
                command = _cell_command(args, campaign, instance, method, seed)
                tag = f"{instance}/{label}"
                jobs.append((command, tag if seed is None else f"{tag}#{seed}"))
    return jobs


def _run_jobs(
    args: argparse.Namespace,
    jobs: list[tuple[list[str], str]],
) -> list[str]:
    """Runs the (command, label) work units, up to ``--jobs`` in parallel.

    Units are fully independent isolated subprocesses, so running several at once
    only changes wall-clock, never results. Failure semantics: without
    ``--continue-on-error`` the first failure stops launching new units
    (already-launched ones are left to finish).
    """
    failures: list[str] = []
    running: list[tuple[subprocess.Popen, str]] = []
    stop = False
    index = 0

    while (index < len(jobs) and not stop) or running:
        while not stop and index < len(jobs) and len(running) < args.jobs:
            command, label = jobs[index]
            index += 1
            # Process is pool-managed and outlives this scope; reaped in _wait_any.
            process = subprocess.Popen(  # pylint: disable=consider-using-with
                command, cwd=str(REPOSITORY_ROOT)
            )
            running.append((process, label))

        finished = _wait_any(running)
        for process, cell_label in finished:
            running.remove((process, cell_label))
            if process.returncode != 0:
                failures.append(cell_label)
                if not args.continue_on_error:
                    stop = True

    return failures


def _wait_any(
    running: list[tuple[subprocess.Popen, str]],
) -> list[tuple[subprocess.Popen, str]]:
    """Blocks until at least one running cell exits; returns the finished ones."""
    if not running:
        return []
    # Block on the oldest, then reap any others that finished in the meantime.
    running[0][0].wait()
    return [pair for pair in running if pair[0].poll() is not None]


def _cell_command(
    args: argparse.Namespace,
    campaign,
    instance: str,
    method: dict,
    seed: int | None,
) -> list[str]:
    """Builds the run_single subprocess command for one work unit."""
    flags = _cell_flags(args, campaign, instance, method, seed)
    return [args.python, "-m", "src.experiments.run_single", *flags]


def _default_jobs() -> int:
    """Default concurrency: leave two cores free, but always at least one job."""
    return max(1, (os.cpu_count() or 2) - 2)


def _run(args: argparse.Namespace, module: str, flags: list[str]) -> int:
    """Runs one experiments subprocess and returns its (non-zero on failure) code."""
    command = [args.python, "-m", f"src.experiments.{module}", *flags]
    return subprocess.run(command, cwd=str(REPOSITORY_ROOT), check=False).returncode


def _baselines(args: argparse.Namespace, campaign, instances: list[str]) -> None:
    """Computes baselines once per instance via the baseline subprocess."""
    _run(
        args,
        "run_baselines",
        [
            "--campaign",
            str(campaign),
            "--milp-time-limit",
            str(args.milp_time_limit),
            "--instances",
            *instances,
        ],
    )


def _cell_flags(
    args: argparse.Namespace,
    campaign,
    instance: str,
    method: dict,
    seed: int | None,
) -> list[str]:
    """Builds the run_single flags for one work unit, forwarding budget/tune flags."""
    flags = [
        "--instance",
        instance,
        "--crossover",
        method["crossover"],
        "--cpm" if method["cpm"] else "--no-cpm",
        "--campaign",
        str(campaign),
    ]
    if args.quick:
        flags.append("--quick")
    if seed is not None:
        flags += ["--sub-cell", "--seed-start", str(seed), "--seeds", "1"]
    elif args.seeds is not None:
        flags += ["--seeds", str(args.seeds), "--seed-start", str(args.seed_start)]

    for attr in _PASSTHROUGH_STORE_TRUE:
        if getattr(args, attr):
            flags.append(f"--{attr.replace('_', '-')}")
    for attr, flag in _PASSTHROUGH_VALUED.items():
        value = getattr(args, attr)
        if value is not None:
            flags += [flag, str(value)]
    return flags


def _unit_exists(campaign, instance: str, label: str, seed: int | None) -> bool:
    """Returns whether a work unit's cell (or sub-cell) is already on disk."""
    if seed is None:
        cell = layout.cell_dir(campaign, instance, label)
    else:
        cell = layout.subcell_dir(campaign, instance, label, seed)
    return (cell / "config.json").exists()


def _merge_subcells(campaign, instances: list[str], seeds: list[int]) -> None:
    """Reassembles per-seed sub-cells into the canonical (instance, method) cell.

    Concatenates each sub-cell's per-seed rows (in seed order) into the cell-level
    runs.csv / convergence.csv / convergence_evals.csv that analyze/plots/figures
    read, and sums the per-seed timings, so downstream analysis is identical to a
    non-per-seed campaign.
    """
    for instance in instances:
        for method in METHODS:
            label = layout.cell_label(method["crossover"], method["cpm"])
            present = [
                seed
                for seed in seeds
                if (
                    layout.subcell_dir(campaign, instance, label, seed) / "runs.csv"
                ).exists()
            ]
            if not present:
                continue
            _merge_one_cell(campaign, instance, label, present)


def _merge_one_cell(campaign, instance: str, label: str, seeds: list[int]) -> None:
    """Merges the present sub-cells of one (instance, method) into its cell."""
    cell = layout.cell_dir(campaign, instance, label)
    sub_dirs = [layout.subcell_dir(campaign, instance, label, s) for s in seeds]

    for filename, fields in (
        ("runs.csv", run_single.RUN_FIELDS),
        ("convergence.csv", run_single.CONVERGENCE_FIELDS),
        ("convergence_evals.csv", run_single.CONVERGENCE_EVAL_FIELDS),
    ):
        rows = [row for sub in sub_dirs for row in _read_csv(sub / filename)]
        layout.write_csv(cell / filename, fields, rows)

    config = layout.read_json(sub_dirs[0] / "config.json")
    config["seed_start"] = min(seeds)
    config["seeds"] = len(seeds)
    layout.write_json(cell / "config.json", config)
    layout.write_json(cell / "timing.json", _merge_timing(instance, label, sub_dirs))


def _read_csv(path) -> list[dict]:
    """Reads a CSV file into a list of dict rows (empty if missing)."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _merge_timing(instance: str, label: str, sub_dirs: list) -> dict:
    """Sums the per-seed sub-cell timings into one cell timing payload."""
    timings = [layout.read_json(sub / "timing.json") for sub in sub_dirs]
    wall = [s for t in timings for s in t.get("per_seed_seconds", [])]
    cpu = [s for t in timings for s in t.get("per_seed_cpu_seconds", [])]
    return {
        "schema_version": run_single.SCHEMA_VERSION,
        "instance": instance,
        "label": label,
        **run_single.timing_seconds_fields(wall, cpu),
        "merged_from_subcells": len(sub_dirs),
    }


def _write_campaign_timing(campaign, started: datetime, timing: dict) -> None:
    """Writes the campaign-level wall-clock timing for this invocation.

    This is the real end-to-end cost under ``--jobs`` parallelism; per-cell
    ``timing.json`` wall times are inflated by contention when jobs > 1, so the
    recorded ``jobs`` says whether those per-cell wall times are isolated.

    The file is named per invocation (``campaign_timing_<UTC stamp>.json``) so
    resuming or re-running a campaign in the same directory never overwrites an
    earlier run's timing, matching the per-cell "nothing clobbers" layout.
    """
    stamp = started.strftime("%Y%m%d_%H%M%S")
    layout.write_json(campaign / f"campaign_timing_{stamp}.json", timing)


def _report(campaign, failures: list[str]) -> None:
    """Prints a short campaign summary."""
    if failures:
        print(
            f"Campaign finished with {len(failures)} failed cell(s): {', '.join(failures)}"
        )
    else:
        print(f"Campaign finished OK -> {campaign}")


def _parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the experiment campaign.")
    layout.add_campaign_args(parser, with_name=True)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use the tiny quick-smoke instance set and budget.",
    )
    parser.add_argument(
        "--instances",
        nargs="+",
        default=None,
        help="Restrict the campaign to these instance names.",
    )
    layout.add_milp_time_limit_arg(parser)
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used for the single-run subprocesses.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=_default_jobs(),
        help="Max cells to run concurrently (cells are independent; default: cores-2).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running remaining cells after a cell fails.",
    )
    parser.add_argument(
        "--skip-existing-cells",
        action="store_true",
        help="Skip cells that already have a config.json (resume faster).",
    )
    parser.add_argument(
        "--no-baselines",
        action="store_true",
        help="Do not compute baselines (EA cells only).",
    )
    parser.add_argument(
        "--per-seed",
        action="store_true",
        help="Parallelise at (instance, method, seed) granularity via per-seed "
        "sub-cells, merged afterwards. Saturates many-core machines.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=None,
        help="Seeds per cell (default: run_single's global default).",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="First seed value.",
    )
    parser.add_argument(
        "--equal-budget",
        action="store_true",
        help="Forward --equal-budget to every run_single (equal eval budget).",
    )
    parser.add_argument(
        "--max-evaluations",
        type=int,
        default=None,
        help="Forward an explicit objective-evaluation budget to every run_single.",
    )
    parser.add_argument(
        "--params-json",
        default=None,
        help="Forward a tuned best_params.json to every run_single.",
    )
    parser.add_argument("--population", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--generations", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--crossover-prob", type=float, default=None, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--mutation-prob", type=float, default=None, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--tournament-size", type=int, default=None, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--local-search-attempts", type=int, default=None, help=argparse.SUPPRESS
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
