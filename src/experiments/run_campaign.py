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
import os
import subprocess
import sys

from src.experiments import layout
from src.experiments.instances import list_instances

REPOSITORY_ROOT = layout.REPOSITORY_ROOT

# The whole campaign: these four methods x every instance. Each runs at the
# global default budget (pure EA = no CPM local search; memetic = CPM on).
METHODS = (
    {"crossover": "ox", "cpm": False},  # pure EA,  order crossover
    {"crossover": "pmx", "cpm": False},  # pure EA,  PMX crossover
    {"crossover": "ox", "cpm": True},  # memetic (CPM), order crossover
    {"crossover": "pmx", "cpm": True},  # memetic (CPM), PMX crossover
)


def main(argv: list[str] | None = None) -> None:
    """Runs the campaign by subprocess-driving the single runner per cell."""
    args = _parse_arguments(argv)
    campaign = layout.resolve_campaign_dir(args.campaign, args.name)
    instances = args.instances or list_instances(quick=args.quick)
    cells = [(instance, method) for instance in instances for method in METHODS]
    print(
        f"Campaign: {len(instances)} instances x {len(METHODS)} methods -> {campaign}"
    )

    if not args.no_baselines:
        _baselines(args, campaign, instances)

    pending = [
        (instance, method)
        for instance, method in cells
        if not (args.skip_existing_cells and _cell_exists(campaign, instance, method))
    ]
    failures = _run_cells(args, campaign, pending)

    _report(campaign, failures)


def _run_cells(
    args: argparse.Namespace,
    campaign,
    cells: list[tuple[str, dict]],
) -> list[str]:
    """Runs the (instance, method) cells, up to ``--jobs`` in parallel.

    Cells are fully independent isolated subprocesses, so running several at once
    only changes wall-clock, never results. Failure semantics match the
    sequential runner: without ``--continue-on-error`` the first failure stops
    launching new cells (already-launched ones are left to finish).
    """
    failures: list[str] = []
    running: list[tuple[subprocess.Popen, str]] = []
    stop = False
    index = 0

    while (index < len(cells) and not stop) or running:
        while not stop and index < len(cells) and len(running) < args.jobs:
            instance, method = cells[index]
            index += 1
            command = _cell_command(args, campaign, instance, method)
            label = layout.cell_label(method["crossover"], method["cpm"])
            # Process is pool-managed and outlives this scope; reaped in _wait_any.
            process = subprocess.Popen(  # pylint: disable=consider-using-with
                command, cwd=str(REPOSITORY_ROOT)
            )
            running.append((process, f"{instance}/{label}"))

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
) -> list[str]:
    """Builds the run_single subprocess command for one cell."""
    flags = _cell_flags(campaign, instance, method, args.quick)
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


def _cell_flags(campaign, instance: str, method: dict, quick: bool) -> list[str]:
    """Builds the run_single flags for one (instance, method) cell."""
    flags = [
        "--instance",
        instance,
        "--crossover",
        method["crossover"],
        "--cpm" if method["cpm"] else "--no-cpm",
        "--campaign",
        str(campaign),
    ]
    if quick:
        flags.append("--quick")
    return flags


def _cell_exists(campaign, instance: str, method: dict) -> bool:
    """Returns whether a cell already has a written config.json."""
    label = layout.cell_label(method["crossover"], method["cpm"])
    return (layout.cell_dir(campaign, instance, label) / "config.json").exists()


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
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
