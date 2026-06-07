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

    failures: list[str] = []
    for instance, method in cells:
        if args.skip_existing_cells and _cell_exists(campaign, instance, method):
            continue
        if _run(
            args, "run_single", _cell_flags(campaign, instance, method, args.quick)
        ):
            label = layout.cell_label(method["crossover"], method["cpm"])
            failures.append(f"{instance}/{label}")
            if not args.continue_on_error:
                break

    _report(campaign, failures)


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
