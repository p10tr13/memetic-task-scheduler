"""Baseline / optimum step for the experiment campaign.

Computes, once per instance, the heuristic baselines (HLFET, MCP), the exact
MILP optimum (small instances only) and the makespan lower bound, and writes one
``baselines/<instance>.json`` file per instance under a campaign directory::

    python -m src.experiments.run_baselines --campaign results/0607_1105 --quick

Baselines are decoupled from the per-config EA runs so the expensive MILP is
never recomputed per configuration. Existing baseline files are skipped unless
``--force`` is given; a rerun overwrites only the targeted instance files.
"""

import argparse
import math
from dataclasses import replace

import networkx as nx

from src.core.decoder import compute_schedule
from src.core.models import ProblemInstance
from src.experiments import layout
from src.experiments.instances import InstanceEntry, get_instance, list_instances
from src.solvers.heuristics import solve_hlfet, solve_mcp
from src.solvers.milp import solve_milp

SCHEMA_VERSION = 1


def main(argv: list[str] | None = None) -> None:
    """Writes one baseline JSON file per requested instance."""
    args = _parse_arguments(argv)
    names = args.instances if args.instances else list_instances(quick=args.quick)

    written = 0
    skipped = 0
    for name in names:
        path = layout.baseline_path(args.campaign, name)
        if path.exists() and not args.force:
            skipped += 1
            continue
        entry = get_instance(name)
        layout.write_json(path, _baseline_row(entry, args.milp_time_limit))
        written += 1
        print(f"[baseline] {name} -> {path}")

    location = layout.baselines_dir(args.campaign)
    print(f"Baselines: {written} written, {skipped} skipped, in {location}")


def _baseline_row(entry: InstanceEntry, milp_time_limit: float) -> dict:
    """Builds the baseline / optimum / lower-bound payload for one instance."""
    milp_makespan, milp_is_optimal = _milp_optimum(entry, milp_time_limit)
    return {
        "schema_version": SCHEMA_VERSION,
        "instance": entry.name,
        "processors": entry.problem_instance.number_of_groups,
        "nodes": entry.metadata.nodes,
        "edges": entry.metadata.edges,
        "density": round(entry.metadata.density, 4),
        "hlfet": solve_hlfet(entry.problem_instance).makespan,
        "mcp": solve_mcp(entry.problem_instance).makespan,
        "milp_makespan": milp_makespan,
        "milp_is_optimal": milp_is_optimal,
        "lower_bound": _lower_bound(entry.problem_instance),
    }


def _milp_optimum(entry: InstanceEntry, time_limit: float) -> tuple[int | None, bool]:
    """Returns the MILP makespan and whether it was proven optimal."""
    if not entry.supports_milp:
        return None, False

    result = solve_milp(entry.problem_instance, time_limit_seconds=time_limit)
    return result.makespan, result.is_optimal


def _lower_bound(problem_instance: ProblemInstance) -> int:
    """Computes ``max(load, critical path)`` as a makespan lower bound.

    The critical path is the makespan with unlimited processors, obtained by
    decoding a topological order with one processor per task (no resource
    contention), reusing the decoder instead of a separate longest-path routine.

    Args:
        problem_instance: Scheduling instance with a configured processor count.

    Returns:
        Lower bound on the optimal makespan.
    """
    task_times = problem_instance.task_times
    total_work = int(sum(int(task_time) for task_time in task_times))
    load_bound = math.ceil(total_work / problem_instance.number_of_groups)

    topological_order = list(
        nx.topological_sort(problem_instance.priority_relationship)
    )
    critical_path = compute_schedule(
        topological_order,
        replace(problem_instance, number_of_groups=max(len(topological_order), 1)),
    ).makespan

    return max(load_bound, critical_path)


def _parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute per-instance baselines for a campaign."
    )
    layout.add_campaign_args(parser, required=True)
    parser.add_argument(
        "--instances",
        nargs="+",
        default=None,
        help="Instance names (default: the full or --quick registry set).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use the quick-smoke instance set when no --instances are given.",
    )
    layout.add_milp_time_limit_arg(parser)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute baselines even when the JSON file already exists.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
