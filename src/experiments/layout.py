"""Shared on-disk layout for experiment campaigns.

Single source of truth for the campaign directory structure and the per-cell
labels, used by the runners (which write) and by analyze/plots (which read) so
the convention can never drift between producers and consumers::

    results/<timestamp-or-name>/
      baselines/<instance>.json          one file per instance
      runs/<instance>__<label>/          one cell per (instance, method)
        config.json  runs.csv  convergence.csv  timing.json
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPOSITORY_ROOT / "results"


def cell_label(crossover: str, use_cpm: bool) -> str:
    """Returns the canonical method label, e.g. ``ox`` or ``ox_cpm``."""
    return f"{crossover}_cpm" if use_cpm else crossover


def cell_dirname(instance: str, label: str) -> str:
    """Returns the cell directory name ``<instance>__<label>``."""
    return f"{instance}__{label}"


def runs_dir(campaign: Path) -> Path:
    """Returns the ``runs/`` subdirectory of a campaign."""
    return campaign / "runs"


def cell_dir(campaign: Path, instance: str, label: str) -> Path:
    """Returns the cell directory for one (instance, method)."""
    return runs_dir(campaign) / cell_dirname(instance, label)


def baselines_dir(campaign: Path) -> Path:
    """Returns the ``baselines/`` subdirectory of a campaign."""
    return campaign / "baselines"


def baseline_path(campaign: Path, instance: str) -> Path:
    """Returns the baseline JSON path for one instance."""
    return baselines_dir(campaign) / f"{instance}.json"


def resolve_campaign_dir(
    campaign: Path | None = None,
    name: str | None = None,
) -> Path:
    """Resolves and creates the campaign directory.

    Args:
        campaign: Explicit campaign directory to use or resume. Takes priority.
        name: Campaign name under ``results/`` when no explicit dir is given.

    Returns:
        The campaign directory (created if missing). Defaults to a fresh
        ``results/<YYYYmmdd_HHMMSS>`` when neither argument is given.
    """
    if campaign is not None:
        target = Path(campaign)
    elif name is not None:
        target = RESULTS_ROOT / name
    else:
        target = RESULTS_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_json(path: Path, payload: dict) -> None:
    """Writes a dict as pretty, key-sorted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2, sort_keys=True)
        json_file.write("\n")


def read_json(path: Path) -> dict:
    """Reads a JSON dict from a file."""
    with path.open(encoding="utf-8") as json_file:
        return json.load(json_file)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    """Writes rows to a CSV file with the given header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def add_campaign_args(
    parser: argparse.ArgumentParser,
    *,
    required: bool = False,
    with_name: bool = False,
) -> None:
    """Adds the shared ``--campaign`` (and optional ``--name``) arguments.

    Args:
        parser: Parser to extend.
        required: Whether ``--campaign`` is required (readers) or optional
            (runners, which fall back to a fresh timestamped directory).
        with_name: Whether to also add ``--name`` for naming a fresh campaign.
    """
    parser.add_argument(
        "--campaign",
        type=Path,
        default=None,
        required=required,
        help="Campaign directory (results/<...>) to write into or read.",
    )
    if with_name:
        parser.add_argument(
            "--name",
            default=None,
            help="Campaign name under results/ when no --campaign is given.",
        )


def add_milp_time_limit_arg(parser: argparse.ArgumentParser) -> None:
    """Adds the shared ``--milp-time-limit`` argument."""
    parser.add_argument(
        "--milp-time-limit",
        type=float,
        default=120.0,
        help="CBC time limit (seconds) for the exact MILP on small instances.",
    )
