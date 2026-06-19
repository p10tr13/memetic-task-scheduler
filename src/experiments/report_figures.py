"""Reproduce every data-driven figure of the final report from a campaign.

The final report contains six figures. Five are data-driven and regenerated
here from a campaign directory; the sixth (the example DAG plus its Gantt chart)
is an editorial schematic and lives, hand-authored, in
``docs/figures/example.tex``.

Two output formats are supported:

``--format pgfplots`` (default)
    Writes data tables (``*.dat``) plus self-contained pgfplots/TikZ *plot
    bodies* (``*.tex``, one ``tikzpicture`` each) under ``--outdir``. The report
    wraps each in a ``figure`` float and ``\\input``s it. Colours and the shared
    ``wykres`` style come from ``docs/figures/_preamble.tex``.

``--format matplotlib``
    Writes PNG renderings of the same five figures under ``<outdir>/png``.

Both formats are computed from the same campaign data, so the two renderings of
a figure are numerically identical.

Run from the repository root, pointing at a campaign directory::

    python -m src.experiments.report_figures --campaign results/<dir>
    python -m src.experiments.report_figures --campaign results/<dir> \\
        --format matplotlib --outdir /tmp/figs

The pgfplots path is dependency-free (standard library only); matplotlib is
imported lazily and only needed for ``--format matplotlib``.
"""

import argparse
import csv
import math
import statistics
from pathlib import Path

from src.experiments import layout

CONFIG_ORDER = ("ox", "pmx", "ox_cpm", "pmx_cpm")

# pgfplots colour macros (defined in docs/figures/_preamble.tex) and the
# matching hex values for the matplotlib renderings.
PGF_COLORS = {"ox": "colOX", "pmx": "colPMX", "ox_cpm": "colOXC", "pmx_cpm": "colPMXC"}
MPL_COLORS = {
    "ox": "#1F77B4",
    "pmx": "#FF7F0E",
    "ox_cpm": "#2CA02C",
    "pmx_cpm": "#D62728",
}
# Underscored config names need escaping inside LaTeX text/legends.
TEX_LABELS = {config: config.replace("_", r"\_") for config in CONFIG_ORDER}

SPARSE_DENSITY_THRESHOLD = 0.1

# Instances shown in the per-instance "gap to LB" bar chart, with the short
# x-axis label used in the report.
GAP_INSTANCES = [
    ("syn_t22_p018", "syn_t22"),
    ("syn_t60_p005", "syn_t60_p005"),
    ("syn_t100_p012", "syn_t100_p012"),
    ("rand0173_t100_p039", "rand0173"),
    ("rand0000_t300_p016", "rand0000_t300"),
    ("robot_t88_p003", "robot_t88"),
    ("fpppp_t334_p002", "fpppp"),
]
# Instances shown in the 2x2 boxplot panel.
BOX_INSTANCES = [
    "syn_t100_p012",
    "rand0173_t100_p039",
    "robot_t88_p003",
    "fpppp_t334_p002",
]

# Relative path (from the report's main .tex) to the generated data tables.
DATA_RELPATH = "figures/data"

FIGURE_NAMES = ("gap", "convergence_aggregate", "boxplots", "aggregate_gap", "cpu_time")

# Equal-budget figures need ``convergence_evals.csv`` (written only by
# equal-budget campaigns), so they are opt-in via ``--figure`` and excluded from
# the default "all" set, which must keep working on older campaigns.
EVAL_FIGURE_NAMES = ("convergence_evals",)

# Output basename per figure, prefixed with its number in the report (the
# hand-authored example schematic is figure 01 and is not generated here).
FIGURE_FILES = {
    "gap": "02_gap",
    "convergence_aggregate": "03_convergence_aggregate",
    "boxplots": "04_boxplots",
    "aggregate_gap": "05_aggregate_gap",
    "cpu_time": "06_cpu_time",
    "convergence_evals": "07_convergence_evals",
}


def main(argv: list[str] | None = None) -> None:
    """Regenerates the report figures in the requested format."""
    arguments = _parse_arguments(argv)
    campaign = arguments.campaign
    outdir = arguments.outdir

    data = _load_campaign(campaign)

    if arguments.format == "pgfplots":
        written = _write_pgfplots(data, outdir, arguments.figure)
    else:
        written = _write_matplotlib(data, outdir, arguments.figure)

    print(f"Wrote {len(written)} files to {outdir}:")
    for path in written:
        print(f"  {path}")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _load_campaign(campaign: Path) -> dict:
    """Loads everything the figures need from a campaign directory."""
    return {
        "gap_percent": _load_gap_percent(campaign),
        "best_makespans": _load_best_makespans(campaign),
        "density": _load_density(campaign),
        "convergence": _load_convergence(campaign),
        "convergence_evals": _load_convergence_evals(campaign),
        "reference": _load_reference(campaign),
        "timing": _load_timing(campaign),
        "nodes": _load_node_counts(campaign),
    }


def _load_gap_percent(campaign: Path) -> dict[tuple[str, str], float]:
    """Loads median gap-to-reference percentages from ``summary.csv``."""
    summary = campaign / "summary.csv"
    if not summary.exists():
        raise SystemExit(
            f"{summary} not found — run "
            "`python -m src.experiments.analyze --campaign <dir>` first."
        )
    gaps: dict[tuple[str, str], float] = {}
    with summary.open(encoding="utf-8") as summary_file:
        for row in csv.DictReader(summary_file):
            gaps[(row["instance"], row["config"])] = float(row["gap_percent"])
    return gaps


def _load_best_makespans(campaign: Path) -> dict[tuple[str, str], list[int]]:
    """Loads per-seed best makespans keyed by (instance, config)."""
    grouped: dict[tuple[str, str], list[int]] = {}
    for runs_csv in sorted(layout.runs_dir(campaign).glob("*/runs.csv")):
        with runs_csv.open(encoding="utf-8") as runs_file:
            for row in csv.DictReader(runs_file):
                key = (row["instance"], row["config"])
                grouped.setdefault(key, []).append(int(row["best_makespan"]))
    return grouped


def _load_density(campaign: Path) -> dict[str, float]:
    """Loads per-instance edge density from the baseline files."""
    return {
        instance: float(row["density"])
        for instance, row in _baselines(campaign).items()
    }


def _load_reference(campaign: Path) -> dict[str, int]:
    """Loads the per-instance reference makespan (MILP optimum or lower bound)."""
    references: dict[str, int] = {}
    for instance, row in _baselines(campaign).items():
        if row.get("milp_is_optimal") and row.get("milp_makespan") is not None:
            references[instance] = int(row["milp_makespan"])
        else:
            references[instance] = int(row["lower_bound"])
    return references


def _baselines(campaign: Path) -> dict[str, dict]:
    """Reads every ``baselines/*.json`` keyed by instance name."""
    return layout.read_baselines(campaign)


def _load_convergence(campaign: Path) -> dict[tuple[str, str], list[float]]:
    """Loads seed-averaged best-makespan curves keyed by (instance, config)."""
    indexed: dict[tuple[str, str], dict[int, list[int]]] = {}
    for conv_csv in sorted(layout.runs_dir(campaign).glob("*/convergence.csv")):
        with conv_csv.open(encoding="utf-8") as convergence_file:
            for row in csv.DictReader(convergence_file):
                key = (row["instance"], row["config"])
                generation = indexed.setdefault(key, {}).setdefault(
                    int(row["generation"]), []
                )
                generation.append(int(row["best_makespan"]))
    return {
        key: [
            sum(values[generation]) / len(values[generation])
            for generation in sorted(values)
        ]
        for key, values in indexed.items()
    }


def _load_convergence_evals(
    campaign: Path,
) -> dict[tuple[str, str], list[list[tuple[int, int]]]]:
    """Loads per-seed ``(evaluations, best_makespan)`` curves per (instance, config).

    Unlike :func:`_load_convergence`, seeds are kept separate (not averaged): the
    curves are sampled at different evaluation counts per run, so they must be
    resampled onto a shared evaluation grid before averaging. Returns ``{}`` when
    the campaign has no ``convergence_evals.csv`` (e.g. an older, non-equal-budget
    campaign), so the rest of the figures still load.
    """
    indexed: dict[tuple[str, str], dict[int, list[tuple[int, int]]]] = {}
    for conv_csv in sorted(layout.runs_dir(campaign).glob("*/convergence_evals.csv")):
        with conv_csv.open(encoding="utf-8") as convergence_file:
            for row in csv.DictReader(convergence_file):
                key = (row["instance"], row["config"])
                seed_curve = indexed.setdefault(key, {}).setdefault(
                    int(row["seed"]), []
                )
                seed_curve.append((int(row["evaluations"]), int(row["best_makespan"])))
    return {
        key: [sorted(seeds[seed]) for seed in sorted(seeds)]
        for key, seeds in indexed.items()
    }


def _load_timing(campaign: Path) -> dict[tuple[str, str], float]:
    """Loads the mean wall-clock seconds per run, keyed by (instance, config)."""
    timing: dict[tuple[str, str], float] = {}
    for timing_json in sorted(layout.runs_dir(campaign).glob("*/timing.json")):
        row = layout.read_json(timing_json)
        timing[(row["instance"], row["label"])] = float(row["mean_seconds"])
    return timing


def _load_node_counts(campaign: Path) -> dict[str, int]:
    """Loads the task (node) count per instance from the cell configs."""
    nodes: dict[str, int] = {}
    for config_json in sorted(layout.runs_dir(campaign).glob("*/config.json")):
        row = layout.read_json(config_json)
        nodes[row["instance"]] = int(row["nodes"])
    return nodes


# --------------------------------------------------------------------------- #
# Derived figure data
# --------------------------------------------------------------------------- #


def _box_stats(values: list[int]) -> dict[str, float]:
    """Returns min, quartiles (inclusive) and max — the report's whisker rule."""
    quartiles = statistics.quantiles(values, n=4, method="inclusive")
    return {
        "lw": min(values),
        "lq": quartiles[0],
        "med": statistics.median(values),
        "uq": quartiles[2],
        "uw": max(values),
    }


def _mean_gap_by_group(data: dict) -> dict[str, dict[str, float]]:
    """Mean gap per config over all instances and the dense/sparse split."""
    gaps = data["gap_percent"]
    density = data["density"]
    instances = sorted({instance for instance, _config in gaps})
    groups = {
        "all": instances,
        "dense": [i for i in instances if density[i] >= SPARSE_DENSITY_THRESHOLD],
        "sparse": [i for i in instances if density[i] < SPARSE_DENSITY_THRESHOLD],
    }
    return {
        group: {
            config: round(
                statistics.mean(gaps[(instance, config)] for instance in members), 2
            )
            for config in CONFIG_ORDER
        }
        for group, members in groups.items()
    }


def _aggregate_convergence(data: dict) -> list[list[float]]:
    """Per-generation mean gap-to-reference (%) across instances, per config.

    Returns rows ``[generation, ox, pmx, ox_cpm, pmx_cpm]``.
    """
    convergence = data["convergence"]
    reference = data["reference"]
    instances = sorted({instance for instance, _config in convergence})
    generations = len(next(iter(convergence.values())))

    rows: list[list[float]] = []
    for generation in range(generations):
        row: list[float] = [generation]
        for config in CONFIG_ORDER:
            gaps = [
                100.0
                * (convergence[(instance, config)][generation] - reference[instance])
                / reference[instance]
                for instance in instances
                if (instance, config) in convergence
            ]
            row.append(round(statistics.mean(gaps), 3))
        rows.append(row)
    return rows


def _aggregate_convergence_evals(  # pylint: disable=too-many-locals
    data: dict, points: int = 40
) -> list[list[float]]:
    """Budget-fair convergence: mean gap-to-reference (%) vs objective evaluations.

    Each seed's best-so-far curve is a step function of evaluations, so it is
    resampled onto a shared evaluation grid (bounded by the smallest last-eval
    across all runs so every curve covers the grid), averaged over seeds, then
    over instances. Returns rows ``[evaluations, ox, pmx, ox_cpm, pmx_cpm]``.
    """
    curves = data["convergence_evals"]
    reference = data["reference"]
    if not curves:
        return []

    instances = sorted({instance for instance, _config in curves})
    seed_curves = [curve for series in curves.values() for curve in series if curve]
    lower = max(curve[0][0] for curve in seed_curves)
    upper = min(curve[-1][0] for curve in seed_curves)
    grid = [round(lower + (upper - lower) * i / (points - 1)) for i in range(points)]

    rows: list[list[float]] = []
    for evaluations in grid:
        row: list[float] = [evaluations]
        for config in CONFIG_ORDER:
            gaps = []
            for instance in instances:
                series = curves.get((instance, config))
                if not series:
                    continue
                mean_best = statistics.mean(
                    _best_at(curve, evaluations) for curve in series
                )
                gaps.append(
                    100.0 * (mean_best - reference[instance]) / reference[instance]
                )
            row.append(round(statistics.mean(gaps), 3) if gaps else 0.0)
        rows.append(row)
    return rows


def _best_at(curve: list[tuple[int, int]], evaluations: int) -> int:
    """Best makespan reached by ``evaluations`` (step-interpolating the curve)."""
    best = curve[0][1]
    for sample_evals, sample_best in curve:
        if sample_evals > evaluations:
            break
        best = sample_best
    return best


def _cpu_rows(data: dict) -> list[list[float]]:
    """CPU-time table rows ``[tasks, ox, pmx, ox_cpm, pmx_cpm]`` sorted by size."""
    timing = data["timing"]
    nodes = data["nodes"]
    instances = sorted(nodes, key=lambda instance: nodes[instance])
    rows: list[list[float]] = []
    for instance in instances:
        row: list[float] = [nodes[instance]]
        for config in CONFIG_ORDER:
            row.append(round(timing[(instance, config)], 3))
        rows.append(row)
    return rows


def _power_law_fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least-squares fit of ``y = a * x^b`` via linear regression in log-log."""
    log_x = [math.log(x) for x in xs]
    log_y = [math.log(y) for y in ys]
    count = len(log_x)
    mean_x = sum(log_x) / count
    mean_y = sum(log_y) / count
    covariance = sum((log_x[i] - mean_x) * (log_y[i] - mean_y) for i in range(count))
    variance = sum((log_x[i] - mean_x) ** 2 for i in range(count))
    slope = covariance / variance
    intercept = math.exp(mean_y - slope * mean_x)
    return intercept, slope


# --------------------------------------------------------------------------- #
# pgfplots output
# --------------------------------------------------------------------------- #


def _write_pgfplots(data: dict, outdir: Path, only: str) -> list[Path]:
    """Writes the requested ``.dat`` tables and ``.tex`` plot bodies."""
    data_dir = outdir / "data"
    outdir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    builders = {
        "gap": lambda: _pgf_gap(data),
        "aggregate_gap": lambda: _pgf_aggregate_gap(data),
        "boxplots": lambda: _pgf_boxplots(data),
        "convergence_aggregate": lambda: _pgf_convergence(data, data_dir, written),
        "convergence_evals": lambda: _pgf_convergence_evals(data, data_dir, written),
        "cpu_time": lambda: _pgf_cpu(data, data_dir, written),
    }
    for name in _selected(only):
        fragment = builders[name]()
        path = outdir / f"{FIGURE_FILES[name]}.tex"
        path.write_text(fragment, encoding="utf-8")
        written.append(path)
    return written


def _write_dat(path: Path, header: list[str], rows: list[list]) -> None:
    """Writes a whitespace-separated pgfplots data table."""
    lines = [" ".join(header)]
    for row in rows:
        lines.append(" ".join(_format_number(value) for value in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_number(value) -> str:
    """Formats integers plainly and floats at a fixed 3-decimal precision."""
    if isinstance(value, int):
        return str(value)
    return f"{value:.3f}"


def _generated_banner() -> str:
    """Returns the header comment stamped on every generated fragment."""
    return (
        "% Generated by src/experiments/report_figures.py — do not edit by hand.\n"
        "% Regenerate: python -m src.experiments.report_figures "
        "--campaign results/<dir>\n"
    )


def _pgf_gap(data: dict) -> str:
    """Builds the per-instance gap-to-LB bar chart body."""
    gaps = data["gap_percent"]
    labels = [short for _full, short in GAP_INSTANCES]
    symbolic = ",".join(label.replace("_", r"\_") for label in labels)

    plots = []
    for config in CONFIG_ORDER:
        coords = " ".join(
            f"({short.replace('_', chr(92) + '_')},{gaps[(full, config)]:g})"
            for full, short in GAP_INSTANCES
        )
        color = PGF_COLORS[config]
        plots.append(f"\\addplot[fill={color},draw={color}] coordinates {{{coords}}};")

    legend = ",".join(TEX_LABELS[config] for config in CONFIG_ORDER)
    body = "\n".join(plots)
    return f"""{_generated_banner()}\\begin{{tikzpicture}}
\\begin{{axis}}[
    wykres, height=4.34cm, ybar, bar width=4.2pt, ymin=0,
    ylabel={{luka do LB [\\%]}},
    symbolic x coords={{{symbolic}}},
    xtick=data, x tick label style={{rotate=35, anchor=east, font=\\footnotesize}},
    legend columns=4, legend style={{at={{(0.5,1.04)}}, anchor=south}},
    enlarge x limits=0.08,
]
{body}
\\legend{{{legend}}}
\\end{{axis}}
\\end{{tikzpicture}}
"""


def _pgf_aggregate_gap(data: dict) -> str:
    """Builds the all/dense/sparse mean-gap grouped bar chart body."""
    grouped = _mean_gap_by_group(data)
    titles = {
        "all": "wszystkie (20)",
        "dense": r"gęste ($\ge 0{,}1$)",
        "sparse": r"rzadkie ($< 0{,}1$)",
    }
    symbolic = ",".join(TEX_LABELS[config] for config in CONFIG_ORDER)

    panels = []
    for index, group in enumerate(("all", "dense", "sparse")):
        ylabel = ", ylabel={średnia luka do LB [\\%]}" if index == 0 else ""
        plots = []
        for config in CONFIG_ORDER:
            color = PGF_COLORS[config]
            value = grouped[group][config]
            plots.append(
                f"\\addplot[fill={color},draw={color}] "
                f"coordinates {{({TEX_LABELS[config]},{value:g})}};"
            )
        panels.append(
            f"\\nextgroupplot[title={{{titles[group]}}}{ylabel}]\n" + "\n".join(plots)
        )

    body = "\n".join(panels)
    return f"""{_generated_banner()}\\begin{{tikzpicture}}
\\begin{{groupplot}}[group style={{group size=3 by 1, horizontal sep=1.0cm}},
    width=0.32\\textwidth, height=4.16cm, ybar, ymin=0, ymax=10,
    symbolic x coords={{{symbolic}}}, xtick={{{symbolic}}}, enlarge x limits=0.18,
    x tick label style={{rotate=25, anchor=east, font=\\footnotesize}},
    tick label style={{font=\\footnotesize}},
    title style={{font=\\small}}, label style={{font=\\small}},
    ymajorgrids=true, grid style={{gray!18}},
    nodes near coords, every node near coord/.append style={{font=\\tiny}},
    every axis plot/.append style={{bar shift=0pt, bar width=9pt}}]
{body}
\\end{{groupplot}}
\\end{{tikzpicture}}
"""


def _pgf_boxplots(data: dict) -> str:
    """Builds the 2x2 boxplot panel body."""
    best = data["best_makespans"]
    panels = []
    for instance in BOX_INSTANCES:
        plots = []
        for config in CONFIG_ORDER:
            stats = _box_stats(best[(instance, config)])
            color = PGF_COLORS[config]
            prepared = (
                f"lower whisker={stats['lw']:g},lower quartile={stats['lq']:g},"
                f"median={stats['med']:g},upper quartile={stats['uq']:g},"
                f"upper whisker={stats['uw']:g}"
            )
            plots.append(
                f"\\addplot[draw={color},fill={color}!25,"
                f"boxplot prepared={{{prepared}}}]coordinates{{}};"
            )
        title = instance.replace("_", r"\_")
        panels.append(
            f"\\nextgroupplot[title={{\\small \\texttt{{{title}}}}}]\n"
            + "\n".join(plots)
        )

    body = "\n".join(panels)
    return f"""{_generated_banner()}\\begin{{tikzpicture}}
\\begin{{groupplot}}[
    group style={{group size=2 by 2, vertical sep=1.7cm, horizontal sep=1.5cm}},
    width=0.5\\textwidth, height=4.4cm, boxplot/draw direction=y,
    xtick={{1,2,3,4}}, xticklabels={{ox,pmx,ox\\_cpm,pmx\\_cpm}},
    x tick label style={{font=\\footnotesize, rotate=25, anchor=east}},
    tick label style={{font=\\footnotesize}}, ylabel style={{font=\\small}},
    ylabel={{$C_{{max}}$}},
    ymajorgrids=true, grid style={{gray!18}},
]
{body}
\\end{{groupplot}}
\\end{{tikzpicture}}
"""


def _pgf_convergence(data: dict, data_dir: Path, written: list[Path]) -> str:
    """Writes ``conv_aggregate.dat`` and builds the convergence-curve body."""
    rows = _aggregate_convergence(data)
    dat = data_dir / "conv_aggregate.dat"
    _write_dat(dat, ["gen", *CONFIG_ORDER], rows)
    written.append(dat)

    plots = []
    for config in CONFIG_ORDER:
        plots.append(
            f"\\addplot[{PGF_COLORS[config]}, thick] "
            f"table[x=gen, y={config}] {{{DATA_RELPATH}/conv_aggregate.dat}};"
        )
    legend = ",".join(TEX_LABELS[config] for config in CONFIG_ORDER)
    body = "\n".join(plots)
    return f"""{_generated_banner()}\\begin{{tikzpicture}}
\\begin{{axis}}[wykres, height=5.6cm, xlabel={{pokolenie}},
    ylabel={{średnia luka do LB [\\%]}},
    xmin=0, xmax=100, ymin=0, legend pos=north east,
    legend style={{font=\\footnotesize}}]
{body}
\\legend{{{legend}}}
\\end{{axis}}
\\end{{tikzpicture}}
"""


def _pgf_convergence_evals(data: dict, data_dir: Path, written: list[Path]) -> str:
    """Writes ``conv_evals_aggregate.dat`` and the budget-fair convergence body."""
    rows = _aggregate_convergence_evals(data)
    if not rows:
        raise SystemExit(
            "No convergence_evals.csv found - run an --equal-budget campaign first."
        )
    dat = data_dir / "conv_evals_aggregate.dat"
    _write_dat(dat, ["evals", *CONFIG_ORDER], rows)
    written.append(dat)

    plots = []
    for config in CONFIG_ORDER:
        plots.append(
            f"\\addplot[{PGF_COLORS[config]}, thick] "
            f"table[x=evals, y={config}] {{{DATA_RELPATH}/conv_evals_aggregate.dat}};"
        )
    legend = ",".join(TEX_LABELS[config] for config in CONFIG_ORDER)
    body = "\n".join(plots)
    return f"""{_generated_banner()}\\begin{{tikzpicture}}
\\begin{{axis}}[wykres, height=5.6cm, xlabel={{ewaluacje funkcji celu}},
    ylabel={{średnia luka do LB [\\%]}},
    xmin={rows[0][0]}, xmax={rows[-1][0]}, ymin=0, legend pos=north east,
    legend style={{font=\\footnotesize}}]
{body}
\\legend{{{legend}}}
\\end{{axis}}
\\end{{tikzpicture}}
"""


def _pgf_cpu(  # pylint: disable=too-many-locals
    data: dict, data_dir: Path, written: list[Path]
) -> str:
    """Writes ``cpu_time.dat`` and builds the log-log CPU scatter body."""
    rows = _cpu_rows(data)
    dat = data_dir / "cpu_time.dat"
    _write_dat(dat, ["tasks", *CONFIG_ORDER], rows)
    written.append(dat)

    tasks = [row[0] for row in rows]
    fits = {}
    for index, config in enumerate(CONFIG_ORDER, start=1):
        values = [row[index] for row in rows]
        fits[config] = _power_law_fit(tasks, values)

    marks = {
        "ox": ("*", "1.7pt"),
        "pmx": ("square*", "1.6pt"),
        "ox_cpm": ("triangle*", "2pt"),
        "pmx_cpm": ("diamond*", "2pt"),
    }
    fit_lines = []
    for config in CONFIG_ORDER:
        intercept, slope = fits[config]
        fit_lines.append(
            f"\\addplot[{PGF_COLORS[config]}, thick, domain=15:520, samples=40, "
            f"forget plot] {{{intercept:.6g}*x^{slope:.4f}}};"
        )
    point_lines = []
    for config in CONFIG_ORDER:
        mark, size = marks[config]
        point_lines.append(
            f"\\addplot[only marks, mark={mark}, {PGF_COLORS[config]}, "
            f"mark size={size}] table[x=tasks,y={config}] "
            f"{{{DATA_RELPATH}/cpu_time.dat}};"
        )
    legend = ",".join(TEX_LABELS[config] for config in CONFIG_ORDER)
    fit_body = "\n".join(fit_lines)
    point_body = "\n".join(point_lines)
    return f"""{_generated_banner()}\\begin{{tikzpicture}}
\\begin{{axis}}[
    width=0.92\\textwidth, height=4.32cm, xmode=log, ymode=log,
    xlabel={{liczba zadań $|T|$}}, ylabel={{średni czas CPU / uruchomienie [s]}},
    tick label style={{font=\\small}}, label style={{font=\\small}},
    grid=major, grid style={{gray!18}},
    xmin=14, xmax=560, ymin=0.05, ymax=2000,
    legend pos=south east, legend style={{font=\\footnotesize, fill=white}},
    legend cell align=left,
]
% power-law fits t = a |T|^b (linear regression in log-log space)
{fit_body}
% measured points
{point_body}
\\legend{{{legend}}}
\\end{{axis}}
\\end{{tikzpicture}}
"""


# --------------------------------------------------------------------------- #
# matplotlib output
# --------------------------------------------------------------------------- #


def _write_matplotlib(data: dict, outdir: Path, only: str) -> list[Path]:
    """Renders the requested figures as PNGs under ``<outdir>/png``."""
    # Imported lazily so the pgfplots path stays dependency-free.
    import matplotlib  # pylint: disable=import-outside-toplevel

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    png_dir = outdir / "png"
    png_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    builders = {
        "gap": _mpl_gap,
        "aggregate_gap": _mpl_aggregate_gap,
        "boxplots": _mpl_boxplots,
        "convergence_aggregate": _mpl_convergence,
        "convergence_evals": _mpl_convergence_evals,
        "cpu_time": _mpl_cpu,
    }
    for name in _selected(only):
        path = png_dir / f"{FIGURE_FILES[name]}.png"
        builders[name](plt, data, path)
        written.append(path)
    return written


def _mpl_gap(plt, data: dict, path: Path) -> None:
    """Renders the per-instance gap bar chart."""
    gaps = data["gap_percent"]
    labels = [short for _full, short in GAP_INSTANCES]
    positions = range(len(labels))
    width = 0.2

    figure, axes = plt.subplots(figsize=(8, 4))
    for offset, config in enumerate(CONFIG_ORDER):
        values = [gaps[(full, config)] for full, _short in GAP_INSTANCES]
        axes.bar(
            [p + (offset - 1.5) * width for p in positions],
            values,
            width=width,
            label=config,
            color=MPL_COLORS[config],
        )
    axes.set_xticks(list(positions))
    axes.set_xticklabels(labels, rotation=35, ha="right")
    axes.set_ylabel("luka do LB [%]")
    axes.legend(ncol=4, loc="upper left")
    _save(plt, figure, path)


def _mpl_aggregate_gap(plt, data: dict, path: Path) -> None:
    """Renders the all/dense/sparse mean-gap panel."""
    grouped = _mean_gap_by_group(data)
    groups = ("all", "dense", "sparse")
    titles = {
        "all": "wszystkie (20)",
        "dense": "gęste (≥0,1)",
        "sparse": "rzadkie (<0,1)",
    }

    figure, axes_list = plt.subplots(1, 3, figsize=(9, 3.4), sharey=True)
    for axes, group in zip(axes_list, groups):
        for offset, config in enumerate(CONFIG_ORDER):
            axes.bar(offset, grouped[group][config], color=MPL_COLORS[config])
            axes.text(
                offset,
                grouped[group][config],
                f"{grouped[group][config]:g}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
        axes.set_xticks(range(len(CONFIG_ORDER)))
        axes.set_xticklabels(CONFIG_ORDER, rotation=25, ha="right", fontsize=8)
        axes.set_title(titles[group], fontsize=9)
        axes.set_ylim(0, 10)
    axes_list[0].set_ylabel("średnia luka do LB [%]")
    _save(plt, figure, path)


def _mpl_boxplots(plt, data: dict, path: Path) -> None:
    """Renders the 2x2 boxplot panel (whiskers at min/max)."""
    best = data["best_makespans"]
    figure, axes_list = plt.subplots(2, 2, figsize=(8, 6))
    for axes, instance in zip(axes_list.flat, BOX_INSTANCES):
        series = [best[(instance, config)] for config in CONFIG_ORDER]
        box = axes.boxplot(
            series, whis=(0, 100), patch_artist=True, tick_labels=list(CONFIG_ORDER)
        )
        for patch, config in zip(box["boxes"], CONFIG_ORDER):
            patch.set_facecolor(MPL_COLORS[config])
            patch.set_alpha(0.35)
        axes.set_title(instance, fontsize=9)
        axes.set_ylabel("$C_{max}$")
        axes.tick_params(axis="x", rotation=25, labelsize=8)
    _save(plt, figure, path)


def _mpl_convergence(plt, data: dict, path: Path) -> None:
    """Renders the aggregate convergence curves."""
    rows = _aggregate_convergence(data)
    generations = [row[0] for row in rows]
    figure, axes = plt.subplots(figsize=(8, 4))
    for index, config in enumerate(CONFIG_ORDER, start=1):
        axes.plot(
            generations,
            [row[index] for row in rows],
            label=config,
            color=MPL_COLORS[config],
            linewidth=1.5,
        )
    axes.set_xlim(0, 100)
    axes.set_ylim(bottom=0)
    axes.set_xlabel("pokolenie")
    axes.set_ylabel("średnia luka do LB [%]")
    axes.grid(color="0.85")
    axes.legend(loc="upper right")
    _save(plt, figure, path)


def _mpl_convergence_evals(plt, data: dict, path: Path) -> None:
    """Renders the budget-fair convergence curves (gap vs evaluations)."""
    rows = _aggregate_convergence_evals(data)
    if not rows:
        raise SystemExit(
            "No convergence_evals.csv found - run an --equal-budget campaign first."
        )
    evaluations = [row[0] for row in rows]
    figure, axes = plt.subplots(figsize=(8, 4))
    for index, config in enumerate(CONFIG_ORDER, start=1):
        axes.plot(
            evaluations,
            [row[index] for row in rows],
            label=config,
            color=MPL_COLORS[config],
            linewidth=1.5,
        )
    axes.set_xlim(rows[0][0], rows[-1][0])
    axes.set_ylim(bottom=0)
    axes.set_xlabel("ewaluacje funkcji celu")
    axes.set_ylabel("średnia luka do LB [%]")
    axes.grid(color="0.85")
    axes.legend(loc="upper right")
    _save(plt, figure, path)


def _mpl_cpu(plt, data: dict, path: Path) -> None:
    """Renders the log-log CPU-time scatter with power-law fits."""
    rows = _cpu_rows(data)
    tasks = [row[0] for row in rows]
    figure, axes = plt.subplots(figsize=(8, 5))
    grid = _logspace(15, 520, 60)
    for index, config in enumerate(CONFIG_ORDER, start=1):
        values = [row[index] for row in rows]
        intercept, slope = _power_law_fit(tasks, values)
        axes.plot(grid, [intercept * x**slope for x in grid], color=MPL_COLORS[config])
        axes.scatter(tasks, values, color=MPL_COLORS[config], s=18, label=config)
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xlabel("liczba zadań |T|")
    axes.set_ylabel("średni czas CPU / uruchomienie [s]")
    axes.grid(which="both", color="0.85")
    axes.legend(loc="lower right")
    _save(plt, figure, path)


def _logspace(start: float, stop: float, count: int) -> list[float]:
    """Returns ``count`` log-spaced points from ``start`` to ``stop``."""
    log_start = math.log(start)
    step = (math.log(stop) - log_start) / (count - 1)
    return [math.exp(log_start + step * index) for index in range(count)]


def _save(plt, figure, path: Path) -> None:
    """Saves and closes a matplotlib figure."""
    figure.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(figure)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _selected(only: str) -> tuple[str, ...]:
    """Returns the figure names to build for the ``--figure`` argument."""
    return FIGURE_NAMES if only == "all" else (only,)


def _parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Reproduce the final report's figures from a campaign."
    )
    layout.add_campaign_args(parser, required=True)
    parser.add_argument(
        "--format",
        choices=("pgfplots", "matplotlib"),
        default="pgfplots",
        help="pgfplots: data tables + TikZ plot bodies; matplotlib: PNGs.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=layout.REPOSITORY_ROOT / "docs" / "figures",
        help="Output directory (default: docs/figures).",
    )
    parser.add_argument(
        "--figure",
        choices=("all", *FIGURE_NAMES, *EVAL_FIGURE_NAMES),
        default="all",
        help="Which figure to (re)generate (default: all; the equal-budget "
        f"figures {EVAL_FIGURE_NAMES} are opt-in).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
