# Results

Full numerical experiments for the memetic scheduler. Reproduce with:

```
python -m src.experiments.run_campaign --name run1   # baselines + OX/PMX × CPM-on/off matrix
python -m src.experiments.analyze --campaign results/run1   # summary.csv, h1_wilcoxon.csv, h2_wilcoxon.csv
python -m src.experiments.plots   --campaign results/run1   # boxplots + convergence curves
```

> ⚠️ **Partial campaign.** This report is generated from campaign `20260607_134535`, which was
> **stopped early at 62/80 cells** (the largest-instance CPM cells were each taking 30–100+ min
> single-threaded). **16 of 20 instances** have EA results; **4 large instances are not yet run**
> (`rand0014_t500_p020`, `robot_t88_p003`, `sparse_t96_p001`, `fpppp_t334_p002`), and
> `rand0000_t300_p016` has only its pure-EA (`ox`/`pmx`) cells, not its CPM cells. Baselines
> (HLFET/MCP/lower bound) were computed for **all 20** instances. The campaign can be finished by
> resuming with `--skip-existing-cells`. Both hypotheses are nonetheless cleanly decided on the
> 16 analysed instances, which already span the small→large (15–302 task) and sparse→dense range.

## Setup

- **Configurations (4):** `ox`, `pmx` (pure EA) and `ox_cpm`, `pmx_cpm` (memetic =
  EA + schedule-aware CPM local search). The CPM analysis uses the critical path of
  the *decoded schedule* (precedence **and** processor bottlenecks).
- **Seeds:** 15 independent runs per configuration (seeds 0–14).
- **EA budget:** population 50, 100 generations, p_c = 0.8, p_m = 0.3 (global defaults).
- **Quality measure:** best `C_max` per run. Tables report the **median** over seeds.
- **Reference:** exact MILP optimum for the small instances (`|T| ≤ 30`, proven optimal
  via CBC: `syn_t15_p010`, `syn_t22_p018`, `syn_t30_p030`), otherwise the lower bound
  `LB = max(⌈total work / M⌉, critical path)`.
- **Significance:** paired Wilcoxon signed-rank test (same seeds), Holm-corrected
  within each hypothesis family, α = 0.05.
- **Plots:** boxplots and convergence curves per instance in
  `results/<campaign>/plots/` (gitignored).

> **Budget caveat (H1).** The memetic configurations perform extra makespan
> evaluations inside the CPM local search, so configurations are compared at equal
> **generations**, not at an equal evaluation budget. The CPM advantage below is
> therefore partly a larger effective search effort per generation.

**Runtime.** This partial campaign ran **~7 h** wall-clock (≈5.5 h CPU summed over the 62
completed cells), single-threaded, on an Apple M4 (10 cores, Python 3.11). The cost is almost
entirely the `*_cpm` cells on larger/denser graphs — the schedule-aware local search re-decodes
the schedule on every attempt, and the runner does not parallelise across cores or seeds. The
pure-EA cells are 1–10 s each; `analyze` and `plots` run in seconds. A parallel runner plus
decoder optimisations (see `docs/optimization-plan.md` on the `optimize` branch) would cut this
to tens of minutes.

## Instances

Baselines for all 20 registry instances; the rightmost column flags whether EA results exist
in this partial campaign. `n` includes the 2 dummy entry/exit tasks. Density = `|E|/|T|²`.

| instance | source | n | edges | density | M | HLFET | MCP | MILP opt | LB | EA run? |
|---|---|---|---|---|---|---|---|---|---|---|
| syn_t15_p010 | synthetic | 17 | 27 | 0.093 | 2 | 54 | 54 | **52** | 52 | ✅ |
| syn_t22_p018 | synthetic | 24 | 52 | 0.090 | 3 | 48 | 48 | **44** | 44 | ✅ |
| syn_t30_p030 | synthetic | 32 | 135 | 0.132 | 3 | 66 | 66 | **66** | 66 | ✅ |
| syn_t40_p006 | synthetic | 42 | 78 | 0.044 | 4 | 55 | 55 | — | 52 | ✅ |
| syn_t40_p035 | synthetic | 42 | 277 | 0.157 | 4 | 108 | 108 | — | 108 | ✅ |
| syn_t60_p005 | synthetic | 62 | 127 | 0.033 | 4 | 74 | 74 | — | 72 | ✅ |
| syn_t60_p015 | synthetic | 62 | 279 | 0.073 | 4 | 88 | 88 | — | 77 | ✅ |
| syn_t100_p004 | synthetic | 102 | 249 | 0.024 | 4 | 126 | 126 | — | 126 | ✅ |
| syn_t100_p012 | synthetic | 102 | 611 | 0.059 | 4 | 153 | 153 | — | 131 | ✅ |
| syn_t150_p004 | synthetic | 152 | 497 | 0.022 | 4 | 241 | 241 | — | 230 | ✅ |
| rand0000_t50_p008 | Kasahara | 52 | 119 | 0.044 | 4 | 77 | 77 | — | 66 | ✅ |
| rand0093_t50_p078 | Kasahara | 52 | 974 | 0.360 | 4 | 73 | 73 | — | 69 | ✅ |
| rand0063_t100_p002 | Kasahara | 102 | 180 | 0.017 | 4 | 137 | 137 | — | 136 | ✅ |
| rand0173_t100_p039 | Kasahara | 102 | 1940 | 0.187 | 4 | 218 | 218 | — | 199 | ✅ |
| rand0071_t300_p001 | Kasahara | 302 | 561 | 0.006 | 4 | 428 | 428 | — | 423 | ✅ |
| rand0000_t300_p016 | Kasahara | 302 | 7265 | 0.080 | 4 | 505 | 505 | — | 448 | ◐ EA only |
| rand0014_t500_p020 | Kasahara | 502 | 24597 | 0.098 | 8 | 893 | 893 | — | 893 | ❌ not run |
| robot_t88_p003 | Kasahara | 90 | 135 | 0.017 | 4 | 694 | 694 | — | 621 | ❌ not run |
| sparse_t96_p001 | Kasahara | 98 | 177 | 0.018 | 4 | 486 | 486 | — | 484 | ❌ not run |
| fpppp_t334_p002 | Kasahara | 336 | 1211 | 0.011 | 4 | 1903 | 1903 | — | 1783 | ❌ not run |

HLFET and MCP coincide on every instance. The memetic EA (`pmx_cpm`) matches or beats both list
heuristics everywhere it ran (e.g. `rand0000_t50` 66 vs 77, `syn_t60_p015` 78 vs 88).

## Median C_max by configuration

Best configuration per instance in **bold** (16 analysed instances; `rand0000_t300_p016` has no
CPM cells yet).

| instance | ref (kind) | ox | pmx | ox_cpm | pmx_cpm |
|---|---|---|---|---|---|
| syn_t15_p010 | 52 (milp) | **52** | **52** | **52** | **52** |
| syn_t22_p018 | 44 (milp) | 46 | 45 | **44** | **44** |
| syn_t30_p030 | 66 (milp) | **66** | **66** | **66** | **66** |
| syn_t40_p006 | 52 (lb) | 55 | **52** | **52** | **52** |
| syn_t40_p035 | 108 (lb) | **108** | **108** | **108** | **108** |
| syn_t60_p005 | 72 (lb) | 81 | 73 | **72** | **72** |
| syn_t60_p015 | 77 (lb) | 89 | 81 | 81 | **78** |
| syn_t100_p004 | 126 (lb) | 128 | **126** | 127 | **126** |
| syn_t100_p012 | 131 (lb) | 162 | 143 | 142 | **135** |
| syn_t150_p004 | 230 (lb) | 234 | **230** | 232 | **230** |
| rand0000_t50_p008 | 66 (lb) | 73 | 67 | **66** | **66** |
| rand0093_t50_p078 | 69 (lb) | **73** | **73** | **73** | **73** |
| rand0063_t100_p002 | 136 (lb) | 137 | **136** | **136** | **136** |
| rand0173_t100_p039 | 199 (lb) | 228 | 214 | 212 | **206** |
| rand0071_t300_p001 | 423 (lb) | 425 | **423** | 424 | **423** |
| rand0000_t300_p016 | 448 (lb) | 531 | **503** | — | — |

`pmx_cpm` is the best (or tied-best) configuration on **every** instance where it ran. Residual
gap of the best config to the reference: 0 % on the instances that reach the optimum/bound, up
to 3.5 % (`rand0173_t100_p039`) and 3.1 % (`syn_t100_p012`). Dense instances (`syn_t30_p030`,
`syn_t40_p035`, `rand0093_t50_p078`) are effectively forced — every config ties — so they carry
no operator/CPM signal. `rand0093` stays 5.8 % above its bound for all configs (a hard dense case).

## H1 — does CPM local search lower C_max?

Paired Wilcoxon, CPM-on vs CPM-off, per crossover per instance (Holm-corrected). Showing the
significant tests; the remainder are ties at the optimum/bound (no headroom for CPM to act).

| instance | crossover | CPM off | CPM on | Δ median | Holm p | significant |
|---|---|---|---|---|---|---|
| syn_t22_p018 | ox | 46 | 44 | 2 | 0.014 | ✅ |
| syn_t22_p018 | pmx | 45 | 44 | 1 | 0.018 | ✅ |
| syn_t40_p006 | ox | 55 | 52 | 3 | 0.012 | ✅ |
| syn_t60_p005 | ox | 81 | 72 | 9 | 0.016 | ✅ |
| syn_t60_p005 | pmx | 73 | 72 | 1 | 0.016 | ✅ |
| syn_t60_p015 | ox | 89 | 81 | 8 | 0.014 | ✅ |
| syn_t60_p015 | pmx | 81 | 78 | 3 | 0.034 | ✅ |
| syn_t100_p004 | ox | 128 | 127 | 1 | 0.016 | ✅ |
| syn_t100_p012 | ox | 162 | 142 | 20 | 0.016 | ✅ |
| syn_t100_p012 | pmx | 143 | 135 | 8 | 0.016 | ✅ |
| syn_t150_p004 | ox | 234 | 232 | 2 | 0.043 | ✅ |
| rand0000_t50_p008 | ox | 73 | 66 | 7 | 0.016 | ✅ |
| rand0000_t50_p008 | pmx | 67 | 66 | 1 | 0.027 | ✅ |
| rand0063_t100_p002 | ox | 137 | 136 | 1 | 0.006 | ✅ |
| rand0173_t100_p039 | ox | 228 | 212 | 16 | 0.016 | ✅ |
| rand0173_t100_p039 | pmx | 214 | 206 | 8 | 0.016 | ✅ |
| rand0071_t300_p001 | ox | 425 | 424 | 1 | 0.036 | ✅ |

**Verdict: H1 supported.** Across all instances with headroom, CPM local search significantly
lowers (or never raises) the median `C_max`: **17/17 significant tests favour CPM-on, none favour
CPM-off.** The effect is largest for OX and on the higher-density mid-size graphs (e.g.
`syn_t100_p012` −20 for OX, `rand0173_t100_p039` −16 for OX). Non-significant cases are ties where
every configuration already reaches the optimum/bound (`syn_t15`, `syn_t30`, `syn_t40_p035`,
`rand0093`) or where PMX alone is already at the bound (small or zero Δ). Subject to the
equal-generations budget caveat above.

## H2 — does OX dominate PMX on sparse graphs?

Paired Wilcoxon, OX vs PMX (pure EA), per instance (Holm-corrected). Sparse = `|E|/|T|² < 0.1`.

| instance | class | ox | pmx | better | Holm p | significant |
|---|---|---|---|---|---|---|
| rand0000_t300_p016 | sparse | 531 | 503 | pmx | 0.008 | ✅ |
| rand0000_t50_p008 | sparse | 73 | 67 | pmx | 0.008 | ✅ |
| rand0063_t100_p002 | sparse | 137 | 136 | pmx | 0.005 | ✅ |
| rand0071_t300_p001 | sparse | 425 | 423 | pmx | 0.008 | ✅ |
| rand0173_t100_p039 | dense | 228 | 214 | pmx | 0.008 | ✅ |
| syn_t100_p004 | sparse | 128 | 126 | pmx | 0.008 | ✅ |
| syn_t100_p012 | sparse | 162 | 143 | pmx | 0.008 | ✅ |
| syn_t150_p004 | sparse | 234 | 230 | pmx | 0.008 | ✅ |
| syn_t22_p018 | sparse | 46 | 45 | pmx | 0.030 | ✅ |
| syn_t40_p006 | sparse | 55 | 52 | pmx | 0.008 | ✅ |
| syn_t60_p005 | sparse | 81 | 73 | pmx | 0.008 | ✅ |
| syn_t60_p015 | sparse | 89 | 81 | pmx | 0.008 | ✅ |
| syn_t15_p010 | sparse | 52 | 52 | tie | 1.000 | — |
| syn_t30_p030 | dense | 66 | 66 | tie | 1.000 | — |
| syn_t40_p035 | dense | 108 | 108 | tie | 1.000 | — |
| rand0093_t50_p078 | dense | 73 | 73 | tie | 1.000 | — |

**Verdict: H2 not supported — the opposite holds.** On **every** non-tied instance PMX
significantly beats OX (**0/11 significant sparse tests favour OX**); the only ties are instances
where both operators already reach the bound. PMX is the stronger crossover across the board for
this problem and decoder. Likely reason: the ready-list repair already enforces precedence, so
OX's order-preservation advantage is muted, while PMX's position-based mixing explores the
priority space more effectively.

## Summary

- **PMX + CPM (`pmx_cpm`) is the best configuration on every instance where it ran.**
- **H1 holds:** schedule-aware CPM significantly improves `C_max` wherever there is headroom
  (17/17 significant tests favour CPM-on; equal-generations budget).
- **H2 fails:** PMX dominates OX, including on sparse graphs (0/11 sparse tests favour OX).
- The memetic EA beats the HLFET/MCP list heuristics on every instance.
- **Incomplete:** 4 large instances (`rand0014_t500`, `robot_t88`, `sparse_t96`, `fpppp_t334`)
  and the CPM cells of `rand0000_t300_p016` are not yet run; finishing them would extend the
  size range but is unlikely to change either verdict.
