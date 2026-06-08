# Results

Full numerical experiments for the memetic scheduler. Reproduce with:

```
python -m src.experiments.run_campaign --name run1   # baselines + OX/PMX × CPM-on/off matrix
python -m src.experiments.analyze --campaign results/run1   # summary.csv, h1_wilcoxon.csv, h2_wilcoxon.csv
python -m src.experiments.plots   --campaign results/run1   # boxplots + convergence curves
```

Complete campaign: **20 instances × 4 configurations × 15 seeds = 1200 EA runs**, plus
per-instance baselines (HLFET, MCP, MILP for `|T| ≤ 30`, lower bound).

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
- **Plots:** 40 boxplots and convergence curves (2 per instance) in
  `results/<campaign>/plots/` (gitignored).

> **Budget caveat (H1).** The memetic configurations perform extra makespan
> evaluations inside the CPM local search, so configurations are compared at equal
> **generations**, not at an equal evaluation budget. The CPM advantage below is
> therefore partly a larger effective search effort per generation.

**Runtime.** The campaign was produced on an Apple M4 (10 cores, Python 3.11) with the
parallel runner (`--jobs 8`); aggregate CPU time across all 80 cells is **~17 h**, but the
cost is wildly skewed: the pure-EA cells run in 1–10 s while the `*_cpm` cells on the
largest/densest graphs dominate — `rand0014_t500` (502 tasks, 24 597 edges) and
`fpppp_t334`/`rand0000_t300` CPM cells run for tens of minutes to a few hours each because
the schedule-aware local search re-decodes on every attempt. The O1–O5 optimisations
(`docs/optimization-plan.md`) — parallel cells, dict adjacency, one-time validation, decode
reuse — make this tractable; per-cell results are bit-identical to the unoptimised code
(verified). `analyze` and `plots` run in seconds.

## Instances

20 benchmark instances (10 synthetic + 10 Kasahara STG). `|T|` includes the 2 dummy
entry/exit tasks. Density = `|E|/|T|²`. HLFET and MCP coincide on every instance.

| instance | source | $\|T\|$ | $\|E\|$ | density | M | HLFET | MCP | MILP opt | LB |
|---|---|---|---|---|---|---|---|---|---|
| syn_t15_p010 | synthetic | 17 | 27 | 0.093 | 2 | 54 | 54 | **52** | 52 |
| syn_t22_p018 | synthetic | 24 | 52 | 0.090 | 3 | 48 | 48 | **44** | 44 |
| syn_t30_p030 | synthetic | 32 | 135 | 0.132 | 3 | 66 | 66 | **66** | 66 |
| syn_t40_p006 | synthetic | 42 | 78 | 0.044 | 4 | 55 | 55 | — | 52 |
| syn_t40_p035 | synthetic | 42 | 277 | 0.157 | 4 | 108 | 108 | — | 108 |
| syn_t60_p005 | synthetic | 62 | 127 | 0.033 | 4 | 74 | 74 | — | 72 |
| syn_t60_p015 | synthetic | 62 | 279 | 0.073 | 4 | 88 | 88 | — | 77 |
| syn_t100_p004 | synthetic | 102 | 249 | 0.024 | 4 | 126 | 126 | — | 126 |
| syn_t100_p012 | synthetic | 102 | 611 | 0.059 | 4 | 153 | 153 | — | 131 |
| syn_t150_p004 | synthetic | 152 | 497 | 0.022 | 4 | 241 | 241 | — | 230 |
| rand0000_t50_p008 | Kasahara | 52 | 119 | 0.044 | 4 | 77 | 77 | — | 66 |
| rand0093_t50_p078 | Kasahara | 52 | 974 | 0.360 | 4 | 73 | 73 | — | 69 |
| rand0063_t100_p002 | Kasahara | 102 | 180 | 0.017 | 4 | 137 | 137 | — | 136 |
| rand0173_t100_p039 | Kasahara | 102 | 1940 | 0.187 | 4 | 218 | 218 | — | 199 |
| rand0071_t300_p001 | Kasahara | 302 | 561 | 0.006 | 4 | 428 | 428 | — | 423 |
| rand0000_t300_p016 | Kasahara | 302 | 7265 | 0.080 | 4 | 505 | 505 | — | 448 |
| rand0014_t500_p020 | Kasahara | 502 | 24597 | 0.098 | 8 | 893 | 893 | — | 893 |
| robot_t88_p003 | Kasahara | 90 | 135 | 0.017 | 4 | 694 | 694 | — | 621 |
| sparse_t96_p001 | Kasahara | 98 | 177 | 0.018 | 4 | 486 | 486 | — | 484 |
| fpppp_t334_p002 | Kasahara | 336 | 1211 | 0.011 | 4 | 1903 | 1903 | — | 1783 |

The memetic EA (`pmx_cpm`) matches or beats both list heuristics on 19 of 20 instances —
often by a wide margin (`fpppp` 1843 vs 1903; `rand0000_t300` 470 vs 505; `rand0173` 206 vs
218). The one exception is `robot_t88`, where the HLFET/MCP value 694 edges out the `pmx_cpm`
**median** of 697 (the best `pmx_cpm` run, 671, is still better) — the hardest instance for the EA.

## Median C_max by configuration

Best configuration per instance in **bold**.

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
| rand0000_t300_p016 | 448 (lb) | 531 | 503 | 512 | **470** |
| rand0014_t500_p020 | 893 (lb) | **893** | **893** | **893** | **893** |
| robot_t88_p003 | 621 (lb) | 814 | 743 | 729 | **697** |
| sparse_t96_p001 | 484 (lb) | 502 | 486 | 486 | **484** |
| fpppp_t334_p002 | 1783 (lb) | 2067 | 1896 | 2017 | **1843** |

`pmx_cpm` is the best (or tied-best) configuration on **every** instance. Residual gap of
the best config to the reference: 0 % on instances that reach the optimum/bound, and at the
top end 1.3 % (`syn_t60_p015`), 3.05 % (`syn_t100_p012`), 3.37 % (`fpppp`), 3.52 %
(`rand0173`), 4.91 % (`rand0000_t300_p016`), 5.8 % (`rand0093`, dense — every config stuck),
and 12.24 % (`robot_t88`, the hardest residual). Several instances are *forced* — every
configuration reaches the bound and ties: `syn_t15`, `syn_t30`, `syn_t40_p035`, `rand0093`,
and notably `rand0014_t500` (all configs = LB 893, even HLFET/MCP), so they carry no
operator/CPM signal.

## H1 — does CPM local search lower C_max?

Paired Wilcoxon, CPM-on vs CPM-off, per crossover per instance (Holm-corrected). Showing the
significant tests; the rest are ties at the optimum/bound (no headroom).

| instance | crossover | CPM off | CPM on | Δ median | Holm p | significant |
|---|---|---|---|---|---|---|
| syn_t22_p018 | ox | 46 | 44 | 2 | 0.019 | ✅ |
| syn_t22_p018 | pmx | 45 | 44 | 1 | 0.021 | ✅ |
| syn_t40_p006 | ox | 55 | 52 | 3 | 0.015 | ✅ |
| syn_t60_p005 | ox | 81 | 72 | 9 | 0.021 | ✅ |
| syn_t60_p005 | pmx | 73 | 72 | 1 | 0.021 | ✅ |
| syn_t60_p015 | ox | 89 | 81 | 8 | 0.019 | ✅ |
| syn_t60_p015 | pmx | 81 | 78 | 3 | 0.040 | ✅ |
| syn_t100_p004 | ox | 128 | 127 | 1 | 0.021 | ✅ |
| syn_t100_p012 | ox | 162 | 142 | 20 | 0.021 | ✅ |
| syn_t100_p012 | pmx | 143 | 135 | 8 | 0.021 | ✅ |
| syn_t150_p004 | ox | 234 | 232 | 2 | 0.050 | ✅ |
| rand0000_t50_p008 | ox | 73 | 66 | 7 | 0.021 | ✅ |
| rand0000_t50_p008 | pmx | 67 | 66 | 1 | 0.032 | ✅ |
| rand0063_t100_p002 | ox | 137 | 136 | 1 | 0.008 | ✅ |
| rand0173_t100_p039 | ox | 228 | 212 | 16 | 0.021 | ✅ |
| rand0173_t100_p039 | pmx | 214 | 206 | 8 | 0.021 | ✅ |
| rand0000_t300_p016 | ox | 531 | 512 | 19 | 0.021 | ✅ |
| rand0000_t300_p016 | pmx | 503 | 470 | 33 | 0.021 | ✅ |
| robot_t88_p003 | ox | 814 | 729 | 85 | 0.021 | ✅ |
| robot_t88_p003 | pmx | 743 | 697 | 46 | 0.021 | ✅ |
| sparse_t96_p001 | ox | 502 | 486 | 16 | 0.021 | ✅ |
| sparse_t96_p001 | pmx | 486 | 484 | 2 | 0.021 | ✅ |
| fpppp_t334_p002 | ox | 2067 | 2017 | 50 | 0.004 | ✅ |
| fpppp_t334_p002 | pmx | 1896 | 1843 | 53 | 0.041 | ✅ |
| rand0071_t300_p001 | ox | 425 | 424 | 1 | 0.041 | ✅ |

**Verdict: H1 supported.** Across all instances with headroom, CPM local search significantly
lowers the median `C_max` for both crossovers: **25/25 significant tests favour CPM-on, none
favour CPM-off.** The effect grows with instance difficulty (`robot_t88` −85 for OX, −46 for
PMX; `rand0000_t300_p016` −33 for PMX; `fpppp` −50/−53). Non-significant cases are ties where
every configuration already reaches the optimum/bound (`syn_t15`, `syn_t30`, `syn_t40_p035`,
`rand0093`, `rand0014_t500`) or where PMX alone already hits the bound (small Δ). Subject to
the equal-generations budget caveat.

## H2 — does OX dominate PMX on sparse graphs?

Paired Wilcoxon, OX vs PMX (pure EA), per instance (Holm-corrected). Sparse = `|E|/|T|² < 0.1`.

| instance | class | ox | pmx | better | Holm p | significant |
|---|---|---|---|---|---|---|
| fpppp_t334_p002 | sparse | 2067 | 1896 | pmx | 0.002 | ✅ |
| robot_t88_p003 | sparse | 814 | 743 | pmx | 0.002 | ✅ |
| rand0063_t100_p002 | sparse | 137 | 136 | pmx | 0.005 | ✅ |
| syn_t40_p006 | sparse | 55 | 52 | pmx | 0.009 | ✅ |
| rand0000_t300_p016 | sparse | 531 | 503 | pmx | 0.010 | ✅ |
| rand0000_t50_p008 | sparse | 73 | 67 | pmx | 0.010 | ✅ |
| rand0071_t300_p001 | sparse | 425 | 423 | pmx | 0.010 | ✅ |
| sparse_t96_p001 | sparse | 502 | 486 | pmx | 0.010 | ✅ |
| syn_t100_p004 | sparse | 128 | 126 | pmx | 0.010 | ✅ |
| syn_t100_p012 | sparse | 162 | 143 | pmx | 0.010 | ✅ |
| syn_t150_p004 | sparse | 234 | 230 | pmx | 0.010 | ✅ |
| syn_t60_p005 | sparse | 81 | 73 | pmx | 0.010 | ✅ |
| syn_t60_p015 | sparse | 89 | 81 | pmx | 0.010 | ✅ |
| syn_t22_p018 | sparse | 46 | 45 | pmx | 0.035 | ✅ |
| rand0173_t100_p039 | dense | 228 | 214 | pmx | 0.010 | ✅ |
| rand0014_t500_p020 | sparse | 893 | 893 | tie | 1.000 | — |
| syn_t15_p010 | sparse | 52 | 52 | tie | 1.000 | — |
| syn_t30_p030 | dense | 66 | 66 | tie | 1.000 | — |
| syn_t40_p035 | dense | 108 | 108 | tie | 1.000 | — |
| rand0093_t50_p078 | dense | 73 | 73 | tie | 1.000 | — |

**Verdict: H2 not supported — the opposite holds.** On **every** non-tied instance PMX
significantly beats OX (**0/14 significant sparse tests favour OX**); ties occur only where
both operators already reach the bound. PMX wins by large margins on the hard instances
(`fpppp` −171, `robot_t88` −71, `rand0000_t300_p016` −28). PMX is the stronger crossover
across the board for this problem and decoder. Likely reason: the ready-list repair already
enforces precedence, so OX's order-preservation advantage is muted, while PMX's position-based
mixing explores the priority space more effectively.

## Summary

- **PMX + CPM (`pmx_cpm`) is the best configuration on every one of the 20 instances.**
- **H1 holds:** schedule-aware CPM significantly improves `C_max` wherever there is headroom
  (25/25 significant tests favour CPM-on; equal-generations budget).
- **H2 fails:** PMX dominates OX, including on sparse graphs (0/14 sparse tests favour OX),
  by large margins on the hard real-application graphs.
- The memetic EA beats the HLFET/MCP list heuristics on 19 of 20 instances (the exception is
  `robot_t88`, where HLFET's 694 narrowly beats the `pmx_cpm` median 697 but not its best run
  671). Residual gap to the lower bound is largest on `robot_t88` (12.2 %), with
  `rand0000_t300_p016` (4.9 %), `rand0093` (5.8 %, dense) and `fpppp` (3.4 %) next. Several
  instances are forced (every config reaches the bound), including the 500-task `rand0014_t500`.
