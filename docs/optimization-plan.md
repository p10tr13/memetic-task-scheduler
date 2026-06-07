# Optimization Plan — memetic-task-scheduler

**Branch:** `optimize` (in the local copy `/Users/woj/memetic-task-scheduler-main`, off `experiments@4d70c2b`)
**Goal:** cut the full-campaign wall-clock from **hours** down to **tens of minutes**, with **no change to results** (same makespans, same H1/H2 verdicts).

All findings below come from profiling the live campaign (a statistical stack sample of a
running `run_single` cell on `rand0173_t100_p039 pmx --cpm`, plus `top`/`ps`).

---

## Context — why it's slow

The campaign is a pure-CPython evolutionary + CPM local-search workload (DEAP + networkx,
integer/dict/heap operations on an irregular task graph). Measured facts:

- **Single-core bound.** The cell process runs at **99% of one core** while `top` shows
  **~55% idle** on a **10-core** M4 (≈8–9 cores idle). `run_campaign` runs the 80 cells
  **strictly sequentially** (`src/experiments/run_campaign.py:48`, a blocking
  `subprocess.run` loop) and `run_single` runs its 15 seeds sequentially
  (`src/experiments/run_single.py:138`). **Zero parallelism anywhere.**
- **Not IO/memory bound.** Process RSS ≈ 25 MB; no IO-wait. Disk is touched only at cell
  boundaries (4 tiny files). Google Drive is not in the hot path.
- **No GPU involvement and none applicable** — branchy graph traversal, not dense array
  math. GPU is not a lever for this algorithm.
- **The CPU hotspot is the list-scheduling decoder, called millions of times per cell.**
  Per CPM cell: 15 seeds × 100 generations × 50 individuals ≈ 75 000 evaluations; each CPM
  individual runs `cpm_local_search` with up to 20 attempts, each attempt doing ~2 full
  `compute_schedule` decodes ⇒ **~3 million full decodes per cell**, each O(tasks). The
  stack sample is dominated by the decoder's `max()` calls (including `max(..., default=)`
  keyword-arg parsing and integer `long_richcompare`) — almost pure interpreter overhead in
  a tight loop.

The non-CPM cells (`ox`, `pmx`) are 1–10 s each; the cost is entirely the `*_cpm` cells on
larger/denser instances (the 100-task dense cell took ~22 min).

---

## Optimizations (ranked by payoff)

### O1 — Parallelize across cores  ⭐ biggest, lowest-risk win
**Where:** `src/experiments/run_campaign.py` (cell loop) and/or
`src/experiments/run_single.py` (seed loop).
**Idea:** cells are fully independent and seeds are independent and individually seeded, so
they can run concurrently with **no effect on results**.
- **Preferred:** add a `--jobs N` flag to `run_campaign` that keeps up to `N` `run_single`
  subprocesses in flight (replace the blocking `for` loop with a bounded pool, e.g.
  `concurrent.futures.ProcessPoolExecutor` over cells, or a simple `subprocess.Popen` window
  of size N). Each cell is already an isolated subprocess, so this is trivially correct and
  reproducible.
- **Also:** parallelize the 15 seeds inside `run_single` via `ProcessPoolExecutor`
  (`run_single.py:138`), since each seed is an independent `run_evolution(random_seed=seed)`.
  Per-seed timings stay valid; only wall-clock changes.
- **Caution:** don't oversubscribe — cap concurrency at ~`cores − 2` (8 on this M4). Running
  cells in parallel is preferable to seeds-in-parallel because it load-balances long CPM
  cells against short non-CPM cells.
- **Expected:** ≈ **8×** wall-clock reduction. **Risk to results: none.**

### O2 — Hoist the per-call DAG check out of the hot path  ⭐ trivial
**Where:** `src/core/cpm_utils.py:34` —
`if not nx.is_directed_acyclic_graph(problem_instance.priority_relationship):`
**Problem:** this O(V+E) acyclicity check runs on **every** `identify_critical_tasks` call
(millions of times); the graph never changes during a run.
**Fix:** validate acyclicity **once** per instance (e.g. at instance load /
`ProblemInstance` construction, or guarded by a module-level "already-checked" set keyed by
graph id) and drop it from the per-call path.
**Expected:** removes a whole graph scan per local-search step. **Risk: none.**

### O3 — Skip per-decode validation in the EA inner loop  ⭐ low effort
**Where:** `src/core/decoder.py:69–70` — `compute_schedule` runs
`_validate_task_permutation` and `_validate_task_time_indexes` on **every** decode, each
building `set(task_permutation)`, comparing to `set(graph.nodes)`, and scanning min/max over
task numbers.
**Problem:** during the EA every permutation is already valid (init + repair guarantee it);
this is pure repeated overhead.
**Fix:** add a fast path — e.g. a `validate: bool = True` parameter (or a thin
`compute_schedule_unchecked`) and have the EA fitness/local-search callers pass
`validate=False`. Public/`main.py`/tests keep validation on.
**Expected:** removes 2 set-builds + comparisons + 2 scans per decode (×3M/cell).
**Risk: none** if scoped to internal EA callers; keep the validating path as the default.

### O4 — Replace networkx predecessor lookups with a prebuilt dict adjacency  ⭐ large
**Where:** `src/core/decoder.py:126` and `src/core/cpm_utils.py:80` — both call
`problem_instance.priority_relationship.predecessors(task)` inside the hot loop.
**Problem:** networkx `.predecessors()` returns an iterator over the adjacency view — far
slower than a plain `dict[int, list[int]]` lookup, and it's hit once per task per decode.
**Fix:** precompute `predecessors_map: dict[int, list[int]]` **once per instance** (cache on
the `ProblemInstance` or pass it into the decoder) and index it directly in the loop. Same
data, same order-independent results.
**Expected:** large constant-factor win on the dominant inner loop. **Risk: none** (identical
predecessor sets).

### O5 — Reuse the decoded schedule inside `cpm_local_search`  ⭐ ~1.5–2×
**Where:** `src/solvers/evolutionary/local_search.py:84–136` / `_movable_critical_tasks`.
**Problem:** each attempt decodes the current permutation inside `identify_critical_tasks`
(via `_movable_critical_tasks`) and **separately** recomputes makespans; the schedule object
is thrown away and re-derived. `current_cmax` is also computed by a full extra decode.
**Fix:** have `identify_critical_tasks`/`_movable_critical_tasks` return (or accept) the
already-computed `Schedule` so the current makespan and critical set come from **one** decode
per attempt instead of two; only the mutant needs a fresh decode.
**Expected:** roughly halves decodes per attempt ⇒ ~1.5–2× on CPM cells. **Risk: none**
(same schedule, same acceptance logic — verify makespans match).

### O6 — Micro-opts in the decoder (optional, after O1–O5)
- `decoder.py:102` — `max(finish_times.values(), default=0)`: track a running `makespan`
  variable during the loop instead of a final O(tasks) `max` with kwarg parsing.
- `decoder.py:87,134` — keep the `max(a, b)` two-arg forms (cheap); the costly ones are the
  `default=`/iterable forms flagged by the profile.
- Consider `__slots__`/plain tuples for hot data if a follow-up profile still shows alloc
  churn (`_PyObject_Malloc` appeared in the sample).
**Expected:** modest. **Risk: none.**

---

## Suggested sequencing

1. **O1 first** (parallelism) — biggest wall-clock win, isolated to the runner, no core-code
   risk. Ship it, re-time a `--quick` run.
2. **O2 + O3** (delete redundant per-call work) — trivial, high confidence.
3. **O4** (dict adjacency) — the main algorithmic constant-factor win.
4. **O5** (reuse schedule) — halves CPM decodes.
5. **O6** if a fresh profile still warrants it.

After O1–O4 the per-cell CPU cost should drop several-fold; combined with O1 the full
campaign should land in **tens of minutes**.

---

## Correctness / verification (must hold for every change)

These optimizations are **behavior-preserving**, so prove it:

1. **Unit tests** — `pytest` (the suite has 80 tests). All must stay green; black/pylint clean.
2. **Decoder equivalence** — add/keep a test asserting `compute_schedule(..., validate=False)`
   returns an identical `Schedule` (start/finish/machine_predecessors/makespan) to the
   validating path on the registry instances.
3. **Determinism check** — run `run_single` on a couple of instances (e.g. `syn_t40_p035`,
   `rand0173_t100_p039`) on the **optimized** code and on the **original** code with the same
   seeds; assert **identical** `runs.csv` (`best_makespan` per seed) and `convergence.csv`.
   This is the key gate: same numbers, faster.
4. **Spot-check the campaign** — a `--quick` campaign on both branches should yield identical
   per-cell makespans; only timings differ.
5. **Re-profile** — sample a CPM cell after O1–O5 to confirm the hotspot moved and to decide
   whether O6 is worth it.

Only after (1)–(4) pass should the optimized code be used to regenerate any results.

---

## Code-change size estimate

Rough, grounded in the current files. "LOC" = net lines added/changed (not counting tests).

| # | Optimization | Files touched | Est. LOC | Complexity |
|---|---|---|---|---|
| O1 | Parallelize cells (`--jobs`) | `run_campaign.py` (+opt. `run_single.py`) | 40–70 | Medium — bounded pool, preserve failure/`--continue-on-error` semantics |
| O2 | Hoist DAG check | `cpm_utils.py` (+ `models.py`/`instances.py` for once-per-instance check) | 5–15 | Trivial |
| O3 | Skip per-decode validation | `decoder.py` + 2–3 EA call sites (`algorithm.py`/`deap_setup.py`, `local_search.py`, `cpm_utils.py`) | 10–25 | Low — add `validate=False` fast path + thread through callers |
| O4 | Dict predecessor adjacency | `decoder.py`, `cpm_utils.py` (+ caching site) | 30–60 | **Highest** — plumb a precomputed map through the call chain (frozen `ProblemInstance` needs a cache, e.g. module-level/`WeakKeyDictionary`, not a new field) |
| O5 | Reuse decoded schedule | `local_search.py`, `cpm_utils.py` | 20–40 | Medium — change `identify_critical_tasks` to accept/return a `Schedule` |
| O6 | Decoder micro-opts | `decoder.py` | 5–10 | Trivial |
| — | Tests (equivalence + determinism gate) | `tests/` | 30–60 | Low |

**Totals:** ≈ **120–220 net LOC of production code** across **~5–6 files**, plus ~30–60 LOC
of tests. This is a **moderate, well-contained change — not a rewrite.** No public API or EA
semantics change; signatures gain optional params (`validate=`, an adjacency/`Schedule`
argument) with backward-compatible defaults.

**Effort split:** O1 is self-contained in the runner and delivers the ~8× alone. O2/O3/O6 are
quick, low-risk edits. O4 carries most of the plumbing risk (passing the adjacency map without
mutating the frozen dataclass). O5 is a focused refactor of the local-search/CPM boundary.
Landing **O1–O4** (~90–170 LOC) gets the bulk of the speedup; O5/O6 are incremental.

## Notes / non-goals

- **Do not touch the EA semantics** — population update, selection, crossover, mutation,
  acceptance rule, hyperparameters, and the `random` seeding all stay exactly as-is. This is
  purely making the same computation cheaper and using idle cores.
- **No GPU work** — not applicable to this algorithm.
- These changes are independent of the currently-running campaign (different repo copy); they
  target the **next** run. The in-flight campaign should finish on the original, unmodified
  code for dataset consistency.
</content>
