---
created: 2026-05-11
modified: 2026-05-11
---

# `VoxelReassigner._assign_unique_matches` switches from Python greedy loop to round-based vectorized intersection

`VoxelReassigner._assign_unique_matches` (in
`nellie/tracking/voxel_reassignment.py`) implements 1-to-1 voxel-pair
selection by greedy descent on distance. The pre-rewrite implementation
sorts matches by ascending distance, then iterates `range(len(order))`
in Python, marking `used_prev[p_idx]` / `used_next[n_idx]` and skipping
rows whose prev_id or next_id has already been claimed. PRD #153's perf
test (`tests/test_voxel_reassignment_perf.py:159`) explicitly flags the
Python loop as the strongest vectorization suspect in the entire
VoxelReassigner stage.

PRD #222 replaces the loop with a **round-based per-prev/per-next argmin
intersection**. Per round: find the per-prev-id argmin-distance row and
the per-next-id argmin-distance row (over still-active rows); a row is
kept iff it is the argmin for BOTH its prev_id AND its next_id;
deactivate kept rows + all rows whose prev or next id was claimed;
repeat until no rows are kept. Two coupled decisions need to be pinned
ahead of the rewrite in Slice 2 (#224): (1) the round-based intersection
is the chosen vectorization (rather than a single first-occurrence-of-both
heuristic, which is faster but **not** equivalent to sequential greedy),
and (2) the test bar is **set-equality** on kept rows (rather than
ordered-equality), because the round-based version returns kept indices
in input-row order while the greedy returns them in distance-ascending
order — both deterministic, both correct, but distinct orderings.

A third decision worth pinning ahead of merge: this PR ships even though
`_assign_unique_matches` is **not called by `_run_reassignment`** — the
production driver uses `_select_best_pairs` (1-best per target via a
single lexsort, no 1-to-1 uniqueness). The function is exposed for
external callers and tests; PRD #153 pre-instrumented it as a hot-path
suspect, so the rewrite pins the perf-test target and is correctness-
preserving for any future caller that wants 1-to-1 unique matches with
global greedy semantics. Status: **Accepted (preventive)** — documents
non-obvious decisions ahead of the rewrite in Slice 2 (#224).

## Equivalence sketch

Sequential greedy (current): process rows in distance-ascending order,
keep iff neither prev_id nor next_id has been claimed by an earlier row.

Round-based (new): per round, keep all rows that are simultaneously the
per-prev argmin and the per-next argmin over the active set; remove
kept rows + rows touching their ids; repeat.

Both produce the same kept SET because:

- A row that is the smallest-distance row for both its prev_id and its
  next_id (over all rows) is kept first by sequential greedy (no earlier
  row touches either id) → kept in round 1 of round-based.
- A row that has contention on either side waits in greedy until the
  contender is resolved (kept or evicted) → kept in a later round of
  round-based, after the contender has been processed.
- Walked example: P1→N1 (d=1), P1→N2 (d=2), P2→N2 (d=3), P2→N3 (d=4),
  P3→N3 (d=5).
  - Greedy: keeps row 0 (P1,N1), skips row 1 (P1 used), keeps row 2
    (P2,N2), skips row 3 (P2 used), keeps row 4 (P3,N3). Result:
    {row 0, row 2, row 4}.
  - Round-based:
    - Round 1: per-prev argmin = {0, 2, 4}; per-next argmin = {0, 1, 3}.
      Intersection = {0}. Kept = {row 0}; deactivate rows touching P1
      or N1 → rows 0, 1.
    - Round 2: active = {2, 3, 4}. Per-prev argmin (over active) =
      {2, 4}; per-next argmin = {2, 3}. Intersection = {2}. Kept =
      {row 2}; deactivate rows touching P2 or N2 → rows 2, 3.
    - Round 3: active = {4}. Per-prev argmin = {4}; per-next argmin =
      {4}. Intersection = {4}. Kept = {row 4}.
  - Result: {row 0, row 2, row 4}. ✓ Matches greedy.

## Considered Options

- **Round-based per-prev/per-next argmin intersection (chosen).** Per
  round, find the per-id argmin via lexsort + first-of-group on `(id,
  distance)`; a row is kept iff it is the argmin for both its prev_id
  and its next_id. Deactivate kept rows + all rows whose prev_id or
  next_id was claimed. Repeat. Provably equivalent to sequential greedy
  (sketch above). Pure NumPy; no new dependencies.
- **First-occurrence-of-both heuristic (rejected).** Sort by distance
  ascending; precompute `prev_first_idx[p]` = index of first occurrence
  of prev_id `p`, same for next; keep row `i` iff `i ==
  prev_first_idx[prev[i]] and i == next_first_idx[next[i]]`. **Not
  equivalent to sequential greedy.** Counter-example: rows
  (P1,N1,d=1), (P1,N2,d=2), (P2,N1,d=3), (P2,N2,d=4). Greedy keeps
  (P1,N1) and (P2,N2). First-of-both keeps only (P1,N1) — for (P2,N2)
  the prev-first-idx for P2 is 2 (row P2,N1), but row 2 was rejected
  (N1 used), and the heuristic doesn't track rejections. Cannot be
  used; ruled out by walked counter-example.
- **Cython / C extension (rejected).** Adds build-time complexity for
  a non-production hot path. The Python loop today takes ~3 ms at
  N=10k per the perf test — fast enough that the round-based vectorized
  version (also a few ms with `lexsort` overhead) is the appropriate
  scope. Not worth a build-system dependency.
- **`numba.njit` JIT compile of the loop (rejected).** Same scope
  argument: heavy dependency for a non-production code path. Numba
  isn't a current Nellie dependency and adding it for one function
  is over-engineering.
- **Defer entirely (rejected).** PRD #153's perf test pre-instruments
  this function as the lead suspect. The rewrite is small (~30 lines
  net) and well-defined (round-based intersection is a textbook trick
  for parallelizing greedy bipartite matching). The cost of doing it
  now is small and the benefit is real — pins the perf-test target so
  future profile-driven work doesn't trip over a well-known suspect.

## Consequences

- **Test bar: set-equality on kept rows, not ordered-equality.** The
  round-based version returns `keep_indices = np.flatnonzero(keep)`
  (input-row order); the pre-rewrite returns indices in
  `keep_indices.append` order (= ascending-distance order over the
  kept subset). Both are deterministic. The Slice 2 (#224) equivalence
  test compares as sets via `set(map(tuple, vox))`. If a future caller
  needs ordered output, sort the result by distance externally — not a
  contract this function has provided.
- **Empty-input contract preserved.** The pre-rewrite returns a 2-tuple
  of `(0, D)` int64 arrays for empty input; the post-rewrite preserves
  this. The `_select_best_pairs` empty-return arity bug fixed in PRD
  #217 was a separate latent issue and does not apply here.
- **`spatial_shape is None` precondition preserved.** The function
  raises `RuntimeError("spatial_shape is not set; ...")` before the
  ravel calls — same behavior pre and post.
- **`_assign_unique_matches` remains not-called by `_run_reassignment`.**
  The PR does not change which callers exist; it only optimizes the
  function for the test/external caller path. The
  `wiki/tracking/voxel-reassignment.md` gotcha at line 46
  (`_allocate_memory()` must run first or these helpers raise) remains
  applicable.
- Future maintainers should not switch to the first-occurrence-of-both
  heuristic without re-verifying equivalence — see the rejection
  rationale and walked counter-example above.
- Future maintainers should not unwind the round-based vectorization
  to "simplify" without measuring the perf test — the pre-rewrite
  Python loop was the lead VoxelReassigner suspect for a reason.

## References

- PRD #222 — vectorize `_assign_unique_matches` Python greedy loop
- Slice 1 #223 — synthetic test suite + this ADR (no production code
  changes)
- Slice 2 #224 — round-based vectorized rewrite + equivalence test +
  perf comparison
- PRD #153 — opt-in perf coverage rollout that pre-instrumented this
  hot spot
- ADR 0010 — preceding tracking-stage rewrite (precedent for
  test-bar pattern + ADR format)
- ADRs 0005 / 0006 / 0007 / 0008 / 0009 — preceding optimization
  ADRs adopted similar slice-plan + considered-options + consequences
  patterns
- `wiki/tracking/voxel-reassignment.md` — documents
  `_assign_unique_matches` as a kept API alongside `_select_best_pairs`
