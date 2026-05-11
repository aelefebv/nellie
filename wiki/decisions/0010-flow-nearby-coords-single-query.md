---
created: 2026-05-12
modified: 2026-05-12
---

# `FlowInterpolator._get_nearby_coords` switches from double KDTree query to single `query_ball_point` + vectorized distance compute, fixes NaN-alignment bug

`FlowInterpolator._get_nearby_coords` (in
`nellie/tracking/flow_interpolation.py`) is the per-frame, per-query-coord
neighbor lookup that drives every interpolation. The current implementation
queries the same `cKDTree` twice — once via `query_ball_point` (just to count
neighbors per coord), once via `query(k=max_k)` (the expensive worker-parallel
traversal that returns distances) — then truncates `distances[pos][:k_all[pos]]`
to keep only the within-radius slice. The second query is pure overhead;
`query_ball_point` already has the in-radius indices, and per-coord distances
are one `linalg.norm` away.

The same loop also has a real correctness bug: the per-coord results are
scattered back to the original-index slots via `for i in range(len(distances)):
if i not in good_coords: continue`. That mapping is correct only when
`good_coords == [0, ..., N-1]` (no NaN coords). With NaN coords present
(routine after the first frame in `interpolate_all_forward`/`_backward`'s
terminal NaN propagation), the `i`/`pos` indexing diverges and per-coord
results land in the wrong slots.

PRD #204 fixes both in one rewrite: single `query_ball_point` + per-coord
`linalg.norm` for distances, and `for pos, i in enumerate(good_coords):` for
correct alignment. Three coupled decisions in the design need to be pinned
ahead of the rewrite in Slice 2 (#207): (1) the rewrite uses a single
`query_ball_point` with manual distance compute over the existing two-query
pattern, (2) the NaN-alignment fix is bundled into the same PR (cannot be
cleanly separated from the same loop body), and (3) the test bar is
bit-identical for the no-NaN case (same kernels, fewer redundant calls — no
BLAS reordering risk) and corrected-and-aligned post-rewrite behavior for the
NaN case (the pre-rewrite NaN behavior was a bug). Status: **Accepted
(preventive)** — documents non-obvious decisions ahead of the rewrite in
Slice 2 (#207).

## Considered Options

- **Single `query_ball_point` + vectorized distance compute (chosen).**
  `cKDTree.query_ball_point` already returns the within-radius neighbor
  indices per query coord. For each per-coord neighbor list (small —
  bounded by spatial-radius density, typically a handful of points),
  compute distances directly via
  `np.linalg.norm(self.scaled_check_coords[idx_arr] - scaled_query[pos], axis=1)`.
  No second tree traversal. The scaled `check_coords` array is materialized
  once when the tree rebuilds (on `current_t` change) and cached as
  `self.scaled_check_coords` for reuse. The loop becomes
  `for pos, i in enumerate(good_coords):` — correct mapping by construction;
  no O(N) `i not in good_coords` ndarray membership scan per iteration.
- **Keep the double-query pattern (rejected).** The redundant
  `query(k=max_k)` traversal is the entire problem PRD #204 exists to fix.
  The Slice 1 perf microbenchmark (#206) measures the wall-clock cost
  ahead of the rewrite.
- **`cKDTree.query` with `distance_upper_bound=max_distance_um` and a
  bounded `k` (rejected).** The `k` parameter is data-dependent (varies
  per query coord based on local density); an upper-bound `k` either
  truncates valid neighbors at high density or wastes work at low
  density. `query_ball_point` is the natural primitive for "all neighbors
  within radius" — using `query` here was over-engineering.
- **`cKDTree.query_ball_tree(other_tree, max_distance_um)` (rejected).**
  Dual-tree variant; useful for many-vs-many queries between two
  pre-built trees. Here the query coords change per call (different
  marker positions per `interpolate_coord` invocation) so building a
  second tree per call would dwarf the savings.
- **Preserve the buggy NaN behavior with an `xfail` test (rejected).**
  The pre-rewrite NaN-coord case produces silently misassigned per-coord
  results. No documented consumer (per `wiki/tracking/flow-interpolation.md`)
  can be relying on misaligned output — the wiki invariant "Per-coordinate
  output is either a finite vector or all-NaN — never a mixed coordinate"
  silently hid the bug because (a) the misassigned coord may still pass
  the all-finite-vs-all-NaN gate at the consumer level, and (b) tracks
  dying mid-sequence is documented as expected behavior. Fix-and-pin is
  the right move; the existing pin tests (#206) capture the corrected
  behavior, not the buggy behavior.
- **Separate the rewrite from the NaN-alignment fix into two PRs
  (rejected).** Both changes touch the same loop body
  (`for pos, i in ...` is the same line that the rewrite restructures).
  Splitting forces an artificial intermediate state where one fix is
  in but the other isn't; the test boundary (no-NaN bit-identical;
  NaN corrected) is cleaner as a single PR.

## Consequences

- The Slice 1 (#206) characterization suite covers the no-NaN code path
  end-to-end: empty input, single coord with no in-radius neighbors,
  single coord with one in-radius neighbor at known distance, multi-coord
  with all coords within radius of each other, scaling-applied distance
  values, 2D and 3D parametrized. All deterministic by construction.
- **Test bar: bit-identical for the no-NaN case.** The new implementation
  calls `query_ball_point` (already called in the existing impl), then
  computes distances via `np.linalg.norm` (a different kernel than the
  existing `cKDTree.query(k=max_k)`-derived distances, but on the same
  small per-coord neighbor list). Both paths use Euclidean (`p=2`) and
  the same scaled coordinates; no BLAS reordering, no precision changes.
  The Slice 2 (#207) equivalence test pins this bit-equality on a
  synthetic no-NaN input.
- **NaN-alignment behavior changes.** The NaN-coord positional-alignment
  test added by Slice 2 (#207) pins the corrected post-rewrite behavior,
  not the buggy pre-rewrite behavior. The wiki gotcha "Per-coordinate
  output is either a finite vector or all-NaN — never a mixed coordinate"
  remains correct (the new impl preserves the NaN-or-finite contract);
  what changes is which slot a given per-coord result lands in when NaN
  coords are interleaved.
- **No cross-platform snapshot test.** `cKDTree` is from `scipy.spatial`;
  while it is platform-stable in practice, Nellie does not assert
  `cKDTree`-stability as an invariant. The Slice 1 synthetic suite +
  the Slice 2 intra-process equivalence test on a deterministic
  synthetic input are sufficient.
- The `self.scaled_check_coords` cache adds one ndarray allocation per
  tree-rebuild; it is the same data the tree was already built from
  (multiplication by `self.scaling`). No incremental memory cost beyond
  the now-explicit reference; previously the multiplication result was
  consumed by `cKDTree(...)` and dropped.
- Future maintainers should not switch back to the double-query pattern
  without verifying that the per-coord `linalg.norm` distance compute
  has become a bottleneck (it shouldn't — per-coord neighbor lists are
  bounded by radius density, not by N).
- Future maintainers tempted to "preserve the original NaN-handling for
  consistency" should read this ADR — the original NaN handling was a
  bug, and fix-and-pin is the explicit decision.
- Future consumers reading `_get_nearby_coords` output should rely on
  the post-rewrite slot-alignment contract: if `coords[i]` has any NaN
  component, then `nearby_idxs_return[i]` and `distance_return[i]` are
  empty; otherwise they reflect that coord's neighbors. No more silent
  index shifting.

## References

- PRD #204 — single-query rewrite of `FlowInterpolator._get_nearby_coords`
  + NaN-alignment fix
- Slice 1 #206 — pin current no-NaN behavior with synthetic-test suite +
  perf benchmark backfill + this ADR (no production code changes)
- Slice 2 #207 — single-query rewrite + NaN-alignment fix + scaled-coords
  cache + perf numbers
- ADR 0008 — `_calculate_normalized_moments` matmul rewrite (precedent
  for slice plan + ADR pattern + intra-platform equivalence test)
- ADR 0009 — `_get_cost_matrix` per-feature streaming (precedent for
  bundling a documented-but-deferred cleanup into a perf rewrite)
- ADRs 0005 / 0006 / 0007 — preceding optimizations adopted similar
  test-bar patterns
- `wiki/tracking/flow-interpolation.md` — gotchas + invariants for
  FlowInterpolator (terminal NaN propagation in
  `interpolate_all_forward`/`_backward`)
