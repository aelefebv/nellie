---
created: 2026-05-11
modified: 2026-05-11
---

# `Markers._local_max_peak` per-sigma loop is threaded with `as_completed`; chunked path stays serial

`Markers._local_max_peak` (in `nellie/segmentation/mocap_marking.py`)
runs a per-sigma loop over `self.sigmas` (default 5), computing
`gaussian_laplace + cast + clamp + maximum_filter + boolean ops` per
sigma and reducing into a running `(peak_mask, best_resp)` pair. Per
the perf test on the 3D yeast fixture, the loop accounts for ~120 ms
of the function's ~161 ms total — the dominant CPU cost in
`Markers.run()` now that PRD #179 brought NMS to sub-ms.

PRD #184 threads the per-sigma loop with
`concurrent.futures.ThreadPoolExecutor`. Each sigma's per-iteration
work is independent, and scipy ndimage releases the GIL on
`gaussian_laplace` and `maximum_filter`. Three coupled decisions in
the design need to be pinned ahead of the rewrite in Slice 2 (#186):
(1) per-sigma combination uses `as_completed` (no need to preserve
original sigma order — the reduction is provably order-independent),
(2) `self.low_memory=True` continues to use the chunked variant
(`_local_max_peak_chunked`) which is **not** threaded, and (3) worker
count is capped at `min(cpu_count, len(sigmas), 8)` per ADR 0005.
Status: **Accepted (preventive)** — documents non-obvious decisions
ahead of the threading rewrite in Slice 2 (#186).

## Considered Options

- **Combine per-sigma results in completion order via `as_completed`
  (chosen).** The final `peak_mask` is **invariant to sigma iteration
  order**:
    - `peak_mask[v]` is True iff at some point in the sigma loop, the
      condition `local_max_k[v] && log_resp_k[v] > best_resp[v]` held.
    - `best_resp[v]` starts at 0; the first sigma `s_k` where
      `local_max_k[v]` is True with `log_resp_k[v] > 0` flips
      `peak_mask[v]` to True (since `log_resp_k[v] > 0` is required by
      the strict `>` against the initial `best_resp = 0`).
    - Subsequent sigmas where `log_resp[v] > best_resp[v]` only update
      `best_resp` — `peak_mask[v]` only goes True → True (never
      unset).
    - Subsequent sigmas where `log_resp[v] <= best_resp[v]` make no
      change.
    - **Therefore the final `peak_mask[v]` is True iff `v` is a local
      maximum for at least one sigma with positive `log_resp`** —
      independent of iteration order.
    - Only `peak_mask` is observed downstream (return value is
      `xp.argwhere(peak_mask)`); `best_resp` is internal state. So
      bit-equivalence is preserved across orderings.
- **Preserve original sigma iteration order in the combination
  (rejected).** Would require either collecting all per-sigma results
  before combining (5× peak memory, no parallelism win) or using a
  fixed-order `result` array indexed by sigma index. Adds complexity
  with no observable benefit since the reduction is order-independent
  (proof above).
- **Process pool instead of thread pool (rejected).** scipy ndimage
  releases the GIL on these ops; thread pool is the right primitive.
  Process pool would pay pickling cost on `use_im` and per-sigma
  results — `use_im` is `O(N)` memory, pickling round-trips it twice
  per task.
- **`low_memory=True` continues to use chunked variant; chunked is
  NOT threaded (chosen).** Threading per-sigma inside a chunk would
  hold K full-volume float32 `log_resp` buffers per chunk
  simultaneously — this re-introduces the very memory pressure
  chunking is designed to bound. The chunked path exists precisely
  for the no-threading-allowed case. `low_memory` is the user's
  signal that K × full-volume working memory is unacceptable.
- **Always thread regardless of `low_memory` (rejected).** Subverts
  the user-facing escape hatch. Same rationale as ADR 0005.
- **Cap workers at `min(cpu_count, len(sigmas), 8)` (chosen).** Per
  ADR 0005 — memory-bandwidth-bound past 4-8 threads on typical
  hardware; 64-core boxes don't gain past 8. Capping at
  `len(sigmas)` avoids spinning idle workers when `num_sigma`
  (default 5) is below `cpu_count`.
- **Skip threading when `len(sigmas) < 2` (chosen).** Threading
  overhead (executor construction, future scheduling) dominates the
  serial path on a single sigma. The threshold is conservative —
  even at `len(sigmas) == 2`, the per-task LoG cost (~17 ms on 3D)
  swamps the few-microsecond executor overhead.

## Consequences

- The Slice 1 synthetic-test suite (#185) covers the high-level
  contracts: empty mask → no peaks, all-zero distance → no peaks,
  peaks gated by mask, peaks gated by distance, single-blob detection.
  All hand-derived from the algorithmic spec; platform-stable in
  shape (count and membership assertions, not exact LoG response
  values which are SIMD-sensitive across platforms).
- **No cross-platform snapshot test.** `gaussian_laplace` is
  SIMD-sensitive across platforms (same root cause as ADR 0005's
  no-snapshot decision). The Slice 1 synthetic suite + the Slice 2
  serial-vs-threaded equivalence test (intra-platform deterministic
  by construction) are sufficient.
- The serial-vs-threaded equivalence test added by Slice 2 pins the
  order-independence proof end-to-end on the yeast 3D fixture: both
  paths are run in the same process and asserted byte-equal via
  `np.array_equal`. If a future maintainer accidentally introduces
  order dependence (e.g., changes the strict `>` to `>=`), this test
  fails.
- Future maintainers should not parallelize the chunked path. If a
  real perf measurement on a future workload shows the chunked LoG
  loop is meaningful overhead, that is a separate PRD that must
  address why the memory pressure of K × chunk-volume × float32 is
  acceptable (and update or remove this ADR).
- Future maintainers should not change the strict `>` in the
  per-sigma reduction to `>=`. The strict `>` is what makes the
  reduction order-independent — `>=` would let a later equal-value
  sigma overwrite an earlier one, which depends on completion order
  in the threaded path.
- Future maintainers tempted to "preserve sigma order for stability"
  in the threaded combination should read this ADR — the order
  doesn't affect any observable output, and forcing an ordering
  defeats the parallelism win.

## References

- PRD #184 — thread `Markers._local_max_peak` per-sigma loop
- Slice 1 #185 — pin current behavior with synthetic-test suite + this
  ADR (no production code changes)
- Slice 2 #186 — threading rewrite + serial-vs-threaded equivalence
  test
- ADR 0005 — `Network._relabel_objects` writeback serialization
  (precedent for `low_memory` gating, worker cap, no-cross-platform-
  snapshot)
- ADR 0006 — `Markers._remove_close_peaks` sparse cKDTree (the
  optimization that made `_local_max_peak` the new dominant cost)
