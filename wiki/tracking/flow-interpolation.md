---
created: 2026-05-06
modified: 2026-05-12
---

# Flow interpolation

Convert sparse [[hu-tracking|marker matches]] (`flow_vector_array`) into a query-able dense flow field for arbitrary coordinates. Used by [[voxel-reassignment]] for per-voxel matching and by `LabelTracks` (`tracking/all_tracks_for_label.py`) for per-label trajectories.

## Why

Hu tracking only matches markers; voxel reassignment and label trajectories need vectors at **any** voxel. Distance-weighted KDTree interpolation fills the gap by averaging nearby marker vectors weighted by inverse spatial distance and inverse matching cost.

## Interactions

- Loads `flow_vector_array` from disk on construction.
- Called by [[voxel-reassignment]] (forward + backward instances per frame) and `LabelTracks` (`interpolate_all_forward/backward`).

## Gotchas

- **Caches per-`current_t` KDTree.** Switching the `forward` flag mid-stream needs a fresh instance — flipping in place returns stale results.
- **Forward looks up markers by origin; backward looks up by destination.** The stored array is `(pre-coord, forward-vector)`. Forward mode queries against marker origins at `t`; backward mode shifts origins by their vector so queries land at predicted positions at `t`. In both cases, "nearby" means "markers whose vector actually touches the query point" — this symmetry is what lets the same weighting code serve both directions.
- **Returns NaN rows for queries with no neighbors within `max_distance_um`.** Callers must handle NaN explicitly.
- **NaN is terminal in `interpolate_all_forward/backward`.** Once a coord's interpolated vector is all-NaN, the driver overwrites the coord with NaN and never revives it on later frames — tracks die silently mid-sequence rather than skipping a gap.
- **`max_distance_um` is scaled by `dim_res['T']` at construction**, with a 0.5 μm floor. The constructor argument is effectively μm-per-frame, so the search radius grows with frame interval. Passing `0.5` does not give a 0.5 μm radius unless `dt == 1`.
- **`flow_vector_array` is pre-bucketed by `t` at `_allocate_memory`**: `self._t_to_rows = {t: row_indices}` is built once. Per-frame `interpolate_coord` lookups are O(1) dict reads; the pre-cleanup O(total_markers) `np.where(flow_vector_array[:, 0] == t)` scan is gone (PR #214).

## Invariants

- `flow_vector_array` must exist on disk before instantiation.
- Per-coordinate output is either a finite vector or all-NaN — never a mixed coordinate.
