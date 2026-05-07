---
created: 2026-05-06
modified: 2026-05-06
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
- **Backward mode shifts `check_coords` by the stored vector** so queries land at predicted positions (since the array stores pre-coords + forward vector).
- **Returns NaN rows for queries with no neighbors within `max_distance_um`.** Callers must handle NaN explicitly.
- **The `__main__` block has a `self.im_info` typo bug** (uses `self` outside a class). Don't run the script standalone.

## Invariants

- `flow_vector_array` must exist on disk before instantiation.
- Per-coordinate output is either a finite vector or all-NaN — never a mixed coordinate.
