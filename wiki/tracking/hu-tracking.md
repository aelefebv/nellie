---
created: 2026-05-06
modified: 2026-05-06
---

# Hu-moment tracking

Match [[mocap-marking|mocap markers]] across consecutive frames and write a `flow_vector_array` (sparse rows of `(t, coord_pre, vec, cost)`). This is the **radius-adaptive pattern matching** step the paper highlights.

## Why

- **Hu moments are translation/scale/rotation invariant** on the ROI image, so shape similarity contributes to matching even when an organelle has rotated.
- **ROI size is radius-adaptive** via the [[mocap-marking|distance transform]]: each marker's matching window scales with local feature size — wide for blobs, thin for tubules.
- Stats (mean / variance of intensity + Frangi) and physical distance are jointly z-scored into a single cost, making the cost a unitless aggregate.

## Interactions

- Inputs: `im_marker`, `im_instance_label`, `im_distance`, `im_preprocessed`, raw — all [[segmentation/index|segmentation]] outputs.
- Output: `flow_vector_array` consumed by [[flow-interpolation]] and [[voxel-reassignment]].
- Backend: [[gpu-runtime|adaptive_run]] cascade.

## Gotchas

- **Dense vs sparse path is auto-selected** by `N_post * N_pre <= max_dense_pairs (1e7)` and an ROI voxel budget. Sparse path is **CPU-only via `cKDTree`**.
- **GPU OOM is caught and falls back to CPU mid-frame** — performance can drop silently.
- **`_find_best_matches` returns the union of row-min and col-min candidates** (not Hungarian), so a target can appear in multiple pairs and downstream code sees duplicates.
- **Hu moment 7 (mirror invariance) is intentionally omitted.**
- **3D ROIs are reduced via 3-axis max projection then stacked into 18 features** (not a true 3D moment).
- **Cost weighting is unjustified in code** — z-scored sum of distance + stats + Hu blocks with no learned weights or rationale committed. Treat as a tuned heuristic; document any change.
- **`test_hu_tracking.py`** covers `_log_hu` finiteness and dense/sparse equivalence on a 2-marker fixture — narrow but pins the matching-mode contract.

## Invariants

- All five input memmaps must be allocated via `ImInfo`; markers must be boolean-coercible.
- Output schema is **positional** (see [[tracking/index|tracking hub]]): 6 cols 2D / 8 cols 3D — adding a column will silently break readers.
