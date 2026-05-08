---
created: 2026-05-06
modified: 2026-05-07
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

- **Two independent dense-vs-sparse axes, each with its own budget and fallback.**
  - **ROI extraction** — dense batched (one tensor of stacked sub-volumes) vs streaming per-marker. Switched by total ROI voxel count vs `max_dense_roi_voxels_gpu` (2e7) / `max_dense_roi_voxels_cpu` (5e7). `low_memory=True` forces streaming.
  - **Matching** — dense pairwise cost matrix vs sparse `cKDTree`. Switched by `N_post * N_pre <= max_dense_pairs` (1e7). Sparse path is **CPU-only**.
- **Adaptive degradation has two layers, both silent.** Inner: GPU OOM during a frame catches and retries that frame on CPU; dense ROI OOM falls to streaming; dense matching OOM falls to sparse. Outer: `run()` walks `adaptive_run.mode_candidates` over `(device, low_memory)` combos and retries the whole pipeline on each OOM. A run can degrade across both layers with only `logger.warning` to show for it.
- **`low_memory` may be auto-enabled** by `adaptive_run.should_use_low_memory(im_info)` based on estimated memory usage — the constructor default is `False` but the actual run may force streaming ROI extraction anyway.
- **Match-acceptance cost cutoff is a hardcoded `1.0`** in both `_find_best_matches` and the sparse path — not a constructor parameter, so changing the cost weighting in one place without changing the other will silently shift acceptance rates.
- **`_find_best_matches` returns the union of row-min and col-min candidates** (not Hungarian), so a target can appear in multiple pairs and downstream code sees duplicates.
- **Hu moment 7 (mirror invariance) is intentionally omitted.**
- **3D ROIs are reduced via 3-axis max projection then stacked into 18 features** (not a true 3D moment).
- **Cost weighting is unjustified in code** — z-scored sum of distance + stats + Hu blocks with no learned weights or rationale committed. Treat as a tuned heuristic; document any change.
- **No tests cover this module.** `test_hu_tracking.py` was removed in `a797323` along with the rest of the legacy suite; the rebuild has only reached `test_filtering.py` so far. The dense/sparse equivalence and `_log_hu` finiteness contracts are currently unpinned.

## Invariants

- All five input memmaps must be allocated via `ImInfo`; markers must be boolean-coercible.
- Output schema is **positional** (see [[tracking/index|tracking hub]]): 6 cols 2D / 8 cols 3D — adding a column will silently break readers.
