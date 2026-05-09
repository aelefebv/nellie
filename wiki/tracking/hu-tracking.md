---
created: 2026-05-06
modified: 2026-05-09
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
- Algorithm config is bundled in a `HuMomentTrackingConfig` frozen dataclass colocated in `hu_tracking.py`. `HuMomentTracking(im_info, HuMomentTrackingConfig(max_distance_um=1.0, ..., cost_cutoff=1.0), viewer=None, num_t=None)` is the construction shape. The cascade may mutate `HuMomentTracking.device` / `HuMomentTracking.low_memory` runtime state; `HuMomentTracking.config` preserves the original intent.

## Gotchas

- **Two independent dense-vs-sparse axes, each with its own budget and fallback.**
  - **ROI extraction** — dense batched (one tensor of stacked sub-volumes) vs streaming per-marker. Switched by total ROI voxel count vs `max_dense_roi_voxels_gpu` (2e7) / `max_dense_roi_voxels_cpu` (5e7). `low_memory=True` forces streaming.
  - **Matching** — dense pairwise cost matrix vs sparse `cKDTree`. Switched by `N_post * N_pre <= max_dense_pairs` (1e7). Sparse path is **CPU-only**.
- **The outer `mode_candidates` cascade in `run()` is the single source of truth for backend switching.** Slice 3 of #91 removed cross-frame backend mutation from the inner cascades:
  - The per-frame `_get_frame_features` cascade is gone (Cascade A deleted). On per-frame OOM, the exception propagates to the outer `adaptive_run.mode_candidates` cascade in `run()`, which retries the whole stage with the next `(device, low_memory)` candidate.
  - The dense→sparse `_match_frames` cascade survives BUT no longer mutates `self.device_type` — later frames keep their original backend. Sparse is CPU-only on the matching axis only; feature extraction for subsequent frames can still run on GPU.
  - The dense-ROI → streaming-ROI fallback inside `_get_frame_features` survives — it only flips a local `use_dense = False` and never touches `self.*`.
  - All three surviving inner cascades log a warning, free GPU memory, and continue without mutating backend attributes. Only the outer `mode_candidates` cascade ever flips `self.xp` / `self.ndi` / `self.device_type`.
- **`low_memory` may be auto-enabled** by `adaptive_run.should_use_low_memory(im_info)` based on estimated memory usage — the constructor default is `False` but the actual run may force streaming ROI extraction anyway.
- **Match-acceptance cost cutoff is the single field `HuMomentTrackingConfig.cost_cutoff` (default `1.0`)**, hot-path-aliased onto `self.cost_cutoff`. Both `_find_best_matches` (dense) and `_match_frames_sparse` (sparse) reference the same attribute, so the dense/sparse acceptance rates can no longer drift apart. `test_cost_cutoff_pinned_in_both_paths` is the authoritative spec — bumping the field flips both paths atomically. Lifted to `HuMomentTrackingConfig.cost_cutoff` in PRD #112 Slice 2 (issue #114). The module constant `_COST_CUTOFF` is gone.
- **`_find_best_matches` returns the union of row-min and col-min candidates** (not Hungarian), so a target can appear in multiple pairs and downstream code sees duplicates.
- **Hu moment 7 (mirror invariance) is intentionally omitted.**
- **3D ROIs are reduced via 3-axis max projection then stacked into 18 features** (not a true 3D moment).
- **Cost weighting is unjustified in code** — z-scored sum of distance + stats + Hu blocks with no learned weights or rationale committed. Treat as a tuned heuristic; document any change.
- **No tests cover this module.** `test_hu_tracking.py` was removed in `a797323` along with the rest of the legacy suite; the rebuild has only reached `test_filtering.py` so far. The dense/sparse equivalence and `_log_hu` finiteness contracts are currently unpinned.

## Invariants

- All five input memmaps must be allocated via `ImInfo`; markers must be boolean-coercible.
- Output schema is **positional** (see [[tracking/index|tracking hub]]): 6 cols 2D / 8 cols 3D — adding a column will silently break readers.
