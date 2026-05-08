---
created: 2026-05-06
modified: 2026-05-07
---

# Voxel reassignment

Relabel branch and object voxels at frame t+1 to inherit labels from frame t, so the same physical structure carries one label across time. Outputs: `im_branch_label_reassigned`, `im_obj_label_reassigned`, plus `voxel_matches` and (optionally) `running_matches` for downstream adjacency work.

## Why

[[segmentation/index|Segmentation]] is independent per frame, so label IDs are arbitrary; tracking needs **label-stable references**. Forward + backward [[flow-interpolation]] provides two candidate matches per voxel; inverse-distance weighted voting resolves conflicts. `_select_best_pairs` and `running_matches` give downstream tools a 1-best mapping per target voxel — `scripts/voxel_reassignment_demo.py` shows the canonical use case (branch/node adjacency remapping), which is the load-bearing reason `store_running_matches` exists.

## How (per frame pair)

1. **Predict.** Two `FlowInterpolator` instances (forward and backward) interpolate motion vectors at every labeled voxel, producing predicted positions in the opposite frame.
2. **Snap to real voxels.** Build a KDTree over actual voxel coords in the target frame, query nearest-neighbor for each predicted position. Drop matches whose physical-distance error exceeds `flow_interpolator.max_distance_um`.
3. **Pool bidirectional candidates.** Concatenate forward (`t→t+1`) and backward (`t+1→t`) matches into one candidate set — same target voxel may collect multiple votes.
4. **Weighted vote per target.** For each target voxel, source labels vote with weight `1/(distance + ε)`. Highest summed-weight label wins. Implementation uses two-stage `lexsort` over `ravel_multi_index`-flattened coords (no Python-level grouping).
5. **Iterate.** Re-vote up to `max_refine_iterations` times on still-unassigned target voxels — earlier iterations seed labels that later iterations can build on.

The expensive matching (steps 1–3) runs **once** per frame pair using the union of branch + object masks; the same candidate set then feeds two independent vote-and-write passes.

## Backend cascade

`_TreeHandle` wraps four interchangeable nearest-neighbor backends, picked at build time and downgraded on failure:

1. **`gpu`** — `cupyx.scipy.spatial.cKDTree` on device.
2. **`gpu_bruteforce`** — pairwise `(query × real)` distance matrix in CuPy, chunked by `max_bruteforce_pairs`. Used when GPU KDTree is unavailable but CuPy works.
3. **`cpu`** — `scipy.spatial.cKDTree` with `workers=-1`.
4. **`cpu_bruteforce`** — chunked NumPy pairwise distances. Final fallback when KDTree allocation fails.

OOM at any tier triggers `_free_gpu_memory()` + `_switch_to_cpu()`; chunk-level OOM in brute-force mode halves `chunk_size` and retries. Outer `run()` wraps the whole pipeline in `adaptive_run.mode_candidates(...)` so a full-frame failure cascades through `(gpu, low=False) → (gpu, low=True) → (cpu, low=False) → (cpu, low=True)` before raising.

## Interactions

- Inputs: [[hu-tracking|`flow_vector_array`]] (via `FlowInterpolator`), [[labelling|`im_instance_label`]], [[networking|`im_skel_relabelled`]].
- Outputs feed [[feature-extraction]] (label identity at the components level via `reassigned_label`) and the [[visualizer|napari visualizer]] (`Reassigned px:` layers).
- Backend orchestration via [[gpu-runtime|`adaptive_run`]] (`normalize_device`, `gpu_available`, `should_use_low_memory`, `mode_candidates`, `is_oom_error`, `is_gpu_unavailable_error`).

## Gotchas

- **Streams over timepoints**, not voxels-by-frame. Only two frames' worth of `argwhere` coordinates plus the candidate match arrays live in memory at once; outputs go straight into memmaps. Loop breaks early if any frame pair produces zero candidates — *all subsequent frames are then unreassigned*.
- **Half-hoisted backend code.** `_resolve_backend`, `_try_import_cupy`, `_is_oom_error` are local copies that shadow the canonical versions in [[gpu-runtime|`adaptive_run`]]. The outer `run()` *does* call the canonical helpers; the inner per-tree path doesn't. Tracked in [[queue]] as part of the per-stage backend hoist.
- **Uses ravel-index tricks on `spatial_shape` for fast uniqueness** — `_allocate_memory()` must run first or `_select_best_pairs` / `_vote_targets` / `_assign_unique_matches` raise.
- **`match_coord_dtype` is auto-selected** from `max(spatial_shape)` (`uint16` / `uint32` / `uint64`). Saved `running_matches` round-trip back to int coordinates correctly only if the dataset's spatial extent fits — bumping image size past 65 535 in any axis silently widens the saved dtype.
- **Low-memory mode rebuilds the second tree only after freeing the first** (and frees the GPU pool between forward/backward passes), trading speed for headroom; it also caps `max_query_points` at 2e5 and `max_bruteforce_pairs` at 2e6.
- **`max_refine_iterations`** lets later iterations fill voxels left unassigned by earlier vote rounds. Stopping early can leave holes.
- **Branch and object label types share one match computation per frame** for efficiency — splitting them would double the work.

## Invariants

- Preserves voxel positions; only label values change.
- Voxels with no valid match keep their (per-frame) label after refinement exits.
- Output dtypes match input label dtypes per layer.
- A target voxel at `t+1` is only assigned if it has a non-zero label in the original `label_memmap[t+1]` (matching never invents labels for background voxels).
