---
created: 2026-05-06
modified: 2026-05-06
---

# Voxel reassignment

Relabel branch and object voxels at frame t+1 to inherit labels from frame t, so the same physical structure carries one label across time. Outputs: `im_branch_label_reassigned`, `im_obj_label_reassigned`, plus `voxel_matches` and (optionally) `running_matches` for downstream adjacency work.

## Why

[[segmentation/index|Segmentation]] is independent per frame, so label IDs are arbitrary; tracking needs **label-stable references**. Forward + backward [[flow-interpolation]] provides two candidate matches per voxel; inverse-distance weighted voting resolves conflicts. `_select_best_pairs` and `running_matches` give downstream tools a 1-best mapping per target voxel — `scripts/voxel_reassignment_demo.py` shows the canonical use case (branch/node adjacency remapping), which is the load-bearing reason `store_running_matches` exists.

## Interactions

- Inputs: [[hu-tracking|`flow_vector_array`]] (via `FlowInterpolator`), [[labelling|`im_instance_label`]], [[networking|`im_skel_relabelled`]].
- Outputs feed [[feature-extraction]] (label identity at the components level via `reassigned_label`) and the [[visualizer|napari visualizer]] (`Reassigned px:` layers).

## Gotchas

- **Uses ravel-index tricks on `spatial_shape` for fast uniqueness** — `_allocate_memory()` must run first or these raise.
- **KDTree matching has CPU/GPU/brute-force fallbacks with chunked queries**; `low_memory` halves chunk sizes.
- **`max_refine_iterations`** lets later iterations fill voxels left unassigned by earlier vote rounds. Stopping early can leave holes.
- **Branch and object label types share one match computation per frame** for efficiency — splitting them would double the work.

## Invariants

- Preserves voxel positions; only label values change.
- Voxels with no valid match keep their (per-frame) label after refinement exits.
- Output dtypes match input label dtypes per layer.
