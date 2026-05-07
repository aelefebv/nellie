---
created: 2026-05-06
modified: 2026-05-06
---

# Tracking

Tracks organelle identity across timepoints by matching [[mocap-marking|mocap markers]] between consecutive frames, then propagating those matches into voxel-level relabeling and per-label trajectories.

## Pipeline shape

Three stages chained through one shared artifact, the `flow_vector_array` saved at `pipeline_paths['flow_vector_array']`:

1. [[hu-tracking|`HuMomentTracking`]] consumes mocap markers + Frangi + intensity + distance-transform memmaps and writes one `(t, coord_pre, vec, cost)` row per matched marker pair. This is the **radius-adaptive** step: each marker's ROI radius comes from the distance-transform value at that voxel, doubled.
2. [[flow-interpolation|`FlowInterpolator`]] turns the sparse marker-to-marker vectors into a dense, query-able flow field via distance-weighted KDTree interpolation. Forward (t → t+1) and backward (t+1 → t) modes share the same array but offset query coords by the stored vector for backward.
3. [[voxel-reassignment|`VoxelReassigner`]] calls a forward and backward `FlowInterpolator` for every labeled voxel in frame t, finds nearest real voxels in t+1, votes by inverse-distance, and writes new labels to `im_branch_label_reassigned` and `im_obj_label_reassigned`.

`tracking/all_tracks_for_label.py` (`LabelTracks`) calls `interpolate_all_forward/backward` directly to assemble per-label napari trajectories — used by the [[visualizer]]. `flow_vector_viz.py` is pure inspection-only reshaping (no tracking logic).

## Interactions

- **Inputs** from [[segmentation/index|segmentation]] (via `ImInfo.pipeline_paths`): `im_marker`, `im_instance_label`, `im_skel_relabelled`, `im_distance` (drives ROI radii), `im_preprocessed`, raw image.
- **Outputs feed:** [[feature-extraction]] (label-stable trajectories needed for per-organelle features), the [[visualizer|napari visualizer]] (via `flow_vector_viz` and `LabelTracks`), and downstream adjacency analyses (`scripts/voxel_reassignment_demo.py` shows branch/node remapping via `running_matches`).
- Backend control flows through [[gpu-runtime|adaptive_run]] for GPU/CPU + low-memory escalation.

## Subsystem invariants and gotchas

- Segmentation must produce nonzero markers, a distance transform large enough to give sensible ROI radii (radii < 1 collapse to empty sub-volumes and are silently skipped), instance labels, and skeleton-relabelled branch labels.
- `dim_res['T']` should be set; missing T resolution forces a 1.0 s fallback with a warning.
- `max_distance_um` is the universal physical gate. In `HuMomentTracking` it's `max(max_distance_um * dt, 0.5)`; in `FlowInterpolator` it's `max(0.5 * dt, 0.5)`. Markers beyond this between frames are unreachable.
- **The `flow_vector_array` schema is positional**: 6 cols 2D `[t, y, x, dy, dx, cost]`, 8 cols 3D `[t, z, y, x, dz, dy, dx, cost]`. `flow_vector_viz` and `FlowInterpolator` both index by absolute column offsets — adding a column will silently break them.
