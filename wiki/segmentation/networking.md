---
created: 2026-05-06
modified: 2026-05-07
---

# Networking

Skeletonize each instance, classify each skeleton voxel (background / isolated / tip / edge / junction), label branches, and propagate branch IDs back to fill each object's volume. Outputs: `im_skel` (int32, parent label ID at skel voxels), `im_pixel_class` (uint8 ∈ {0,1,2,3,4}), `im_skel_relabelled` (uint32, every voxel of an object gets a branch ID).

## Why

Topology — branches, junctions, tips — is what enables network metrics (lengths, branching, connectivity). The per-object EDT propagation step assigns every voxel of an object to its nearest branch ID, giving a "branch-painted" volume usable by [[feature-extraction]] for branch-level stats.

## Pipeline (per timepoint)

`_run_frame_backend` runs six stages in order:

1. **Skeletonize** (`_skeletonize`) — `skimage.morphology.skeletonize` on each label, with a per-object fallback (`_skeletonize_per_object`) when memory is tight.
2. **Clean ambiguous voxels** (`_remove_connected_label_pixels`) — drop skel voxels that touch ≥2 different object labels via vectorized 3×3(×3) min/max neighborhood filters.
3. **Patch missing** (`_add_missing_skeleton_labels`) — for any label whose skeleton vanished, plant one seed at the Frangi-maximum voxel inside that label.
4. **Classify** (`_get_pixel_class`) — 3×3(×3) convolution counts neighbors per skel voxel; clipped at 4.
5. **Identify branches** (`_get_branch_skel_labels`) — connected components on non-junction skel voxels.
6. **Project to volume** (`_relabel_objects`) — per-object EDT (anisotropic, on a bounding-box crop) assigns every object voxel the label of its nearest branch seed.

## Interactions

- Inputs: [[labelling|`im_instance_label`]], raw, [[filtering|`im_preprocessed`]].
- Outputs consumed by [[feature-extraction]] (branches/components level), [[voxel-reassignment]] (branch labels), and the [[visualizer|napari visualizer]] (label layers).
- Plugs into the [[gpu-runtime|adaptive_run cascade]]: `run()` iterates `mode_candidates` over `(device, low_memory)` pairs, retrying the whole pipeline if a stage raises an OOM or GPU-unavailable error.

## Gotchas

- **Skeletonization runs on CPU only** (skimage). The GPU path is only for neighborhood / CC ops. Don't expect end-to-end GPU here.
- **Per-stage device choreography is uneven.** Stages 1–3 (skeletonize, clean, patch) and stage 6 (per-object EDT) are forced CPU. Only stages 4–5 (`_get_pixel_class`, `_get_branch_skel_labels`) take the GPU path, and only when `device_type == "cuda" and not low_memory`. Setting `device="gpu"` does not move the heavy stages off CPU.
- **Frame-level OOM fallback is separate from `run()`'s cascade.** `_run_frame` catches GPU OOM mid-pipeline, calls `_switch_to_cpu()`, and retries that frame. After this fires, all subsequent frames run on CPU for the rest of the call — there's no switch back.
- **Low-memory mode swaps in chunked variants** (`_remove_connected_label_pixels_chunked`, `_get_pixel_class_chunked`) that tile via `_iter_chunks` with a 1-voxel halo, then trim. Chunking is per-stage, not pipeline-wide.
- **`_remove_connected_label_pixels` deletes skel voxels touching multiple object IDs** (vectorized via min/max filters). Boundary voxels are intentionally preserved.
- **`_add_missing_skeleton_labels` guarantees every label has ≥1 skel voxel.** Without this, small/thin objects vanish from the skeleton; the fallback plants one voxel at the Frangi maximum within the label.
- **Pixel class uses 3×3(×3) convolution count, clipped at 4** — so "junction" means "≥ 4 neighbors", not specifically "exactly 4".
- **Per-object EDT uses anisotropic `sampling=self.scaling`**; the [[mocap-marking|marker-stage EDT]] does **not**. Different design choices in the two stages — be aware when comparing distance values.
- **`_clean_junctions` exists but is never called from the main path** (dead-but-kept code).

## Invariants

- `im_skel`: int32; skel voxels carry parent label ID, 0 elsewhere.
- `im_pixel_class`: uint8, ∈ {0, 1, 2, 3, 4} where 0 = background, 1 = isolated, 2 = tip, 3 = edge, 4 = junction.
- `im_skel_relabelled`: uint32; every voxel of an object gets a branch ID; voxels outside any object are 0.
