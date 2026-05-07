---
created: 2026-05-06
modified: 2026-05-06
---

# Networking

Skeletonize each instance, classify each skeleton voxel (background / isolated / tip / edge / junction), label branches, and propagate branch IDs back to fill each object's volume. Outputs: `im_skel` (int32, parent label ID at skel voxels), `im_pixel_class` (uint8 ∈ {0,1,2,3,4}), `im_skel_relabelled` (uint32, every voxel of an object gets a branch ID).

## Why

Topology — branches, junctions, tips — is what enables network metrics (lengths, branching, connectivity). The per-object EDT propagation step assigns every voxel of an object to its nearest branch ID, giving a "branch-painted" volume usable by [[feature-extraction]] for branch-level stats.

## Interactions

- Inputs: [[labelling|`im_instance_label`]], raw, [[filtering|`im_preprocessed`]].
- Outputs consumed by [[feature-extraction]] (branches/components level), [[voxel-reassignment]] (branch labels), and the [[visualizer|napari visualizer]] (label layers).

## Gotchas

- **Skeletonization runs on CPU only** (skimage). The GPU path is only for neighborhood / CC ops. Don't expect end-to-end GPU here.
- **`_remove_connected_label_pixels` deletes skel voxels touching multiple object IDs** (vectorized via min/max filters). Boundary voxels are intentionally preserved.
- **`_add_missing_skeleton_labels` guarantees every label has ≥1 skel voxel.** Without this, small/thin objects vanish from the skeleton; the fallback plants one voxel at the Frangi maximum within the label.
- **Pixel class uses 3×3(×3) convolution count, clipped at 4** — so "junction" means "≥ 4 neighbors", not specifically "exactly 4".
- **Per-object EDT uses anisotropic `sampling=self.scaling`**; the [[mocap-marking|marker-stage EDT]] does **not**. Different design choices in the two stages — be aware when comparing distance values.
- **`_clean_junctions` exists but is never called from the main path** (dead-but-kept code).

## Invariants

- `im_skel`: int32; skel voxels carry parent label ID, 0 elsewhere.
- `im_pixel_class`: uint8, ∈ {0, 1, 2, 3, 4} where 0 = background, 1 = isolated, 2 = tip, 3 = edge, 4 = junction.
- `im_skel_relabelled`: uint32; every voxel of an object gets a branch ID; voxels outside any object are 0.
