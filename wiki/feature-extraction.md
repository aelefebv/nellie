---
created: 2026-05-06
modified: 2026-05-07
---

# Feature extraction

Extracts a multi-level, per-frame feature table from an already-segmented and skeletonized stack, so downstream analysis can reason about a labeled organelle at any spatial granularity from a single voxel up to a whole image. Driven by `Hierarchy` in `nellie/feature_extraction/hierarchical.py`.

## Hierarchy

Five levels, all driven by `Hierarchy._get_hierarchies()` in this fixed order so each level can read the previous one:

1. **Voxels** — every foreground voxel. Coords (x/y/z), raw intensity, structure-image value, voxel-to-node/branch/component label maps, and motility (`linear_vel`, `angular_vel`, `linear_acc`, `angular_acc` plus their `rel_*` reference-frame counterparts and `rel_directionality`). Motility uses forward/backward [[flow-interpolation|`FlowInterpolator`]].
2. **Nodes** — skeleton "junction-radius" patches. Per-node thickness (KD-tree distance to border mask), plus flow-derived `divergence`, `convergence`, `vergere`. Each node's bbox is sized by its `im_distance` value (its skeleton radius); voxels falling inside that bbox become its children, and the resulting `num_nodes × chunk_size` mask is processed in adaptive chunks bounded by `max_node_mask_elems`. **Skipped by default** (`skip_nodes=True`); the [[settings|napari settings widget]] exposes the toggle but [[pipeline]] hard-codes `False`.
3. **Branches** — connected skeleton segments. Centerline `branch_length` from an O(N) neighborhood walk over the skeleton (sums physical edge lengths between adjacent same-label voxels via shifted-array compares; CuPy backend with CPU fallback) plus end-cap corrections that add the local distance-transform radius at each tip. Median `branch_thickness`, `branch_aspect_ratio`, `branch_tortuosity` (chord-over-arc), plus `regionprops` morphology on the relabelled branch mask — also the source of branch x/y/z centroids (skeleton coords are only used for length/thickness, not position). Aggregates voxel and node stats by branch label.
4. **Components / Organelles** — connected instance labels. `regionprops` morphology suite (`organelle_area`, axis lengths, extent, solidity) and a `reassigned_label` carrying the [[voxel-reassignment|tracked identity]]. Aggregates voxel, node, and branch stats by component label.
5. **Image** — whole-frame summary. Holds no inherent metrics; rolls all four lower levels into single-row-per-frame aggregates.

## Why hierarchical

A mitochondrial network is interesting at multiple scales simultaneously: a single tubule's tortuosity, the branch-thickness distribution within one organelle, network-wide motility. The same primitive measurement (e.g. a voxel's velocity) needs to surface as raw data, branch mean, organelle mean, and image mean **without recomputation**. The shared `aggregate_stats_for_class` helper enforces that every parent level is a deterministic roll-up of its children's `stats_to_aggregate` list (mean / std_dev / min / max / sum) — so a branch's velocity stats are by construction the same as if you grouped the voxel CSV by branch label yourself.

## Interactions

- **Consumes** (via `ImInfo.pipeline_paths` memmaps): raw, [[filtering|`im_preprocessed`]], [[mocap-marking|`im_distance`]], [[networking|`im_skel`]], [[networking|`im_pixel_class`]], [[labelling|`im_instance_label`]], [[networking|`im_skel_relabelled`]], [[mocap-marking|`im_border`]], and (when temporal) [[voxel-reassignment|tracked-reassigned label volumes]].
- Motility pulls forward/backward flow through [[flow-interpolation|`FlowInterpolator`]].
- **Outputs** read by the [[analysis|napari analysis widget]] and [[processor]].

## Output shape

Five per-level CSVs (`features_voxels` / `nodes` / `branches` / `organelles` / `image`), streamed one frame at a time with `t` and `label` as the leading columns and `feature_stat`-style headers (e.g. `branch_length_mean`). A pickled edge-list dict at `adjacency_maps` carries v→b, v→n, v→o, n→b, n→o, b→o membership for joining levels back together.

## Gotchas

- `regionprops`, skimage, and numpy NaN-warnings are silenced at module import.
- **Empty frames/regions short-circuit to empty lists**; aggregates of empty groups become NaN; multi-dimensional stats are silently skipped during aggregation.
- **All coords are scaled by `self.spacing` (Z, Y, X `dim_res`), velocities divided by `dim_res["T"]`** — output is in **physical units**, not pixels.
- **2D vs 3D paths diverge throughout** (`im_info.no_z`); 2D drops the z column to NaN.
- **Node-to-voxel assignment chunks adaptively** (`_resolve_node_chunk_size`, MemoryError retry halving). The whole `Hierarchy.run()` retries across `[gpu, cpu] × [high-mem, low-mem]` combinations on OOM/GPU-unavailable.
- **Aspect-ratio path swaps thickness and length when thickness exceeds length** (preserving "long axis" semantics for blobby segments).
- **`enable_motility=False` short-circuits all flow-derived voxel features to NaN** (and every `*_vel` / `*_acc` / `rel_*` aggregate above it inherits NaNs). `enable_adjacency=False` skips writing `adjacency_maps.pkl` entirely. Both default `True`; flipping them is the cheap way to skip whole feature families when you only need morphology.
- **`FlowInterpolator` instances are built lazily** — only when `enable_motility and not no_t and num_t > 1`. Single-frame stacks skip flow loading even when motility is "enabled," and the per-voxel motility helper fills NaN in that branch.
- **Per-branch reference voxel for `rel_*` motility is the one with minimum-magnitude flow vector in that branch** (`_get_min_euc_dist`). All `rel_linear_vel` / `rel_angular_vel` / `rel_directionality` are measured against that single representative — interpret accordingly.

## Invariants

- `test_aggregate_stats_low_memory_parity` pins the core guarantee: the fast vectorized path and the low-memory path produce identical mean/std/min/max/sum (NaN-equal) for the same `(child_class, t, list_of_idxs)`, and identical column headers from `append_to_array`.
- NaNs in inputs propagate via `nan*` reductions rather than corrupting aggregates.
- CSV header order is **stable across frames** (set on first frame, reused in append mode).
- Adjacency edge lists are 0-indexed for level-internal indices but use raw label values for component columns, matching how the CSVs key their `label` field.
