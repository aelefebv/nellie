---
created: 2026-05-06
modified: 2026-05-06
---

# Labelling

Threshold the [[filtering|Frangi]] volume and emit per-frame instance labels (connected components) with min-size pruning. Output: `im_instance_label`, int32, background = 0.

## Why

Frangi response is continuous; tracking and features need discrete object IDs. Combining triangle and Otsu in **log10 domain** handles the heavy-tailed Frangi histogram. Intensity Otsu on the raw image optionally pre-masks before Frangi thresholding so dim-but-Frangi-positive noise gets dropped.

## Interactions

- Input: raw + Frangi memmaps (both from [[im-info|ImInfo]] paths).
- Output `im_instance_label` consumed by [[networking]], [[mocap-marking]], [[tracking/index|tracking]], and [[feature-extraction]].
- Threshold helpers from [[gpu-runtime|gpu_functions]].

## Gotchas

- **Z-chunked path uses union-find to stitch label IDs across chunk boundaries** (boundary slice pair lookup), then a relabel pass renumbers densely. Only triggered when `had_merges` is true. If you tweak chunking, re-verify equivalence with the unchunked path.
- **Min-area is the pixel area/volume of a sphere of `min_radius_um`** — not a literal pixel count. Anisotropy is honored.
- **A smoothing pass after pruning** (`uniform_filter` then `> 0.5`) re-runs CC, so the final label count can differ from the initial count.
- **Label IDs are not stable across frames** — pinned by `test_label_ids_reset_per_frame`. Cross-frame identity is the job of [[voxel-reassignment]].

## Invariants

- Output is int32; background = 0; IDs dense within a frame; small components (< sphere of `min_radius_um`) are removed.
- Mutating intensity / Frangi inputs is forbidden — `test_masking_does_not_mutate_inputs` pins this.
- Input shape == output shape.
