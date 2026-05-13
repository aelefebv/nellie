---
created: 2026-05-06
modified: 2026-05-09
---

# Labelling

Threshold the [[filtering|Frangi]] volume and emit per-frame instance labels (connected components) with min-size pruning. Output: `im_instance_label`, int32, background = 0.

## Why

Frangi response is continuous; tracking and features need discrete object IDs. Combining triangle and Otsu in **log10 domain** (taking the `min()` — the more conservative threshold) handles the heavy-tailed Frangi histogram. Intensity Otsu on the raw image optionally pre-masks before Frangi thresholding so dim-but-Frangi-positive noise gets suppressed — the mask shifts foreground voxel coverage but doesn't necessarily reduce label count (on the yeast fixture, count is unchanged; only voxel coverage shifts).

## Interactions

- Input: raw + Frangi memmaps (both from [[im-info|ImInfo]] paths).
- Output `im_instance_label` consumed by [[networking]], [[mocap-marking]], [[tracking/index|tracking]], and [[feature-extraction]].
- Threshold helpers from [[gpu-runtime|gpu_functions]].
- Backend selection and OOM cascade go through [[gpu-runtime|`adaptive_run`]].
- Algorithm config is bundled in a `LabelConfig` frozen dataclass colocated in `labelling.py`. `Label(im_info, LabelConfig(threshold=..., ...), viewer=None, num_t=None)` is the construction shape. The cascade may mutate `Label.device` / `Label.low_memory` runtime state; `Label.config` preserves the original intent.

## Adaptive backend & low-memory mode

Two layers of fallback wrap the per-frame work:

- **Outer (`run()`)** iterates `(device, low_memory)` candidates from `adaptive_run.mode_candidates`, re-running `_set_backend` / `_set_low_memory` / `_allocate_memory` / `_run_segmentation` for each. `low_memory` may be auto-enabled at start by `adaptive_run.should_use_low_memory(im_info, ...)` even if the user didn't ask for it.
- **Inner (per-frame)** OOM during full-volume labeling falls back to **chunked-Z with `initial_chunk = full Z`** in 3D, or to CPU in 2D. Inside the chunked-Z loop, OOM **halves `chunk_z`**; at `chunk_z=1` on CUDA it switches to CPU; at `chunk_z=1` on CPU it aborts.

The two layers can interact non-obviously. An OOM the inner cascade **silently recovers from** (full→chunked, halving `chunk_z`, cuda→cpu) never reaches the outer `run()` retry loop. So an inner CPU switch can leave the run finishing on CPU even though the outer mode is still `gpu` — the outer loop only sees what the inner loop chose to re-raise.

Chunk size: when `low_memory=True` and no explicit `chunk_z` is given, `_infer_chunk_z` derives it from `max_chunk_voxels // (Y*X)`. **Gotcha:** an explicit user `chunk_z` is stashed in `_user_chunk_z` and reapplied on every `_set_low_memory` call — toggling `low_memory` does **not** override an explicit chunk. The stash is captured *after* the `no_z` gate, however, so passing `chunk_z=...` against a 2D image silently records `None` — intentionally a no-op since there's no Z to chunk along.

## Gotchas

- **Z-chunked path uses union-find to stitch label IDs across chunk boundaries** (boundary slice pair lookup), then a relabel pass renumbers densely. Only triggered when `had_merges` is true. If you tweak chunking, re-verify component-count equivalence per frame with the unchunked path — full and chunked outputs are **not** byte-equivalent: the `uniform_filter` smoothing pass in `_get_labels` runs on different neighborhoods at chunk seams, so individual mask voxels can flicker. Component count per frame is the contract; mask identity is not.
- **Threshold sampling is strided, not exhaustive.** `_sample_nonzero` strides through the flat frame at two offsets, capped at `threshold_sampling_pixels` (default 1M). Falls back to a full-array scan only if the strided pass returns nothing. Threshold values can therefore drift slightly across volumes of different sizes.
- **`threshold` parameter has two simultaneous effects.** (a) It gates the Frangi-threshold sample pool via `_compute_frangi_threshold(mask_thresh=...)`, restricting which voxels feed the triangle/Otsu computation; AND (b) it multiplies into the per-frame Frangi during labeling (`frangi_in_mem * mask`). Setting `threshold` high can therefore *increase* component count by lowering the computed Frangi cutoff on the now-smaller sample — counterintuitive.
- **Min-area is the pixel area/volume of a sphere of `min_radius_um`** — not a literal pixel count. Anisotropy is honored. **`min_radius_um` is floored at `x_res`** in the constructor; you can't ask for objects smaller than one X pixel.
- **2D vs 3D differ inside `_get_labels`:** only 3D runs `binary_fill_holes`, and the structuring footprint is `(3,3)` vs `(3,3,3)`.
- **A smoothing pass after pruning** (`uniform_filter` then `> 0.5`) re-runs CC, so the final label count can differ from the initial count.
- **Label IDs are not stable across frames.** Cross-frame identity is the job of [[voxel-reassignment]].
- `flush_interval` controls how often `instance_label_memmap` is flushed during the per-frame loop (default = every frame).
- **`_run_frame_full_volume` returns `labels | None`; `_run_frame_chunked_z` returns `None` always.** Same shape, different meanings: `None` from full-volume signals "I already wrote chunked instead, don't write again"; `None` from chunked is just side-effect convention. Anyone replacing either method has to preserve this asymmetry or move the memmap write into the orchestrator.

## Invariants

- Output is int32; background = 0; IDs dense within a frame; small components (< sphere of `min_radius_um`) are removed.
- Mutating intensity / Frangi inputs is forbidden.
- Input shape == output shape.
