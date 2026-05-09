---
created: 2026-05-06
modified: 2026-05-08
---

# Mocap marking

Detect motion-tracking anchor points (multi-scale LoG peaks) inside each label, plus produce the distance transform and the outer-shell border mask. Outputs: `im_marker` (uint8, 1 at peak voxels), `im_distance` (float32 ≥0), `im_border` (uint8 binary).

## Why

[[tracking/index|Tracking]] needs sparse, repeatable, in-object anchors. **LoG-of-distance-transform peaks correspond to local thickness maxima** (tube centers, blob centers) and are robust across frames. Distance and border are deliberately co-produced here — [[feature-extraction]] needs them too, and recomputing in two places would be wasteful and error-prone.

## Interactions

- Inputs: [[labelling|`im_instance_label`]] + raw intensity (+ [[filtering|Frangi]] if `use_im='frangi'`).
- Outputs feed [[hu-tracking|Hu-moment tracking]] (markers, distance), [[voxel-reassignment]] (markers indirectly via the flow array), and [[feature-extraction]] (distance, border).
- Algorithm config is bundled in a `MarkersConfig` frozen dataclass colocated in `mocap_marking.py`. `Markers(im_info, MarkersConfig(use_im=..., ...), viewer=None, num_t=None)` is the construction shape. The cascade may mutate `Markers.device` / `Markers.low_memory` runtime state; `Markers.config` preserves the original intent.

## Gotchas

- **`use_im` switches between LoG-of-distance (default) and LoG-of-Frangi.** Different rationales — distance is shape-agnostic; Frangi is structure-aware. Default works for the paper's data.
- **NMS is morphological max-filter** (window `2 * peak_min_distance + 1`), not KD-tree. Chosen for GPU-friendliness, not theoretical purity.
- **Distance is clamped to `2 * max_radius_px`.** Mimics the legacy KD-tree's behavior for infinities; if you remove the clamp, downstream LoG can pick up runaway peaks.
- **Empty mask short-circuits to all-zero outputs.** No error.
- **Low-memory chunked LoG/NMS uses `_log_halo` (= `truncate * sigma_max`) and `_nms_halo`** to keep results identical to the unchunked path. `test_low_memory_matches_full_2d` / `_3d` pin this equivalence.
- **Multi-scale reduction is per-pixel "best response wins", with scales streamed.** No 4D `(scale, z, y, x)` array; per sigma the scale-normalized `-LoG · σ²` is computed and a running best-response mask is updated in place. Dropping the `σ²` factor breaks cross-scale comparison (small scales dominate); stacking instead of streaming will OOM on real volumes.
- **OOM handling is the outer [[gpu-runtime|`adaptive_run.mode_candidates`]] cascade only.** Slice 3 of #84 deleted the per-frame inner cascade and the cross-frame state-mutation it caused. On OOM the whole stage now restarts from frame 0 with the next `(device, low_memory)` candidate — partial progress on long time series is lost in exchange for predictable per-call state.
- **`prefer_gpu` was dropped in Slice 2 of #84** (matching Network/Label/Filter); pass `device="cpu"` instead. The constructor now normalizes `device` through `adaptive_run.normalize_device` so `"cuda"` is accepted as a synonym for `"gpu"`. The constructor also indexes `dim_res['X']` / `dim_res['Z']` directly — a missing X (or Z, when the image has Z) now raises `KeyError` instead of silently falling back to a 1.0 µm scale.

## Invariants

- `im_marker`: uint8 binary (1 at peak voxels).
- `im_distance`: float32, ≥0.
- `im_border`: uint8 binary; **`border ∩ mask = ∅`** (`test_border_is_outside_mask` pins this).
