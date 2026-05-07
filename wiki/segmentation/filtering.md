---
created: 2026-05-06
modified: 2026-05-07
---

# Filtering

Multi-scale Frangi vesselness on raw intensities. For 2D, additionally fuses a multi-scale LoG ("blobness") response. Output: `im_preprocessed`, float32, ≥0 everywhere.

## Why

Tubular organelles (mitochondria, ER tubules) are low-contrast and varied in radius. Frangi uses Hessian eigenvalue ratios (`Ra`, `Rb`, `S`) to suppress plates and blobs and reward elongated darker-than-background structures across scales. Cascaded Gaussians widen scale incrementally rather than recomputing from raw — much cheaper for stacks of σ values.

## Interactions

- Input: any [[im-info|ImInfo]] raw memmap.
- Output `im_preprocessed` consumed by [[labelling]], [[networking]], [[mocap-marking]] (when `use_im='frangi'`), and [[feature-extraction]] (as the "structure image").
- Calls `triangle_threshold` / `otsu_threshold` from [[gpu-runtime]].
- Algorithm config is bundled in a `FrangiConfig` frozen dataclass colocated in `filtering.py`. `Filter(im_info, FrangiConfig(alpha_sq=0.3, ...), viewer=None, num_t=None)` is the construction shape. The cascade may mutate `Filter.device` / `Filter.low_memory` runtime state; `Filter.config` preserves the original intent.
- Pure math primitives (Frangi formula, Hessian, multi-scale LoG, γ estimation) live in `nellie/segmentation/frangi_math.py` — Filter is the thin caller that holds `xp`, `ndi`, `alpha_sq`, `beta_sq`, etc. and passes them through.
- Chunking primitives (`iter_chunks`, `compute_chunk_shape`, `safe_eigvalsh`, `subsample_for_thresholds`) live in `nellie/utils/chunking.py` — stage-agnostic, default `is_oom` predicate handles both NumPy and CuPy errors.
- Backend resolution (`resolve_backend`, `try_import_cupy`, `free_gpu_memory`, `is_oom_error`) lives in [[gpu-runtime|`adaptive_run`]] — Filter's `_set_backend` / `_set_low_memory` / `_switch_to_cpu` are thin mutator wrappers required by the cascade contract.

## Gotchas

- **`gamma` is auto-rescaled by `spacing_geomean ** 2`.** Hessian eigenvalues live in `intensity / spacing²` units while triangle/Otsu thresholds run on raw intensity. Without the rescale the `(1 - exp(-S² / γ²))` term saturates and magnitudes blow up. This is a deliberate fix; do not remove.
- **Cascaded Gaussian must `copy=True` from the memmap.** In-place writes would corrupt the on-disk OME-TIFF.
- **Frangi only keeps voxels where `λ₂ ≤ 0`** (and `λ₃ ≤ 0` in 3D). This bakes in a **bright-on-dark assumption** — dark-on-bright structures are silently zeroed.
- **`remove_edges` zeroes a 15-px border around the bbox per slice.** Tunable but easy to forget.
- **`alpha`, `beta`, min/max radius are dataset-sensitive.** No auto-tuning; defaults work for the paper's mitochondrial data.

## Invariants

- Output shape matches input shape.
- Output is float32, ≥0, NaN/Inf scrubbed.
- Input memmap is never mutated.
