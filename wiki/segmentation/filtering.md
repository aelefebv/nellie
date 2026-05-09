---
created: 2026-05-06
modified: 2026-05-09
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
- Chunking primitives (`iter_chunks`, `compute_chunk_shape`, `safe_eigvalsh`, `eigvalsh_3x3_symmetric`, `eigvalsh_3x3_components`, `subsample_for_thresholds`) live in `nellie/utils/chunking.py` — stage-agnostic, default `is_oom` predicate handles both NumPy and CuPy errors. Filter's 3D path uses `eigvalsh_3x3_components` directly; see [[filtering#performance|Performance]].
- Backend resolution (`resolve_backend`, `try_import_cupy`, `free_gpu_memory`, `is_oom_error`) lives in [[gpu-runtime|`adaptive_run`]] — Filter's `_set_backend` / `_set_low_memory` / `_switch_to_cpu` are thin mutator wrappers required by the cascade contract.

## Gotchas

- **`gamma` is auto-rescaled by `spacing_geomean ** 2`.** Hessian eigenvalues live in `intensity / spacing²` units while triangle/Otsu thresholds run on raw intensity. Without the rescale the `(1 - exp(-S² / γ²))` term saturates and magnitudes blow up. This is a deliberate fix; do not remove.
- **Cascaded Gaussian must `copy=True` from the memmap.** In-place writes would corrupt the on-disk OME-TIFF.
- **Frangi only keeps voxels where `λ₂ ≤ 0`** (and `λ₃ ≤ 0` in 3D). This bakes in a **bright-on-dark assumption** — dark-on-bright structures are silently zeroed.
- **`remove_edges` zeroes a 15-px border around the bbox per slice.** Originally added for **snouty** (oblique light-sheet) data — deskew warps the boundaries and Frangi latches onto the interpolation artifacts without it. The 15-px margin is tuned for that artifact width; for non-snouty data the side effect is "small structures (bbox height ≤ 30) get fully wiped." Default is `False` so most users never hit this.
- **`alpha`, `beta`, min/max radius are dataset-sensitive.** No auto-tuning; defaults work for the paper's mitochondrial data.

## Invariants

- Output shape matches input shape.
- Output is float32, ≥0, NaN/Inf scrubbed.
- Input memmap is never mutated.

## Performance

The hot loop is per-scale Hessian eigenvalue extraction; secondary costs are mask construction and per-scale reductions. Several optimizations have landed; collectively a 3D run is ~2.4× faster than before.

- **Closed-form 3×3 symmetric eigenvalues for the 3D path.** Replaces the per-chunk LAPACK `eigvalsh` (`chunking.safe_eigvalsh`) with Smith's 1961 trigonometric formula. Lives in `chunking.eigvalsh_3x3_symmetric` (tensor input) and `chunking.eigvalsh_3x3_components` (1D-components input — the variant Filter calls, since it has the components on hand and skips the `(N, 3, 3)` materialization). On the 3D yeast fixture: ~10× faster than LAPACK on the isolated eigenvalue call (320 ms → 31 ms), ~2.4× end-to-end. Numerical contract preserved: returns `(N, 3)` sorted by absolute value ascending, matching `safe_eigvalsh`. `safe_eigvalsh` itself stays in [[gpu-runtime|chunking.py]] as the gold-standard reference (used by tests and as a fallback for general N×N symmetric inputs); Filter no longer wraps it.
  - Edge cases: `arccos(clip(r, -1, 1))` guards against float32 rounding pushing the discriminant outside [-1, 1]; a degenerate-case branch handles scalar-multiple-of-identity inputs (where `p = 0` would otherwise divide-by-zero).
- **Dense `h_mask` fast path.** `_compute_vesselness_chunkwise` is now a dispatcher: when the Frobenius mask covers every voxel (`bool(h_mask.all())`), it routes to `_compute_vesselness_dense`, which iterates contiguous flat slices of the components — no `xp.where` coordinate materialization, no per-chunk fancy indexing, no final scatter. Sparse path (`_compute_vesselness_sparse`, the original logic) retained for partial masks. The all-True check costs one O(N) reduction; the savings dominate. On the 3D fixture (post-eigvalsh swap): dense ~43 ms vs sparse ~64 ms.
- **In-place reductions.** `xp.maximum(vesselness, scale, out=vesselness)` and `vesselness *= masks` replace `vesselness = ...` allocations across `_compute_vesselness`, `_run_frame`, and `_run_frame_chunked`. Saves one full-volume / per-chunk allocation per scale.
- **Inf handling in `_get_frob_mask` no longer copies the volume.** Infs would break `triangle_threshold` / `otsu_threshold` (both call `xp.histogram(range=(min, max))`). The original code copied `frobenius_norm` and replaced infs with `max_finite`; now it just excludes infs from the threshold input — the final `> thresh` comparison naturally admits them (inf > finite is True). Pathological all-inf case explicitly returns an all-False mask to match prior behavior.
- **CuPy backend probe is module-level cached.** `_backend_for_array` was a per-call try/except + import; now `_CUPY_BACKEND = _probe_cupy_backend()` at import time, dispatch is a single `isinstance` against the cached `cupy.ndarray` type. Hot in `_run_filter` (per frame), `_mask_volume`, and `_bbox`.
- **Intentional: full-frame on-device LoG allocation in the 2D chunked path.** `_run_frame_chunked` allocates `frame_xp = self.xp.asarray(frame_cpu)` for LoG fusion *after* the Hessian work has gone chunked, which looks like a regression on the low-memory contract. It's deliberate — per-chunk LoG would require global normalization across chunks (LoG response is scale-fused via `xp.maximum`, then divided by global max). For 2D it's one Y×X plane, small relative to the chunked Hessian working set we just freed. Comment-pinned in code.

### Benchmarks

`tests/test_filtering_perf.py` carries opt-in microbenchmarks under the `benchmark` pytest marker (deselected by default; run with `pytest -m benchmark`). Three checks pin the wins:

- `test_dense_path_at_least_as_fast_as_sparse[2d|3d]` — relative timing assertion (1.2× margin) so a regression where dense becomes slower than sparse on a fully-true mask fails the suite.
- `test_h_mask_all_check_is_cheap_relative_to_dispatch[2d|3d]` — pins the dispatcher's `h_mask.all()` reduction at <10% of the dense path on CPU. (Note: would force a D→H sync on GPU; revisit if profiling shows it matters.)
- `test_closed_form_3d_eigvalsh_beats_lapack` — asserts ≥1.5× speedup of `eigvalsh_3x3_components` over `safe_eigvalsh` on the full 3D Hessian. Currently 10×.

Plus `test_filter_run_baseline_prints_wall_clock[2d|3d]` for an informational end-to-end timing baseline (no assertion).
