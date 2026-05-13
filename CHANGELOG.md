# Changelog
All notable changes to this project will be documented in this file.
The format is based on Keep a Changelog and this project adheres to Semantic Versioning.

## [Unreleased]
### Changed (breaking)
- **napari moved to optional `gui` extra.** `pip install nellie` and `uv sync` now install a headless install — usable on Linux servers without libEGL/libgl/Qt. GUI users opt in via `pip install 'nellie[gui]'` or `uv sync --extra gui`. The headless `nellie` package was already napari-free; only `nellie_napari/` and `main.py` need it. The `napari.manifest` plugin entry-point is unchanged.

### Added
- **Apple Silicon / MPS GPU acceleration via PyTorch** (PRD [#140](https://github.com/aelefebv/nellie/issues/140)). Mac researchers opt in via `pip install 'nellie[mps]'`; pass `device="mps"` (explicit pin) or `device="gpu"` (platform-aware on Darwin) to any pipeline-stage constructor. Four stages onboarded in v1: filtering ([PR #147](https://github.com/aelefebv/nellie/pull/147)), labelling ([PR #148](https://github.com/aelefebv/nellie/pull/148)), networking ([PR #149](https://github.com/aelefebv/nellie/pull/149)), hu_tracking. Convolutional ndi ops dispatch through `torch.nn.functional.conv*`; structural ops (`binary_fill_holes`, `label`, skeletonization) round-trip to scipy on CPU since unified-memory transfer cost on Apple Silicon is negligible. Stages not onboarded for v1 (`hierarchical`, `voxel_reassignment`, `mocap_marking`) handle a Mac user passing `device="mps"` by falling back to CPU via the cascade. Float64 silently coerces to float32 with a one-time log notice — MPS does not support double precision; documented determinism risk specific to hu_tracking's moment-distance matrix.
- `nellie.utils.adaptive_run.to_numpy(arr)` — backend-agnostic host-transfer helper. Handles numpy (identity), cupy (`.get()`), and torch (`.detach().cpu().numpy()`) uniformly. Replaces the previous duck-typed `hasattr(arr, "get")` pattern in hu_tracking call sites that needed to land a host numpy array (per PRD #140 § Implementation Decisions: polluting torch tensors with a cupy idiom would be hostile to torch users).
- Repo wiki under `wiki/` — bootstrapped articles for every algorithmic stage, glossary, queue of open architectural questions, and a `now.md` snapshot of in-flight work.
- Per-stage frozen `*Config` dataclasses (`FrangiConfig`, `LabelConfig`, `NetworkConfig`, `MarkersConfig`, `HuMomentTrackingConfig`, `VoxelReassignerConfig`, `HierarchyConfig`) with `__post_init__` validation — bad device strings, allowed-set strings, non-positive numerics, and inverted radius ranges all raise at construction time.
- `nellie.im_info.load_image(path)` — canonical one-call construct-and-load entry point used by `run.py`, tests, and demos.
- Pytest characterization suites for every algorithmic stage (filtering, labelling, networking, mocap_marking, hu_tracking, voxel_reassignment, hierarchical) plus verifier and extractors. Suite size grew from 0 (legacy tests removed) to 410 tests.
- `nellie/im_info/extractors/` subpackage — `MetadataExtractor` Protocol + 5 per-format implementations (OME, ImageJ, ImageJ-tif-tags, ND2, raw-TIFF) + `detect_extractor(filepath)` factory. Replaces the 5-way string-dispatch into per-format `_get_*_metadata` methods and unblocks bioio integration.
- Filter perf pass:
    - Closed-form 3×3 symmetric eigenvalues (`chunking.eigvalsh_3x3_symmetric` / `eigvalsh_3x3_components`) replacing LAPACK `eigvalsh` in the 3D path. ~10× faster on the isolated math; ~2.4× faster end-to-end on the 3D yeast fixture (CPU).
    - Dense `h_mask` fast path in `_compute_vesselness_chunkwise` — skips `xp.where` + per-chunk fancy indexing + scatter when the Frobenius mask covers every voxel.
    - Module-level cached cupy backend probe (`_backend_for_array` was per-call try/except + import).
    - In-place reductions (`xp.maximum(out=)`, `*=`) across `_compute_vesselness`, `_run_frame`, and `_run_frame_chunked`.
    - `_get_frob_mask` no longer copies the volume to mask infs — excludes them from threshold input only.
- Opt-in `benchmark` pytest marker (registered in `pyproject.toml`, deselected by default) and `tests/test_filtering_perf.py` carrying 7 microbenchmarks with relative-comparison assertions.
- `nellie/utils/chunking.py` and `nellie/utils/adaptive_run.py` — stage-agnostic chunking primitives and runtime device/memory arbitrator extracted from the old monolithic `Filter`.
- `nellie/segmentation/frangi_math.py` — pure Frangi/Hessian/LoG/γ math primitives extracted from `Filter`.

### Changed
- Filter: 3D `Filter.run()` ~2.4× faster on CPU (yeast fixture) — see Added entry above.
- All 7 algorithmic stages share unified backend resolution via `adaptive_run` — per-stage `_resolve_backend` / `_try_import_cupy` / `_is_oom_error` / `_free_gpu_memory` / `_switch_to_cpu` duplicates deleted (~5 methods × 7 stages consolidated).
- Verifier (`nellie/im_info/verifier.py`) untangled into 8 modular slices: characterization tests → clarify pass → constructor I/O extraction (`from_file_info`) → unified axes normalizer (`infer_t_axis` + `transform_to_axes`) → validation split (`compute_errors` + `apply_defaults`, no more asymmetric raise) → extractor strategy → OME-TIFF writer extraction (`_write_ome_tiff`) → `load_image()` orchestrator. Pyright errors on `verifier.py`: 93 → 38 (−59%).
- Stage construction shape unified across all stages: `Stage(im_info, config, viewer, num_t)`. `Stage.config` preserves the original intent across cascade-mutated runtime state (`device`, `low_memory`).
- napari processor / settings widget seam: all 7 `get_*_params` methods return typed Configs (6 return `tuple[Config, int | None]`, `get_feature_params` returns just `HierarchyConfig`); processor `_run_*` methods take Configs directly. The dict→pop→`Config(**kwargs)` triplet is gone.
- `ImInfo` constructor is now thin (no I/O); use `ImInfo.from_file_info(file_info)` for the canonical construct-and-load entry point.
- Verifier `change_axes` length gate restored (originally added in `bb2b0b7`, disabled by `492edfb` for napari fileselect partial-typing UX — but napari pre-validates, so the gate was redundant for napari and missing for programmatic callers).
- HuMomentTracking `cost_cutoff` lifted from module constant to `HuMomentTrackingConfig.cost_cutoff` field.
- Markers `prefer_gpu`, Hierarchical / mocap_marking `use_gpu` constructor params dropped — folded into the unified `device` arg.

### Fixed
- Filter OOM-fallback `NameError` when the per-frame path raised on first invocation.
- Filter low-memory chunked path silently dropped 2D LoG blobness fusion — now applies the same fusion as the non-chunked path.
- VoxelReassigner GPU OOM cascade silently swallowed non-OOM exceptions at sites 3 & 4 — now explicitly re-raises.
- Hierarchical GPU OOM handler now catches the broader OOM family via `adaptive_run.is_oom_error` (was: narrow `cp.cuda.memory.OutOfMemoryError`) and propagates non-OOM exceptions instead of silently swallowing.

### Removed
- `nellie/cli.py` — broken since `run.py`'s signature change; no console-script entry point referenced it.
- `nellie/run copy.py` — confirmed orphan, no importers.
- `Filter._safe_eigvalsh` (replaced by direct closed-form call), `Filter._compute_chunkwise_eigenvalues` (dead), `Filter._bbox` 3D path (only ever called on 2D slices).
- Verifier polymorphic `self.metadata` field + 6 per-format `_find_*` / `_get_*_metadata` methods (subsumed by the extractor strategy).
- Network `_clean_junctions`, `_local_max_peak`, LoG/sigma machinery (`_set_default_sigmas`, `_get_sigma_vec`, `self.sigmas`), dead `_get_t`, `force_cpu` parameter on `_remove_connected_label_pixels`.
- Hierarchical `_cupy_available`, `_resolve_device`, module-level `try: import cupy as cp` block (consolidated into `adaptive_run`).

## [1.0.0] - 01/12/2026
### Added
- A changelog
- Major speed ups across the board for segmentation and tracking (all tests performed on CUDA / Windows):
    - **GPU**
        - Filter: 9x faster
        - Label: 7x faster
        - Network: 14x faster
        - Markers: 10x faster 
        - HuMomentTracking: 6x faster 
        - VoxelReassigner: 2x faster 
        - Hierarchy: no change
        - Total: 3.5x faster 
    - **CPU**
        - Filter: 1.5x faster 
        - Label: 1.5x faster
        - Network: 11x faster
        - Markers: no change
        - HuMomentTracking: no change
        - VoxelReassigner: 1.5x faster
        - Hierarchy: no change
        - Total: 1.5x faster 
- UV compatibility
- Added a dropdown for which stat to visualize for the feature of interest in the analysis plugin tab
- Added a tab for advanced settings to tweak specific parameters for each step of the pipeline.
### Changed
- Docs to mkdocs format
- Defaulted logger to INFO level
### Fixed
- Lots of little things in the napari GUI (e.g. buttons turning on and off, things like that)
- Removed some annoying test strings
- Track visualization to only tracks that end up at a valid mask pixel
- Version verification
- Discover nellie entrypoint plugins when importlib is outdated.
