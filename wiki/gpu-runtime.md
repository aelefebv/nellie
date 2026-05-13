---
created: 2026-05-06
modified: 2026-05-12
---

# GPU / runtime

Thin compatibility layer providing CPU/GPU-agnostic thresholding primitives, runtime device negotiation with OOM/availability fallback, and package-wide logging. Lives in `nellie/utils/`.

Three backends now coexist behind one dispatch contract: **NumPy/SciPy** (CPU), **CuPy/cupyx.scipy.ndimage** (CUDA), and **PyTorch + MPS** (Apple Silicon, opt-in via `pip install 'nellie[mps]'`). Per-stage device negotiation arbitrates which one each pipeline stage uses; cascade behavior catches OOM and falls back accordingly.

## Two device-detection systems coexist

1. **Module-level static detect** in `nellie/__init__.py`. On import: tries `cupy` / `cupyx.scipy.ndimage` as `xp` / `ndi`, falls back to `numpy` / `scipy.ndimage`. Sets `device_type ∈ {cuda, cpu}` and `is_gpu`. **macOS stays pinned to NumPy here** by design — the previously-commented-out static MPS block was removed in slice 1 of [[decisions/0002-adaptive-run-extension-over-static-torch-xp]] because the per-stage path is now canonical.
2. **Per-component runtime resolve** via `adaptive_run.py`. Each [[pipeline|pipeline]] stage accepts `device="auto"|"cpu"|"gpu"|"cuda"|"mps"`, calls `normalize_device` → `device_cascade`, then resolves its own `(xp, ndi, device_type)` backend triple via `resolve_backend`.

These can disagree. Importing `nellie` on a CUDA-less Linux box still attempts the cupy import. Importing it on a Mac with `nellie[mps]` installed still gets numpy from the static path — MPS only flows through if a stage explicitly requests it (or `device="auto"` resolves to it via the cascade). The pipeline stages independently re-resolve at construction time and that's the resolution that matters.

**Libraries**: CuPy for CUDA, PyTorch (`nellie/utils/torch_xp.py` + `nellie/utils/torch_ndi.py`) for MPS, NumPy + SciPy for CPU. The `xp = numpy | cupy | torch_xp` (and `ndi = scipy.ndimage | cupyx.scipy.ndimage | torch_ndi`) swap is the dispatch contract.

## Adaptive run

Not a chunking primitive — a **device/memory mode arbitrator**. It exposes:

- **Backend resolution**: `resolve_backend(device)` returns `(xp, ndi, device_type)` for `"auto" | "cpu" | "gpu" | "cuda" | "mps"`. `device="gpu"` is platform-aware: returns torch+MPS on Darwin (if torch available), cupy on Linux/Windows (if available), CPU otherwise. `try_import_cupy(require=False)` returns `(cupy, cupy_ndi)` or `(None, None)`. `try_import_torch_mps(require=False)` mirrors it for torch.
- **Cascade construction**: `device_cascade(device) → list[str]` — single-source ordering for the OOM/unavailability retry loop. Returns `["mps", "cpu"]` on Mac with torch+MPS, `["gpu", "cpu"]` on Linux/Windows with cupy, `["cpu"]` otherwise. Honors explicit `"mps"` / `"cuda"` to pin a single backend (no fallback). All cascade-using stages call this — adding a new backend is now a one-line change to `device_cascade`.
- **Memory probes**: `get_gpu_free_bytes` via `cupy.cuda.runtime.memGetInfo`; `get_mps_free_bytes` via `get_cpu_available_bytes() - torch.mps.driver_allocated_memory()`; `get_cpu_available_bytes` via `psutil` then `os.sysconf` fallback.
- **Peak-memory heuristic** (`should_use_low_memory`): peak ≈ frame_bytes × **6.0**, refuses if peak > free × headroom. **Headroom is device-aware**: 0.7 for CUDA's discrete VRAM, 0.5 for MPS's system-shared RAM (must leave room for the OS, IDE, browser).
- **Mode candidates** (`mode_candidates(device_order, start_low_memory)`): produces ordered `(device, low_memory)` retry plan, e.g. `[(mps, False), (mps, True), (cpu, False), (cpu, True)]` on Mac, `[(gpu, False), (gpu, True), (cpu, False), (cpu, True)]` on Linux+CUDA.
- **OOM/unavailable classifiers**: `is_oom_error` matches `MemoryError`, `cupy.cuda.memory.OutOfMemoryError`, MPS's `RuntimeError("MPS backend out of memory ...")` substring, and **string-sniffs** `"out of memory"`. `is_gpu_unavailable_error` catches `cupy` ImportError, MPS-not-built / no-MPS-device patterns, **and shim `NotImplementedError("torch_xp/torch_ndi: ... not implemented")`** so that non-MPS-onboarded stages cascade to CPU when a Mac user passes `device="mps"` to them.
- **Backend-agnostic host transfer**: `to_numpy(arr)` returns a numpy view of a numpy / cupy / torch tensor uniformly. Used by call sites that previously special-cased `arr.get()` (cupy idiom) — preferred over polluting torch tensors with a `.get()` method.
- **Pool cleanup**: `free_gpu_memory(xp)` is a no-op on NumPy, calls cupy's pool cleanup on CuPy, calls `torch.mps.empty_cache()` on MPS.

Solves: pipelines don't crash hard on OOM or missing GPU — they cascade through device/memory modes until something fits.

## MPS path (PyTorch on Apple Silicon)

Opt-in via `pip install 'nellie[mps]'` (the extra is `sys_platform == 'darwin'`-constrained because MPS doesn't exist elsewhere). When active, four pipeline stages dispatch to torch on the MPS device:

- [[segmentation/filtering|filtering]] — full MPS path including the inner Frangi sigma loop (`gaussian_filter` is the dominant op)
- labelling — convolutional `uniform_filter` runs on MPS; structural ops (`binary_fill_holes`, `label`) round-trip to scipy on CPU per-frame
- networking — `convolve`, `maximum_filter`, `minimum_filter` on MPS; per-frame `label` on CPU
- hu_tracking — per-frame `maximum_filter` on MPS

Other stages (`hierarchical`, `voxel_reassignment`, `mocap_marking`) handle `device="mps"` gracefully by cascading to CPU; they're not MPS-onboarded in v1 but don't crash if a user picks the device.

The two new shim modules are in `nellie/utils/`:

- `torch_xp.py` — numpy-API shim (~40 ops covering the union of what the four onboarded stages touch). Lazy torch import. Float64 in shim ops silently coerces to float32 with a one-time module-load log message ("MPS backend: float64 → float32 coercion is in effect; results may differ from CPU within float32 tolerance"). Unimplemented ops raise `NotImplementedError("torch_xp: <op> not implemented; this stage isn't MPS-onboarded yet")` — the cascade catches that as `is_gpu_unavailable_error` and falls back to CPU. Patches `torch.Tensor` with `.astype()`, `.get()`, `.copy()` methods (cupy/numpy duck-typing fix). `_LazyDtype` proxies allow `xp.float32(3.0)` to construct a 0-D tensor.
- `torch_ndi.py` — scipy.ndimage-API shim. Convolutional ops dispatch via `torch.nn.functional.conv*` and `*_pool*`. Structural ops (`binary_opening`, `binary_fill_holes`, `label`) round-trip to scipy on CPU. **scipy "reflect" semantics** (half-sample symmetric: boundary value duplicated) are implemented via a custom `_scipy_reflect_pad_along_axis` helper that uses `torch.index_select` — torch's `pad(mode='reflect')` matches scipy "mirror", *not* scipy "reflect", and the difference is significant for filter-driven segmentation. `gaussian_laplace` uses scipy's algorithm exactly (per-axis convolve with second-derivative-of-Gaussian kernel) rather than the naive "smooth then finite-diff" — the two diverge at typical sigmas.

The MPS path is **opt-in for tests** via the `mps` pytest marker (registered alongside `benchmark`, deselected by default). Maintainers run `pytest -m mps` locally on a Mac before tagging an MPS-affecting release — there are no Apple Silicon CI runners. The `mps`-marker tests cover smoke (does the stage run end-to-end?), equivalence (within-tolerance vs CPU on a small fixture), and benchmark (perf measurement). See [[decisions/0001-pytorch-mps-over-mlx]] for the library-choice rationale, [[decisions/0002-adaptive-run-extension-over-static-torch-xp]] for the dispatch architecture, [[decisions/0003-device-gpu-platform-aware]] for the API surface.

## Logger

`base_logger` calls `logging.basicConfig` at import time (root-level mutation — first import wins) with a verbose timestamped format. Demotes the noisy `xmlschema` logger to WARNING. The module aliases `logger = logging` (not a named logger), so callers get the root logger.

## Consumers

- **`adaptive_run`**: [[feature-extraction|`hierarchical`]], [[labelling]], [[filtering]], [[networking]], `mocap-marking`, [[hu-tracking]], [[voxel-reassignment]].
- **`gpu_functions`** (`otsu_threshold`, `triangle_threshold`): [[labelling]], [[filtering]] (re-exported via `utils/__init__.py`).
- **MPS-onboarded** (additional torch dispatch): [[filtering]], [[labelling]], [[networking]], [[hu-tracking]].

## Gotchas

- **Apple Silicon: MPS is opt-in.** `pip install 'nellie[mps]'` adds the `torch` extra. Without it, Mac users still get NumPy on every stage even if `torch.backends.mps.is_available()` would have returned True. The extra is `sys_platform == 'darwin'`-constrained — `pip install 'nellie[mps]'` on Linux is a no-op.
- **MPS = float32 only.** `torch_xp` silently coerces float64 to float32 with a one-time log message. Affects accuracy for stages that explicitly cast to float64 (notably `hu_tracking`'s moment-distance matrix at `hu_tracking.py:762-765`); equivalence tests use looser tolerances there. For byte-exact reference results, use `device="cpu"`.
- **MPS structural ndimage ops round-trip to CPU.** `binary_fill_holes`, `binary_opening`, `label` have no MPS equivalent. The shim moves data to CPU, runs scipy, moves back. Cost is small on unified memory but per-frame round-trips can dominate when the frame is small enough that the convolutional MPS work doesn't amortize.
- **MPS speedup is fixture-size-dependent.** On small yeast fixtures: filtering wins clearly on MPS (the inner Frangi sigma loop is the dominant cost); labelling and hu_tracking can be slower on MPS than CPU due to dispatch overhead vs the small per-frame work; networking is roughly break-even. Production-sized volumes shift the balance toward MPS — don't generalize from microbenchmarks.
- **String-sniffing OOM** (`"out of memory" in repr(exc).lower()`) is fragile across CuPy/CUDA driver versions AND torch MPS releases — new error wordings silently break the fallback. Both backends now contribute to this gotcha.
- **Silent CPU fallback masks GPU bugs.** A cupy import error or missing CUDA device drops you to numpy; missing torch on Mac drops MPS to numpy; a shim `NotImplementedError` cascades to CPU. All silent. Performance regressions look like "it just got slow" — if MPS speedups vanish unexpectedly, check whether a stage you onboarded is actually flowing through `torch_xp`/`torch_ndi` or quietly degraded.
- **6.0× peak multiplier and headroom constants are magic** — tuned empirically, not per-stage. CUDA uses 0.7 headroom (discrete VRAM); MPS uses 0.5 (system-shared RAM, leaves room for the OS).
- **scipy "reflect" ≠ torch "reflect"** — half-sample vs full-sample symmetric. The `torch_ndi` shim handles this for filter ops; if you ever bypass the shim and call `torch.nn.functional.pad(mode='reflect')` directly while expecting scipy semantics, results will drift at boundaries.
- **`logger = logging`** means `from nellie.utils import logger` aliases the root logger module; `logger.basicConfig` runs as an import side effect.
- **Determinism**: results may differ subtly across all three backends — `otsu` / `triangle` threshold values are not byte-identical (binning edge cases); MPS adds a third axis of float32 drift. The `mps` pytest marker's equivalence tests are the canonical tolerance contract per stage.

## Invariants

- Anything in `gpu_functions` must accept a numpy, cupy, OR torch array; backend is auto-detected via `_get_xp(matrix)` (or explicit `xp=` override).
- Any consumer of `adaptive_run` must accept `device` and `low_memory` constructor kwargs and expose `_set_backend` / `_set_low_memory` mutators (so the retry loop can re-resolve).
- Any cascade-using stage must call `adaptive_run.device_cascade(device)` for its retry order — never reconstruct `device_order = [...]` inline. Adding a future backend is then a one-line change.
- Pipeline stages must raise OOM/unavailable errors with **recognizable signatures** — swallowing them defeats the cascade.
- `_get_xp` always returns a usable namespace (numpy minimum) — never `None`.
- Stage-side host transfers must use `adaptive_run.to_numpy(arr)` rather than `arr.get()` — preserves backend-agnostic ergonomics for torch users.
- Cuda-only branches in stage code (`if device_type == "cuda": ...`) must widen to `("cuda", "mps")` if they handle device-resident tensors that need the same treatment on MPS.
