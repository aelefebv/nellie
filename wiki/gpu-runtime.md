---
created: 2026-05-06
modified: 2026-05-07
---

# GPU / runtime

Thin compatibility layer providing CPU/GPU-agnostic thresholding primitives, runtime device negotiation with OOM/availability fallback, and package-wide logging. Lives in `nellie/utils/`.

## Two device-detection systems coexist

1. **Module-level static detect** in `nellie/__init__.py`. On import: tries `cupy` / `cupyx.scipy.ndimage` as `xp` / `ndi`, falls back to `numpy` / `scipy.ndimage`. Sets `device_type ∈ {cuda, cpu}` and `is_gpu`. **macOS is hard-pinned to CPU** (the MPS / `torch_xp` block is commented out).
2. **Per-component runtime resolve** via `adaptive_run.py`. Each [[pipeline|pipeline]] stage accepts `device="auto"|"cpu"|"gpu"`, calls `normalize_device` → `gpu_available` (probes `cupy.cuda.runtime.getDeviceCount()`), then resolves its own `xp` / `ndi` backend.

These can disagree. Importing `nellie` on a CUDA-less Linux box still attempts the cupy import. The pipeline stages then independently re-resolve at construction time.

**Library**: CuPy only (no torch in the active path). The `xp = cupy or numpy` swap is the dispatch contract.

## Adaptive run

Not a chunking primitive — a **device/memory mode arbitrator**. It exposes:

- Memory probes (`get_gpu_free_bytes` via `cupy.cuda.runtime.memGetInfo`; `get_cpu_available_bytes` via `psutil` then `os.sysconf` fallback).
- A peak-memory heuristic (`should_use_low_memory`): peak ≈ frame_bytes × **6.0**, refusing if peak > free × **0.7** headroom.
- `mode_candidates(device_order, start_low_memory)` → ordered `(device, low_memory)` retry plan, e.g. `[(gpu, False), (gpu, True), (cpu, False), (cpu, True)]`.
- OOM/unavailable classifiers: `is_oom_error` matches `MemoryError`, `cupy.cuda.memory.OutOfMemoryError`, and **string-sniffs** `"out of memory"`. `is_gpu_unavailable_error` catches `cupy` ImportError and message patterns.
- Backend resolution: `resolve_backend(device)` returns `(xp, ndi, device_type)` for `"auto" | "cpu" | "gpu" | "cuda"`. `try_import_cupy(require=False)` returns `(cupy, cupy_ndi)` or `(None, None)`. `free_gpu_memory(xp)` is a no-op on NumPy.

Solves: pipelines don't crash hard on OOM or missing CUDA — they cascade through device/memory modes until something fits.

## Logger

`base_logger` calls `logging.basicConfig` at import time (root-level mutation — first import wins) with a verbose timestamped format. Demotes the noisy `xmlschema` logger to WARNING. The module aliases `logger = logging` (not a named logger), so callers get the root logger.

## Consumers

- **`adaptive_run`**: [[feature-extraction|`hierarchical`]], [[labelling]], [[filtering]], [[networking]], [[mocap-marking]], [[hu-tracking]], [[voxel-reassignment]].
- **`gpu_functions`** (`otsu_threshold`, `triangle_threshold`): [[labelling]], [[filtering]] (re-exported via `utils/__init__.py`).

## Gotchas

- **macOS = CPU, full stop.** The MPS branch is commented out; Mac users get no GPU even if `torch.backends.mps.is_available()`. Don't promise users Apple Silicon acceleration.
- **String-sniffing OOM** (`"out of memory" in repr(exc).lower()`) is fragile across CuPy/CUDA driver versions — new error wordings silently break the fallback.
- **Silent CPU fallback masks GPU bugs.** A cupy import error or missing CUDA device drops you to numpy with no exception, only a (commented-out) log line. Performance regressions look like "it just got slow."
- **6.0× peak multiplier and 0.7 headroom are magic constants** — tuned empirically, not per-stage.
- **`logger = logging`** means `from nellie.utils import logger` aliases the root logger module; `logger.basicConfig` runs as an import side effect.
- **Determinism**: results may differ subtly between cupy/numpy histograms (binning edge cases) — `otsu` / `triangle` threshold values are not byte-identical across backends.

## Invariants

- Anything in `gpu_functions` must accept either a numpy or cupy array; backend is auto-detected via `_get_xp(matrix)` (or explicit `xp=` override).
- Any consumer of `adaptive_run` must accept `device` and `low_memory` constructor kwargs and expose `_set_device` / `_set_low_memory` mutators (so the retry loop can re-resolve).
- Pipeline stages must raise OOM/unavailable errors with **recognizable signatures** — swallowing them defeats the cascade.
- `_get_xp` always returns a usable namespace (numpy minimum) — never `None`.
