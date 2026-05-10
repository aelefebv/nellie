"""
Helpers for adaptive device/memory selection and fallback retries.
"""
from __future__ import annotations

import math
import os
import platform
from typing import Any


_ESTIMATED_PEAK_MULTIPLIER = 6.0
# Per-device headroom: CUDA gets 0.7 of discrete VRAM; MPS gets 0.5 of
# system-shared RAM (must leave room for the OS, IDE, browser). CPU
# follows the CUDA value historically; this is the public contract for
# the free-memory cascade.
_MEMORY_HEADROOM_BY_DEVICE = {
    "cuda": 0.7,
    "mps": 0.5,
    "cpu": 0.7,
}
# Backwards-compatible alias for callers (and tests) that read the legacy
# scalar. Resolves to the CUDA/CPU value, which matches prior behavior.
_MEMORY_HEADROOM = _MEMORY_HEADROOM_BY_DEVICE["cuda"]


def _headroom(device_type: str) -> float:
    return _MEMORY_HEADROOM_BY_DEVICE.get(device_type, _MEMORY_HEADROOM)


def try_import_cupy(require: bool = False) -> tuple[Any | None, Any | None]:
    """Return ``(cupy, cupyx.scipy.ndimage)`` or ``(None, None)`` if unavailable.

    With ``require=True``, raises ``RuntimeError`` instead of returning
    None when CuPy is missing or no CUDA device is present.
    """
    try:
        import cupy
        import cupyx.scipy.ndimage as cupy_ndi
    except ModuleNotFoundError as exc:
        if require:
            raise RuntimeError("GPU backend requested but CuPy is not installed.") from exc
        return None, None

    try:
        device_count = cupy.cuda.runtime.getDeviceCount()
    except Exception as exc:
        if require:
            raise RuntimeError("GPU backend requested but CUDA is not available.") from exc
        return None, None

    if device_count <= 0:
        if require:
            raise RuntimeError("GPU backend requested but no CUDA devices were found.")
        return None, None

    return cupy, cupy_ndi


def try_import_torch_mps(require: bool = False) -> tuple[Any | None, Any | None]:
    """Return ``(torch_xp, torch_ndi)`` or ``(None, None)`` if MPS is unavailable.

    With ``require=True``, raises ``RuntimeError`` instead of returning
    None when torch is missing or MPS is not available on this platform
    or torch wheel.
    """
    try:
        import torch
    except ModuleNotFoundError as exc:
        if require:
            raise RuntimeError(
                "MPS backend requested but torch is not installed. "
                "Install with `pip install 'nellie[mps]'`."
            ) from exc
        return None, None

    # Probe MPS availability — torch may be built without MPS support
    # (e.g. CPU-only wheel) or the device may be unavailable on this OS.
    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None) if backends is not None else None
    is_built = bool(getattr(mps, "is_built", lambda: False)()) if mps is not None else False
    is_available = bool(getattr(mps, "is_available", lambda: False)()) if mps is not None else False
    if not is_available or not is_built:
        if require:
            raise RuntimeError(
                "MPS backend requested but no MPS device is available "
                "(torch.backends.mps.is_available() is False)."
            )
        return None, None

    # Lazy import — avoids loading torch at module import time on CUDA/CPU boxes.
    from nellie.utils import torch_ndi, torch_xp

    return torch_xp, torch_ndi


def _cpu_backend() -> tuple[Any, Any, str]:
    import numpy as np
    import scipy.ndimage as ndi

    return np, ndi, "cpu"


def _is_darwin() -> bool:
    return platform.system() == "Darwin"


def resolve_backend(device: str) -> tuple[Any, Any, str]:
    """Resolve a device string to ``(xp, ndi, device_type)``.

    Accepts ``"auto"``, ``"cpu"``, ``"gpu"``, ``"cuda"``, ``"mps"``.

    - ``"cuda"`` forces CuPy (a no-fallback explicit pin).
    - ``"mps"`` forces torch+MPS (a no-fallback explicit pin).
    - ``"gpu"`` is **platform-aware**: MPS on Darwin, CUDA elsewhere
      (per [[wiki/decisions/0003-device-gpu-platform-aware]]).
    - ``"auto"`` tries the platform's preferred GPU first, then CPU.

    The returned ``device_type`` is the canonical backend identifier
    (``"cuda"``, ``"mps"``, or ``"cpu"``) used by per-stage code.
    """
    device = (device or "auto").lower()
    if device not in ("auto", "cpu", "gpu", "cuda", "mps"):
        raise ValueError(
            f"Unsupported device '{device}'. Use 'auto', 'cpu', 'gpu', 'cuda', or 'mps'."
        )

    if device == "cpu":
        return _cpu_backend()

    if device == "cuda":
        xp, ndi = try_import_cupy(require=True)
        return xp, ndi, "cuda"

    if device == "mps":
        xp, ndi = try_import_torch_mps(require=True)
        return xp, ndi, "mps"

    if device == "gpu":
        # Platform-aware: MPS on Darwin, CUDA elsewhere.
        if _is_darwin():
            xp, ndi = try_import_torch_mps(require=True)
            return xp, ndi, "mps"
        xp, ndi = try_import_cupy(require=True)
        return xp, ndi, "cuda"

    # auto — try platform's preferred GPU, fall back to CPU.
    if _is_darwin():
        xp, ndi = try_import_torch_mps(require=False)
        if xp is not None:
            return xp, ndi, "mps"
    else:
        xp, ndi = try_import_cupy(require=False)
        if xp is not None:
            return xp, ndi, "cuda"
    return _cpu_backend()


def free_gpu_memory(xp: Any) -> None:
    """Free the active backend's memory pool. No-op on NumPy.

    Detects the backend via duck-typing rather than importing cupy or
    torch unconditionally. CuPy: ``get_default_memory_pool().free_all_blocks()``.
    Torch+MPS: ``torch.mps.empty_cache()``.
    """
    pool_fn = getattr(xp, "get_default_memory_pool", None)
    if pool_fn is not None:
        try:
            pool_fn().free_all_blocks()
        except Exception:
            pass
        return
    # Torch path: detect by module-name suffix (the torch_xp shim is a
    # module, not a class — use its name to disambiguate from numpy).
    mod_name = getattr(xp, "__name__", "")
    if "torch_xp" in mod_name:
        try:
            import torch
            empty = getattr(getattr(torch, "mps", None), "empty_cache", None)
            if empty is not None:
                empty()
        except Exception:
            pass


def to_numpy(arr: Any) -> Any:
    """Convert a backend array to a host numpy array.

    Backend-agnostic helper for the common "I have an ``xp.ndarray`` and
    I need a ``np.ndarray`` for downstream scipy / sklearn / pure-Python
    code" idiom. Handles all three backends uniformly:

    - **numpy**: identity (no copy).
    - **cupy**: ``arr.get()`` (cupy's cupy→numpy host transfer).
    - **torch**: ``arr.detach().cpu().numpy()`` (detach autograd graph,
      move from MPS/CUDA to CPU, view as numpy).

    Anything else (Python sequence, scalar, ...) falls through to
    ``np.asarray(arr)``.

    The slice 1 shim added a duck-typed ``arr.get()`` to ``torch.Tensor``
    so existing cupy-style ``.get()`` call sites would keep working, but
    per PRD #140 § Implementation Decisions polluting torch tensors with
    a cupy idiom is hostile to torch users — call sites that need a host
    numpy array should use this helper explicitly so the intent is
    discoverable. (The ``.get()`` patch on ``torch.Tensor`` remains for
    other backends-not-fully-onboarded paths; this helper is the
    preferred way for new code.)
    """
    import numpy as np

    if isinstance(arr, np.ndarray):
        return arr

    # Cupy + torch both expose ``.get`` (cupy natively; torch via the
    # slice 1 shim patch). Torch additionally needs detach + cpu before
    # ``.numpy()``; using its native API path avoids any reliance on the
    # ``.get`` patch staying in place.
    try:
        import torch

        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu().numpy()
    except ModuleNotFoundError:
        pass

    get = getattr(arr, "get", None)
    if callable(get):
        return get()

    return np.asarray(arr)


def normalize_device(device: str | None) -> str:
    """Normalize a user-supplied device string to a canonical form.

    Returns one of ``"auto"``, ``"cpu"``, ``"gpu"``, ``"mps"``. ``"cuda"``
    is normalized to ``"gpu"`` (it stays user-explicit at the outer API
    surface but we don't carry the alias inside the cascade). Unknown
    values raise ``ValueError``.
    """
    device = (device or "auto").lower()
    if device == "cuda":
        device = "gpu"
    if device not in ("auto", "cpu", "gpu", "mps"):
        raise ValueError(
            f"Unsupported device '{device}'. Use 'auto', 'cpu', 'gpu', or 'mps'."
        )
    return device


def gpu_available() -> bool:
    """Legacy CUDA-availability probe. Use :func:`device_cascade` for new code.

    Retained for stages that haven't migrated to ``device_cascade`` yet.
    """
    try:
        import cupy
    except ModuleNotFoundError:
        return False
    try:
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def mps_available() -> bool:
    """Probe whether torch+MPS is available on this machine.

    Returns False when torch is missing, MPS is not built into the wheel,
    or no MPS device is exposed.
    """
    try:
        import torch
    except ModuleNotFoundError:
        return False
    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None) if backends is not None else None
    if mps is None:
        return False
    try:
        return bool(mps.is_built()) and bool(mps.is_available())
    except Exception:
        return False


def device_cascade(device: str | None) -> list[str]:
    """Return the device retry order for the OOM/availability cascade.

    The single source of truth for "what backends should this stage try,
    in what order?" — per :doc:`/wiki/decisions/0002-adaptive-run-extension-over-static-torch-xp`.
    Adding a new backend in the future is a one-line change here, not a
    refactor across every cascade-using stage.

    Semantics:

    - ``"auto"``: platform's preferred GPU first (if available), then CPU.
      Mac with torch+MPS → ``["mps", "cpu"]``; Linux/Windows with cupy →
      ``["gpu", "cpu"]``; no GPU → ``["cpu"]``.
    - ``"gpu"``: same as ``"auto"`` (platform-aware).
    - ``"mps"``: pinned to ``["mps"]`` (no fallback — explicit user pin).
    - ``"cuda"``: pinned to ``["gpu"]`` (no fallback — explicit user pin
      for cupy).
    - ``"cpu"``: ``["cpu"]``.

    The cascade list uses the *outer* device names (``"gpu"`` rather
    than ``"cuda"``) for compatibility with the existing per-stage
    ``_set_backend`` implementations. ``"mps"`` is its own outer name
    because no other backend resolves to it.
    """
    raw = (device or "auto").lower()
    if raw == "cpu":
        return ["cpu"]
    if raw == "mps":
        return ["mps"]
    if raw == "cuda":
        # Explicit pin — the user wants CuPy on CUDA, no fallback.
        return ["gpu"]
    if raw not in ("auto", "gpu"):
        raise ValueError(
            f"device_cascade: unsupported device '{device}'. "
            f"Use 'auto', 'cpu', 'gpu', 'cuda', or 'mps'."
        )

    # 'auto' / 'gpu' — platform-aware best-effort with CPU fallback.
    if _is_darwin():
        if mps_available():
            return ["mps", "cpu"]
        return ["cpu"]
    if gpu_available():
        return ["gpu", "cpu"]
    return ["cpu"]


def get_gpu_free_bytes() -> int | None:
    try:
        import cupy
    except ModuleNotFoundError:
        return None
    try:
        free_bytes, _ = cupy.cuda.runtime.memGetInfo()
        return int(free_bytes)
    except Exception:
        return None


def get_mps_free_bytes() -> int | None:
    """Return free MPS memory in bytes, or None if torch is unavailable.

    Apple Silicon's MPS shares system RAM with the OS and other apps.
    We approximate "free MPS bytes" as ``cpu_available - torch.mps.driver_allocated_memory()``,
    capped by the ``PYTORCH_MPS_HIGH_WATERMARK_RATIO`` env var if set.
    Returns None if torch is missing (so callers can short-circuit).
    """
    try:
        import torch
    except ModuleNotFoundError:
        return None

    cpu_free = get_cpu_available_bytes()
    if cpu_free is None:
        return None

    # Subtract torch's already-allocated MPS memory (if torch+MPS is built).
    allocated = 0
    mps = getattr(torch, "mps", None)
    if mps is not None:
        try:
            allocated = int(mps.driver_allocated_memory())
        except Exception:
            allocated = 0

    free = max(0, cpu_free - allocated)

    # Honor PYTORCH_MPS_HIGH_WATERMARK_RATIO when set (a fraction in [0, 1]).
    watermark_env = os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO")
    if watermark_env:
        try:
            watermark = float(watermark_env)
            if 0.0 < watermark <= 1.0:
                free = min(free, int(cpu_free * watermark))
        except ValueError:
            pass

    return free


def _sysconf(name: str) -> int | None:
    try:
        return os.sysconf(name)
    except (AttributeError, ValueError, OSError):
        return None


def get_cpu_available_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        pass

    page_size = _sysconf("SC_PAGE_SIZE") or _sysconf("SC_PAGESIZE")
    if page_size is None:
        return None
    avail_pages = _sysconf("SC_AVPHYS_PAGES")
    if avail_pages is not None:
        return int(avail_pages * page_size)
    total_pages = _sysconf("SC_PHYS_PAGES")
    if total_pages is not None:
        return int(total_pages * page_size)
    return None


def estimate_frame_bytes(im_info) -> int | None:
    if im_info is None or im_info.axes is None or im_info.shape is None:
        return None
    frame_shape = tuple(
        dim for axis, dim in zip(im_info.axes, im_info.shape) if axis != "T"
    )
    if not frame_shape:
        return None
    try:
        itemsize = im_info.im.dtype.itemsize
    except Exception:
        return None
    return int(math.prod(frame_shape) * itemsize)


def should_use_low_memory(im_info, include_gpu: bool, *, device_order=None) -> bool:
    """Heuristic: should this stage start in low-memory mode?

    Compares the estimated peak (``frame_bytes * _ESTIMATED_PEAK_MULTIPLIER``)
    against the active device's free memory budget, scaled by the
    device-specific headroom in ``_MEMORY_HEADROOM_BY_DEVICE``.

    ``include_gpu`` is the legacy boolean toggle — when True we consult
    the CUDA budget. When ``device_order`` is provided we additionally
    consult the MPS budget if MPS is in the cascade. Both probes are
    independent OR-conditions (any one being tight triggers low-memory).
    """
    frame_bytes = estimate_frame_bytes(im_info)
    if frame_bytes is None:
        return False
    peak_bytes = frame_bytes * _ESTIMATED_PEAK_MULTIPLIER

    if include_gpu:
        gpu_free = get_gpu_free_bytes()
        if gpu_free is not None and peak_bytes > gpu_free * _headroom("cuda"):
            return True

    if device_order and "mps" in device_order:
        mps_free = get_mps_free_bytes()
        if mps_free is not None and peak_bytes > mps_free * _headroom("mps"):
            return True

    cpu_free = get_cpu_available_bytes()
    if cpu_free is not None and peak_bytes > cpu_free * _headroom("cpu"):
        return True
    return False


def mode_candidates(device_order, start_low_memory: bool):
    modes = []
    for dev in device_order:
        modes.append((dev, False))
        modes.append((dev, True))
    if start_low_memory and modes:
        first_device = device_order[0]
        for idx, (dev, low) in enumerate(modes):
            if dev == first_device and low:
                return modes[idx:]
    return modes


def is_oom_error(exc: Exception) -> bool:
    if isinstance(exc, MemoryError):
        return True
    try:
        import cupy

        if isinstance(exc, cupy.cuda.memory.OutOfMemoryError):
            return True
    except Exception:
        pass
    msg = repr(exc).lower()
    # CuPy historically; "mps backend out of memory" is the torch wording.
    return (
        "out of memory" in msg
        or "outofmemory" in msg
        or "mps backend out of memory" in msg
    )


def is_gpu_unavailable_error(exc: Exception) -> bool:
    if isinstance(exc, (ModuleNotFoundError, ImportError)):
        name = getattr(exc, "name", "") or ""
        if "cupy" in name or "torch" in name:
            return True
    # NotImplementedError from the torch_xp/torch_ndi shim signals "this
    # stage isn't MPS-onboarded yet" — treat it as backend-unavailable so
    # the cascade falls back to CPU rather than crashing the user's run.
    # This is what makes ``device="auto"`` safe to pass to non-onboarded
    # stages on a Mac with torch+MPS installed.
    if isinstance(exc, NotImplementedError):
        msg = str(exc).lower()
        if "torch_xp" in msg or "torch_ndi" in msg:
            return True
    msg = repr(exc).lower()
    return (
        # CuPy / generic
        "gpu backend requested" in msg
        or "cupy is not installed" in msg
        or "cuda is not available" in msg
        or "no cuda devices" in msg
        # Torch + MPS
        or "mps backend requested" in msg
        or "torch is not installed" in msg
        or "no mps device" in msg
        or "mps is not available" in msg
        or "is_available() is false" in msg
        or "mps not built" in msg
    )
