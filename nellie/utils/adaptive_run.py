"""
Helpers for adaptive device/memory selection and fallback retries.
"""
from __future__ import annotations

import math
import os
from typing import Any


_ESTIMATED_PEAK_MULTIPLIER = 6.0
_MEMORY_HEADROOM = 0.7


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


def _cpu_backend() -> tuple[Any, Any, str]:
    import numpy as np
    import scipy.ndimage as ndi

    return np, ndi, "cpu"


def resolve_backend(device: str) -> tuple[Any, Any, str]:
    """Resolve a device string to ``(xp, ndi, device_type)``.

    Accepts ``"auto"``, ``"cpu"``, ``"gpu"``, ``"cuda"``. ``"cuda"`` is a
    synonym for ``"gpu"``. ``"auto"`` uses GPU if available, otherwise CPU.

    The returned ``device_type`` is ``"cuda"`` for GPU and ``"cpu"`` for CPU,
    matching the per-stage convention.
    """
    device = (device or "auto").lower()
    if device not in ("auto", "cpu", "gpu", "cuda"):
        raise ValueError(f"Unsupported device '{device}'. Use 'auto', 'cpu', or 'gpu'.")

    if device in ("gpu", "cuda"):
        xp, ndi = try_import_cupy(require=True)
        return xp, ndi, "cuda"
    if device == "cpu":
        return _cpu_backend()

    # auto
    xp, ndi = try_import_cupy(require=False)
    if xp is not None:
        return xp, ndi, "cuda"
    return _cpu_backend()


def free_gpu_memory(xp: Any) -> None:
    """Free CuPy's default memory pool. No-op when ``xp`` is NumPy.

    Detects the backend via duck-typing (``get_default_memory_pool``)
    rather than importing cupy unconditionally.
    """
    pool_fn = getattr(xp, "get_default_memory_pool", None)
    if pool_fn is None:
        return
    try:
        pool_fn().free_all_blocks()
    except Exception:
        return


def normalize_device(device: str | None) -> str:
    device = (device or "auto").lower()
    if device == "cuda":
        device = "gpu"
    if device not in ("auto", "cpu", "gpu"):
        raise ValueError(f"Unsupported device '{device}'. Use 'auto', 'cpu', or 'gpu'.")
    return device


def gpu_available() -> bool:
    try:
        import cupy
    except ModuleNotFoundError:
        return False
    try:
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


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


def should_use_low_memory(im_info, include_gpu: bool) -> bool:
    frame_bytes = estimate_frame_bytes(im_info)
    if frame_bytes is None:
        return False
    peak_bytes = frame_bytes * _ESTIMATED_PEAK_MULTIPLIER
    if include_gpu:
        gpu_free = get_gpu_free_bytes()
        if gpu_free is not None and peak_bytes > gpu_free * _MEMORY_HEADROOM:
            return True
    cpu_free = get_cpu_available_bytes()
    if cpu_free is not None and peak_bytes > cpu_free * _MEMORY_HEADROOM:
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
    return "out of memory" in msg or "outofmemory" in msg


def is_gpu_unavailable_error(exc: Exception) -> bool:
    if isinstance(exc, (ModuleNotFoundError, ImportError)):
        name = getattr(exc, "name", "") or ""
        if "cupy" in name:
            return True
    msg = repr(exc).lower()
    return (
        "gpu backend requested" in msg
        or "cupy is not installed" in msg
        or "cuda is not available" in msg
        or "no cuda devices" in msg
    )
