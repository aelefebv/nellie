"""
Stage-agnostic chunking and adaptive-memory primitives.

These helpers are extracted from `nellie.segmentation.filtering.Filter`
but don't depend on Filter, ImInfo, or any stage-specific config.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np


def _default_is_oom(exc: BaseException) -> bool:
    """Recognize NumPy ``MemoryError`` and CuPy ``OutOfMemoryError``.

    Self-contained — does not import cupy at module load. Caller may
    override via the ``is_oom`` parameter on `safe_eigvalsh`.
    """
    if isinstance(exc, MemoryError):
        return True
    try:
        import cupy

        if isinstance(exc, cupy.cuda.memory.OutOfMemoryError):
            return True
    except ImportError:
        pass
    return False


def compute_chunk_shape(
    shape: tuple[int, ...],
    max_chunk_voxels: int | None,
) -> tuple[int, ...]:
    """Halve the largest axis until the chunk fits in ``max_chunk_voxels``.

    Returns ``shape`` unchanged if ``max_chunk_voxels`` is None or ≤ 0,
    or if the input already fits.
    """
    if max_chunk_voxels is None or max_chunk_voxels <= 0:
        return tuple(shape)
    chunk = list(shape)
    while int(np.prod(chunk)) > max_chunk_voxels:
        idx = int(np.argmax(chunk))
        chunk[idx] = max(1, int(np.ceil(chunk[idx] / 2)))
    return tuple(chunk)


def iter_chunks(
    shape: tuple[int, ...],
    chunk_shape: tuple[int, ...],
    halo: tuple[int, ...] | None = None,
) -> Iterator[tuple[tuple[slice, ...], tuple[slice, ...], tuple[slice, ...]]]:
    """Yield ``(core, ext, core_in_ext)`` slice tuples covering ``shape``.

    - ``core``: the chunk's slice in the full array (no halo).
    - ``ext``: the chunk's slice extended by ``halo`` voxels on each side
      (clipped at array bounds).
    - ``core_in_ext``: where the core sits inside the extracted ``ext``
      sub-array — useful for trimming the halo back off after processing.
    """
    if halo is None or len(halo) != len(shape):
        halo = (0,) * len(shape)
    ranges = [range(0, dim, step) for dim, step in zip(shape, chunk_shape)]
    for starts in itertools.product(*ranges):
        ends = [min(start + step, dim) for start, step, dim in zip(starts, chunk_shape, shape)]
        core = tuple(slice(s, e) for s, e in zip(starts, ends))
        ext_starts = [max(0, s - h) for s, h in zip(starts, halo)]
        ext_ends = [min(dim, e + h) for e, h, dim in zip(ends, halo, shape)]
        ext = tuple(slice(s, e) for s, e in zip(ext_starts, ext_ends))
        core_in_ext = tuple(
            slice(s - es, e - es) for s, e, es in zip(starts, ends, ext_starts)
        )
        yield core, ext, core_in_ext


def safe_eigvalsh(
    H: np.ndarray,
    xp: Any,
    *,
    is_oom: Callable[[BaseException], bool] = _default_is_oom,
    force_device_gpu: bool = False,
) -> np.ndarray:
    """Compute Hessian eigenvalues sorted by absolute value, with OOM fallback.

    On CPU (``xp`` is NumPy), this is a thin wrapper around ``eigvalsh``.

    On GPU (``xp`` is CuPy), an OOM error triggers recursive halving of
    the chunk; if a single matrix still OOMs and ``force_device_gpu`` is
    False, the function copies that single chunk to CPU as a last resort.
    Set ``force_device_gpu=True`` to disable the CPU fallback.
    """

    def _eig_backend(arr: np.ndarray) -> np.ndarray:
        ev = xp.linalg.eigvalsh(arr)
        order = xp.argsort(xp.abs(ev), axis=1)
        return xp.take_along_axis(ev, order, axis=1)

    on_cuda = hasattr(xp, "cuda")
    if not on_cuda:
        return _eig_backend(H)

    try:
        return _eig_backend(H)
    except Exception as e:
        if not is_oom(e):
            raise

        n = H.shape[0]
        if n > 1:
            mid = n // 2
            ev1 = safe_eigvalsh(
                H[:mid], xp, is_oom=is_oom, force_device_gpu=force_device_gpu
            )
            ev2 = safe_eigvalsh(
                H[mid:], xp, is_oom=is_oom, force_device_gpu=force_device_gpu
            )
            return xp.concatenate([ev1, ev2], axis=0)

        if force_device_gpu:
            raise

        # Single-matrix CPU fallback
        try:
            H_cpu = xp.asnumpy(H)
        except Exception:
            H_cpu = np.asarray(H)
        ev_cpu = np.linalg.eigvalsh(H_cpu)
        order = np.argsort(np.abs(ev_cpu), axis=1)
        ev_cpu = np.take_along_axis(ev_cpu, order, axis=1)
        return xp.asarray(ev_cpu)


def eigvalsh_3x3_symmetric(H: np.ndarray, xp: Any) -> np.ndarray:
    """Closed-form eigenvalues of batched 3x3 symmetric matrices.

    Drop-in replacement for `safe_eigvalsh` on 3x3 symmetric inputs.
    Returns shape ``(N, 3)`` sorted by absolute value ascending — the
    contract `frangi_math.frangi` (and `safe_eigvalsh`) rely on. Uses
    Smith's 1961 trigonometric formula; vectorizes cleanly on both
    NumPy and CuPy and skips the per-batch LAPACK call entirely.
    """
    return eigvalsh_3x3_components(
        H[..., 0, 0],
        H[..., 0, 1],
        H[..., 0, 2],
        H[..., 1, 1],
        H[..., 1, 2],
        H[..., 2, 2],
        xp,
    )


def eigvalsh_3x3_components(
    h11: np.ndarray,
    h12: np.ndarray,
    h13: np.ndarray,
    h22: np.ndarray,
    h23: np.ndarray,
    h33: np.ndarray,
    xp: Any,
) -> np.ndarray:
    """`eigvalsh_3x3_symmetric` variant that takes the six unique components directly.

    Avoids materializing a full ``(N, 3, 3)`` tensor when the caller
    already has the components as 1D arrays (e.g., the Frangi filter's
    chunked Hessian path).
    """
    # Smith 1961: characteristic polynomial via the trigonometric ID
    # 4·cos³(θ) − 3·cos(θ) = cos(3θ). For symmetric A:
    #   p₁ = Σ aᵢⱼ²  (off-diagonal)
    #   q  = trace(A) / 3
    #   p² = (Σ (aᵢᵢ − q)² + 2·p₁) / 6
    #   B  = (A − q·I) / p
    #   r  = det(B) / 2
    # eigenvalues = q + 2p·cos(arccos(r)/3 + k·2π/3) for k=0,1,2.
    p1 = h12 * h12 + h13 * h13 + h23 * h23
    q = (h11 + h22 + h33) / xp.float32(3.0)

    h11q = h11 - q
    h22q = h22 - q
    h33q = h33 - q
    p2 = h11q * h11q + h22q * h22q + h33q * h33q + xp.float32(2.0) * p1
    p = xp.sqrt(p2 / xp.float32(6.0))

    # When p == 0, the matrix is a scalar multiple of the identity (or
    # zero) — every eigenvalue equals q. Substitute 1.0 to avoid div-by-
    # zero in the formula; restore the correct value afterward.
    is_degenerate = p <= xp.float32(0.0)
    safe_p = xp.where(is_degenerate, xp.float32(1.0), p)
    inv_p = xp.float32(1.0) / safe_p

    b11 = h11q * inv_p
    b12 = h12 * inv_p
    b13 = h13 * inv_p
    b22 = h22q * inv_p
    b23 = h23 * inv_p
    b33 = h33q * inv_p

    # Cofactor expansion along the first row (B is symmetric)
    det_b = (
        b11 * (b22 * b33 - b23 * b23)
        - b12 * (b12 * b33 - b23 * b13)
        + b13 * (b12 * b23 - b22 * b13)
    )
    r = det_b / xp.float32(2.0)
    # Float rounding can push r marginally outside [-1, 1] — arccos of
    # that is NaN. Clip before the trig call.
    r = xp.clip(r, xp.float32(-1.0), xp.float32(1.0))

    phi = xp.arccos(r) / xp.float32(3.0)
    two_p = xp.float32(2.0) * p

    eig_max = q + two_p * xp.cos(phi)
    eig_min = q + two_p * xp.cos(phi + xp.float32(2.0 * np.pi / 3.0))
    eig_mid = xp.float32(3.0) * q - eig_max - eig_min

    eig_max = xp.where(is_degenerate, q, eig_max)
    eig_min = xp.where(is_degenerate, q, eig_min)
    eig_mid = xp.where(is_degenerate, q, eig_mid)

    eigenvalues = xp.stack([eig_min, eig_mid, eig_max], axis=-1)
    order = xp.argsort(xp.abs(eigenvalues), axis=-1)
    return xp.take_along_axis(eigenvalues, order, axis=-1)


def _sample_strides(shape: tuple[int, ...], max_samples: int | None) -> tuple[int, ...]:
    if max_samples is None or max_samples <= 0:
        return (1,) * len(shape)
    total = int(np.prod(shape))
    if total <= max_samples:
        return (1,) * len(shape)
    ndim = len(shape)
    stride = int(np.ceil((total / max_samples) ** (1.0 / ndim)))
    strides = [max(1, stride) for _ in range(ndim)]
    while int(np.prod([int(np.ceil(s / st)) for s, st in zip(shape, strides)])) > max_samples:
        idx = int(np.argmax([s / st for s, st in zip(shape, strides)]))
        strides[idx] += 1
    return tuple(strides)


def _downsample(arr: np.ndarray, strides: tuple[int, ...]) -> np.ndarray:
    if all(s == 1 for s in strides):
        return arr
    slices = tuple(slice(None, None, s) for s in strides)
    return arr[slices]


def _size(arr: Any) -> int:
    """Return total element count of an array-like, backend-agnostic.

    numpy/cupy expose ``.size`` as an integer attribute. torch.Tensor
    exposes ``.size()`` as a method (returns the shape tuple), so the
    plain ``arr.size`` access blows up under MPS. Reduce via
    ``arr.shape`` instead — supported by every backend.
    """
    return int(np.prod(arr.shape))


def subsample_for_thresholds(
    arr: np.ndarray,
    max_samples: int,
    xp: Any,  # noqa: ARG001 -- reserved for backend-aware variants
) -> np.ndarray:
    """Strided positive-voxel subsample for triangle/Otsu threshold estimation.

    Reduces memory and runtime when computing thresholds on very large
    volumes. Returns the positive voxels of a strided downsample, capped
    at ``max_samples``.

    ``xp`` is currently unused but reserved for future backend-aware
    optimizations (e.g., GPU-side fancy indexing variants).
    """
    if _size(arr) == 0:
        return arr
    strides = _sample_strides(arr.shape, max_samples)
    arr = _downsample(arr, strides)
    arr = arr[arr > 0]
    n = _size(arr)
    if n == 0:
        return arr
    if n > max_samples:
        stride = max(1, n // max_samples)
        arr = arr[::stride]
    return arr
