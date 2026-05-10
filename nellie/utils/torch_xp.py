"""numpy-API shim over PyTorch + MPS.

Implements the union of ``xp.*`` ops used by nellie's four MPS-onboarded
stages (filtering, labelling, networking, hu_tracking) plus their helper
modules (``frangi_math``, ``chunking``, ``gpu_functions``). Operations not
listed in :data:`_SUPPORTED_OPS` raise :class:`NotImplementedError` with a
message identifying the missing op so future maintainers can extend the
shim deliberately.

Float64 silently coerces to float32 — MPS does not support double
precision. The coercion is announced at module load time via a single
log message (see :func:`_announce_float64_coercion`).

This module imports torch lazily: the import only happens when an op is
actually called, so installs without the ``mps`` extra (where torch is
absent) don't pay an import cost or raise at module load.

See also:
    - PRD #140 (Apple Silicon / MPS)
    - wiki/decisions/0001-pytorch-mps-over-mlx.md
    - wiki/decisions/0002-adaptive-run-extension-over-static-torch-xp.md
    - :mod:`nellie.utils.torch_ndi`
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Set to True once the float64 coercion notice has been emitted.
_FLOAT64_NOTICE_EMITTED = False

# Cached torch module + dtypes once the first op imports torch successfully.
_TORCH = None
_TORCH_FLOAT32 = None
_TORCH_FLOAT16 = None
_TORCH_BOOL = None


def _announce_float64_coercion() -> None:
    """Emit the one-time float64 coercion notice.

    Idempotent — subsequent calls are no-ops. The exact wording is part
    of the public contract per the slice acceptance criteria.
    """
    global _FLOAT64_NOTICE_EMITTED
    if _FLOAT64_NOTICE_EMITTED:
        return
    _FLOAT64_NOTICE_EMITTED = True
    logger.info(
        "MPS backend: float64 → float32 coercion is in effect; results may "
        "differ from CPU within float32 tolerance."
    )


def _torch():
    """Return the lazily-imported torch module, raising a clear error if missing."""
    global _TORCH, _TORCH_FLOAT32, _TORCH_FLOAT16, _TORCH_BOOL
    if _TORCH is not None:
        return _TORCH
    try:
        import torch as _t
    except ModuleNotFoundError as exc:  # pragma: no cover - defensive
        raise ModuleNotFoundError(
            "torch is not installed. Install the MPS extra: "
            "`pip install 'nellie[mps]'`."
        ) from exc
    _TORCH = _t
    _TORCH_FLOAT32 = _t.float32
    _TORCH_FLOAT16 = _t.float16
    _TORCH_BOOL = _t.bool
    _patch_tensor_methods(_t)
    return _TORCH


def _patch_tensor_methods(torch_mod) -> None:
    """Add numpy/cupy-style ``astype`` and ``get`` methods to torch.Tensor.

    Pipeline stages were written against the numpy/cupy contract where
    ``arr.astype(dtype, copy=...)`` and ``arr.get()`` (cupy-only — moves
    data to host as a numpy array) are both available as methods. Torch
    spells the equivalents differently (``arr.to(dtype)`` and
    ``arr.detach().cpu().numpy()``), so without these patches stages
    that pass torch tensors through code that calls ``.astype`` or
    ``.get()`` blow up with ``AttributeError``.

    Patching the Tensor class (rather than wrapping every call site) is
    the smallest possible diff to onboard existing stages to MPS without
    touching their internals. The methods only attach if torch.Tensor
    doesn't already have them — future torch versions could in
    principle add ``astype`` natively, in which case this becomes a
    no-op.
    """
    Tensor = torch_mod.Tensor

    if not hasattr(Tensor, "astype"):
        def astype(self, dtype, copy: bool = True):
            """numpy/cupy-style ``astype``: cast to ``dtype`` with optional copy.

            ``dtype`` accepts torch.dtype, numpy dtype, _LazyDtype proxies,
            and Python scalar-class strings (``"float32"`` etc.) per
            :func:`_resolve_dtype`. ``copy=False`` returns ``self`` when
            the requested dtype already matches; ``copy=True`` (numpy's
            default) always allocates.
            """
            target = _resolve_dtype(dtype)
            if target is None or self.dtype == target:
                return self.clone() if copy else self
            return self.to(target)

        Tensor.astype = astype

    if not hasattr(Tensor, "get"):
        def get(self):
            """cupy-style ``get``: copy this tensor's contents to a host numpy array.

            Mirrors :func:`cupy.ndarray.get`. Detaches and moves to CPU
            first so MPS-resident tensors round-trip cleanly. Existing
            ``hasattr(arr, "get")`` checks in stage code (originally
            written for the cupy backend) now also fire on MPS, which
            is the desired symmetry.
            """
            return self.detach().cpu().numpy()

        Tensor.get = get

    if not hasattr(Tensor, "copy"):
        def copy(self):
            """numpy-style ``copy``: return a clone of this tensor.

            Mirrors ``np.ndarray.copy()`` / ``cupy.ndarray.copy()``.
            ``hu_tracking._get_frame_features`` calls ``.copy()`` on the
            Frangi and distance frames before mutating them in place;
            torch's native equivalent is ``.clone()``, so this method
            forwards. Same patching pattern as ``astype`` / ``get``.
            """
            return self.clone()

        Tensor.copy = copy


# -----------------------------------------------------------------------------
# Dtype attributes
# -----------------------------------------------------------------------------
#
# These are exposed as module attributes via ``__getattr__`` because the
# real torch dtype objects can only be created after torch is imported.
# Special-cased: ``float64`` resolves to torch.float32 (with the one-time
# log notice); MPS does not support double precision.


class _LazyDtype:
    """Sentinel that resolves to a torch dtype on first attribute access.

    Equality and hashing pass through to the underlying torch dtype so
    callers using ``dtype is xp.float32`` get sensible behavior. Calling
    the proxy as a constructor (``xp.float32(3.0)``) returns a 0-d
    tensor of that dtype — mirrors numpy's
    ``np.float32(3.0)`` scalar-construction idiom that the chunking /
    closed-form eigenvalue helpers rely on.
    """

    __slots__ = ("_name", "_resolve_via")

    def __init__(self, name: str, resolve_via: str) -> None:
        self._name = name
        self._resolve_via = resolve_via

    def _resolve(self):
        torch = _torch()
        if self._resolve_via == "float64":
            _announce_float64_coercion()
            return torch.float32
        return getattr(torch, self._resolve_via)

    def __repr__(self) -> str:
        return f"<torch_xp.{self._name} (resolves to torch.{self._resolve_via})>"

    def __eq__(self, other: object) -> bool:
        try:
            return self._resolve() == other
        except Exception:
            return NotImplemented

    def __hash__(self) -> int:
        return hash(("torch_xp", self._name))

    def __call__(self, value):
        """Construct a 0-d tensor of this dtype — matches ``np.float32(value)``.

        ``chunking.eigvalsh_3x3_components`` (and any future caller that
        wants a typed scalar constant) writes ``xp.float32(3.0)`` to
        keep arithmetic in float32. For numpy this returns a float32
        scalar; here we return ``torch.tensor(value, dtype=...)`` which
        is the closest equivalent (torch's elementwise ops accept it as
        a scalar without upcasting).
        """
        torch = _torch()
        return torch.tensor(value, dtype=self._resolve())


float32 = _LazyDtype("float32", "float32")
# float64 is intentionally aliased to torch.float32 with a logged notice.
float64 = _LazyDtype("float64", "float64")
float16 = _LazyDtype("float16", "float16")

# -----------------------------------------------------------------------------
# numpy API constants
# -----------------------------------------------------------------------------

# `xp.newaxis` is `None` in numpy — same in torch (used for indexing only).
newaxis = None

# `xp.inf` is the float; numpy/torch interop is via raw float.
import math as _math
inf = _math.inf


# -----------------------------------------------------------------------------
# Array class (for isinstance checks and type hints)
# -----------------------------------------------------------------------------


class _NdarrayProxy:
    """Lazy proxy for ``torch.Tensor`` so ``isinstance(x, xp.ndarray)`` works.

    Defining ``__instancecheck__`` on the metaclass lets the proxy stand
    in for the real Tensor class without forcing torch to be imported at
    module load.
    """

    class _Meta(type):
        def __instancecheck__(cls, instance) -> bool:
            try:
                torch = _torch()
            except Exception:
                return False
            return isinstance(instance, torch.Tensor)


class ndarray(metaclass=_NdarrayProxy._Meta):
    """Proxy class for ``torch.Tensor`` — for isinstance checks only.

    ``isinstance(x, ndarray)`` returns True iff ``x`` is a torch.Tensor.
    Instantiating this class is not supported; construct tensors via the
    shim's ``zeros``/``asarray``/etc. ops instead.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError(
            "torch_xp.ndarray is a proxy for torch.Tensor — use torch_xp.zeros, "
            "torch_xp.asarray, etc. to construct tensors."
        )


# -----------------------------------------------------------------------------
# Dtype coercion helper (silently downgrade float64 to float32)
# -----------------------------------------------------------------------------


def _resolve_dtype(dtype):
    """Convert a dtype-ish value into a torch.dtype, coercing float64→float32.

    Accepts torch.dtype, numpy dtype, _LazyDtype proxies, Python `bool`,
    and `None` (returns None). Anything else falls through to torch's
    dtype machinery (which will raise if it doesn't understand it).
    """
    if dtype is None:
        return None
    torch = _torch()
    # Already a torch dtype.
    if isinstance(dtype, torch.dtype):
        if dtype == torch.float64:
            _announce_float64_coercion()
            return torch.float32
        return dtype
    # _LazyDtype proxies.
    if isinstance(dtype, _LazyDtype):
        return dtype._resolve()
    # Python `bool` (numpy uses this for boolean arrays).
    if dtype is bool:
        return torch.bool
    # Numpy dtype-likes: map to torch via numpy's standardized name.
    try:
        import numpy as np
        np_dtype = np.dtype(dtype)
    except Exception:
        return dtype  # let torch deal with it
    name = np_dtype.name
    if name == "float64":
        _announce_float64_coercion()
        return torch.float32
    name_to_torch = {
        "float32": torch.float32,
        "float16": torch.float16,
        "int64": torch.int64,
        "int32": torch.int32,
        "int16": torch.int16,
        "int8": torch.int8,
        "uint8": torch.uint8,
        "bool": torch.bool,
        "complex64": torch.complex64,
        "complex128": torch.complex128,
    }
    return name_to_torch.get(name, dtype)


# -----------------------------------------------------------------------------
# Construction
# -----------------------------------------------------------------------------


def asarray(obj, dtype=None):
    torch = _torch()
    target_dtype = _resolve_dtype(dtype)
    if isinstance(obj, torch.Tensor):
        if target_dtype is not None and obj.dtype != target_dtype:
            return obj.to(target_dtype)
        return obj
    # Fall through to torch.as_tensor for numpy / list / scalar inputs.
    if target_dtype is not None:
        return torch.as_tensor(obj, dtype=target_dtype)
    return torch.as_tensor(obj)


def zeros(shape, dtype=None):
    torch = _torch()
    target_dtype = _resolve_dtype(dtype) if dtype is not None else torch.float32
    if isinstance(shape, int):
        shape = (shape,)
    return torch.zeros(*shape, dtype=target_dtype) if shape else torch.zeros((), dtype=target_dtype)


def zeros_like(arr, dtype=None):
    torch = _torch()
    if dtype is None:
        return torch.zeros_like(arr)
    return torch.zeros_like(arr, dtype=_resolve_dtype(dtype))


def ones(shape, dtype=None):
    torch = _torch()
    target_dtype = _resolve_dtype(dtype) if dtype is not None else torch.float32
    if isinstance(shape, int):
        shape = (shape,)
    return torch.ones(*shape, dtype=target_dtype) if shape else torch.ones((), dtype=target_dtype)


def ones_like(arr, dtype=None):
    torch = _torch()
    if dtype is None:
        return torch.ones_like(arr)
    return torch.ones_like(arr, dtype=_resolve_dtype(dtype))


def full_like(arr, fill_value, dtype=None):
    torch = _torch()
    if dtype is None:
        return torch.full_like(arr, fill_value)
    return torch.full_like(arr, fill_value, dtype=_resolve_dtype(dtype))


def arange(*args, dtype=None):
    """``arange(stop)`` / ``arange(start, stop)`` / ``arange(start, stop, step)``."""
    torch = _torch()
    target_dtype = _resolve_dtype(dtype)
    if target_dtype is None:
        return torch.arange(*args)
    return torch.arange(*args, dtype=target_dtype)


def meshgrid(*xi, indexing: str = "xy"):
    """numpy-style meshgrid; defaults to 'xy' indexing for parity."""
    torch = _torch()
    return tuple(torch.meshgrid(*xi, indexing=indexing))


# -----------------------------------------------------------------------------
# Element-wise math
# -----------------------------------------------------------------------------


def abs(x):  # noqa: A001 - matches numpy API
    return _torch().abs(x)


def sqrt(x):
    return _torch().sqrt(x)


def exp(x):
    return _torch().exp(x)


def log10(x):
    return _torch().log10(x)


def cos(x):
    return _torch().cos(x)


def arccos(x):
    return _torch().arccos(x)


def sign(x):
    return _torch().sign(x)


def ceil(x):
    return _torch().ceil(x)


def clip(x, a_min, a_max):
    return _torch().clamp(x, min=a_min, max=a_max)


def isfinite(x):
    return _torch().isfinite(x)


def isinf(x):
    return _torch().isinf(x)


def nan_to_num(x, nan=0.0, posinf=None, neginf=None):
    return _torch().nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)


def maximum(a, b, *, out=None):
    """numpy-style elementwise max with optional ``out=``.

    Supports the ``out=`` write-target convention used in nellie's
    in-place reductions (see Filter._compute_vesselness loop) and
    accepts Python-scalar second-args (numpy lets ``np.maximum(arr, 0)``
    promote 0 implicitly; torch refuses unless the scalar is wrapped).
    """
    torch = _torch()
    if not isinstance(a, torch.Tensor):
        a = torch.as_tensor(a)
    if not isinstance(b, torch.Tensor):
        b = torch.as_tensor(b, dtype=a.dtype, device=a.device)
    if out is not None:
        return torch.maximum(a, b, out=out)
    return torch.maximum(a, b)


# -----------------------------------------------------------------------------
# Reductions and scans
# -----------------------------------------------------------------------------


def any(x, axis=None):  # noqa: A001 - matches numpy API
    torch = _torch()
    if axis is None:
        return torch.any(x)
    return torch.any(x, dim=axis)


def sum(x, axis=None, dtype=None):  # noqa: A001 - matches numpy API
    torch = _torch()
    target_dtype = _resolve_dtype(dtype)
    if axis is None:
        if target_dtype is not None:
            return torch.sum(x, dtype=target_dtype)
        return torch.sum(x)
    if target_dtype is not None:
        return torch.sum(x, dim=axis, dtype=target_dtype)
    return torch.sum(x, dim=axis)


def nansum(x, axis=None):
    torch = _torch()
    if axis is None:
        return torch.nansum(x)
    return torch.nansum(x, dim=axis)


def max(x, axis=None):  # noqa: A001 - matches numpy API
    torch = _torch()
    if axis is None:
        return torch.max(x)
    out = torch.max(x, dim=axis)
    return out.values  # numpy returns just the values


def min(x, axis=None):  # noqa: A001 - matches numpy API
    torch = _torch()
    if axis is None:
        return torch.min(x)
    out = torch.min(x, dim=axis)
    return out.values


def argmin(x, axis=None):
    torch = _torch()
    if axis is None:
        return torch.argmin(x)
    return torch.argmin(x, dim=axis)


def argmax(x, axis=None):
    torch = _torch()
    if axis is None:
        return torch.argmax(x)
    return torch.argmax(x, dim=axis)


def cumsum(x, axis=None, dtype=None):
    torch = _torch()
    target_dtype = _resolve_dtype(dtype)
    if axis is None:
        flat = x.reshape(-1)
        if target_dtype is not None:
            return torch.cumsum(flat, dim=0, dtype=target_dtype)
        return torch.cumsum(flat, dim=0)
    if target_dtype is not None:
        return torch.cumsum(x, dim=axis, dtype=target_dtype)
    return torch.cumsum(x, dim=axis)


def percentile(a, q):
    """numpy-style percentile (q in [0, 100]).

    torch's ``quantile`` takes q in [0, 1], so we rescale and forward.
    Return a 0-d tensor matching numpy's scalar return for a scalar q.
    """
    torch = _torch()
    q_tensor = torch.as_tensor(q, dtype=torch.float32) / 100.0
    return torch.quantile(a.to(torch.float32) if a.dtype != torch.float32 else a, q_tensor)


# -----------------------------------------------------------------------------
# Comparison / selection
# -----------------------------------------------------------------------------


def where(condition, x=None, y=None):
    torch = _torch()
    if x is None and y is None:
        # numpy.where(cond) returns a tuple of 1-D index tensors (per dim).
        return torch.where(condition)
    return torch.where(condition, x, y)


def argwhere(x):
    """Return a (N, ndim) tensor of indices where ``x`` is truthy.

    Matches numpy's ``argwhere`` shape contract.
    """
    return _torch().argwhere(x)


def flatnonzero(x):
    """Return 1-D tensor of indices into the flattened array where ``x`` is nonzero."""
    torch = _torch()
    return torch.nonzero(x.reshape(-1), as_tuple=False).reshape(-1)


def take_along_axis(arr, indices, axis):
    return _torch().take_along_dim(arr, indices, dim=axis)


def argsort(x, axis=-1):
    return _torch().argsort(x, dim=axis)


# -----------------------------------------------------------------------------
# Combining / shape
# -----------------------------------------------------------------------------


def stack(arrays, axis=0):
    return _torch().stack(list(arrays), dim=axis)


def concatenate(arrays, axis=0):
    return _torch().cat(list(arrays), dim=axis)


def flip(x, axis=None):
    torch = _torch()
    if axis is None:
        return torch.flip(x, dims=tuple(range(x.ndim)))
    if isinstance(axis, int):
        return torch.flip(x, dims=(axis,))
    return torch.flip(x, dims=tuple(axis))


# -----------------------------------------------------------------------------
# Histogram / counting
# -----------------------------------------------------------------------------


def histogram(a, bins=10, range=None):  # noqa: A002 - matches numpy API
    """numpy-style histogram returning ``(counts, bin_edges)``.

    torch.histogram has the same return shape but slightly different
    parameter handling for ``range``; we forward via the keyword
    ``min``/``max`` form to keep the contract aligned.
    """
    torch = _torch()
    a_flat = a.reshape(-1).to(torch.float32) if a.dtype not in (torch.float32, torch.float64) else a.reshape(-1)
    if range is not None:
        lo, hi = range
        return torch.histogram(a_flat, bins=bins, range=(float(lo), float(hi)))
    return torch.histogram(a_flat, bins=bins)


def bincount(x, weights=None, minlength: int = 0):
    return _torch().bincount(x, weights=weights, minlength=minlength)


# -----------------------------------------------------------------------------
# Gradient (numpy.gradient with axis= and per-axis spacing)
# -----------------------------------------------------------------------------


def gradient(f, *varargs, axis=None):
    """Implements numpy.gradient's central-difference behavior.

    nellie's ``frangi_math.compute_hessian`` uses three call shapes:

    1. ``xp.gradient(image, h, axis=0)`` — single-axis with scalar spacing.
    2. ``xp.gradient(image, h0, h1)`` — multi-axis (returns tuple) with
       per-axis scalar spacing, no ``axis`` arg.
    3. ``xp.gradient(image, h0, h1, h2)`` — same, 3D.

    This implementation covers all three. Multi-axis returns a tuple of
    tensors (one per axis), single-axis returns a single tensor.
    """
    torch = _torch()
    ndim = f.ndim

    if axis is not None:
        # Single-axis path. varargs is at most 1 spacing scalar.
        h = float(varargs[0]) if varargs else 1.0
        return _gradient_axis(f, axis, h, torch)

    # Multi-axis path: one spacing per dimension (must match ndim).
    spacings = [float(v) for v in varargs] if varargs else [1.0] * ndim
    if len(spacings) == 1 and ndim > 1:
        spacings = spacings * ndim
    return tuple(
        _gradient_axis(f, ax, spacings[ax], torch) for ax in range(ndim)
    )


def _gradient_axis(f, axis: int, h: float, torch_mod):
    """Central differences along ``axis`` with first-order forward/backward edges.

    Mirrors ``numpy.gradient``'s default ``edge_order=1`` behavior: central
    differences in the interior, simple forward/backward at the boundaries.
    For len-1 axes we return zeros to match numpy's behavior on that edge case.
    """
    n = f.shape[axis]
    if n < 2:
        return torch_mod.zeros_like(f)

    # Build slices for centered-diff (shifted indices along axis).
    def _idx(i_start, i_end):
        sl = [slice(None)] * f.ndim
        sl[axis] = slice(i_start, i_end)
        return tuple(sl)

    out = torch_mod.empty_like(f)
    if n == 2:
        delta = (f[_idx(1, 2)] - f[_idx(0, 1)]) / h
        out[_idx(0, 1)] = delta
        out[_idx(1, 2)] = delta
        return out

    # Interior: central difference.
    interior = (f[_idx(2, n)] - f[_idx(0, n - 2)]) / (2.0 * h)
    out[_idx(1, n - 1)] = interior
    # Left edge: first-order forward — (f1 - f0) / h
    out[_idx(0, 1)] = (f[_idx(1, 2)] - f[_idx(0, 1)]) / h
    # Right edge: first-order backward — (fN-1 - fN-2) / h
    out[_idx(n - 1, n)] = (f[_idx(n - 1, n)] - f[_idx(n - 2, n - 1)]) / h
    return out


# -----------------------------------------------------------------------------
# Linalg
# -----------------------------------------------------------------------------


class _Linalg:
    """Lazy linalg sub-namespace mirroring ``numpy.linalg`` / ``cupy.linalg``."""

    def eigvalsh(self, a):
        return _torch().linalg.eigvalsh(a)


linalg = _Linalg()


# -----------------------------------------------------------------------------
# Misc helpers
# -----------------------------------------------------------------------------


def finfo(dtype):
    """Mirror numpy.finfo for the float dtypes the pipeline asks about."""
    torch = _torch()
    target = _resolve_dtype(dtype) if not isinstance(dtype, torch.dtype) else dtype
    return torch.finfo(target)


def asnumpy(arr):
    """Convert a tensor (or anything else) to a numpy array on CPU.

    Mirrors ``cupy.asnumpy``. For numpy inputs this is a no-op identity.
    """
    torch = _torch()
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    import numpy as np
    return np.asarray(arr)


# -----------------------------------------------------------------------------
# Unimplemented op trap (preserves stage-onboarding clarity)
# -----------------------------------------------------------------------------


def __getattr__(name: str) -> Any:
    """Trap unknown ``xp.*`` access with a clear ``NotImplementedError``.

    The shim covers the union of ops used by the four onboarded stages
    (filtering, labelling, networking, hu_tracking) — see the function
    definitions above. Anything else fires this trap. ``adaptive_run``'s
    cascade recognizes the message and treats it as "stage not
    MPS-onboarded yet" so the cascade can fall back to CPU.

    Only fires for names that aren't already module attributes (Python
    only consults ``__getattr__`` after the normal lookup fails).
    """
    raise NotImplementedError(
        f"torch_xp.{name} is not implemented. The MPS shim only covers the "
        f"union of ops used by the four onboarded stages (filtering, "
        f"labelling, networking, hu_tracking). If you're onboarding a new "
        f"stage to MPS, add the op here. See PRD #140."
    )
