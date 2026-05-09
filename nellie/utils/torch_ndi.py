"""scipy.ndimage-API shim over PyTorch + MPS.

Implements the 9 ``scipy.ndimage`` ops nellie's four MPS-onboarded stages
(filtering, labelling, networking, hu_tracking) need:

- **Convolutional ops** dispatch via :mod:`torch.nn.functional` conv/pool
  variants on the active torch device. They support the ``mode=`` and
  ``cval=`` parameters scipy uses (``"reflect"``, ``"constant"``).

  - :func:`gaussian_filter` (separable 1D Gaussian along each axis)
  - :func:`gaussian_laplace` (Gaussian then Laplacian)
  - :func:`uniform_filter` (separable mean filter)
  - :func:`convolve` (general N-D correlation, scipy semantics)
  - :func:`maximum_filter` (max pool with footprint shape)
  - :func:`minimum_filter` (min pool — implemented as ``-max(-x)``)

- **Structural ops** have no MPS-native equivalent and round-trip to
  scipy on CPU. Cost is small on Apple Silicon's unified memory.

  - :func:`binary_opening`
  - :func:`binary_fill_holes`
  - :func:`label`

This module imports torch lazily (only when an op is called), matching
the install model in :mod:`nellie.utils.torch_xp`.

See also:
    - PRD #140 (Apple Silicon / MPS)
    - wiki/decisions/0001-pytorch-mps-over-mlx.md
"""

from __future__ import annotations

import math
from typing import Any, Sequence


# Cached torch + functional module once an op imports them. Typed as ``Any``
# so that ``F.conv2d(...)`` / ``F.pad(...)`` call sites don't trip Pyright's
# Optional-narrowing — the lazy-import contract guarantees these are non-None
# by the time any op runs.
_TORCH: Any = None
_F: Any = None


def _torch():
    global _TORCH, _F
    if _TORCH is not None:
        return _TORCH
    try:
        import torch as _t
        import torch.nn.functional as _f
    except ModuleNotFoundError as exc:  # pragma: no cover - defensive
        raise ModuleNotFoundError(
            "torch is not installed. Install the MPS extra: "
            "`pip install 'nellie[mps]'`."
        ) from exc
    _TORCH = _t
    _F = _f
    return _TORCH


def _functional():
    if _F is None:
        _torch()
    return _F


# -----------------------------------------------------------------------------
# Padding mode translation: scipy.ndimage <-> torch.nn.functional.pad
# -----------------------------------------------------------------------------

_SCIPY_TO_TORCH_PAD = {
    "constant": "constant",
    "reflect": "reflect",
    "mirror": "reflect",  # scipy "mirror" matches torch "reflect"
    "nearest": "replicate",
    "wrap": "circular",
}


def _pad_mode(mode: str) -> str:
    if mode not in _SCIPY_TO_TORCH_PAD:
        raise ValueError(
            f"torch_ndi: unsupported boundary mode '{mode}'. "
            f"Supported: {sorted(_SCIPY_TO_TORCH_PAD)}"
        )
    return _SCIPY_TO_TORCH_PAD[mode]


def _add_batch_channel(x):
    """Promote an N-D tensor to (1, 1, *shape) for conv/pool ops."""
    return x.unsqueeze(0).unsqueeze(0)


def _strip_batch_channel(x):
    return x.squeeze(0).squeeze(0)


def _pad_for_kernel(x, kernel_shape: Sequence[int], mode: str, cval: float):
    """Pad an N-D tensor by half-kernel on each side along its spatial axes.

    Mirrors scipy's "extend then convolve" handling. ``mode`` is in
    scipy parlance ('constant', 'reflect', 'nearest', 'wrap', 'mirror');
    ``cval`` is only consulted when ``mode == "constant"``.
    """
    torch = _torch()
    F = _functional()
    pad_mode = _pad_mode(mode)
    # F.pad takes the spatial pads in REVERSED axis order, two values per axis.
    pad: list[int] = []
    for ks in reversed(kernel_shape):
        before = ks // 2
        after = ks - 1 - before
        pad.extend([before, after])
    if pad_mode == "constant":
        return F.pad(x, pad, mode="constant", value=float(cval))
    return F.pad(x, pad, mode=pad_mode)


def _to_4d_or_5d(x):
    """Wrap an (H,W) or (D,H,W) input into a 4-D/5-D tensor for torch convN."""
    if x.ndim == 2:
        return _add_batch_channel(x), False  # (1,1,H,W)
    if x.ndim == 3:
        return _add_batch_channel(x), True   # (1,1,D,H,W)
    raise ValueError(
        f"torch_ndi: expected 2-D or 3-D input, got ndim={x.ndim}"
    )


def _conv(x_padded, kernel, *, dim_3d: bool):
    """Run conv2d/conv3d with the padded input and a (kD,kH,kW)-style kernel."""
    F = _functional()
    if dim_3d:
        # kernel: shape (1, 1, kD, kH, kW)
        return F.conv3d(x_padded, kernel)
    return F.conv2d(x_padded, kernel)


# -----------------------------------------------------------------------------
# Sigma normalization
# -----------------------------------------------------------------------------


def _normalize_sigma(sigma, ndim: int):
    """Convert sigma to a per-axis tuple of floats."""
    if isinstance(sigma, (int, float)):
        return (float(sigma),) * ndim
    sigma_list = [float(s) for s in sigma]
    if len(sigma_list) != ndim:
        raise ValueError(
            f"torch_ndi: sigma length {len(sigma_list)} != ndim {ndim}"
        )
    return tuple(sigma_list)


def _gaussian_kernel_1d(sigma: float, truncate: float = 4.0):
    """1-D Gaussian kernel that mirrors scipy.ndimage._ni_support semantics."""
    torch = _torch()
    if sigma <= 0:
        # scipy returns the input unchanged; we represent this with a length-1
        # identity kernel (caller can short-circuit for efficiency).
        return torch.ones(1, dtype=torch.float32)
    radius = int(truncate * sigma + 0.5)
    if radius < 1:
        radius = 1
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    return kernel


# -----------------------------------------------------------------------------
# Convolutional ops
# -----------------------------------------------------------------------------


def gaussian_filter(
    input,  # noqa: A002 - matches scipy API
    sigma,
    *,
    output=None,
    mode: str = "reflect",
    cval: float = 0.0,
    truncate: float = 4.0,
):
    """Separable Gaussian filter dispatched via 1-D conv along each axis.

    Mirrors :func:`scipy.ndimage.gaussian_filter`. The ``output=`` argument
    accepts a tensor of the right shape; the result is written in-place
    AND returned (matching scipy's behavior).
    """
    torch = _torch()
    sigmas = _normalize_sigma(sigma, input.ndim)

    work = input
    for axis, s in enumerate(sigmas):
        if s <= 0:
            continue
        work = _gaussian_filter_1d_along_axis(work, axis, s, truncate, mode, cval)

    if output is not None:
        output.copy_(work)
        return output
    return work


def _gaussian_filter_1d_along_axis(
    x, axis: int, sigma: float, truncate: float, mode: str, cval: float
):
    """Apply a 1-D Gaussian along ``axis`` using a depthwise conv1d trick."""
    torch = _torch()
    F = _functional()
    kernel_1d = _gaussian_kernel_1d(sigma, truncate).to(x.dtype if x.dtype.is_floating_point else torch.float32)
    radius = (kernel_1d.numel() - 1) // 2

    # Move the target axis to the last position, collapse the rest into batch.
    perm = list(range(x.ndim))
    perm.append(perm.pop(axis))
    x_perm = x.permute(perm).contiguous()
    leading_shape = x_perm.shape[:-1]
    n = x_perm.shape[-1]
    flat = x_perm.reshape(-1, 1, n)  # (B, 1, N)

    # Pad along the last dim only.
    pad_mode = _pad_mode(mode)
    if pad_mode == "constant":
        padded = F.pad(flat, [radius, radius], mode="constant", value=float(cval))
    else:
        padded = F.pad(flat, [radius, radius], mode=pad_mode)

    conv_kernel = kernel_1d.reshape(1, 1, -1)
    out_flat = F.conv1d(padded, conv_kernel)  # (B, 1, N)
    out_perm = out_flat.reshape(*leading_shape, n)

    # Restore axis order.
    inverse_perm = [0] * x.ndim
    for new_pos, orig_pos in enumerate(perm):
        inverse_perm[orig_pos] = new_pos
    return out_perm.permute(inverse_perm).contiguous()


def uniform_filter(
    input,  # noqa: A002 - matches scipy API
    size,
    *,
    output=None,
    mode: str = "reflect",
    cval: float = 0.0,
):
    """Mean filter with a box of given ``size``."""
    torch = _torch()

    if isinstance(size, int):
        sizes = (size,) * input.ndim
    else:
        sizes = tuple(int(s) for s in size)
        if len(sizes) != input.ndim:
            raise ValueError(
                f"torch_ndi.uniform_filter: size length {len(sizes)} != ndim {input.ndim}"
            )

    work = input
    for axis, s in enumerate(sizes):
        if s <= 1:
            continue
        # Build a uniform 1-D kernel along this axis.
        kernel = torch.full((s,), 1.0 / s, dtype=torch.float32)
        work = _conv1d_along_axis(work, axis, kernel, mode, cval)

    if output is not None:
        output.copy_(work)
        return output
    return work


def _conv1d_along_axis(x, axis: int, kernel_1d, mode: str, cval: float):
    """Generic 1-D conv along ``axis`` (centered) with scipy-style padding."""
    torch = _torch()
    F = _functional()
    if not x.dtype.is_floating_point:
        x = x.to(torch.float32)
    perm = list(range(x.ndim))
    perm.append(perm.pop(axis))
    x_perm = x.permute(perm).contiguous()
    leading_shape = x_perm.shape[:-1]
    n = x_perm.shape[-1]
    flat = x_perm.reshape(-1, 1, n)

    radius = (kernel_1d.numel() - 1) // 2
    pad_mode = _pad_mode(mode)
    if pad_mode == "constant":
        padded = F.pad(flat, [radius, radius], mode="constant", value=float(cval))
    else:
        padded = F.pad(flat, [radius, radius], mode=pad_mode)

    out_flat = F.conv1d(padded, kernel_1d.reshape(1, 1, -1).to(flat.dtype))
    out_perm = out_flat.reshape(*leading_shape, n)
    inverse_perm = [0] * x.ndim
    for new_pos, orig_pos in enumerate(perm):
        inverse_perm[orig_pos] = new_pos
    return out_perm.permute(inverse_perm).contiguous()


def convolve(
    input,  # noqa: A002 - matches scipy API
    weights,
    *,
    output=None,
    mode: str = "reflect",
    cval: float = 0.0,
):
    """N-D convolution mirroring :func:`scipy.ndimage.convolve` semantics.

    scipy's ``convolve`` does correlation flipped (true convolution).
    We dispatch via padded ``conv2d``/``conv3d``. Note: torch's ``conv2d``
    is *cross-correlation* (no flip), so for the symmetric kernels
    nellie's networking stage uses (``ones((3, 3))``) the result is
    identical to scipy's convolve. For asymmetric kernels we pre-flip
    the weights to match scipy's true-convolution semantics.
    """
    torch = _torch()

    if input.ndim != weights.ndim:
        raise ValueError(
            f"torch_ndi.convolve: input ndim {input.ndim} != weights ndim {weights.ndim}"
        )
    if input.ndim not in (2, 3):
        raise ValueError(
            f"torch_ndi.convolve: only 2-D or 3-D input supported, got ndim={input.ndim}"
        )

    work_dtype = input.dtype if input.dtype.is_floating_point else torch.float32
    x = input.to(work_dtype)
    w = weights.to(work_dtype)

    # scipy.convolve = true convolution = correlation with flipped kernel.
    w_flipped = torch.flip(w, dims=tuple(range(w.ndim)))
    kernel = w_flipped.reshape(1, 1, *w_flipped.shape)

    x4d, dim_3d = _to_4d_or_5d(x)
    padded = _pad_for_kernel(x4d, w_flipped.shape, mode, cval)
    out_4d = _conv(padded, kernel, dim_3d=dim_3d)
    out = _strip_batch_channel(out_4d)

    # Cast back to input dtype if needed.
    if out.dtype != input.dtype:
        out = out.to(input.dtype)

    if output is not None:
        output.copy_(out)
        return output
    return out


def gaussian_laplace(
    input,  # noqa: A002 - matches scipy API
    sigma,
    *,
    mode: str = "reflect",
    cval: float = 0.0,
):
    """Laplacian-of-Gaussian, mirroring :func:`scipy.ndimage.gaussian_laplace`.

    Implementation follows scipy: smooth by Gaussian, then sum the
    second derivatives along each axis.
    """
    torch = _torch()
    sigmas = _normalize_sigma(sigma, input.ndim)
    smoothed = gaussian_filter(input, sigmas, mode=mode, cval=cval)
    # Sum of second derivatives along each axis.
    out = None
    for axis in range(smoothed.ndim):
        d2 = _second_derivative_along_axis(smoothed, axis, mode, cval)
        if out is None:
            out = d2
        else:
            out = out + d2
    return out


def _second_derivative_along_axis(x, axis: int, mode: str, cval: float):
    """Second-derivative kernel ``[1, -2, 1]`` along ``axis`` with scipy padding."""
    torch = _torch()
    kernel = torch.tensor([1.0, -2.0, 1.0], dtype=torch.float32)
    return _conv1d_along_axis(x, axis, kernel, mode, cval)


def maximum_filter(
    input,  # noqa: A002 - matches scipy API
    size=None,
    *,
    footprint=None,
    output=None,
    mode: str = "reflect",
    cval: float = 0.0,
):
    """Max filter via depthwise N-D max pool.

    ``size`` follows scipy's convention: scalar => same size on every
    axis; tuple => per-axis size. ``footprint`` is supported only for
    fully-true rectangular footprints (the only shape nellie uses); other
    shapes raise NotImplementedError.
    """
    torch = _torch()
    F = _functional()

    if footprint is not None:
        # Validate it's a fully-true rectangular footprint.
        if not bool(footprint.all()):
            raise NotImplementedError(
                "torch_ndi.maximum_filter: non-rectangular footprints are not "
                "supported by the MPS shim."
            )
        sizes = tuple(footprint.shape)
    elif size is not None:
        if isinstance(size, int):
            sizes = (size,) * input.ndim
        else:
            sizes = tuple(int(s) for s in size)
    else:
        raise ValueError("torch_ndi.maximum_filter: size or footprint required")

    if len(sizes) != input.ndim:
        raise ValueError(
            f"torch_ndi.maximum_filter: size/footprint ndim {len(sizes)} != input ndim {input.ndim}"
        )

    work_dtype = input.dtype if input.dtype.is_floating_point else torch.float32
    x = input.to(work_dtype)
    x4d, dim_3d = _to_4d_or_5d(x)

    padded = _pad_for_kernel(x4d, sizes, mode, cval)
    if dim_3d:
        result = F.max_pool3d(padded, kernel_size=sizes, stride=1)
    else:
        result = F.max_pool2d(padded, kernel_size=sizes, stride=1)
    out = _strip_batch_channel(result)
    if out.dtype != input.dtype:
        out = out.to(input.dtype)

    if output is not None:
        output.copy_(out)
        return output
    return out


def minimum_filter(
    input,  # noqa: A002 - matches scipy API
    size=None,
    *,
    footprint=None,
    output=None,
    mode: str = "reflect",
    cval: float = 0.0,
):
    """Min filter implemented as ``-max(-x)``.

    The negation flips ``cval`` semantics: a constant-pad min filter pads
    with ``+cval`` to ignore those positions, so negating means we pad
    with ``-cval`` for the underlying max pool.
    """
    torch = _torch()
    work_dtype = input.dtype if input.dtype.is_floating_point else torch.float32
    neg = (-input.to(work_dtype))
    neg_max = maximum_filter(
        neg, size=size, footprint=footprint, mode=mode, cval=-float(cval)
    )
    result = -neg_max
    if result.dtype != input.dtype:
        result = result.to(input.dtype)

    if output is not None:
        output.copy_(result)
        return output
    return result


# -----------------------------------------------------------------------------
# Structural ops — round-trip to scipy on CPU
# -----------------------------------------------------------------------------


def _to_numpy(x):
    """Move a tensor to host numpy for the scipy round-trip."""
    torch = _torch()
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def _from_numpy_like(arr_np, template):
    """Wrap a numpy result back as a tensor on the same device as ``template``."""
    torch = _torch()
    if isinstance(template, torch.Tensor):
        return torch.as_tensor(arr_np, device=template.device)
    return arr_np


def binary_opening(input, structure=None, **kwargs):  # noqa: A002 - matches scipy API
    """Round-trip to ``scipy.ndimage.binary_opening`` on CPU.

    No MPS-native morphology op exists. Cost on Apple Silicon's unified
    memory is small.
    """
    import scipy.ndimage as scipy_ndi
    in_np = _to_numpy(input)
    struct_np = _to_numpy(structure) if structure is not None else None
    result_np = scipy_ndi.binary_opening(in_np, structure=struct_np, **kwargs)
    return _from_numpy_like(result_np, input)


def binary_fill_holes(input, structure=None, **kwargs):  # noqa: A002 - matches scipy API
    """Round-trip to ``scipy.ndimage.binary_fill_holes`` on CPU."""
    import scipy.ndimage as scipy_ndi
    in_np = _to_numpy(input)
    struct_np = _to_numpy(structure) if structure is not None else None
    result_np = scipy_ndi.binary_fill_holes(in_np, structure=struct_np, **kwargs)
    return _from_numpy_like(result_np, input)


def label(input, structure=None, output=None):  # noqa: A002 - matches scipy API
    """Round-trip to ``scipy.ndimage.label`` on CPU.

    Returns ``(labels, num_features)`` to match scipy's contract. Connected
    components has no equivalent on MPS without third-party libs (kornia
    etc.) — see PRD #140 § Out of Scope.
    """
    import scipy.ndimage as scipy_ndi
    in_np = _to_numpy(input)
    struct_np = _to_numpy(structure) if structure is not None else None
    labels_np, num_features = scipy_ndi.label(in_np, structure=struct_np)
    labels_out = _from_numpy_like(labels_np, input)
    if output is not None:
        output.copy_(labels_out if hasattr(output, "copy_") else labels_np)
        return output, num_features
    return labels_out, num_features
