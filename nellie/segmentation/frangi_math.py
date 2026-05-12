"""
Pure functions for the Frangi vesselness pipeline.

Each function takes its xp/ndi backend explicitly and derives 2D/3D
behavior from input array shape — no `ImInfo` or stage-instance state.
Suitable for callers outside `nellie.segmentation.filtering.Filter`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from nellie.utils.gpu_functions import otsu_threshold, triangle_threshold


def frangi(
    eigenvalues: np.ndarray,
    alpha_sq: float,
    beta_sq: float,
    gamma_sq: float,
    xp: Any,
) -> np.ndarray:
    """Apply the Frangi vesselness formula to Hessian eigenvalues.

    Parameters
    ----------
    eigenvalues : (N, 2) for 2D or (N, 3) for 3D
        Sorted by absolute value, ascending.
    alpha_sq, beta_sq, gamma_sq : float
        Squared Frangi parameters. ``alpha_sq`` is unused in 2D.
    xp : numpy or cupy module

    Returns
    -------
    response : (N,) float array, NaN/Inf-scrubbed.

    Dark structures (positive λ₂, and positive λ₃ in 3D) are zeroed —
    this implementation is tuned for **bright vessels on a dark
    background** (λ₂ ≤ 0, and λ₃ ≤ 0 in 3D). The second derivative
    perpendicular to a bright tubular structure is negative (image
    curves downward outward from the bright center), so the Hessian
    eigenvalues across the vessel are non-positive; positive λ₂/λ₃
    correspond to dark structures on a bright background and are
    rejected.
    """
    ndim = eigenvalues.shape[1]
    if ndim == 2:
        l1 = eigenvalues[:, 0]
        l2 = eigenvalues[:, 1]
        rb_sq = (xp.abs(l1) / (xp.abs(l2) + 1e-12)) ** 2
        s_sq = l1 ** 2 + l2 ** 2
        filtered_im = xp.exp(-(rb_sq / beta_sq)) * (
            1.0 - xp.exp(-(s_sq / gamma_sq))
        )
    elif ndim == 3:
        l1 = eigenvalues[:, 0]
        l2 = eigenvalues[:, 1]
        l3 = eigenvalues[:, 2]
        ra_sq = (xp.abs(l2) / (xp.abs(l3) + 1e-12)) ** 2
        rb_sq = (xp.abs(l2) / (xp.sqrt(xp.abs(l2 * l3)) + 1e-12)) ** 2
        s_sq = l1 ** 2 + l2 ** 2 + l3 ** 2
        filtered_im = (
            (1.0 - xp.exp(-(ra_sq / alpha_sq)))
            * xp.exp(-(rb_sq / beta_sq))
            * (1.0 - xp.exp(-(s_sq / gamma_sq)))
        )
    else:
        raise ValueError(
            f"frangi expects eigenvalues of shape (N, 2) or (N, 3); got (..., {ndim})"
        )

    if ndim == 3:
        filtered_im[eigenvalues[:, 2] > 0] = 0.0
    filtered_im[eigenvalues[:, 1] > 0] = 0.0

    return xp.nan_to_num(filtered_im, nan=0.0, posinf=0.0, neginf=0.0)


def compute_hessian(
    image: np.ndarray,
    spacing: tuple[float, ...],
    low_memory: bool,
    xp: Any,
    work_dtype: str = "float32",
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Compute Hessian components and a normalized Frobenius-norm volume.

    ``low_memory=True`` computes gradients axis-by-axis with intermediate
    `del`s; ``low_memory=False`` computes all axes at once (faster, more
    peak memory).

    Returns
    -------
    h_components : dict
        Keys ``hxx, hxy, hyy`` for 2D; adds ``hxz, hyz, hzz`` for 3D.
    frobenius_norm : ndarray
        ``sqrt(Σ hᵢⱼ²) / max(|hᵢⱼ|)`` for stability. Same shape as ``image``.
    """
    image = image.astype(work_dtype, copy=False)

    if image.ndim == 2:
        if low_memory:
            g0 = xp.gradient(image, spacing[0], axis=0)
            hxx = xp.gradient(g0, spacing[0], axis=0).astype(work_dtype, copy=False)
            hxy = xp.gradient(g0, spacing[1], axis=1).astype(work_dtype, copy=False)
            del g0
            g1 = xp.gradient(image, spacing[1], axis=1)
            hyy = xp.gradient(g1, spacing[1], axis=1).astype(work_dtype, copy=False)
            del g1
        else:
            g0, g1 = xp.gradient(image, *spacing)
            hxx = xp.gradient(g0, spacing[0], axis=0).astype(work_dtype, copy=False)
            hxy = xp.gradient(g0, spacing[1], axis=1).astype(work_dtype, copy=False)
            hyy = xp.gradient(g1, spacing[1], axis=1).astype(work_dtype, copy=False)

        frob_sq = hxx ** 2 + hyy ** 2 + 2.0 * (hxy ** 2)
        h_components = {"hxx": hxx, "hxy": hxy, "hyy": hyy}
    elif image.ndim == 3:
        if low_memory:
            g0 = xp.gradient(image, spacing[0], axis=0)
            hxx = xp.gradient(g0, spacing[0], axis=0).astype(work_dtype, copy=False)
            hxy = xp.gradient(g0, spacing[1], axis=1).astype(work_dtype, copy=False)
            hxz = xp.gradient(g0, spacing[2], axis=2).astype(work_dtype, copy=False)
            del g0
            g1 = xp.gradient(image, spacing[1], axis=1)
            hyy = xp.gradient(g1, spacing[1], axis=1).astype(work_dtype, copy=False)
            hyz = xp.gradient(g1, spacing[2], axis=2).astype(work_dtype, copy=False)
            del g1
            g2 = xp.gradient(image, spacing[2], axis=2)
            hzz = xp.gradient(g2, spacing[2], axis=2).astype(work_dtype, copy=False)
            del g2
        else:
            g0, g1, g2 = xp.gradient(image, *spacing)
            hxx = xp.gradient(g0, spacing[0], axis=0).astype(work_dtype, copy=False)
            hxy = xp.gradient(g0, spacing[1], axis=1).astype(work_dtype, copy=False)
            hxz = xp.gradient(g0, spacing[2], axis=2).astype(work_dtype, copy=False)
            hyy = xp.gradient(g1, spacing[1], axis=1).astype(work_dtype, copy=False)
            hyz = xp.gradient(g1, spacing[2], axis=2).astype(work_dtype, copy=False)
            hzz = xp.gradient(g2, spacing[2], axis=2).astype(work_dtype, copy=False)

        frob_sq = (
            hxx ** 2
            + hyy ** 2
            + hzz ** 2
            + 2.0 * (hxy ** 2 + hxz ** 2 + hyz ** 2)
        )
        h_components = {
            "hxx": hxx,
            "hxy": hxy,
            "hxz": hxz,
            "hyy": hyy,
            "hyz": hyz,
            "hzz": hzz,
        }
    else:
        raise ValueError(f"Unsupported number of dimensions: {image.ndim}")

    max_abs = 0.0
    for comp in h_components.values():
        # ``.size`` is a method on torch.Tensor (returns shape) but an
        # int attribute on numpy/cupy arrays. Reduce via ``.shape`` so
        # the empty-component check works on all three backends.
        if int(np.prod(comp.shape)) > 0:
            max_abs = max(max_abs, float(xp.max(xp.abs(comp))))
    if max_abs <= 0:
        max_abs = 1.0
    frobenius_norm = xp.sqrt(frob_sq) / max_abs

    return h_components, frobenius_norm


def log_blobness(
    image: np.ndarray,
    sigmas: list[float],
    sigma_vec_fn: Callable[[float], tuple[float, ...]],
    mask: np.ndarray,
    xp: Any,
    ndi: Any,
    work_dtype: str = "float32",
) -> np.ndarray:
    """Multi-scale Laplacian-of-Gaussian, max-fused across scales, scaled to /10.

    The ``/10`` divisor keeps the LoG response from dominating Frangi
    vesselness when the two are max-fused in the 2D path.
    """
    image = image.astype(work_dtype, copy=False)
    if not sigmas:
        return xp.zeros_like(image)

    def _scale_response(s: float) -> np.ndarray:
        sigma_vec = sigma_vec_fn(s)
        return (-ndi.gaussian_laplace(image, sigma_vec) * (float(s) ** 2)) * mask

    lapofg = _scale_response(sigmas[0])
    for s in sigmas[1:]:
        lapofg = xp.maximum(lapofg, _scale_response(s))

    lapofg[lapofg < 0] = 0.0
    lapofg_max = xp.max(lapofg)
    return (lapofg / (lapofg_max + 1e-12)) / 10.0


def calculate_gamma(
    gauss_volume: np.ndarray,
    spacing: tuple[float, ...],
    subsample_fn: Callable[[np.ndarray], np.ndarray],
    xp: Any,
) -> float:
    """Estimate γ from triangle/Otsu thresholds, rescaled by spacing_geomean².

    The rescale exists because Hessian eigenvalues live in
    ``intensity / spacing²`` units while the triangle/Otsu thresholds run on
    raw intensity. Without the rescale the ``(1 - exp(-S²/γ²))`` term in
    `frangi()` saturates and response magnitudes blow up.

    ``subsample_fn`` is the caller-provided positive-voxel subsampler; this
    keeps `calculate_gamma` independent of the chunking primitives that
    own the subsampling logic.
    """
    positive = subsample_fn(gauss_volume)
    # See note in ``compute_hessian`` on ``.size`` cross-backend gotcha.
    if int(np.prod(positive.shape)) == 0:
        return float(np.finfo(np.float32).eps)

    gamma_tri = triangle_threshold(positive, xp=xp)
    gamma_otsu, _ = otsu_threshold(positive, xp=xp)
    gamma = float(min(gamma_tri, gamma_otsu))
    if gamma <= 0:
        gamma = float(np.finfo(np.float32).eps)

    ndim = len(spacing)
    spacing_geomean = float(np.prod(spacing)) ** (1.0 / ndim)
    if spacing_geomean > 0:
        gamma = gamma / (spacing_geomean ** 2)
    return gamma
