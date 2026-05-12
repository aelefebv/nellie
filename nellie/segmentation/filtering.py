"""
Frangi-like vesselness filter for 3D/4D microscopy image data.

This module provides the Filter class, which implements a multi-scale Frangi filtering approach
optimized for large datasets with optional GPU acceleration.
"""

from dataclasses import dataclass

import numpy as np
import scipy.ndimage as scipy_ndi

from nellie.im_info.verifier import ImInfo
from nellie.segmentation import frangi_math
from nellie.utils import adaptive_run, chunking
from nellie.utils.base_logger import logger
from nellie.utils.gpu_functions import otsu_threshold, triangle_threshold


def _probe_cupy_backend():
    """Resolve the CuPy backend tuple once at import time.

    Returns ``(cupy, cupyx.scipy.ndimage, cupy.ndarray)`` or ``None``
    when CuPy is unavailable. Cached so `_backend_for_array` can dispatch
    via a single `isinstance` check instead of re-importing per call.
    """
    try:
        import cupy
        import cupyx.scipy.ndimage as cupy_ndi

        return (cupy, cupy_ndi, cupy.ndarray)
    except Exception:
        return None


def _probe_torch_backend():
    """Resolve the torch+MPS backend tuple once at import time.

    Returns ``(torch_xp, torch_ndi, torch.Tensor)`` or ``None`` when
    torch is unavailable. Mirrors :func:`_probe_cupy_backend` so
    ``_backend_for_array`` can dispatch torch tensors through the
    correct shim instead of (incorrectly) falling through to numpy.
    """
    try:
        import torch

        from nellie.utils import torch_ndi as _torch_ndi
        from nellie.utils import torch_xp as _torch_xp

        return (_torch_xp, _torch_ndi, torch.Tensor)
    except Exception:
        return None


_CUPY_BACKEND = _probe_cupy_backend()
_TORCH_BACKEND = _probe_torch_backend()


@dataclass(frozen=True)
class FrangiConfig:
    """Algorithm configuration for `Filter`.

    Frozen — represents the user's intent at construction time. Filter
    copies these values into mutable instance attributes that the
    OOM/availability cascade can update mid-run (`device`, `low_memory`).
    Inspect `Filter.config` to see the original intent regardless of any
    cascade-driven runtime fallbacks.
    """

    remove_edges: bool = False
    min_radius_um: float = 0.25
    max_radius_um: float = 1.0
    alpha_sq: float = 0.5
    beta_sq: float = 0.5
    frob_thresh: float | None = None
    frob_thresh_division: float = 2
    device: str = "auto"
    low_memory: bool = False
    max_chunk_voxels: int = int(1e6)
    max_threshold_samples: int = int(1e6)

    def __post_init__(self) -> None:
        adaptive_run.normalize_device(self.device)
        for name, value in (
            ("min_radius_um", self.min_radius_um),
            ("max_radius_um", self.max_radius_um),
            ("alpha_sq", self.alpha_sq),
            ("beta_sq", self.beta_sq),
            ("frob_thresh_division", self.frob_thresh_division),
            ("max_chunk_voxels", self.max_chunk_voxels),
            ("max_threshold_samples", self.max_threshold_samples),
        ):
            if value <= 0:
                raise ValueError(f"FrangiConfig.{name} must be > 0, got {value}")
        if self.frob_thresh is not None and self.frob_thresh <= 0:
            raise ValueError(
                f"FrangiConfig.frob_thresh must be > 0 if set, got {self.frob_thresh}"
            )
        if self.min_radius_um > self.max_radius_um:
            raise ValueError(
                f"FrangiConfig.min_radius_um ({self.min_radius_um}) must be <= "
                f"max_radius_um ({self.max_radius_um})"
            )


class Filter:
    """
    Frangi-like vesselness filter for 3D or 4D microscopy image data, optimized for
    large datasets and optional GPU acceleration.
    """

    def __init__(
        self,
        im_info: ImInfo,
        config: FrangiConfig = FrangiConfig(),
        viewer=None,
        num_t: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        im_info : ImInfo
            Image metadata and file paths.
        config : FrangiConfig
            Algorithm configuration. Defaults to ``FrangiConfig()``.
        viewer : object or None
            Optional GUI viewer with a `.status` attribute.
        num_t : int, optional
            Number of timepoints to process. If None, inferred from image.
        """
        self.im_info = im_info
        self.config = config
        self.viewer = viewer

        # Cascade-mutable runtime state. Initial values come from config;
        # `_set_backend` and `_set_low_memory` may update them on retry.
        self.device = config.device
        self.low_memory = config.low_memory

        self.xp, self.ndi, self.device_type = adaptive_run.resolve_backend(self.device)
        self.force_device = self.device.lower() in ("cpu", "gpu", "cuda", "mps")
        self.truncate = 3.0
        if not self.im_info.no_z:
            z_res = self.im_info.dim_res.get("Z") or self.im_info.dim_res.get("X") or 1.0
            x_res = self.im_info.dim_res.get("X") or 1.0
            self.z_ratio = float(z_res) / float(x_res)
        self.num_t = num_t
        if num_t is None and not self.im_info.no_t:
            self.num_t = im_info.shape[im_info.axes.index("T")]

        # Aliases for hot path readability — config remains the source of truth.
        self.remove_edges = config.remove_edges
        # either (roughly) diffraction limit, or pixel size, whichever is larger
        # self.min_radius_um = max(config.min_radius_um, self.im_info.dim_res["X"])
        self.min_radius_um = config.min_radius_um
        self.max_radius_um = config.max_radius_um
        self.min_radius_px = self.min_radius_um / self.im_info.dim_res["X"]
        self.max_radius_px = self.max_radius_um / self.im_info.dim_res["X"]

        self.alpha_sq = float(config.alpha_sq)
        self.beta_sq = float(config.beta_sq)
        self.frob_thresh = config.frob_thresh
        self.frob_thresh_division = config.frob_thresh_division
        self.max_chunk_voxels = int(config.max_chunk_voxels)
        self.max_threshold_samples = int(config.max_threshold_samples)

        self.im_memmap = None
        self.frangi_memmap = None
        self.sigmas = None

        # Dtypes
        self.work_dtype = "float32"
        self.out_dtype = "float32"

        # Cached per-run values
        self.halo = None

    def _switch_to_cpu(self):
        self.xp, self.ndi, self.device_type = adaptive_run.resolve_backend("cpu")

    def _set_backend(self, device):
        device = adaptive_run.normalize_device(device)
        self.device = device
        self.xp, self.ndi, self.device_type = adaptive_run.resolve_backend(device)
        self.force_device = device in ("cpu", "gpu", "cuda", "mps")

    def _set_low_memory(self, low_memory):
        self.low_memory = bool(low_memory)

    # -------------------------------------------------------------------------
    # Setup helpers
    # -------------------------------------------------------------------------
    def _get_t(self):
        """Determine the number of timepoints to process."""
        if self.num_t is None:
            if self.im_info.no_t:
                self.num_t = 1
            else:
                self.num_t = self.im_info.shape[self.im_info.axes.index("T")]

    def _allocate_memory(self):
        """
        Allocate memory-mapped storage for the Frangi-filtered image.

        The output is stored as float32 to reduce memory footprint.
        """
        logger.debug("Allocating memory for frangi filter.")
        self.im_memmap = self.im_info.get_memmap(self.im_info.im_path)

        im_frangi_path = self.im_info.pipeline_paths["im_preprocessed"]
        self.frangi_memmap = self.im_info.allocate_memory(
            im_frangi_path,
            dtype=self.out_dtype,
            description="frangi filtered im",
            return_memmap=True,
        )

    def _bbox(self, im):
        # Only ever called from `_remove_edges` on a single Y-X slice;
        # the 3D path iterates over Z and passes one 2D plane at a time.
        if len(im.shape) != 2:
            raise ValueError(f"_bbox expects a 2D image; got shape {im.shape}")
        xp, _ = self._backend_for_array(im)
        rows = xp.any(im, axis=1)
        cols = xp.any(im, axis=0)
        if (not rows.any()) or (not cols.any()):
            return 0, 0, 0, 0
        rmin, rmax = xp.where(rows)[0][[0, -1]]
        cmin, cmax = xp.where(cols)[0][[0, -1]]
        return int(rmin), int(rmax), int(cmin), int(cmax)

    def _backend_for_array(self, arr):
        # `_CUPY_BACKEND` / `_TORCH_BACKEND` are probed once at module
        # import; this dispatcher is hot (called per frame in
        # `_run_filter`, per call in `_mask_volume` / `_bbox`), so the
        # per-call try/except + import the original did was pure
        # overhead on CuPy-less installs. The torch arm makes Filter
        # treat ``device="mps"`` the same way: a torch tensor flowing
        # through ``_mask_volume`` / ``_bbox`` / ``_run_filter`` gets
        # the torch_xp / torch_ndi shim back, not numpy + scipy.
        if _CUPY_BACKEND is not None and isinstance(arr, _CUPY_BACKEND[2]):
            return _CUPY_BACKEND[0], _CUPY_BACKEND[1]
        if _TORCH_BACKEND is not None and isinstance(arr, _TORCH_BACKEND[2]):
            return _TORCH_BACKEND[0], _TORCH_BACKEND[1]
        return np, scipy_ndi

    def _get_spacing(self, ndim):
        if ndim == 2:
            y = self.im_info.dim_res.get("Y") or 1.0
            x = self.im_info.dim_res.get("X") or 1.0
            return (float(y), float(x))
        if ndim == 3:
            z = self.im_info.dim_res.get("Z") or self.im_info.dim_res.get("X") or 1.0
            y = self.im_info.dim_res.get("Y") or 1.0
            x = self.im_info.dim_res.get("X") or 1.0
            return (float(z), float(y), float(x))
        raise ValueError(f"Unsupported number of dimensions: {ndim}")

    def _get_sigma_vec(self, sigma: float):
        """
        Generate the sigma vector in (Z, Y, X) or (Y, X) depending on dimensionality.
        """
        if self.im_info.no_z:
            return (float(sigma), float(sigma))
        # scale Z by resolution ratio
        return (float(sigma) / self.z_ratio, float(sigma), float(sigma))

    def _set_default_sigmas(self):
        """
        Set default sigma values based on the radius range, with a minimum
        step size to avoid oversampling scales for large volumes.
        """
        logger.debug("Setting Frangi sigma values.")
        min_sigma_step_size = 0.2
        num_sigma = 5

        sigma_1 = self.min_radius_px / 2.0
        sigma_2 = self.max_radius_px / 3.0
        self.sigma_min = min(sigma_1, sigma_2)
        self.sigma_max = max(sigma_1, sigma_2)

        if self.sigma_max <= self.sigma_min:
            self.sigma_max = self.sigma_min + min_sigma_step_size

        sigma_step_size_calculated = (self.sigma_max - self.sigma_min) / float(num_sigma)
        sigma_step_size = max(
            min_sigma_step_size, sigma_step_size_calculated
        )  # Avoid too small steps.

        self.sigmas = list(np.arange(self.sigma_min, self.sigma_max, sigma_step_size, dtype=float))
        self.sigmas.sort()
        self.halo = self._compute_halo()

        logger.debug(
            f"Calculated sigma step size = {sigma_step_size_calculated}. Sigmas = {self.sigmas}"
        )

    def _compute_halo(self):
        if not self.sigmas:
            return None
        max_sigma = max(self.sigmas)
        sigma_vec = self._get_sigma_vec(max_sigma)
        return tuple(int(np.ceil(self.truncate * float(s))) for s in sigma_vec)

    # -------------------------------------------------------------------------
    # Threshold helpers
    # -------------------------------------------------------------------------
    def _subsample_for_thresholds(self, arr):
        return chunking.subsample_for_thresholds(arr, self.max_threshold_samples, self.xp)

    # -------------------------------------------------------------------------
    # Hessian Frobenius mask (policy)
    # -------------------------------------------------------------------------
    def _get_frob_mask(self, frobenius_norm):
        """
        Threshold a Frobenius-norm volume to produce a boolean mask.

        Parameters
        ----------
        frobenius_norm : xp.ndarray
            Frobenius norm (possibly rescaled) of the Hessian at each voxel.

        Returns
        -------
        mask : xp.ndarray of bool
        """
        if not self.frob_thresh_division:
            return frobenius_norm > 0

        if self.frob_thresh is None:
            # Infs would break `triangle_threshold` / `otsu_threshold`,
            # both of which use `xp.histogram(range=(min, max))`. Filter
            # them on the SUBSAMPLE only — the final `> thresh`
            # comparison keeps them in the mask anyway (inf > finite is
            # True). Subsample-side filter avoids a full-volume `isinf`
            # scan + `any` reduction in the common (finite-only) case;
            # see ADR 0012 for the approx-equivalence trade-off when
            # the input does contain infs.
            positive = self._subsample_for_thresholds(frobenius_norm)
            # ``.size`` is a method on torch.Tensor, an int on numpy/cupy
            # arrays. Reduce via ``.shape`` for backend-agnostic count.
            if int(np.prod(positive.shape)) > 0:
                finite_mask = self.xp.isfinite(positive)
                if not bool(finite_mask.all()):
                    positive = positive[finite_mask]

            if int(np.prod(positive.shape)) == 0:
                # Subsample empty or all-inf. Preserve the all-inf →
                # all-False contract by checking the full volume; this
                # is the only path that pays the full-volume `isinf`
                # scan and only fires when the subsample tells us infs
                # are dense (or there are no positive voxels at all).
                if bool(self.xp.isinf(frobenius_norm).all()):
                    return self.xp.zeros_like(frobenius_norm, dtype=bool)
                frobenius_threshold = 0.0
            else:
                frob_triangle_thresh = triangle_threshold(positive, xp=self.xp)
                frob_otsu_thresh, _ = otsu_threshold(positive, xp=self.xp)
                frobenius_threshold = float(min(frob_triangle_thresh, frob_otsu_thresh))
        else:
            frobenius_threshold = float(self.frob_thresh)

        return frobenius_norm > (frobenius_threshold / self.frob_thresh_division)

    # -------------------------------------------------------------------------
    # Eigenvalues and vesselness
    # -------------------------------------------------------------------------
    def _eigenvalues_from_chunk(self, h_chunks):
        """Hessian eigenvalues from a dict of 1D component chunks.

        Returns shape ``(N, 2)`` for 2D, ``(N, 3)`` for 3D, sorted by
        absolute value ascending — what `frangi_math.frangi` expects.
        Shared by the sparse and dense vesselness paths so both routes
        stay in lockstep numerically.
        """
        if self.im_info.no_z:
            hxx_c = h_chunks["hxx"]
            hxy_c = h_chunks["hxy"]
            hyy_c = h_chunks["hyy"]
            trace = hxx_c + hyy_c
            diff = hxx_c - hyy_c
            delta = self.xp.sqrt(diff * diff + 4.0 * (hxy_c * hxy_c))
            l1 = 0.5 * (trace - delta)
            l2 = 0.5 * (trace + delta)
            abs1 = self.xp.abs(l1)
            abs2 = self.xp.abs(l2)
            swap = abs1 > abs2
            eig1 = self.xp.where(swap, l2, l1)
            eig2 = self.xp.where(swap, l1, l2)
            return self.xp.stack([eig1, eig2], axis=1)

        # Closed-form Smith's formula for symmetric 3x3 eigenvalues —
        # vectorizes cleanly over the chunk and skips both the (N, 3, 3)
        # tensor materialization and the per-batch LAPACK `eigvalsh`
        # that the prior `_safe_eigvalsh` path required. Same sort
        # contract (abs ascending), same shape `(N, 3)`.
        return chunking.eigvalsh_3x3_components(
            h_chunks["hxx"], h_chunks["hxy"], h_chunks["hxz"],
            h_chunks["hyy"], h_chunks["hyz"], h_chunks["hzz"],
            self.xp,
        )

    def _compute_vesselness_chunkwise(self, h_components, h_mask, gamma_sq, is_dense=None):
        """Dispatch to the dense fast path when h_mask covers every voxel.

        The sparse path (the original implementation) is correct in all
        cases but pays an `xp.where` coordinate materialization plus
        per-chunk fancy indexing plus a final scatter — pure overhead
        when the mask is fully True. `bool(h_mask.all())` costs one
        reduction; the dense path saves the rest.

        ``is_dense`` lets the caller (``_compute_vesselness``) thread
        the all-True flag through from a single ``xp.sum(h_mask)`` it
        already paid for the skip-check; passing ``None`` falls back
        to a local ``bool(h_mask.all())`` reduction (preserves the
        contract for direct unit-test callers in
        ``tests/test_filtering.py``).
        """
        if is_dense is None:
            is_dense = bool(h_mask.all())
        if is_dense:
            return self._compute_vesselness_dense(h_components, gamma_sq)
        return self._compute_vesselness_sparse(h_components, h_mask, gamma_sq)

    def _compute_vesselness_sparse(self, h_components, h_mask, gamma_sq):
        """Vesselness over masked voxels only, scattered back into a full volume.

        Used when the Frobenius mask is partial — paying the indexing
        overhead is worth it to avoid evaluating frangi on voxels that
        will be zeroed anyway.
        """
        coords = self.xp.where(h_mask)
        # Torch's ``.size`` is a method, not an attribute — reduce via shape.
        total_voxels = int(np.prod(coords[0].shape))
        template = next(iter(h_components.values()))
        if total_voxels == 0:
            return self.xp.zeros_like(template, dtype=self.work_dtype)

        chunk_size = self.max_chunk_voxels
        if chunk_size is None or chunk_size <= 0:
            chunk_size = total_voxels

        vessel_masked = self.xp.zeros(total_voxels, dtype=self.work_dtype)

        for start in range(0, total_voxels, chunk_size):
            end = min(start + chunk_size, total_voxels)
            idx_chunk = tuple(c[start:end] for c in coords)
            h_chunks = {k: h_components[k][idx_chunk] for k in h_components}
            eigenvalues = self._eigenvalues_from_chunk(h_chunks)
            v_chunk = frangi_math.frangi(
                eigenvalues, self.alpha_sq, self.beta_sq, gamma_sq, self.xp
            )
            vessel_masked[start:end] = v_chunk.astype(self.work_dtype, copy=False)

        vesselness = self.xp.zeros_like(template, dtype=self.work_dtype)
        vesselness[coords] = vessel_masked
        return vesselness

    def _compute_vesselness_dense(self, h_components, gamma_sq):
        """Vesselness over the full volume via flat slicing — no `xp.where` indirection.

        Iterating contiguous flat slices of the components keeps reads
        contiguous and avoids both the coord arrays the sparse path
        materializes and the per-chunk fancy indexing it performs.
        """
        template = next(iter(h_components.values()))
        shape = template.shape
        total = int(np.prod(shape))
        if total == 0:
            return self.xp.zeros_like(template, dtype=self.work_dtype)

        flat = {k: v.ravel() for k, v in h_components.items()}

        chunk_size = self.max_chunk_voxels
        if chunk_size is None or chunk_size <= 0:
            chunk_size = total

        vessel_flat = self.xp.zeros(total, dtype=self.work_dtype)

        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            h_chunks = {k: flat[k][start:end] for k in flat}
            eigenvalues = self._eigenvalues_from_chunk(h_chunks)
            v_chunk = frangi_math.frangi(
                eigenvalues, self.alpha_sq, self.beta_sq, gamma_sq, self.xp
            )
            vessel_flat[start:end] = v_chunk.astype(self.work_dtype, copy=False)

        return vessel_flat.reshape(shape)

    # -------------------------------------------------------------------------
    # Per-frame processing
    # -------------------------------------------------------------------------
    def _compute_vesselness(self, frame, mask=True):
        vesselness = self.xp.zeros_like(frame, dtype=self.work_dtype)
        masks = self.xp.ones_like(frame, dtype=bool)
        spacing = self._get_spacing(frame.ndim)

        # Start from raw frame and build Gaussian scales incrementally.
        # Must copy: the loop below uses output=gauss for in-place cascaded
        # Gaussian, which would otherwise mutate the caller's `frame`. When
        # `frame` is a view of the writable raw-image memmap (which happens
        # whenever the raw dtype already matches work_dtype), in-place writes
        # propagate to the OME-TIFF on disk and corrupt the raw file.
        gauss = frame.astype(self.work_dtype, copy=True)
        prev_sigma = 0.0

        for sigma in self.sigmas:
            # Compute incremental sigma to go from prev_sigma -> sigma
            sigma_vec_prev = self._get_sigma_vec(prev_sigma)
            sigma_vec_curr = self._get_sigma_vec(sigma)

            sigma_vec_delta = []
            for sp, sc in zip(sigma_vec_prev, sigma_vec_curr):
                sp2 = float(sp) ** 2
                sc2 = float(sc) ** 2
                diff = max(0.0, sc2 - sp2)
                sigma_vec_delta.append(np.sqrt(diff))
            sigma_vec_delta = tuple(sigma_vec_delta)

            if any(s > 0 for s in sigma_vec_delta):
                self.ndi.gaussian_filter(
                    gauss,
                    sigma=sigma_vec_delta,
                    output=gauss,
                    mode="reflect",
                    cval=0.0,
                    truncate=self.truncate,
                )

            prev_sigma = sigma

            gamma = frangi_math.calculate_gamma(
                gauss, spacing, self._subsample_for_thresholds, self.xp
            )
            gamma_sq = 2.0 * (float(gamma) ** 2)

            h_components, frobenius_norm = frangi_math.compute_hessian(
                gauss, spacing, self.low_memory, self.xp, self.work_dtype
            )
            if mask:
                h_mask = self._get_frob_mask(frobenius_norm)
            else:
                h_mask = self.xp.ones_like(gauss, dtype=bool)
            # Single reduction replaces the prior `xp.any(h_mask)` skip-check
            # plus `_compute_vesselness_chunkwise`'s internal
            # `bool(h_mask.all())` dispatch. The sum gives both answers:
            # is_empty (== 0) and is_dense (== total). Saves one
            # full-volume reduction per sigma.
            true_count = int(self.xp.sum(h_mask))
            total = int(np.prod(h_mask.shape))
            if true_count == 0:
                continue

            vessel_scale = self._compute_vesselness_chunkwise(
                h_components, h_mask, gamma_sq=gamma_sq,
                is_dense=(true_count == total),
            )

            self.xp.maximum(vesselness, vessel_scale, out=vesselness)
            masks &= h_mask

        return vesselness, masks

    def _run_frame_chunked(self, t, mask=True, max_chunk_voxels=None):
        frame_cpu = self.im_memmap[t, ...]
        shape = frame_cpu.shape
        chunk_voxels = int(max_chunk_voxels or self.max_chunk_voxels or int(np.prod(shape)))
        halo = self.halo or (0,) * len(shape)
        # 2D path also fuses a full-frame multi-scale LoG response (matches
        # `_run_frame`). Per-chunk LoG would require global normalization
        # across chunks; full-frame LoG sidesteps that and the extra plane
        # is small relative to the chunked Hessian/eigenvalue working set.
        fuse_blobness = self.im_info.no_z

        while True:
            try:
                chunk_shape = chunking.compute_chunk_shape(shape, chunk_voxels)
                vessel_out = np.zeros(shape, dtype=self.work_dtype)
                mask_out = np.ones(shape, dtype=bool) if fuse_blobness else None
                for core, ext, core_in_ext in chunking.iter_chunks(shape, chunk_shape, halo):
                    chunk = frame_cpu[ext]
                    chunk_xp = self.xp.asarray(chunk, dtype=self.work_dtype)
                    vessel_chunk, mask_chunk = self._compute_vesselness(
                        chunk_xp, mask=mask
                    )
                    vessel_chunk *= mask_chunk
                    # ``hasattr(arr, "get")`` is the canonical "is this on a
                    # GPU backend?" duck-type check — cupy.ndarray has it
                    # natively, and ``torch_xp._patch_tensor_methods`` adds
                    # it to torch.Tensor so MPS goes through the same path.
                    if hasattr(vessel_chunk, "get"):
                        vessel_chunk = vessel_chunk.get()
                        if mask_out is not None:
                            mask_chunk = mask_chunk.get()
                    vessel_out[core] = vessel_chunk[core_in_ext]
                    if mask_out is not None:
                        mask_out[core] = mask_chunk[core_in_ext]

                if fuse_blobness:
                    # Allocating the full frame on-device after we just
                    # went chunked looks wasteful, but it's intentional:
                    # per-chunk LoG would require global normalization
                    # across chunks, and for 2D this is a single Y×X
                    # plane — small relative to the chunked Hessian
                    # working set we just freed.
                    frame_xp = self.xp.asarray(frame_cpu, dtype=self.work_dtype)
                    log_mask = self.xp.asarray(mask_out)
                    blobness = frangi_math.log_blobness(
                        frame_xp, self.sigmas, self._get_sigma_vec, log_mask,
                        self.xp, self.ndi, self.work_dtype,
                    )
                    blobness = self.xp.maximum(blobness, 0)
                    # Same hasattr-driven duck-type as the chunked vessel
                    # path above — covers cupy and torch+MPS uniformly.
                    if hasattr(blobness, "get"):
                        blobness = blobness.get()
                    np.maximum(vessel_out, blobness, out=vessel_out)

                if self.remove_edges:
                    vessel_out = self._remove_edges(vessel_out)
                return vessel_out
            except Exception as exc:
                if not adaptive_run.is_oom_error(exc):
                    raise
                adaptive_run.free_gpu_memory(self.xp)
                if chunk_voxels <= 1:
                    raise
                chunk_voxels = max(1, chunk_voxels // 2)

    def _run_frame(self, t, mask=True):
        """
        Run the Frangi filter for a single timepoint using Gaussian scale-space
        with a cascaded Gaussian to avoid recomputing from raw at each sigma.
        """
        logger.info(f"Running Frangi filter on t={t}.")

        frame_cpu = self.im_memmap[t, ...]

        if self.low_memory:
            return self._run_frame_chunked(t, mask=mask)

        try:
            frame = self.xp.asarray(frame_cpu, dtype=self.work_dtype)
            vesselness, masks = self._compute_vesselness(frame, mask=mask)
            vesselness *= masks
            if self.im_info.no_z:
                log_mask = masks if mask else self.xp.ones_like(frame, bool)
                blobness = frangi_math.log_blobness(
                    frame, self.sigmas, self._get_sigma_vec, log_mask,
                    self.xp, self.ndi, self.work_dtype,
                )
                blobness = self.xp.maximum(blobness, 0)  # keep bright-blob response only
                self.xp.maximum(vesselness, blobness, out=vesselness)
            if self.remove_edges:
                vesselness = self._remove_edges(vesselness)
            return vesselness
        except Exception as exc:
            if not adaptive_run.is_oom_error(exc):
                raise
            adaptive_run.free_gpu_memory(self.xp)
            # Try chunked on current backend
            try:
                return self._run_frame_chunked(t, mask=mask)
            except Exception as exc2:
                if not adaptive_run.is_oom_error(exc2):
                    raise
                if self.device_type in ("cuda", "mps") and not self.force_device:
                    self._switch_to_cpu()
                    return self._run_frame_chunked(t, mask=mask)
                raise

    # -------------------------------------------------------------------------
    # Post-processing helpers
    # -------------------------------------------------------------------------
    def _mask_volume(self, frangi_frame):
        """
        Apply a simple percentile-based threshold and binary opening to refine
        the vesselness mask.
        """
        xp, ndi = self._backend_for_array(frangi_frame)
        positive = self._subsample_for_thresholds(frangi_frame)
        # Backend-agnostic empty check — see ``.size`` cross-backend note.
        if int(np.prod(positive.shape)) == 0:
            return frangi_frame

        # Use a low percentile to keep faint vessels
        thr = xp.percentile(positive, 1)
        frangi_mask = frangi_frame > thr
        frangi_mask = ndi.binary_opening(frangi_mask)
        frangi_frame = frangi_frame * frangi_mask
        return frangi_frame

    def _remove_edges(self, frangi_frame):
        """
        Remove edges from the detected structures by zeroing out a border
        around the bounding box.
        """
        if self.im_info.no_z:
            # 2D case. ``.size`` is a method on torch.Tensor and an int
            # on numpy/cupy — reduce via shape for backend-agnostic count.
            if int(np.prod(frangi_frame.shape)) == 0:
                return frangi_frame
            rmin, rmax, cmin, cmax = self._bbox(frangi_frame)
            height = max(0, rmax - rmin + 1)
            if height <= 0:
                return frangi_frame
            margin = min(15, height)
            frangi_frame[rmin : rmin + margin, :] = 0
            frangi_frame[rmax - margin + 1 : rmax + 1, :] = 0
        else:
            # 3D case: assume Z is axis 0
            num_z = frangi_frame.shape[0]
            margin = 15
            for z_idx in range(num_z):
                slice_im = frangi_frame[z_idx, ...]
                if int(np.prod(slice_im.shape)) == 0:
                    continue
                rmin, rmax, cmin, cmax = self._bbox(slice_im)
                height = max(0, rmax - rmin + 1)
                if height <= 0:
                    continue
                use_margin = min(margin, height)
                frangi_frame[z_idx, rmin : rmin + use_margin, :] = 0
                frangi_frame[z_idx, rmax - use_margin + 1 : rmax + 1, :] = 0
        return frangi_frame

    # -------------------------------------------------------------------------
    # Top-level loops
    # -------------------------------------------------------------------------
    def _run_filter(self, mask=True):
        """Run the Frangi filter over all timepoints."""
        for t in range(self.num_t):
            if self.viewer is not None:
                self.viewer.status = (
                    f"Preprocessing. Frame: {t + 1} of {self.num_t}."
                )
            frangi_frame = self._run_frame(t, mask=mask)

            # Only apply percentile-based masking if there is any signal
            xp, _ = self._backend_for_array(frangi_frame)
            total = float(xp.sum(frangi_frame))
            if total > 0.0:
                frangi_frame = self._mask_volume(frangi_frame)

            filtered_im = frangi_frame

            # Move result back to CPU for memmap storage when using GPU
            if hasattr(filtered_im, "get"):
                filtered_im = filtered_im.get()

            if self.im_info.no_t or self.num_t == 1:
                self.frangi_memmap[:] = filtered_im[:]
            else:
                self.frangi_memmap[t, ...] = filtered_im

            self.frangi_memmap.flush()

    def run(self, mask: bool = True) -> None:
        """
        Main entry point: run the Frangi filter over the image.
        """
        logger.info("Running Frangi filter.")
        device = adaptive_run.normalize_device(self.device)
        device_order = adaptive_run.device_cascade(self.device)
        if device == "gpu" and device_order == ["cpu"]:
            logger.warning("Filter: GPU requested but not available; falling back to CPU.")

        start_low_memory = bool(self.low_memory) or adaptive_run.should_use_low_memory(
            self.im_info,
            include_gpu="gpu" in device_order,
            device_order=device_order,
        )
        if start_low_memory and not self.low_memory:
            logger.info("Filter: enabling low-memory mode based on estimated usage.")

        last_exc = None
        for dev, low in adaptive_run.mode_candidates(device_order, start_low_memory):
            try:
                self._set_backend(dev)
                self._set_low_memory(low)
                self._get_t()
                self._allocate_memory()
                self._set_default_sigmas()
                self._run_filter(mask=mask)
                return
            except Exception as exc:
                last_exc = exc
                if adaptive_run.is_gpu_unavailable_error(exc) and dev in ("gpu", "mps"):
                    logger.warning("Filter: GPU backend unavailable; retrying on CPU.")
                    continue
                if adaptive_run.is_oom_error(exc):
                    logger.warning(
                        "Filter: OOM on %s/%s; retrying with lower settings.",
                        dev,
                        "low-memory" if low else "high-memory",
                    )
                    continue
                raise
        raise last_exc
