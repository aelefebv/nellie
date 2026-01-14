"""
Frangi-like vesselness filter for 3D/4D microscopy image data.

This module provides the Filter class, which implements a multi-scale Frangi filtering approach
optimized for large datasets with optional GPU acceleration.
"""

import itertools
import numpy as np

from nellie.im_info.verifier import ImInfo
from nellie.utils import adaptive_run
from nellie.utils.base_logger import logger
from nellie.utils.gpu_functions import triangle_threshold, otsu_threshold


class Filter:
    """
    Frangi-like vesselness filter for 3D or 4D microscopy image data, optimized for
    large datasets and optional GPU acceleration.
    """

    def __init__(
        self,
        im_info: ImInfo,
        num_t=None,
        remove_edges: bool = False,
        min_radius_um: float = 0.25,
        max_radius_um: float = 1.0,
        alpha_sq: float = 0.5,
        beta_sq: float = 0.5,
        frob_thresh=None,
        frob_thresh_division=2,
        viewer=None,
        device: str = "auto",
        # New optimization-related parameters
        low_memory: bool = False,
        max_chunk_voxels: int = int(1e6),
        max_threshold_samples: int = int(1e6),
    ):
        """
        Parameters
        ----------
        im_info : ImInfo
            Image metadata and file paths.
        num_t : int, optional
            Number of timepoints to process. If None, inferred from image.
        remove_edges : bool
            If True, aggressively zero out bounding-box edges.
        min_radius_um, max_radius_um : float
            Expected structure radius range in micrometers.
        alpha_sq, beta_sq : float
            Frangi parameters controlling sensitivity to blobness and plate-likeness.
        frob_thresh : float or None
            If given, fixed Frobenius norm threshold. Otherwise auto-estimated.
        viewer : object or None
            Optional GUI viewer with a `.status` attribute.
        device : {"auto", "cpu", "gpu"}
            Backend selection. "auto" uses GPU if available, otherwise CPU.
            "cpu" forces NumPy/Scipy, and "gpu" forces CuPy/CuPyX (error if unavailable).
        low_memory : bool
            If True, prefer strategies that reduce peak memory at the cost of speed
            (e.g. smaller eigen-decomposition chunks).
        max_chunk_voxels : int
            Maximum number of voxels per processing chunk and eigen-decomposition chunk.
        max_threshold_samples : int
            Maximum number of samples to use when estimating thresholds
            (triangle / Otsu) from very large arrays.
        """
        self.im_info = im_info
        self.device = device
        self.xp, self.ndi, self.device_type = self._resolve_backend(device)
        self.force_device = device is not None and device.lower() in ("cpu", "gpu", "cuda")
        self.truncate = 3.0
        if not self.im_info.no_z:
            z_res = self.im_info.dim_res.get("Z") or self.im_info.dim_res.get("X") or 1.0
            x_res = self.im_info.dim_res.get("X") or 1.0
            self.z_ratio = float(z_res) / float(x_res)
        self.num_t = num_t
        if num_t is None and not self.im_info.no_t:
            self.num_t = im_info.shape[im_info.axes.index("T")]
        self.remove_edges = remove_edges
        # either (roughly) diffraction limit, or pixel size, whichever is larger
        # self.min_radius_um = max(min_radius_um, self.im_info.dim_res["X"])
        self.min_radius_um = min_radius_um
        self.max_radius_um = max_radius_um

        self.min_radius_px = self.min_radius_um / self.im_info.dim_res["X"]
        self.max_radius_px = self.max_radius_um / self.im_info.dim_res["X"]

        self.im_memmap = None
        self.frangi_memmap = None

        self.sigma_vec = None
        self.sigmas = None

        self.alpha_sq = float(alpha_sq)
        self.beta_sq = float(beta_sq)

        self.frob_thresh = frob_thresh
        self.frob_thresh_division = frob_thresh_division

        self.viewer = viewer

        # Optimization-related settings
        self.low_memory = low_memory
        self.max_chunk_voxels = int(max_chunk_voxels)
        self.max_threshold_samples = int(max_threshold_samples)

        # Dtypes
        self.work_dtype = "float32"
        self.out_dtype = "float32"

        # Cached per-run values
        self.halo = None

    def _resolve_backend(self, device):
        device = (device or "auto").lower()
        if device not in ("auto", "cpu", "gpu", "cuda"):
            raise ValueError(f"Unsupported device '{device}'. Use 'auto', 'cpu', or 'gpu'.")

        if device in ("gpu", "cuda"):
            xp, ndi = self._try_import_cupy(require=True)
            return xp, ndi, "cuda"
        if device == "cpu":
            import numpy as np
            import scipy.ndimage as ndi
            return np, ndi, "cpu"

        # auto
        xp, ndi = self._try_import_cupy(require=False)
        if xp is not None:
            return xp, ndi, "cuda"
        import numpy as np
        import scipy.ndimage as ndi
        return np, ndi, "cpu"

    def _try_import_cupy(self, require):
        try:
            import cupy
            import cupyx.scipy.ndimage as ndi
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

        return cupy, ndi

    def _is_oom_error(self, exc):
        if isinstance(exc, MemoryError):
            return True
        if self.device_type != "cuda":
            return False
        try:
            import cupy

            return isinstance(exc, cupy.cuda.memory.OutOfMemoryError)
        except Exception:
            return "OutOfMemory" in repr(exc)

    def _free_gpu_memory(self):
        if self.device_type != "cuda":
            return
        try:
            self.xp.get_default_memory_pool().free_all_blocks()
        except Exception:
            return

    def _switch_to_cpu(self):
        import numpy as np
        import scipy.ndimage as ndi

        self.xp = np
        self.ndi = ndi
        self.device_type = "cpu"

    def _set_backend(self, device):
        device = adaptive_run.normalize_device(device)
        self.device = device
        self.xp, self.ndi, self.device_type = self._resolve_backend(device)
        self.force_device = device in ("cpu", "gpu")

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
        self.shape = self.im_memmap.shape

        im_frangi_path = self.im_info.pipeline_paths["im_preprocessed"]
        self.frangi_memmap = self.im_info.allocate_memory(
            im_frangi_path,
            dtype=self.out_dtype,
            description="frangi filtered im",
            return_memmap=True,
        )

    def _bbox(self, im):
        xp, _ = self._backend_for_array(im)
        if len(im.shape) == 2:
            rows = xp.any(im, axis=1)
            cols = xp.any(im, axis=0)
            if (not rows.any()) or (not cols.any()):
                return 0, 0, 0, 0
            rmin, rmax = xp.where(rows)[0][[0, -1]]
            cmin, cmax = xp.where(cols)[0][[0, -1]]
            return int(rmin), int(rmax), int(cmin), int(cmax)

        if len(im.shape) == 3:
            r = xp.any(im, axis=(1, 2))
            c = xp.any(im, axis=(0, 2))
            z = xp.any(im, axis=(0, 1))
            if (not r.any()) or (not c.any()) or (not z.any()):
                return 0, 0, 0, 0, 0, 0
            rmin, rmax = xp.where(r)[0][[0, -1]]
            cmin, cmax = xp.where(c)[0][[0, -1]]
            zmin, zmax = xp.where(z)[0][[0, -1]]
            return int(rmin), int(rmax), int(cmin), int(cmax), int(zmin), int(zmax)

        logger.warning("Image not 2D or 3D... Cannot get bounding box.")
        return None

    def _backend_for_array(self, arr):
        try:
            import cupy
            import cupyx.scipy.ndimage as cupy_ndi

            if isinstance(arr, cupy.ndarray):
                return cupy, cupy_ndi
        except Exception:
            pass
        import numpy as np
        import scipy.ndimage as ndi
        return np, ndi

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
            self.sigma_vec = (float(sigma), float(sigma))
        else:
            # scale Z by resolution ratio
            self.sigma_vec = (float(sigma) / self.z_ratio, float(sigma), float(sigma))
        return self.sigma_vec

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
    def _sample_strides(self, shape, max_samples):
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

    def _downsample(self, arr, strides):
        if all(s == 1 for s in strides):
            return arr
        slices = tuple(slice(None, None, s) for s in strides)
        return arr[slices]

    def _subsample_for_thresholds(self, arr):
        """
        Subsample large arrays before passing to triangle / Otsu threshold
        to reduce memory and runtime.
        """
        if arr.size == 0:
            return arr
        strides = self._sample_strides(arr.shape, self.max_threshold_samples)
        arr = self._downsample(arr, strides)
        arr = arr[arr > 0]
        if arr.size == 0:
            return arr
        if arr.size > self.max_threshold_samples:
            stride = max(1, arr.size // self.max_threshold_samples)
            arr = arr[::stride]
        return arr

    def _calculate_gamma(self, gauss_volume):
        """
        Estimate gamma using triangle and Otsu thresholds on the positive voxels.
        """
        positive = self._subsample_for_thresholds(gauss_volume)
        if positive.size == 0:
            # Fallback to a very small positive value to avoid division by zero.
            return np.finfo(np.float32).eps

        gamma_tri = triangle_threshold(positive, xp=self.xp)
        gamma_otsu, _ = otsu_threshold(positive, xp=self.xp)
        gamma = float(min(gamma_tri, gamma_otsu))
        # Avoid division by zero downstream
        if gamma <= 0:
            gamma = np.finfo(np.float32).eps
        return gamma

    def _estimate_gamma(self, frame, sigma):
        """
        Estimate gamma from a deterministic downsample of the Gaussian volume.
        """
        if frame.size == 0:
            return np.finfo(np.float32).eps
        strides = self._sample_strides(frame.shape, self.max_threshold_samples)
        sample = self._downsample(frame, strides)
        sample = self.xp.asarray(sample, dtype=self.work_dtype)
        sigma_vec = self._get_sigma_vec(sigma)
        if len(sigma_vec) != sample.ndim:
            raise ValueError("Sigma vector does not match sample dimensionality")
        sigma_vec_sample = tuple(float(s) / st for s, st in zip(sigma_vec, strides))
        gauss_sample = self.ndi.gaussian_filter(
            sample,
            sigma=sigma_vec_sample,
            mode="reflect",
            cval=0.0,
            truncate=self.truncate,
        )
        return self._calculate_gamma(gauss_sample)

    # -------------------------------------------------------------------------
    # Hessian and Frobenius norm
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
        # Replace infs with max finite value to keep thresholds valid
        inf_mask = self.xp.isinf(frobenius_norm)
        if self.xp.any(inf_mask):
            finite_vals = frobenius_norm[~inf_mask]
            max_finite = float(self.xp.max(finite_vals)) if finite_vals.size > 0 else 0.0
            frobenius_norm = frobenius_norm.copy()
            frobenius_norm[inf_mask] = max_finite

        if not self.frob_thresh_division:
            mask = frobenius_norm > 0
            return mask

        if self.frob_thresh is None:
            positive = self._subsample_for_thresholds(frobenius_norm)
            if positive.size == 0:
                frobenius_threshold = 0.0
            else:
                frob_triangle_thresh = triangle_threshold(positive, xp=self.xp)
                frob_otsu_thresh, _ = otsu_threshold(positive, xp=self.xp)
                frobenius_threshold = float(min(frob_triangle_thresh, frob_otsu_thresh))
        else:
            frobenius_threshold = float(self.frob_thresh)

        mask = frobenius_norm > (frobenius_threshold / self.frob_thresh_division)
        return mask

    def _compute_hessian(self, image, mask=True):
        """
        Compute Hessian components and an optional Frobenius-based mask.

        Returns
        -------
        h_mask : xp.ndarray[bool]
        h_components : dict
            For 3D: keys 'hxx','hxy','hxz','hyy','hyz','hzz'
            For 2D: keys 'hxx','hxy','hyy'
        """
        # Ensure working dtype
        image = image.astype(self.work_dtype, copy=False)
        spacing = self._get_spacing(image.ndim)

        if image.ndim == 2:  # 2D dataset
            if self.low_memory:
                g0 = self.xp.gradient(image, spacing[0], axis=0)
                hxx = self.xp.gradient(g0, spacing[0], axis=0).astype(
                    self.work_dtype, copy=False
                )
                hxy = self.xp.gradient(g0, spacing[1], axis=1).astype(
                    self.work_dtype, copy=False
                )
                del g0
                g1 = self.xp.gradient(image, spacing[1], axis=1)
                hyy = self.xp.gradient(g1, spacing[1], axis=1).astype(
                    self.work_dtype, copy=False
                )
                del g1
            else:
                g0, g1 = self.xp.gradient(image, *spacing)  # axes 0,1
                hxx = self.xp.gradient(g0, spacing[0], axis=0).astype(
                    self.work_dtype, copy=False
                )
                hxy = self.xp.gradient(g0, spacing[1], axis=1).astype(
                    self.work_dtype, copy=False
                )
                hyy = self.xp.gradient(g1, spacing[1], axis=1).astype(
                    self.work_dtype, copy=False
                )

            # Frobenius norm: hxx^2 + hyy^2 + 2*hxy^2
            frob_sq = hxx ** 2 + hyy ** 2 + 2.0 * (hxy ** 2)
            h_components = {"hxx": hxx, "hxy": hxy, "hyy": hyy}
        elif image.ndim == 3:  # 3D dataset
            if self.low_memory: 
                g0 = self.xp.gradient(image, spacing[0], axis=0)
                hxx = self.xp.gradient(g0, spacing[0], axis=0).astype(
                    self.work_dtype, copy=False
                )
                hxy = self.xp.gradient(g0, spacing[1], axis=1).astype(
                    self.work_dtype, copy=False
                )
                hxz = self.xp.gradient(g0, spacing[2], axis=2).astype(
                    self.work_dtype, copy=False
                )
                del g0
                g1 = self.xp.gradient(image, spacing[1], axis=1)
                hyy = self.xp.gradient(g1, spacing[1], axis=1).astype(
                    self.work_dtype, copy=False
                )
                hyz = self.xp.gradient(g1, spacing[2], axis=2).astype(
                    self.work_dtype, copy=False
                )
                del g1
                g2 = self.xp.gradient(image, spacing[2], axis=2)
                hzz = self.xp.gradient(g2, spacing[2], axis=2).astype(
                    self.work_dtype, copy=False
                )
                del g2
            else:
                g0, g1, g2 = self.xp.gradient(image, *spacing)  # axes 0,1,2
                hxx = self.xp.gradient(g0, spacing[0], axis=0).astype(
                    self.work_dtype, copy=False
                )
                hxy = self.xp.gradient(g0, spacing[1], axis=1).astype(
                    self.work_dtype, copy=False
                )
                hxz = self.xp.gradient(g0, spacing[2], axis=2).astype(
                    self.work_dtype, copy=False
                )
                hyy = self.xp.gradient(g1, spacing[1], axis=1).astype(
                    self.work_dtype, copy=False
                )
                hyz = self.xp.gradient(g1, spacing[2], axis=2).astype(
                    self.work_dtype, copy=False
                )
                hzz = self.xp.gradient(g2, spacing[2], axis=2).astype(
                    self.work_dtype, copy=False
                )

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

        # Normalize Frobenius norm by max absolute Hessian component for stability
        max_abs = 0.0
        for comp in h_components.values():
            if comp.size > 0:
                max_abs = max(max_abs, float(self.xp.max(self.xp.abs(comp))))
        if max_abs <= 0:
            max_abs = 1.0
        frobenius_norm = self.xp.sqrt(frob_sq) / max_abs

        if mask:
            h_mask = self._get_frob_mask(frobenius_norm)
        else:
            h_mask = self.xp.ones_like(image, dtype=bool)

        return h_mask, h_components

    # -------------------------------------------------------------------------
    # Eigenvalues and vesselness
    # -------------------------------------------------------------------------
    def _safe_eigvalsh(self, H_chunk):
        """
        Compute eigenvalues robustly, with GPU OOM fallback to smaller chunks
        or CPU if necessary. Eigenvalues are sorted by absolute value.
        """

        def _eig_backend(arr):
            ev = self.xp.linalg.eigvalsh(arr)
            # sort by absolute value as in original implementation
            order = self.xp.argsort(self.xp.abs(ev), axis=1)
            ev = self.xp.take_along_axis(ev, order, axis=1)
            return ev

        if self.device_type != "cuda":
            return _eig_backend(H_chunk)

        # GPU case
        try:
            return _eig_backend(H_chunk)
        except Exception as e:
            if not self._is_oom_error(e):
                raise

            n = H_chunk.shape[0]
            if n > 1:
                # Split chunk and recurse to stay on GPU if possible
                mid = n // 2
                ev1 = self._safe_eigvalsh(H_chunk[:mid])
                ev2 = self._safe_eigvalsh(H_chunk[mid:])
                return self.xp.concatenate([ev1, ev2], axis=0)

            if self.force_device:
                raise

            # Fall back to CPU for this chunk
            try:
                H_cpu = self.xp.asnumpy(H_chunk)
            except Exception:
                H_cpu = np.asarray(H_chunk)
            ev_cpu = np.linalg.eigvalsh(H_cpu)
            order = np.argsort(np.abs(ev_cpu), axis=1)
            ev_cpu = np.take_along_axis(ev_cpu, order, axis=1)
            return self.xp.asarray(ev_cpu)

    def _compute_chunkwise_eigenvalues(self, hessian_matrices, chunk_size=1e6):
        """
        Backwards-compatible helper: compute eigenvalues of Hessian matrices in chunks.

        Parameters
        ----------
        hessian_matrices : xp.ndarray, shape (N, D, D)
        chunk_size : int
            Number of voxels per chunk.

        Returns
        -------
        eigenvalues_flat : xp.ndarray, shape (N, D)
        """
        chunk_size = int(chunk_size) if chunk_size is not None else hessian_matrices.shape[0]
        total_voxels = int(hessian_matrices.shape[0])

        eigenvalues_list = []
        if chunk_size <= 0:
            chunk_size = total_voxels

        for start_idx in range(0, total_voxels, chunk_size):
            end_idx = min(start_idx + chunk_size, total_voxels)
            H_chunk = hessian_matrices[start_idx:end_idx]
            eig_chunk = self._safe_eigvalsh(H_chunk)
            eigenvalues_list.append(eig_chunk)

        if len(eigenvalues_list) == 1:
            return eigenvalues_list[0]

        eigenvalues_flat = self.xp.concatenate(eigenvalues_list, axis=0)
        return eigenvalues_flat

    def _compute_vesselness_chunkwise(self, h_components, h_mask, gamma_sq):
        """
        Compute vesselness in chunks over the masked voxels to control memory.
        """
        # Coordinates of voxels where we evaluate the Hessian
        coords = self.xp.where(h_mask)
        total_voxels = int(coords[0].size)
        if total_voxels == 0:
            # Return an all-zero volume
            template = next(iter(h_components.values()))
            return self.xp.zeros_like(template, dtype=self.work_dtype)

        chunk_size = self.max_chunk_voxels
        if chunk_size is None or chunk_size <= 0:
            chunk_size = total_voxels

        # Preallocate 1D buffer for vesselness values on masked voxels
        vessel_masked = self.xp.zeros(total_voxels, dtype=self.work_dtype)

        # Iterate over chunks
        for start in range(0, total_voxels, chunk_size):
            end = min(start + chunk_size, total_voxels)
            idx_chunk = tuple(c[start:end] for c in coords)

            if self.im_info.no_z:
                # 2D Hessian
                hxx_c = h_components["hxx"][idx_chunk]
                hxy_c = h_components["hxy"][idx_chunk]
                hyy_c = h_components["hyy"][idx_chunk]
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
                eigenvalues = self.xp.stack([eig1, eig2], axis=1)
            else:
                # 3D Hessian
                hxx_c = h_components["hxx"][idx_chunk]
                hxy_c = h_components["hxy"][idx_chunk]
                hxz_c = h_components["hxz"][idx_chunk]
                hyy_c = h_components["hyy"][idx_chunk]
                hyz_c = h_components["hyz"][idx_chunk]
                hzz_c = h_components["hzz"][idx_chunk]
                H_chunk = self.xp.stack(
                    [
                        self.xp.stack([hxx_c, hxy_c, hxz_c], axis=-1),
                        self.xp.stack([hxy_c, hyy_c, hyz_c], axis=-1),
                        self.xp.stack([hxz_c, hyz_c, hzz_c], axis=-1),
                    ],
                    axis=-2,
                )
                eigenvalues = self._safe_eigvalsh(H_chunk)
            v_chunk = self._filter_hessian(eigenvalues, gamma_sq=gamma_sq)
            vessel_masked[start:end] = v_chunk.astype(self.work_dtype, copy=False)

        # Scatter back into full volume
        template = next(iter(h_components.values()))
        vesselness = self.xp.zeros_like(template, dtype=self.work_dtype)
        vesselness[coords] = vessel_masked
        return vesselness

    def _filter_hessian(self, eigenvalues, gamma_sq):
        """
        Apply the Frangi filter to Hessian eigenvalues to detect vessel-like structures.

        Parameters
        ----------
        eigenvalues : xp.ndarray, shape (N, 2) or (N, 3)
            Eigenvalues sorted by absolute value.
        gamma_sq : float
            Squared gamma for vesselness calculation.

        Returns
        -------
        filtered_im : xp.ndarray, shape (N,)
        """
        if self.im_info.no_z:
            # 2D: eigenvalues[:, 0] is smallest |λ|, eigenvalues[:, 1] largest |λ|
            l1 = eigenvalues[:, 0]
            l2 = eigenvalues[:, 1]

            rb_sq = (self.xp.abs(l1) / (self.xp.abs(l2) + 1e-12)) ** 2
            s_sq = l1 ** 2 + l2 ** 2
            filtered_im = self.xp.exp(-(rb_sq / self.beta_sq)) * (
                1.0 - self.xp.exp(-(s_sq / gamma_sq))
            )
        else:
            # 3D: eigenvalues[:, 0] <= eigenvalues[:, 1] <= eigenvalues[:, 2] in |·|
            l1 = eigenvalues[:, 0]
            l2 = eigenvalues[:, 1]
            l3 = eigenvalues[:, 2]

            ra_sq = (self.xp.abs(l2) / (self.xp.abs(l3) + 1e-12)) ** 2
            rb_sq = (self.xp.abs(l2) / (self.xp.sqrt(self.xp.abs(l2 * l3)) + 1e-12)) ** 2
            s_sq = l1 ** 2 + l2 ** 2 + l3 ** 2

            filtered_im = (
                (1.0 - self.xp.exp(-(ra_sq / self.alpha_sq)))
                * self.xp.exp(-(rb_sq / self.beta_sq))
                * (1.0 - self.xp.exp(-(s_sq / gamma_sq)))
            )

        # Exclude bright structures (vessels expected to be darker than background)
        if not self.im_info.no_z:
            filtered_im[eigenvalues[:, 2] > 0] = 0.0
        filtered_im[eigenvalues[:, 1] > 0] = 0.0

        # Clean up NaNs/Infs
        filtered_im = self.xp.nan_to_num(
            filtered_im, nan=0.0, posinf=0.0, neginf=0.0
        )
        return filtered_im

    # -------------------------------------------------------------------------
    # LoG filter
    # -------------------------------------------------------------------------
    def _filter_log(self, frame, mask):
        """
        Apply a Laplacian-of-Gaussian filter across scales and retain the minimum
        response (as in a multi-scale LoG).
        """
        frame = frame.astype(self.work_dtype, copy=False)
        lapofg = None
        for i, s in enumerate(self.sigmas):
            sigma_vec = self._get_sigma_vec(s)
            current_lapofg = -self.ndi.gaussian_laplace(frame, sigma_vec) * (float(s) ** 2)
            current_lapofg = current_lapofg * mask
            if i == 0:
                lapofg = current_lapofg
            else:
                min_indices = current_lapofg > lapofg
                # min_indices = current_lapofg < lapofg
                lapofg[min_indices] = current_lapofg[min_indices]
                
        # Scale lapofg between 0 and 1
        # lapofg_min = self.xp.min(lapofg)
        lapofg[lapofg < 0] = 0.0
        lapofg_max = self.xp.max(lapofg)
        lapofg = (lapofg) / (lapofg_max + 1e-12)
        return lapofg / 10.0

    # -------------------------------------------------------------------------
    # Per-frame processing
    # -------------------------------------------------------------------------
    def _precompute_gammas(self, frame):
        gammas = []
        for sigma in self.sigmas:
            gammas.append(self._estimate_gamma(frame, sigma))
        return gammas

    def _compute_vesselness(self, frame, mask=True):
        vesselness = self.xp.zeros_like(frame, dtype=self.work_dtype)
        masks = self.xp.ones_like(frame, dtype=bool)

        # Start from raw frame and build Gaussian scales incrementally
        gauss = frame.astype(self.work_dtype, copy=False)
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

            gamma = self._calculate_gamma(gauss)
            gamma_sq = 2.0 * (float(gamma) ** 2)

            h_mask, h_components = self._compute_hessian(gauss, mask=mask)
            if not self.xp.any(h_mask):
                continue

            vessel_scale = self._compute_vesselness_chunkwise(
                h_components, h_mask, gamma_sq=gamma_sq
            )

            vesselness = self.xp.maximum(vesselness, vessel_scale)
            masks &= h_mask

        return vesselness, masks

    def _compute_chunk_shape(self, shape, max_chunk_voxels):
        if max_chunk_voxels is None or max_chunk_voxels <= 0:
            return tuple(shape)
        chunk = list(shape)
        while int(np.prod(chunk)) > max_chunk_voxels:
            idx = int(np.argmax(chunk))
            chunk[idx] = max(1, int(np.ceil(chunk[idx] / 2)))
        return tuple(chunk)

    def _iter_chunks(self, shape, chunk_shape, halo):
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

    def _run_frame_chunked(self, t, mask=True, max_chunk_voxels=None):
        frame_cpu = self.im_memmap[t, ...]
        shape = frame_cpu.shape
        chunk_voxels = int(max_chunk_voxels or self.max_chunk_voxels or int(np.prod(shape)))
        halo = self.halo or (0,) * len(shape)

        while True:
            try:
                chunk_shape = self._compute_chunk_shape(shape, chunk_voxels)
                vessel_out = np.zeros(shape, dtype=self.work_dtype)
                for core, ext, core_in_ext in self._iter_chunks(shape, chunk_shape, halo):
                    chunk = frame_cpu[ext]
                    chunk_xp = self.xp.asarray(chunk, dtype=self.work_dtype)
                    vessel_chunk, mask_chunk = self._compute_vesselness(
                        chunk_xp, mask=mask
                    )
                    vessel_chunk = vessel_chunk * mask_chunk
                    if self.device_type == "cuda":
                        vessel_chunk = vessel_chunk.get()
                    vessel_out[core] = vessel_chunk[core_in_ext]
                if self.remove_edges:
                    vessel_out = self._remove_edges(vessel_out)
                return vessel_out
            except Exception as exc:
                if not self._is_oom_error(exc):
                    raise
                self._free_gpu_memory()
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
        # gammas = self._precompute_gammas(frame_cpu)

        if self.low_memory:
            return self._run_frame_chunked(t, mask=mask)

        try:
            frame = self.xp.asarray(frame_cpu, dtype=self.work_dtype)
            vesselness, masks = self._compute_vesselness(frame, mask=mask)
            vesselness = vesselness * masks
            if self.im_info.no_z:
                blobness = self._filter_log(frame, mask=masks if mask else self.xp.ones_like(frame, bool))
                blobness = self.xp.maximum(blobness, 0)  # keep bright-blob response only
                vesselness = self.xp.maximum(vesselness, blobness)
            if self.remove_edges:
                vesselness = self._remove_edges(vesselness)
            return vesselness
        except Exception as exc:
            if not self._is_oom_error(exc):
                raise
            self._free_gpu_memory()
            # Try chunked on current backend
            try:
                return self._run_frame_chunked(t, gammas, mask=mask)
            except Exception as exc2:
                if not self._is_oom_error(exc2):
                    raise
                if self.device_type == "cuda" and not self.force_device:
                    self._switch_to_cpu()
                    return self._run_frame_chunked(t, gammas, mask=mask)
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
        if positive.size == 0:
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
            # 2D case
            if frangi_frame.size == 0:
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
                if slice_im.size == 0:
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

    def run(self, mask=True):
        """
        Main entry point: run the Frangi filter over the image.
        """
        logger.info("Running Frangi filter.")
        device = adaptive_run.normalize_device(self.device)
        gpu_ok = adaptive_run.gpu_available()
        if device == "gpu" and not gpu_ok:
            logger.warning("Filter: GPU requested but not available; falling back to CPU.")
        if device == "cpu" or not gpu_ok:
            device_order = ["cpu"]
        else:
            device_order = ["gpu", "cpu"]

        start_low_memory = bool(self.low_memory) or adaptive_run.should_use_low_memory(
            self.im_info, include_gpu="gpu" in device_order
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
                if adaptive_run.is_gpu_unavailable_error(exc) and dev == "gpu":
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
