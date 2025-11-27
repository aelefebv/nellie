from itertools import combinations_with_replacement  # kept for compatibility, may be unused

from nellie import logger
from nellie.im_info.verifier import ImInfo
from nellie.utils.general import bbox
import numpy as np
from nellie import ndi, xp, device_type

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
        min_radius_um: float = 0.20,
        max_radius_um: float = 1.0,
        alpha_sq: float = 0.5,
        beta_sq: float = 0.5,
        frob_thresh=None,
        viewer=None,
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
        low_memory : bool
            If True, prefer strategies that reduce peak memory at the cost of speed
            (e.g. smaller eigen-decomposition chunks).
        max_chunk_voxels : int
            Maximum number of voxels per eigen-decomposition chunk.
        max_threshold_samples : int
            Maximum number of samples to use when estimating thresholds
            (triangle / Otsu) from very large arrays.
        """
        self.im_info = im_info
        if not self.im_info.no_z:
            self.z_ratio = self.im_info.dim_res["Z"] / self.im_info.dim_res["X"]
        self.num_t = num_t
        if num_t is None and not self.im_info.no_t:
            self.num_t = im_info.shape[im_info.axes.index("T")]
        self.remove_edges = remove_edges
        # either (roughly) diffraction limit, or pixel size, whichever is larger
        self.min_radius_um = max(min_radius_um, self.im_info.dim_res["X"])
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

        self.viewer = viewer

        # Optimization-related settings
        self.low_memory = low_memory
        self.max_chunk_voxels = int(max_chunk_voxels)
        self.max_threshold_samples = int(max_threshold_samples)

        # Dtypes
        self.work_dtype = "float32"
        self.out_dtype = "float32"

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

        logger.debug(
            f"Calculated sigma step size = {sigma_step_size_calculated}. Sigmas = {self.sigmas}"
        )

    # -------------------------------------------------------------------------
    # Threshold helpers
    # -------------------------------------------------------------------------
    def _subsample_for_thresholds(self, arr):
        """
        Subsample large arrays before passing to triangle / Otsu threshold
        to reduce memory and runtime.
        """
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

        gamma_tri = triangle_threshold(positive)
        gamma_otsu, _ = otsu_threshold(positive)
        gamma = float(min(gamma_tri, gamma_otsu))
        # Avoid division by zero downstream
        if gamma <= 0:
            gamma = np.finfo(np.float32).eps
        return gamma

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
        inf_mask = xp.isinf(frobenius_norm)
        if xp.any(inf_mask):
            finite_vals = frobenius_norm[~inf_mask]
            max_finite = float(xp.max(finite_vals)) if finite_vals.size > 0 else 0.0
            frobenius_norm = frobenius_norm.copy()
            frobenius_norm[inf_mask] = max_finite

        if self.frob_thresh is None:
            positive = self._subsample_for_thresholds(frobenius_norm)
            if positive.size == 0:
                frobenius_threshold = 0.0
            else:
                frob_triangle_thresh = triangle_threshold(positive)
                frob_otsu_thresh, _ = otsu_threshold(positive)
                frobenius_threshold = float(min(frob_triangle_thresh, frob_otsu_thresh))
        else:
            frobenius_threshold = float(self.frob_thresh)

        mask = frobenius_norm > frobenius_threshold
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
        gradients = xp.gradient(image)

        if image.ndim == 2:
            g0, g1 = gradients  # axes 0,1
            hxx = xp.gradient(g0, axis=0).astype(self.work_dtype, copy=False)
            hxy = xp.gradient(g0, axis=1).astype(self.work_dtype, copy=False)
            hyy = xp.gradient(g1, axis=1).astype(self.work_dtype, copy=False)

            # Frobenius norm: hxx^2 + hyy^2 + 2*hxy^2
            frob_sq = hxx ** 2 + hyy ** 2 + 2.0 * (hxy ** 2)
            h_components = {"hxx": hxx, "hxy": hxy, "hyy": hyy}
        elif image.ndim == 3:
            g0, g1, g2 = gradients  # axes 0,1,2
            hxx = xp.gradient(g0, axis=0).astype(self.work_dtype, copy=False)
            hxy = xp.gradient(g0, axis=1).astype(self.work_dtype, copy=False)
            hxz = xp.gradient(g0, axis=2).astype(self.work_dtype, copy=False)
            hyy = xp.gradient(g1, axis=1).astype(self.work_dtype, copy=False)
            hyz = xp.gradient(g1, axis=2).astype(self.work_dtype, copy=False)
            hzz = xp.gradient(g2, axis=2).astype(self.work_dtype, copy=False)

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
                max_abs = max(max_abs, float(xp.max(xp.abs(comp))))
        if max_abs <= 0:
            max_abs = 1.0
        frobenius_norm = xp.sqrt(frob_sq) / max_abs

        if mask:
            h_mask = self._get_frob_mask(frobenius_norm)
        else:
            h_mask = xp.ones_like(image, dtype=bool)

        if self.remove_edges:
            h_mask = self._remove_edges(h_mask)

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
            ev = xp.linalg.eigvalsh(arr)
            # sort by absolute value as in original implementation
            order = xp.argsort(xp.abs(ev), axis=1)
            ev = xp.take_along_axis(ev, order, axis=1)
            return ev

        if device_type != "cuda":
            return _eig_backend(H_chunk)

        # GPU case
        try:
            return _eig_backend(H_chunk)
        except Exception as e:
            # Try to detect out-of-memory and fall back
            is_oom = False
            try:
                # CuPy-style OOM class if xp is cupy
                import cupy  # type: ignore  # noqa

                is_oom = isinstance(e, cupy.cuda.memory.OutOfMemoryError)
            except Exception:
                # Fallback heuristic if CuPy is not importable here
                is_oom = "OutOfMemory" in repr(e)

            if not is_oom:
                raise

            n = H_chunk.shape[0]
            if n > 1 and not self.low_memory:
                # Split chunk and recurse to stay on GPU
                mid = n // 2
                ev1 = self._safe_eigvalsh(H_chunk[:mid])
                ev2 = self._safe_eigvalsh(H_chunk[mid:])
                return xp.concatenate([ev1, ev2], axis=0)

            # Fall back to CPU for this chunk
            H_cpu = xp.asnumpy(H_chunk)
            ev_cpu = np.linalg.eigvalsh(H_cpu)
            order = np.argsort(np.abs(ev_cpu), axis=1)
            ev_cpu = np.take_along_axis(ev_cpu, order, axis=1)
            return xp.asarray(ev_cpu)

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

        eigenvalues_flat = xp.concatenate(eigenvalues_list, axis=0)
        return eigenvalues_flat

    def _compute_vesselness_chunkwise(self, h_components, h_mask, gamma_sq):
        """
        Compute vesselness in chunks over the masked voxels to control memory.
        """
        # Coordinates of voxels where we evaluate the Hessian
        coords = xp.where(h_mask)
        total_voxels = int(coords[0].size)
        if total_voxels == 0:
            # Return an all-zero volume
            template = next(iter(h_components.values()))
            return xp.zeros_like(template, dtype=self.work_dtype)

        chunk_size = self.max_chunk_voxels
        if chunk_size is None or chunk_size <= 0:
            chunk_size = total_voxels

        # Preallocate 1D buffer for vesselness values on masked voxels
        vessel_masked = xp.zeros(total_voxels, dtype=self.work_dtype)

        # Iterate over chunks
        for start in range(0, total_voxels, chunk_size):
            end = min(start + chunk_size, total_voxels)
            idx_chunk = tuple(c[start:end] for c in coords)

            if self.im_info.no_z:
                # 2D Hessian
                hxx_c = h_components["hxx"][idx_chunk]
                hxy_c = h_components["hxy"][idx_chunk]
                hyy_c = h_components["hyy"][idx_chunk]

                H_chunk = xp.stack(
                    [
                        xp.stack([hxx_c, hxy_c], axis=-1),
                        xp.stack([hxy_c, hyy_c], axis=-1),
                    ],
                    axis=-2,
                )
            else:
                # 3D Hessian
                hxx_c = h_components["hxx"][idx_chunk]
                hxy_c = h_components["hxy"][idx_chunk]
                hxz_c = h_components["hxz"][idx_chunk]
                hyy_c = h_components["hyy"][idx_chunk]
                hyz_c = h_components["hyz"][idx_chunk]
                hzz_c = h_components["hzz"][idx_chunk]

                H_chunk = xp.stack(
                    [
                        xp.stack([hxx_c, hxy_c, hxz_c], axis=-1),
                        xp.stack([hxy_c, hyy_c, hyz_c], axis=-1),
                        xp.stack([hxz_c, hyz_c, hzz_c], axis=-1),
                    ],
                    axis=-2,
                )

            eigenvalues = self._safe_eigvalsh(H_chunk)
            v_chunk = self._filter_hessian(eigenvalues, gamma_sq=gamma_sq)
            vessel_masked[start:end] = v_chunk.astype(self.work_dtype, copy=False)

        # Scatter back into full volume
        template = next(iter(h_components.values()))
        vesselness = xp.zeros_like(template, dtype=self.work_dtype)
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

            rb_sq = (xp.abs(l1) / (xp.abs(l2) + 1e-12)) ** 2
            s_sq = l1 ** 2 + l2 ** 2
            filtered_im = xp.exp(-(rb_sq / self.beta_sq)) * (1.0 - xp.exp(-(s_sq / gamma_sq)))
        else:
            # 3D: eigenvalues[:, 0] <= eigenvalues[:, 1] <= eigenvalues[:, 2] in |·|
            l1 = eigenvalues[:, 0]
            l2 = eigenvalues[:, 1]
            l3 = eigenvalues[:, 2]

            ra_sq = (xp.abs(l2) / (xp.abs(l3) + 1e-12)) ** 2
            rb_sq = (xp.abs(l2) / (xp.sqrt(xp.abs(l2 * l3)) + 1e-12)) ** 2
            s_sq = l1 ** 2 + l2 ** 2 + l3 ** 2

            filtered_im = (
                (1.0 - xp.exp(-(ra_sq / self.alpha_sq)))
                * xp.exp(-(rb_sq / self.beta_sq))
                * (1.0 - xp.exp(-(s_sq / gamma_sq)))
            )

        # Exclude bright structures (vessels expected to be darker than background)
        if not self.im_info.no_z:
            filtered_im[eigenvalues[:, 2] > 0] = 0.0
        filtered_im[eigenvalues[:, 1] > 0] = 0.0

        # Clean up NaNs/Infs
        filtered_im = xp.nan_to_num(
            filtered_im, nan=0.0, posinf=0.0, neginf=0.0
        )
        return filtered_im

    # -------------------------------------------------------------------------
    # LoG filter (optional, kept mostly intact but with small cleanups)
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
            current_lapofg = -ndi.gaussian_laplace(frame, sigma_vec) * (float(s) ** 2)
            current_lapofg = current_lapofg * mask
            if i == 0:
                lapofg = current_lapofg
            else:
                min_indices = current_lapofg < lapofg
                lapofg[min_indices] = current_lapofg[min_indices]
        return lapofg

    # -------------------------------------------------------------------------
    # Per-frame processing
    # -------------------------------------------------------------------------
    def _run_frame(self, t, mask=True):
        """
        Run the Frangi filter for a single timepoint using Gaussian scale-space
        with a cascaded Gaussian to avoid recomputing from raw at each sigma.
        """
        logger.info(f"Running Frangi filter on t={t}.")

        # Load this frame once into working dtype
        frame = xp.asarray(self.im_memmap[t, ...], dtype=self.work_dtype)

        vesselness = xp.zeros_like(frame, dtype=self.work_dtype)
        masks = xp.ones_like(frame, dtype=bool)

        # Start from raw frame and build Gaussian scales incrementally
        gauss = frame.copy()
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
                ndi.gaussian_filter(
                    gauss,
                    sigma=sigma_vec_delta,
                    output=gauss,
                    mode="reflect",
                    cval=0.0,
                    truncate=3.0,
                )

            prev_sigma = sigma

            # Compute gamma and Hessian-based vesselness for this scale
            gamma = self._calculate_gamma(gauss)
            gamma_sq = 2.0 * (float(gamma) ** 2)

            h_mask, h_components = self._compute_hessian(gauss, mask=mask)
            if not xp.any(h_mask):
                # Nothing to do at this scale
                continue

            vessel_scale = self._compute_vesselness_chunkwise(
                h_components, h_mask, gamma_sq=gamma_sq
            )

            # Take maximum vesselness across scales
            max_indices = vessel_scale > vesselness
            vesselness[max_indices] = vessel_scale[max_indices]

            # Update accumulated mask (voxels must be valid at all scales)
            masks &= h_mask

        vesselness = vesselness * masks
        return vesselness

    # -------------------------------------------------------------------------
    # Post-processing helpers
    # -------------------------------------------------------------------------
    def _mask_volume(self, frangi_frame):
        """
        Apply a simple percentile-based threshold and binary opening to refine
        the vesselness mask.
        """
        positive = frangi_frame[frangi_frame > 0]
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
            rmin, rmax, cmin, cmax = bbox(frangi_frame)
            height = max(0, rmax - rmin + 1)
            if height <= 0:
                return frangi_frame
            margin = min(15, height)
            frangi_frame[rmin : rmin + margin, :] = 0
            frangi_frame[rmax - margin + 1 : rmax + 1, :] = 0
        else:
            # 3D case: assume Z is axis 0
            num_z = self.im_info.shape[self.im_info.axes.index("Z")]
            margin = 15
            for z_idx in range(num_z):
                slice_im = frangi_frame[z_idx, ...]
                if slice_im.size == 0:
                    continue
                rmin, rmax, cmin, cmax = bbox(slice_im)
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
            total = float(xp.sum(frangi_frame))
            if total > 0.0:
                frangi_frame = self._mask_volume(frangi_frame)

            filtered_im = frangi_frame

            # Move result back to CPU for memmap storage when using GPU
            if device_type == "cuda":
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
        self._get_t()
        self._allocate_memory()
        self._set_default_sigmas()
        self._run_filter(mask=mask)


if __name__ == "__main__":
    im_path = r"F:\2024_06_26_SD_ExM_nhs_u2OS_488+578_cropped.tif"
    im_info = ImInfo(
        im_path, dim_res={"T": 1, "Z": 0.2, "Y": 0.1, "X": 0.1}, dimension_order="ZYX"
    )
    filter_im = Filter(im_info)
    filter_im.run()