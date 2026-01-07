"""
Hu moment-based tracking for labeled objects across timepoints.

This module provides the HuMomentTracking class for tracking objects using
Hu moment invariants and optical flow interpolation.
"""
import numpy as np
import scipy.ndimage as sp_ndi
from dataclasses import dataclass
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

from nellie.utils.base_logger import logger

from nellie.im_info.verifier import ImInfo

# Optional: GPU OOM handling (safe if CuPy is not installed)
try:
    import cupy  # type: ignore
    GPU_OOM_ERRORS = (cupy.cuda.memory.OutOfMemoryError,)
except Exception:  # ImportError or others
    GPU_OOM_ERRORS = ()


@dataclass
class _FrameFeatures:
    """Internal container for per-frame features."""
    coords_voxel: np.ndarray  # (N, 2) or (N, 3) indices in voxel space
    coords_phys: np.ndarray   # (N, 2) or (N, 3) in micrometers
    stats: object             # xp.ndarray, shape (N, F_stats)
    hu: object                # xp.ndarray, shape (N, F_hu)


class HuMomentTracking:
    """
    A class for tracking objects in microscopy images using Hu moments and distance-based matching.

    This version is optimized for:
      - Large images (memory-aware ROI extraction, streaming modes).
      - Optional GPU acceleration (via CuPy).
      - Fallback to sparse KDTree-based matching when dense matching is too large.

    Attributes
    ----------
    im_info : ImInfo
        An object containing image metadata and memory-mapped image data.
    device : {"auto", "cpu", "gpu"}
        Backend selection. "auto" uses GPU if available, otherwise CPU.
    device_type : str
        Resolved backend type ("cuda" or "cpu").
    num_t : int
        Number of timepoints in the image.
    max_distance_um : float
        Maximum allowed velocity (micrometers/second), scaled by time resolution for per-frame distance.
    low_memory : bool
        If True, prefer streaming ROI extraction and sparse matching to reduce peak memory.
    scaling : tuple
        Scaling factors for Z, Y, and X dimensions.
    shape : tuple
        Shape of the input image.
    im_memmap : np.ndarray or None
        Memory-mapped original image data.
    im_frangi_memmap : np.ndarray or None
        Memory-mapped Frangi-filtered image data.
    im_distance_memmap : np.ndarray or None
        Memory-mapped distance transform data.
    im_marker_memmap : np.ndarray or None
        Memory-mapped marker data for object tracking.
    flow_vector_array_path : str or None
        Path to save the flow vector array.

    Internal tuning parameters
    --------------------------
    mode : {"auto", "dense", "sparse"}
        Matching mode. "auto" selects dense for small problems, sparse for large ones.
    max_dense_pairs : int
        Maximum number of pairwise matches (N_post * N_pre) before switching to sparse matching.
    max_dense_roi_voxels_cpu : int
        Rough upper bound on total ROI voxels for dense ROI extraction on CPU.
    max_dense_roi_voxels_gpu : int
        Rough upper bound on total ROI voxels for dense ROI extraction on GPU.
    """

    def __init__(self, im_info: ImInfo, num_t=None,
                 max_distance_um=1.0,
                 viewer=None,
                 device: str = "auto",
                 mode: str = "auto",
                 max_dense_pairs: int = int(1e7),
                 max_dense_roi_voxels_cpu: int = int(5e7),
                 max_dense_roi_voxels_gpu: int = int(2e7),
                 low_memory: bool = False):
        self.im_info = im_info

        # If no time dimension, nothing to do.
        if self.im_info.no_t:
            return

        self.num_t = num_t
        if num_t is None and not self.im_info.no_t:
            self.num_t = im_info.shape[im_info.axes.index('T')]

        # Voxel-to-physical scaling (per axis)
        if self.im_info.no_z:
            self.scaling = (im_info.dim_res['Y'], im_info.dim_res['X'])
        else:
            self.scaling = (im_info.dim_res['Z'], im_info.dim_res['Y'], im_info.dim_res['X'])

        # Max velocity (um/s) scaled by dt to per-frame distance; enforce a sensible minimum
        dt = self.im_info.dim_res.get('T') or 1.0
        if self.im_info.dim_res.get('T') is None:
            logger.warning("Time resolution missing; assuming 1.0s for max_distance_um scaling.")
        self.max_distance_um = max(max_distance_um * dt, 0.5)

        self.shape = ()

        self.im_memmap = None
        self.im_frangi_memmap = None
        self.im_distance_memmap = None
        self.im_marker_memmap = None
        self.flow_vector_array_path = None

        self.debug = None
        self.viewer = viewer

        # Backend / device info
        self.device = device
        self.xp, self.ndi, self.device_type = self._resolve_backend(device)
        self._on_gpu = self.device_type == "cuda"

        # Matching / ROI mode tuning
        self.mode = mode  # "auto", "dense", "sparse"
        self.low_memory = bool(low_memory)
        self.max_dense_pairs = int(max_dense_pairs)
        self.max_dense_roi_voxels_cpu = int(max_dense_roi_voxels_cpu)
        self.max_dense_roi_voxels_gpu = int(max_dense_roi_voxels_gpu)

    # -------------------------------------------------------------------------
    # Backend helpers
    # -------------------------------------------------------------------------

    def _resolve_backend(self, device):
        device = (device or "auto").lower()
        if device not in ("auto", "cpu", "gpu", "cuda"):
            raise ValueError(f"Unsupported device '{device}'. Use 'auto', 'cpu', or 'gpu'.")

        if device in ("gpu", "cuda"):
            xp_mod, ndi_mod = self._try_import_cupy(require=True)
            return xp_mod, ndi_mod, "cuda"
        if device == "cpu":
            return np, sp_ndi, "cpu"

        # auto
        xp_mod, ndi_mod = self._try_import_cupy(require=False)
        if xp_mod is not None:
            return xp_mod, ndi_mod, "cuda"
        return np, sp_ndi, "cpu"

    def _try_import_cupy(self, require):
        try:
            import cupy
            import cupyx.scipy.ndimage as ndi_mod
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

        return cupy, ndi_mod

    def _is_oom_error(self, exc):
        if isinstance(exc, MemoryError):
            return True
        if not self._on_gpu:
            return False
        try:
            import cupy
        except Exception:
            return "OutOfMemory" in repr(exc)
        return isinstance(exc, cupy.cuda.memory.OutOfMemoryError)

    def _free_gpu_memory(self):
        if not self._on_gpu:
            return
        try:
            self.xp.get_default_memory_pool().free_all_blocks()
        except Exception:
            return

    def _switch_to_cpu(self):
        self.xp = np
        self.ndi = sp_ndi
        self.device_type = "cpu"
        self._on_gpu = False

    def _to_cpu_array(self, arr):
        if isinstance(arr, np.ndarray):
            return arr
        if hasattr(arr, "get"):
            return arr.get()
        return np.asarray(arr)

    # -------------------------------------------------------------------------
    # Moment and feature computations
    # -------------------------------------------------------------------------

    def _calculate_normalized_moments(self, images):
        """
        Calculates the normalized moments for a set of 2D images.

        Parameters
        ----------
        images : xp.ndarray, shape (N, H, W)

        Returns
        -------
        xp.ndarray
            Normalized moments eta for each image, shape (N, 4, 4).
        """
        xp = self.xp
        # Broadcasting-heavy implementation for speed.
        num_images, height, width = images.shape
        extended_images = images[:, :, :, None, None]  # (N, H, W, 1, 1)

        # pre-compute meshgrid
        x, y = xp.meshgrid(xp.arange(width), xp.arange(height))
        x = x[None, :, :, None, None]
        y = y[None, :, :, None, None]

        powers = xp.arange(4)
        powers_x = powers[None, None, None, :, None]
        powers_y = powers[None, None, None, None, :]

        # raw moments M_{pq}
        M = xp.sum(extended_images * (x ** powers_x) * (y ** powers_y), axis=(1, 2))  # (N, 4, 4)

        # centroids
        x_bar = M[:, 1, 0] / (M[:, 0, 0] + 1e-12)
        y_bar = M[:, 0, 1] / (M[:, 0, 0] + 1e-12)
        x_bar = x_bar[:, None, None, None, None]
        y_bar = y_bar[:, None, None, None, None]

        # central moments mu_{pq}
        mu = xp.sum(
            extended_images *
            (x - x_bar) ** powers_x *
            (y - y_bar) ** powers_y,
            axis=(1, 2)
        )  # (N, 4, 4)

        # normalized moments eta_{pq}
        i_plus_j = xp.arange(4)[:, None] + xp.arange(4)[None, :]
        denom = (M[:, 0, 0][:, None, None] ** ((i_plus_j[None, :, :] + 2) / 2.0)) + 1e-12
        eta = mu / denom
        return eta

    def _calculate_hu_moments(self, eta):
        """
        Calculates the first six Hu moments for a set of images.

        Parameters
        ----------
        eta : xp.ndarray
            The normalized moments for each image (N, 4, 4).

        Returns
        -------
        xp.ndarray
            The first six Hu moments for each image, shape (N, 6).
        """
        xp = self.xp
        num_images = eta.shape[0]
        hu = xp.zeros((num_images, 6), dtype=eta.dtype)

        eta20 = eta[:, 2, 0]
        eta02 = eta[:, 0, 2]
        eta11 = eta[:, 1, 1]
        eta30 = eta[:, 3, 0]
        eta12 = eta[:, 1, 2]
        eta21 = eta[:, 2, 1]
        eta03 = eta[:, 0, 3]

        hu[:, 0] = eta20 + eta02
        hu[:, 1] = (eta20 - eta02) ** 2 + 4 * eta11 ** 2
        hu[:, 2] = (eta30 - 3 * eta12) ** 2 + (3 * eta21 - eta03) ** 2
        hu[:, 3] = (eta30 + eta12) ** 2 + (eta21 + eta03) ** 2
        hu[:, 4] = ((eta30 - 3 * eta12) * (eta30 + eta12) *
                    ((eta30 + eta12) ** 2 - 3 * (eta21 + eta03) ** 2) +
                    (3 * eta21 - eta03) * (eta21 + eta03) *
                    (3 * (eta30 + eta12) ** 2 - (eta21 + eta03) ** 2))
        hu[:, 5] = ((eta20 - eta02) *
                    ((eta30 + eta12) ** 2 - (eta21 + eta03) ** 2) +
                    4 * eta11 * (eta30 + eta12) * (eta21 + eta03))

        # we intentionally do not compute Hu moment 7 (mirror invariance)
        return hu

    def _log_hu(self, hu):
        """Stable log-Hu transform that avoids NaNs/inf for zero moments."""
        xp = self.xp
        if hu.size == 0:
            return hu
        abs_hu = xp.abs(hu)
        eps = xp.finfo(hu.dtype).tiny
        abs_hu = xp.maximum(abs_hu, eps)
        log_hu = -xp.sign(hu) * xp.log10(abs_hu)
        log_hu = xp.where(xp.isfinite(log_hu), log_hu, 0.0)
        return log_hu

    def _normalize_features(self, pre, post, xp_mod):
        """Normalize features jointly for pre/post frames using mean/std."""
        if pre.size == 0 or post.size == 0:
            return pre, post
        combined = xp_mod.concatenate((pre, post), axis=0)
        mean = xp_mod.nanmean(combined, axis=0)
        std = xp_mod.nanstd(combined, axis=0) + 1e-8
        mean = xp_mod.where(xp_mod.isfinite(mean), mean, 0.0)
        std = xp_mod.where(xp_mod.isfinite(std), std, 1.0)
        pre_norm = (pre - mean) / std
        post_norm = (post - mean) / std
        return pre_norm, post_norm

    def _calculate_mean_and_variance(self, images):
        """
        Calculates the mean and variance of intensity for a set of images.

        Parameters
        ----------
        images : xp.ndarray
            Input image data, shape (N, H, W) or (N, Z, Y, X).

        Returns
        -------
        xp.ndarray
            Array containing [mean, variance] for each image, shape (N, 2).
        """
        xp = self.xp
        if images.size == 0:
            return xp.zeros((0, 2), dtype=xp.float32)

        num_images = images.shape[0]
        features = xp.zeros((num_images, 2), dtype=xp.float32)

        mask = images != 0
        if self.im_info.no_z:
            axis = (1, 2)
        else:
            axis = (1, 2, 3)

        count_nonzero = xp.sum(mask, axis=axis)
        count_nonzero_safe = xp.where(count_nonzero == 0, 1, count_nonzero)

        sum_nonzero = xp.sum(images * mask, axis=axis)
        sumsq_nonzero = xp.sum((images * mask) ** 2, axis=axis)

        mean = sum_nonzero / count_nonzero_safe
        variance = (sumsq_nonzero - (sum_nonzero ** 2) / count_nonzero_safe) / count_nonzero_safe

        # Where there are no nonzero pixels, set mean and variance to 0 explicitly.
        mean = xp.where(count_nonzero == 0, 0.0, mean)
        variance = xp.where(count_nonzero == 0, 0.0, variance)

        features[:, 0] = mean
        features[:, 1] = variance
        return features

    # -------------------------------------------------------------------------
    # ROI / sub-volume handling
    # -------------------------------------------------------------------------

    def _get_im_bounds(self, markers, distance_frame):
        """
        Calculates the bounds of the region around each marker in the image.

        Parameters
        ----------
        markers : xp.ndarray, shape (N, 2) or (N, 3)
        distance_frame : xp.ndarray

        Returns
        -------
        tuple of xp.ndarray
            Boundaries for sub-volumes around each marker.
        """
        xp = self.xp
        if not self.im_info.no_z:
            radii = distance_frame[markers[:, 0], markers[:, 1], markers[:, 2]]
        else:
            radii = distance_frame[markers[:, 0], markers[:, 1]]
        marker_radii = xp.ceil(radii)

        low_0 = xp.clip(markers[:, 0] - marker_radii, 0, self.shape[1])
        high_0 = xp.clip(markers[:, 0] + (marker_radii + 1), 0, self.shape[1])
        low_1 = xp.clip(markers[:, 1] - marker_radii, 0, self.shape[2])
        high_1 = xp.clip(markers[:, 1] + (marker_radii + 1), 0, self.shape[2])

        if not self.im_info.no_z:
            low_2 = xp.clip(markers[:, 2] - marker_radii, 0, self.shape[3])
            high_2 = xp.clip(markers[:, 2] + (marker_radii + 1), 0, self.shape[3])
            return low_0, high_0, low_1, high_1, low_2, high_2
        return low_0, high_0, low_1, high_1

    def _get_sub_volumes(self, im_frame, im_bounds, max_radius):
        """
        Extracts sub-volumes from the image within the specified bounds (dense/batched ROI path).

        Parameters
        ----------
        im_frame : xp.ndarray
            Image data for a single frame (2D or 3D).
        im_bounds : tuple
            Bounds for extracting sub-volumes.
        max_radius : int
            Maximum radius for the sub-volumes.

        Returns
        -------
        xp.ndarray
            Extracted sub-volumes from the image, shape:
            (N, max_radius, max_radius) or (N, max_radius, max_radius, max_radius).
        """
        xp = self.xp
        if self.im_info.no_z:
            y_low, y_high, x_low, x_high = im_bounds
        else:
            z_low, z_high, y_low, y_high, x_low, x_high = im_bounds

        num_markers = len(y_low)

        if self.im_info.no_z:
            sub_volumes = xp.zeros((num_markers, max_radius, max_radius), dtype=im_frame.dtype)
        else:
            sub_volumes = xp.zeros((num_markers, max_radius, max_radius, max_radius), dtype=im_frame.dtype)

        for i in range(num_markers):
            if self.im_info.no_z:
                yl, yh = int(y_low[i]), int(y_high[i])
                xl, xh = int(x_low[i]), int(x_high[i])
                if yl >= yh or xl >= xh:
                    continue
                sub_volumes[i, :yh - yl, :xh - xl] = im_frame[yl:yh, xl:xh]
            else:
                zl, zh = int(z_low[i]), int(z_high[i])
                yl, yh = int(y_low[i]), int(y_high[i])
                xl, xh = int(x_low[i]), int(x_high[i])
                if zl >= zh or yl >= yh or xl >= xh:
                    continue
                sub_volumes[i, :zh - zl, :yh - yl, :xh - xl] = im_frame[zl:zh, yl:yh, xl:xh]
        return sub_volumes

    def _get_orthogonal_projections(self, sub_volumes):
        """
        Computes the orthogonal projections of 3D sub-volumes along each axis.

        Parameters
        ----------
        sub_volumes : xp.ndarray, shape (N, Z, Y, X)

        Returns
        -------
        tuple of xp.ndarray
            Z, Y, and X projections of the sub-volumes, each shape (N, H, W).
        """
        xp = self.xp
        z_projections = xp.max(sub_volumes, axis=1)
        y_projections = xp.max(sub_volumes, axis=2)
        x_projections = xp.max(sub_volumes, axis=3)
        return z_projections, y_projections, x_projections

    # -------------------------------------------------------------------------
    # Meta / memory allocation
    # -------------------------------------------------------------------------

    def _get_t(self):
        """Determines the number of timepoints to process."""
        if self.num_t is None:
            if self.im_info.no_t:
                self.num_t = 1
            else:
                self.num_t = self.im_info.shape[self.im_info.axes.index('T')]

    def _allocate_memory(self):
        """
        Allocates memory / memmaps for the necessary image data.
        """
        logger.debug('Allocating memory for Hu-moment tracking.')
        self.label_memmap = self.im_info.get_memmap(self.im_info.pipeline_paths['im_instance_label'])
        self.im_memmap = self.im_info.get_memmap(self.im_info.im_path)
        self.im_frangi_memmap = self.im_info.get_memmap(self.im_info.pipeline_paths['im_preprocessed'])
        self.im_marker_memmap = self.im_info.get_memmap(self.im_info.pipeline_paths['im_marker'])
        self.im_distance_memmap = self.im_info.get_memmap(self.im_info.pipeline_paths['im_distance'])
        self.shape = self.label_memmap.shape
        self.flow_vector_array_path = self.im_info.pipeline_paths['flow_vector_array']

    def _get_hu_moments(self, sub_volumes):
        """
        Calculates Hu moments for the given sub-volumes of the image.

        Parameters
        ----------
        sub_volumes : xp.ndarray
            Sub-volumes of the image (2D or 3D).

        Returns
        -------
        xp.ndarray
            The Hu moments for the sub-volumes.
        """
        xp = self.xp
        if self.im_info.no_z:
            etas = self._calculate_normalized_moments(sub_volumes)
            hu_moments = self._calculate_hu_moments(etas)
            return hu_moments

        intensity_projections = self._get_orthogonal_projections(sub_volumes)
        etas_z = self._calculate_normalized_moments(intensity_projections[0])
        etas_y = self._calculate_normalized_moments(intensity_projections[1])
        etas_x = self._calculate_normalized_moments(intensity_projections[2])

        hu_moments_z = self._calculate_hu_moments(etas_z)
        hu_moments_y = self._calculate_hu_moments(etas_y)
        hu_moments_x = self._calculate_hu_moments(etas_x)

        hu_moments = xp.concatenate((hu_moments_z, hu_moments_y, hu_moments_x), axis=1)
        return hu_moments

    def _concatenate_hu_matrices(self, hu_matrices):
        """Concatenates multiple feature matrices along the feature axis."""
        return self.xp.concatenate(hu_matrices, axis=1)

    # -------------------------------------------------------------------------
    # Per-frame feature extraction (stats + Hu + coordinates)
    # -------------------------------------------------------------------------

    def _get_frame_features(self, t) -> _FrameFeatures:
        """
        Extracts all features (mean, variance, Hu moments) and coordinates for a given timepoint.

        This method adaptively chooses between dense/batched ROI extraction and
        streaming per-ROI extraction depending on estimated memory usage and
        optionally falls back if dense extraction runs out of memory.

        Parameters
        ----------
        t : int
            Timepoint index.

        Returns
        -------
        _FrameFeatures
        """
        try:
            return self._get_frame_features_impl(t)
        except GPU_OOM_ERRORS + (MemoryError,) as exc:
            if self._on_gpu and self._is_oom_error(exc):
                logger.warning(
                    f"Frame {t}: GPU OOM during feature extraction; switching to CPU."
                )
                self._free_gpu_memory()
                self._switch_to_cpu()
                return self._get_frame_features_impl(t)
            raise

    def _get_frame_features_impl(self, t) -> _FrameFeatures:
        xp = self.xp
        ndi = self.ndi

        # Load frames into xp arrays
        intensity_frame = xp.asarray(self.im_memmap[t])
        frangi_frame = xp.asarray(self.im_frangi_memmap[t]).copy()
        distance_frame = xp.asarray(self.im_distance_memmap[t])

        # Frangi normalization
        positive_mask = frangi_frame > 0
        if xp.any(positive_mask):
            frangi_frame[positive_mask] = xp.log10(frangi_frame[positive_mask])
        negative_mask = frangi_frame < 0
        if xp.any(negative_mask):
            min_neg = xp.min(frangi_frame[negative_mask])
            frangi_frame[negative_mask] -= min_neg

        # Distance transform dilation (in-place)
        distance_max_frame = distance_frame.copy()
        ndi.maximum_filter(distance_max_frame, size=3, output=distance_max_frame)
        distance_max_frame *= 2

        # Markers
        marker_frame = xp.asarray(self.im_marker_memmap[t]) > 0
        marker_indices_xp = xp.argwhere(marker_frame)

        if marker_indices_xp.size == 0:
            dims = 2 if self.im_info.no_z else 3
            empty_coords_voxel = np.zeros((0, dims), dtype=int)
            empty_coords_phys = np.zeros((0, dims), dtype=float)
            empty_stats = xp.zeros((0, 0), dtype=xp.float32)
            empty_hu = xp.zeros((0, 0), dtype=xp.float32)
            return _FrameFeatures(empty_coords_voxel, empty_coords_phys, empty_stats, empty_hu)

        # Convert indices to numpy for physical coordinates and later matching
        if hasattr(marker_indices_xp, "get"):
            marker_indices_np = marker_indices_xp.get()
        else:
            marker_indices_np = np.asarray(marker_indices_xp)
        scaling = np.asarray(self.scaling, dtype=float)
        coords_phys = marker_indices_np * scaling  # (N, dim)

        # ROI bounds in voxel space
        region_bounds = self._get_im_bounds(marker_indices_xp, distance_max_frame)

        # Decide whether to use dense ROI extraction or streaming per-ROI
        num_markers = marker_indices_xp.shape[0]
        marker_mask = marker_frame
        max_radius = int(xp.ceil(xp.max(distance_max_frame[marker_mask])).item()) * 2 + 1

        dim = 2 if self.im_info.no_z else 3
        voxels_per_roi = max_radius ** dim
        total_voxels = int(num_markers * voxels_per_roi)
        dense_limit = self.max_dense_roi_voxels_gpu if self._on_gpu else self.max_dense_roi_voxels_cpu
        use_dense = total_voxels <= dense_limit
        if self.low_memory:
            use_dense = False

        logger.debug(
            f"Frame {t}: {num_markers} markers, max_radius={max_radius}, "
            f"total_roi_voxels~{total_voxels}, use_dense={use_dense}"
        )

        stats_dim = 4
        hu_dim = 6 if self.im_info.no_z else 18
        stats_feature_matrix = xp.zeros((num_markers, stats_dim), dtype=xp.float32)
        log_hu_feature_matrix = xp.zeros((num_markers, hu_dim), dtype=xp.float32)

        # Dense / batched ROI extraction (fast, more memory)
        if use_dense:
            try:
                intensity_sub_volumes = self._get_sub_volumes(intensity_frame, region_bounds, max_radius)
                frangi_sub_volumes = self._get_sub_volumes(frangi_frame, region_bounds, max_radius)

                intensity_stats = self._calculate_mean_and_variance(intensity_sub_volumes)
                frangi_stats = self._calculate_mean_and_variance(frangi_sub_volumes)
                stats_feature_matrix = self._concatenate_hu_matrices([intensity_stats, frangi_stats])

                intensity_hus = self._get_hu_moments(intensity_sub_volumes)
                log_hu_feature_matrix = self._log_hu(intensity_hus)

                del intensity_sub_volumes, frangi_sub_volumes, intensity_stats, frangi_stats, intensity_hus
            except GPU_OOM_ERRORS + (MemoryError,) as exc:
                if self._on_gpu and self._is_oom_error(exc):
                    self._free_gpu_memory()
                logger.warning(
                    f"Frame {t}: dense ROI extraction OOM; falling back to streaming ROI extraction."
                )
                use_dense = False

        # Streaming per-ROI extraction (slower, minimal memory)
        if not use_dense:
            logger.debug(f"Frame {t}: using streaming ROI extraction for {num_markers} markers.")
            stats_feature_matrix, log_hu_feature_matrix = self._compute_features_streaming(
                intensity_frame, frangi_frame, region_bounds, num_markers
            )

        coords_voxel = marker_indices_np.astype(int)
        return _FrameFeatures(coords_voxel, coords_phys, stats_feature_matrix, log_hu_feature_matrix)

    def _compute_features_streaming(self, intensity_frame, frangi_frame, region_bounds, num_markers):
        """
        Streaming per-marker ROI extraction to reduce memory usage.

        Parameters
        ----------
        intensity_frame : xp.ndarray
        frangi_frame : xp.ndarray
        region_bounds : tuple of xp.ndarray
        num_markers : int

        Returns
        -------
        (stats_feature_matrix, log_hu_feature_matrix) : xp.ndarray, xp.ndarray
        """
        xp = self.xp
        stats_dim = 4
        hu_dim = 6 if self.im_info.no_z else 18
        stats_feature_matrix = xp.zeros((num_markers, stats_dim), dtype=xp.float32)
        log_hu_feature_matrix = xp.zeros((num_markers, hu_dim), dtype=xp.float32)

        if self.im_info.no_z:
            y_low, y_high, x_low, x_high = region_bounds
        else:
            z_low, z_high, y_low, y_high, x_low, x_high = region_bounds

        for i in range(num_markers):
            if self.im_info.no_z:
                yl, yh = int(y_low[i]), int(y_high[i])
                xl, xh = int(x_low[i]), int(x_high[i])
                if yl >= yh or xl >= xh:
                    continue
                intensity_roi = intensity_frame[yl:yh, xl:xh]
                frangi_roi = frangi_frame[yl:yh, xl:xh]
            else:
                zl, zh = int(z_low[i]), int(z_high[i])
                yl, yh = int(y_low[i]), int(y_high[i])
                xl, xh = int(x_low[i]), int(x_high[i])
                if zl >= zh or yl >= yh or xl >= xh:
                    continue
                intensity_roi = intensity_frame[zl:zh, yl:yh, xl:xh]
                frangi_roi = frangi_frame[zl:zh, yl:yh, xl:xh]

            if intensity_roi.size == 0 or frangi_roi.size == 0:
                continue

            intensity_roi_b = intensity_roi[xp.newaxis, ...]
            frangi_roi_b = frangi_roi[xp.newaxis, ...]

            intensity_stats = self._calculate_mean_and_variance(intensity_roi_b)[0]
            frangi_stats = self._calculate_mean_and_variance(frangi_roi_b)[0]
            stats_row = xp.concatenate((intensity_stats, frangi_stats), axis=0)

            intensity_hu = self._get_hu_moments(intensity_roi_b)[0]
            log_hu_row = self._log_hu(intensity_hu)

            stats_feature_matrix[i] = stats_row
            log_hu_feature_matrix[i] = log_hu_row

        return stats_feature_matrix, log_hu_feature_matrix

    # -------------------------------------------------------------------------
    # Matching utilities
    # -------------------------------------------------------------------------

    def _get_distance_mask(self, coords_post_phys, coords_pre_phys):
        """
        Computes the distance matrix and mask between objects in consecutive frames.

        Parameters
        ----------
        coords_post_phys : array-like, shape (N_post, dim)
        coords_pre_phys : array-like, shape (N_pre, dim)

        Returns
        -------
        (distance_matrix, distance_mask) : xp.ndarray, xp.ndarray
            distance_matrix is normalized by max_distance_um (range ~[0, 1]).
            distance_mask is boolean, True where distances < max_distance_um.
        """
        xp = self.xp
        coords_post_phys = np.asarray(coords_post_phys, dtype=float)
        coords_pre_phys = np.asarray(coords_pre_phys, dtype=float)

        if coords_post_phys.size == 0 or coords_pre_phys.size == 0:
            return xp.zeros((0, 0), dtype=xp.float32), xp.zeros((0, 0), dtype=bool)

        if self._on_gpu:
            A = xp.asarray(coords_post_phys)
            B = xp.asarray(coords_pre_phys)
            diff = A[:, None, :] - B[None, :, :]
            distance_matrix = xp.sqrt(xp.sum(diff ** 2, axis=2))
        else:
            distance_matrix_np = cdist(coords_post_phys, coords_pre_phys)
            distance_matrix = xp.asarray(distance_matrix_np)

        distance_mask = distance_matrix < self.max_distance_um
        distance_matrix = distance_matrix / self.max_distance_um
        return distance_matrix, distance_mask

    def _get_difference_matrix(self, m1, m2):
        """
        Computes the absolute difference matrix between two feature matrices.

        Parameters
        ----------
        m1 : xp.ndarray, shape (N_post, F)
        m2 : xp.ndarray, shape (N_pre, F)

        Returns
        -------
        xp.ndarray
            Difference matrix, shape (N_post, N_pre, F).
        """
        xp = self.xp
        if m1.size == 0 or m2.size == 0:
            return xp.zeros((0, 0, 0), dtype=xp.float64)

        m1_reshaped = m1[:, xp.newaxis, :].astype(xp.float64)
        m2_reshaped = m2[xp.newaxis, :, :].astype(xp.float64)
        difference_matrix = xp.abs(m1_reshaped - m2_reshaped)
        return difference_matrix

    def _zscore_normalize(self, m, mask):
        """
        Z-score normalizes the values in a matrix over masked entries.

        Parameters
        ----------
        m : xp.ndarray, shape (N_post, N_pre, F)
        mask : xp.ndarray, shape (N_post, N_pre), boolean

        Returns
        -------
        xp.ndarray
            Z-score normalized matrix with masked entries set to +inf.
        """
        xp = self.xp
        if m.size == 0:
            return m

        mask_exp = mask[..., None]
        sum_mask = xp.sum(mask_exp)
        if float(sum_mask) == 0.0:
            # No valid pairs; everything is "infinite" cost.
            return xp.full_like(m, xp.inf)

        mean_vals = xp.sum(m * mask_exp, axis=(0, 1)) / sum_mask
        var_vals = xp.sum((m - mean_vals) ** 2 * mask_exp, axis=(0, 1)) / sum_mask
        std_vals = xp.sqrt(var_vals) + 1e-8

        m = (m - mean_vals) / std_vals
        m = xp.where(mask_exp, m, xp.inf)
        return m

    def _get_cost_matrix(self, coords_post_phys, coords_pre_phys,
                         stats_vecs, pre_stats_vecs, hu_vecs, pre_hu_vecs):
        """
        Calculates the dense cost matrix for matching objects between consecutive frames.

        This is used in the "dense" matching path; for very large problems,
        sparse KDTree-based matching is used instead.

        Parameters
        ----------
        coords_post_phys, coords_pre_phys : np.ndarray
            Physical coordinates for post- and pre-frame markers.
        stats_vecs, pre_stats_vecs : xp.ndarray
            Feature matrices for current and previous frames.
        hu_vecs, pre_hu_vecs : xp.ndarray
            Hu matrices for current and previous frames.

        Returns
        -------
        xp.ndarray
            Cost matrix, shape (N_post, N_pre).
        """
        xp = self.xp
        if stats_vecs.size == 0 or pre_stats_vecs.size == 0 or hu_vecs.size == 0 or pre_hu_vecs.size == 0:
            return xp.zeros((0, 0), dtype=xp.float16)

        distance_matrix, distance_mask = self._get_distance_mask(coords_post_phys, coords_pre_phys)

        # Distance feature
        z_score_distance_matrix = self._zscore_normalize(distance_matrix[..., xp.newaxis],
                                                         distance_mask).astype(xp.float16)

        # Stats feature differences
        stats_matrix = self._get_difference_matrix(stats_vecs, pre_stats_vecs)
        z_score_stats_matrix = self._zscore_normalize(stats_matrix, distance_mask)
        z_score_stats_matrix = (z_score_stats_matrix / stats_matrix.shape[2]).astype(xp.float16)
        del stats_matrix

        # Hu feature differences
        hu_matrix = self._get_difference_matrix(hu_vecs, pre_hu_vecs)
        z_score_hu_matrix = self._zscore_normalize(hu_matrix, distance_mask)
        z_score_hu_matrix = (z_score_hu_matrix / hu_matrix.shape[2]).astype(xp.float16)
        del hu_matrix, distance_mask

        z_score_matrix = xp.concatenate(
            (z_score_distance_matrix, z_score_stats_matrix, z_score_hu_matrix), axis=2
        ).astype(xp.float16)
        cost_matrix = xp.nansum(z_score_matrix, axis=2).astype(xp.float16)
        del z_score_distance_matrix, z_score_stats_matrix, z_score_hu_matrix, z_score_matrix

        return cost_matrix.astype(xp.float32)

    def _find_best_matches(self, cost_matrix):
        """
        Finds the best object matches between two frames based on a dense cost matrix.

        Parameters
        ----------
        cost_matrix : xp.ndarray, shape (N_post, N_pre)

        Returns
        -------
        (row_matches, col_matches, costs) : list[int], list[int], list[float]
        """
        xp = self.xp
        if cost_matrix.size == 0:
            return [], [], []

        cost_cutoff = 1.0

        # Row-wise minima
        row_min_idx = xp.argmin(cost_matrix, axis=1)
        row_min_val = xp.min(cost_matrix, axis=1)

        # Column-wise minima
        col_min_idx = xp.argmin(cost_matrix, axis=0)
        col_min_val = xp.min(cost_matrix, axis=0)

        row_matches = []
        col_matches = []
        costs = []

        # Row candidates
        for i, (r_idx, r_val) in enumerate(zip(row_min_idx, row_min_val)):
            val = float(r_val)
            if val > cost_cutoff:
                continue
            row_matches.append(int(i))
            col_matches.append(int(r_idx))
            costs.append(val)

        # Column candidates
        for j, (c_idx, c_val) in enumerate(zip(col_min_idx, col_min_val)):
            val = float(c_val)
            if val > cost_cutoff:
                continue
            row_matches.append(int(c_idx))
            col_matches.append(int(j))
            costs.append(val)

        return row_matches, col_matches, costs

    # -------------------------------------------------------------------------
    # Matching modes: dense vs sparse (KDTree)
    # -------------------------------------------------------------------------

    def _match_frames_sparse(self, coords_post_phys, coords_pre_phys,
                             stats_vecs, pre_stats_vecs, hu_vecs, pre_hu_vecs):
        """
        Sparse KDTree-based matching for large problems.

        Operates entirely on CPU/NumPy to reduce GPU memory pressure.
        Uses dense-like z-score normalization over candidate pairs to keep scoring consistent.
        """
        coords_post = np.asarray(coords_post_phys, dtype=float)
        coords_pre = np.asarray(coords_pre_phys, dtype=float)

        n_post = coords_post.shape[0]
        n_pre = coords_pre.shape[0]
        if n_post == 0 or n_pre == 0:
            return [], [], []

        stats_post = np.asarray(stats_vecs, dtype=np.float32)
        stats_pre = np.asarray(pre_stats_vecs, dtype=np.float32)
        hu_post = np.asarray(hu_vecs, dtype=np.float32)
        hu_pre = np.asarray(pre_hu_vecs, dtype=np.float32)

        if stats_post.size == 0 or stats_pre.size == 0 or hu_post.size == 0 or hu_pre.size == 0:
            return [], [], []

        logger.debug(
            f"Using sparse KDTree matching: N_post={n_post}, N_pre={n_pre}, "
            f"max_distance_um={self.max_distance_um}"
        )

        # KDTree for spatial gating
        tree = cKDTree(coords_pre)
        candidates_per_row = tree.query_ball_point(coords_post, self.max_distance_um)

        # First pass: estimate mean/std over candidate pairs (dense-like normalization)
        n_pairs = 0
        sum_dist = 0.0
        sumsq_dist = 0.0
        sum_stats = np.zeros(stats_post.shape[1], dtype=np.float64)
        sumsq_stats = np.zeros(stats_post.shape[1], dtype=np.float64)
        sum_hu = np.zeros(hu_post.shape[1], dtype=np.float64)
        sumsq_hu = np.zeros(hu_post.shape[1], dtype=np.float64)

        for i_post, pre_list in enumerate(candidates_per_row):
            if not pre_list:
                continue
            pre_idx = np.asarray(pre_list, dtype=np.int64)
            if pre_idx.size == 0:
                continue

            delta = coords_post[i_post] - coords_pre[pre_idx]
            d_geom = np.linalg.norm(delta, axis=1) / self.max_distance_um

            stats_diff = np.abs(stats_post[i_post][None, :] - stats_pre[pre_idx])
            hu_diff = np.abs(hu_post[i_post][None, :] - hu_pre[pre_idx])

            sum_dist += float(np.sum(d_geom))
            sumsq_dist += float(np.sum(d_geom * d_geom))
            sum_stats += np.sum(stats_diff, axis=0, dtype=np.float64)
            sumsq_stats += np.sum(stats_diff * stats_diff, axis=0, dtype=np.float64)
            sum_hu += np.sum(hu_diff, axis=0, dtype=np.float64)
            sumsq_hu += np.sum(hu_diff * hu_diff, axis=0, dtype=np.float64)
            n_pairs += int(d_geom.size)

        if n_pairs == 0:
            return [], [], []

        mean_dist = sum_dist / n_pairs
        var_dist = max(0.0, (sumsq_dist / n_pairs) - (mean_dist ** 2))
        std_dist = np.sqrt(var_dist) + 1e-8

        mean_stats = sum_stats / n_pairs
        var_stats = (sumsq_stats / n_pairs) - (mean_stats ** 2)
        var_stats = np.maximum(var_stats, 0.0)
        std_stats = np.sqrt(var_stats) + 1e-8

        mean_hu = sum_hu / n_pairs
        var_hu = (sumsq_hu / n_pairs) - (mean_hu ** 2)
        var_hu = np.maximum(var_hu, 0.0)
        std_hu = np.sqrt(var_hu) + 1e-8

        # Row and column minima
        row_min_val = np.full(n_post, np.inf, dtype=np.float32)
        row_min_idx = np.full(n_post, -1, dtype=np.int64)
        col_min_val = np.full(n_pre, np.inf, dtype=np.float32)
        col_min_idx = np.full(n_pre, -1, dtype=np.int64)

        cost_cutoff = 1.0

        for i_post, pre_list in enumerate(candidates_per_row):
            if not pre_list:
                continue

            pre_idx = np.asarray(pre_list, dtype=np.int64)
            if pre_idx.size == 0:
                continue

            delta = coords_post[i_post] - coords_pre[pre_idx]
            d_geom = np.linalg.norm(delta, axis=1) / self.max_distance_um

            stats_diff = np.abs(stats_post[i_post][None, :] - stats_pre[pre_idx])
            hu_diff = np.abs(hu_post[i_post][None, :] - hu_pre[pre_idx])

            z_dist = (d_geom - mean_dist) / std_dist
            z_stats = (stats_diff - mean_stats) / std_stats
            z_hu = (hu_diff - mean_hu) / std_hu

            cost = z_dist + np.mean(z_stats, axis=1) + np.mean(z_hu, axis=1)
            valid_mask = cost <= cost_cutoff
            if not np.any(valid_mask):
                continue

            cost_valid = cost[valid_mask]
            pre_idx_valid = pre_idx[valid_mask]

            # Row minimum
            best_idx_local = int(np.argmin(cost_valid))
            best_cost = float(cost_valid[best_idx_local])
            best_pre = int(pre_idx_valid[best_idx_local])

            if best_cost < row_min_val[i_post]:
                row_min_val[i_post] = best_cost
                row_min_idx[i_post] = best_pre

            # Column minima
            for j_pre, c in zip(pre_idx_valid, cost_valid):
                c = float(c)
                if c < col_min_val[j_pre]:
                    col_min_val[j_pre] = c
                    col_min_idx[j_pre] = i_post

        row_matches = []
        col_matches = []
        costs = []

        # Row-based candidates
        for i_post, (j_pre, c) in enumerate(zip(row_min_idx, row_min_val)):
            if j_pre >= 0 and c <= cost_cutoff:
                row_matches.append(int(i_post))
                col_matches.append(int(j_pre))
                costs.append(float(c))

        # Column-based candidates
        for j_pre, (i_post, c) in enumerate(zip(col_min_idx, col_min_val)):
            if i_post >= 0 and c <= cost_cutoff:
                row_matches.append(int(i_post))
                col_matches.append(int(j_pre))
                costs.append(float(c))

        return row_matches, col_matches, costs

    def _match_frames(self, frame_t: _FrameFeatures, frame_prev: _FrameFeatures):
        """
        Dispatch between dense and sparse matching modes for two consecutive frames.
        """
        stats_vecs = frame_t.stats
        pre_stats_vecs = frame_prev.stats
        hu_vecs = frame_t.hu
        pre_hu_vecs = frame_prev.hu

        if self._on_gpu:
            stats_vecs = self.xp.asarray(stats_vecs)
            pre_stats_vecs = self.xp.asarray(pre_stats_vecs)
            hu_vecs = self.xp.asarray(hu_vecs)
            pre_hu_vecs = self.xp.asarray(pre_hu_vecs)
        else:
            stats_vecs = self._to_cpu_array(stats_vecs)
            pre_stats_vecs = self._to_cpu_array(pre_stats_vecs)
            hu_vecs = self._to_cpu_array(hu_vecs)
            pre_hu_vecs = self._to_cpu_array(pre_hu_vecs)

        n_post = stats_vecs.shape[0]
        n_pre = pre_stats_vecs.shape[0]
        if n_post == 0 or n_pre == 0:
            return [], [], []

        num_pairs = n_post * n_pre
        logger.debug(
            f"Matching frames: N_post={n_post}, N_pre={n_pre}, "
            f"pairs={num_pairs}, mode={self.mode}"
        )

        # Decide mode
        use_dense = (self.mode == "dense") or (
            self.mode == "auto" and num_pairs <= self.max_dense_pairs
        )

        if use_dense and self.mode != "sparse":
            try:
                cost_matrix = self._get_cost_matrix(
                    frame_t.coords_phys, frame_prev.coords_phys,
                    stats_vecs, pre_stats_vecs, hu_vecs, pre_hu_vecs
                )
                return self._find_best_matches(cost_matrix)
            except GPU_OOM_ERRORS + (MemoryError,) as exc:
                if self._on_gpu and self._is_oom_error(exc):
                    logger.warning(
                        "Dense matching OOM; switching to CPU and falling back to sparse KDTree matching."
                    )
                    self._free_gpu_memory()
                    self._switch_to_cpu()
                else:
                    logger.warning(
                        "Dense matching OOM; falling back to sparse KDTree matching."
                    )

        # Sparse fallback or forced sparse mode
        return self._match_frames_sparse(
            frame_t.coords_phys, frame_prev.coords_phys,
            stats_vecs, pre_stats_vecs, hu_vecs, pre_hu_vecs
        )

    # -------------------------------------------------------------------------
    # Main tracking loop
    # -------------------------------------------------------------------------

    def _run_hu_tracking(self):
        """
        Runs the full tracking algorithm over all timepoints, saving the results to disk.
        """
        prev_frame_features: _FrameFeatures | None = None
        frame_vectors = []

        for t in range(self.num_t):
            if self.viewer is not None:
                self.viewer.status = f'Tracking markers. Frame: {t + 1} of {self.num_t}.'
            logger.debug(f'Running Hu-moment tracking for frame {t + 1} of {self.num_t}')

            frame_features = self._get_frame_features(t)

            # First frame: nothing to match yet
            if prev_frame_features is None:
                prev_frame_features = frame_features
                continue

            # Match current frame to previous
            row_indices, col_indices, costs = self._match_frames(frame_features, prev_frame_features)

            if len(row_indices) == 0:
                prev_frame_features = frame_features
                continue

            row_indices = np.asarray(row_indices, dtype=np.int64)
            col_indices = np.asarray(col_indices, dtype=np.int64)
            costs = np.asarray(costs, dtype=np.float32)

            pre_marker_indices = prev_frame_features.coords_voxel[col_indices]
            marker_indices = frame_features.coords_voxel[row_indices]
            vecs = marker_indices - pre_marker_indices

            if self.im_info.no_z:
                idx0_y, idx0_x = pre_marker_indices.T
                vec_y, vec_x = vecs.T
                frame_vector_array = np.column_stack([
                    np.full(len(vec_y), t - 1, dtype=np.int64),
                    idx0_y.astype(np.int64),
                    idx0_x.astype(np.int64),
                    vec_y.astype(np.int64),
                    vec_x.astype(np.int64),
                    costs
                ])
            else:
                idx0_z, idx0_y, idx0_x = pre_marker_indices.T
                vec_z, vec_y, vec_x = vecs.T
                frame_vector_array = np.column_stack([
                    np.full(len(vec_z), t - 1, dtype=np.int64),
                    idx0_z.astype(np.int64),
                    idx0_y.astype(np.int64),
                    idx0_x.astype(np.int64),
                    vec_z.astype(np.int64),
                    vec_y.astype(np.int64),
                    vec_x.astype(np.int64),
                    costs
                ])

            frame_vectors.append(frame_vector_array)
            prev_frame_features = frame_features

        # Save accumulated vectors
        if frame_vectors:
            flow_vector_array = np.concatenate(frame_vectors, axis=0)
        else:
            # Fallback: no vectors at all
            if self.im_info.no_z:
                flow_vector_array = np.empty((0, 6), dtype=np.float32)
            else:
                flow_vector_array = np.empty((0, 8), dtype=np.float32)

        np.save(self.flow_vector_array_path, flow_vector_array)
        logger.debug(f"Saved flow vector array to {self.flow_vector_array_path}")

    def run(self):
        """
        Main method to execute the Hu moment-based tracking process over the image data.
        """
        if self.im_info.no_t:
            logger.info("Skipping Hu moment tracking for non-temporal dataset.")
            return
            
        self._get_t()
        self._allocate_memory()
        self._run_hu_tracking()


if __name__ == "__main__":
    im_path = r"D:\test_files\nelly_smorgasbord\deskewed-iono_pre.ome.tif"
    im_info = ImInfo(im_path)
    hu = HuMomentTracking(im_info, num_t=2)
    hu.run()
