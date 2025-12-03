"""
Motion capture marker generation for microscopy images.

This module provides the Markers class for detecting and marking key points in segmented
structures using distance transforms and multi-scale peak detection.
"""
import numpy as np
from scipy import ndimage as sp_ndi  # CPU ndimage backend

from nellie import xp, ndi, device_type
from nellie.utils.base_logger import logger
from nellie.im_info.verifier import ImInfo

# Try to capture GPU OOM errors if cupy is used under the hood
try:
    import cupy
    GPU_OOM_ERRORS = (cupy.cuda.memory.OutOfMemoryError, MemoryError)
except Exception:
    GPU_OOM_ERRORS = (MemoryError,)


class Markers:
    """
    A class for generating motion capture markers in microscopy images using distance transforms and peak detection.

    Optimizations:
    - Uses distance_transform_edt instead of KD-tree for distance transform.
    - Streams over scales for LoG (no large 4D arrays).
    - Uses morphological non-max suppression instead of KD-tree for peak pruning.
    - Supports GPU via nellie.xp/ndi with automatic fallback to CPU on OOM.

    Attributes
    ----------
    im_info : ImInfo
        An object containing image metadata and memory-mapped image data.
    num_t : int
        Number of timepoints in the image.
    min_radius_um : float
        Minimum radius of detected objects in micrometers.
    max_radius_um : float
        Maximum radius of detected objects in micrometers.
    min_radius_px : float
        Minimum radius of detected objects in pixels.
    max_radius_px : float
        Maximum radius of detected objects in pixels.
    use_im : str
        Specifies which image to use for peak detection ('distance' or 'frangi').
    num_sigma : int
        Number of sigma steps for multi-scale filtering.
    shape : tuple
        Shape of the input image.
    im_memmap : np.ndarray or None
        Memory-mapped original image data.
    im_frangi_memmap : np.ndarray or None
        Memory-mapped Frangi-filtered image data.
    label_memmap : np.ndarray or None
        Memory-mapped label data from instance segmentation.
    im_marker_memmap : np.ndarray or None
        Memory-mapped output for motion capture markers.
    im_distance_memmap : np.ndarray or None
        Memory-mapped output for distance transform.
    im_border_memmap : np.ndarray or None
        Memory-mapped output for image borders.
    debug : dict or None
        Debugging information for tracking the marking process.
    viewer : object or None
        Viewer object for displaying status during processing.
    save_distance : bool
        Whether to save the distance transform image.
    save_border : bool
        Whether to save the border image.
    use_gpu : bool
        Whether to use the GPU backend (if available). Automatically set to False on GPU OOM.
    peak_min_distance : int
        Minimum separation (in pixels) between peaks in morphological NMS.
    """

    def __init__(self, im_info: ImInfo, num_t=None,
                 min_radius_um=0.20, max_radius_um=1, use_im='distance', num_sigma=5,
                 viewer=None, prefer_gpu=True, peak_min_distance=2):
        """
        Initializes the Markers object with image metadata and marking parameters.

        Parameters
        ----------
        im_info : ImInfo
            An instance of the ImInfo class, containing metadata and paths for the image file.
        num_t : int, optional
            Number of timepoints to process. If None, defaults to the number of timepoints in the image.
        min_radius_um : float, optional
            Minimum radius of detected objects in micrometers (default is 0.20).
        max_radius_um : float, optional
            Maximum radius of detected objects in micrometers (default is 1).
        use_im : str, optional
            Specifies which image to use for peak detection ('distance' or 'frangi', default is 'distance').
        num_sigma : int, optional
            Number of sigma steps for multi-scale filtering (default is 5).
        viewer : object or None, optional
            Viewer object for displaying status during processing (default is None).
        prefer_gpu : bool, optional
            Whether to prefer GPU backend when available (default is True).
        peak_min_distance : int, optional
            Minimum distance (in pixels) between peaks for NMS (default is 2).
        """
        self.im_info = im_info

        self.num_t = num_t
        if self.im_info.no_t:
            self.num_t = 1
        elif num_t is None:
            self.num_t = im_info.shape[im_info.axes.index('T')]

        if not self.im_info.no_z:
            self.z_ratio = self.im_info.dim_res['Z'] / self.im_info.dim_res['X']
        else:
            self.z_ratio = 1.0

        self.min_radius_um = max(min_radius_um, self.im_info.dim_res['X'])
        self.max_radius_um = max_radius_um

        self.min_radius_px = self.min_radius_um / self.im_info.dim_res['X']
        self.max_radius_px = self.max_radius_um / self.im_info.dim_res['X']

        self.use_im = use_im
        self.num_sigma = num_sigma

        self.shape = ()

        self.im_memmap = None
        self.im_frangi_memmap = None
        self.label_memmap = None
        self.im_marker_memmap = None
        self.im_distance_memmap = None
        self.im_border_memmap = None

        self.debug = None

        self.viewer = viewer

        # Backend selection; will be overridden to CPU on GPU OOM.
        self.use_gpu = prefer_gpu and (device_type == 'cuda')

        # Morphological NMS radius
        self.peak_min_distance = peak_min_distance

    # -------------------------------------------------------------------------
    # Backend helpers
    # -------------------------------------------------------------------------
    @property
    def xp(self):
        """Array module: xp (nellie backend) if using GPU, else numpy."""
        return xp if self.use_gpu else np

    @property
    def ndi_backend(self):
        """Ndimage backend: ndi (nellie backend) if using GPU, else scipy.ndimage."""
        return ndi if self.use_gpu else sp_ndi

    def _to_cpu(self, arr):
        """Convert an array (xp or numpy) to a numpy.ndarray."""
        if isinstance(arr, np.ndarray):
            return arr
        # Try xp.asnumpy if available (e.g. cupy), otherwise fall back to np.asarray
        try:
            asnumpy = xp.asnumpy
        except AttributeError:
            return np.asarray(arr)
        else:
            return asnumpy(arr)

    # -------------------------------------------------------------------------
    # Core logic
    # -------------------------------------------------------------------------
    def _get_sigma_vec(self, sigma):
        """
        Computes the sigma vector for multi-scale filtering based on image dimensions.

        Parameters
        ----------
        sigma : float
            The sigma value to use for filtering.

        Returns
        -------
        tuple
            Sigma vector for Gaussian filtering in (Z, Y, X) or (Y, X).
        """
        if self.im_info.no_z:
            sigma_vec = (sigma, sigma)
        else:
            sigma_vec = (sigma / self.z_ratio, sigma, sigma)
        return sigma_vec

    def _set_default_sigmas(self):
        """
        Sets the default sigma values for multi-scale filtering based on the minimum and maximum radius in pixels.
        """
        logger.debug('Setting sigma values.')
        min_sigma_step_size = 0.2

        self.sigma_min = self.min_radius_px / 2.0
        self.sigma_max = self.max_radius_px / 3.0

        sigma_range = self.sigma_max - self.sigma_min
        if sigma_range <= 0:
            logger.warning(
                "Non-positive sigma range (min=%f, max=%f). Check radius settings.",
                self.sigma_min, self.sigma_max
            )
            self.sigmas = [self.sigma_min]
            return

        sigma_step_size_calculated = sigma_range / max(self.num_sigma, 1)
        sigma_step_size = max(min_sigma_step_size, sigma_step_size_calculated)

        xp_mod = self.xp
        self.sigmas = list(xp_mod.arange(self.sigma_min, self.sigma_max, sigma_step_size))

        logger.debug(
            'Calculated sigma step size = %f. Sigmas (%d) = %s',
            sigma_step_size_calculated,
            len(self.sigmas),
            self.sigmas
        )

        if len(self.sigmas) == 0:
            self.sigmas = [self.sigma_min]
            logger.warning("No sigma values generated; falling back to a single sigma=%f.", self.sigma_min)

    def _get_t(self):
        """
        Determines the number of timepoints to process.

        If `num_t` is not set and the image contains a temporal dimension, it sets `num_t` to the number of timepoints.
        """
        if self.num_t is None:
            if self.im_info.no_t:
                self.num_t = 1
            else:
                self.num_t = self.im_info.shape[self.im_info.axes.index('T')]

    def _allocate_memory(self):
        """
        Allocates memory for motion capture markers, distance transform, and border images.

        This method creates memory-mapped arrays for the instance label data, original image data,
        Frangi-filtered data (if needed), markers, distance transforms, and borders.
        """
        logger.debug('Allocating memory for mocap marking.')

        self.label_memmap = self.im_info.get_memmap(self.im_info.pipeline_paths['im_instance_label'])
        self.im_memmap = self.im_info.get_memmap(self.im_info.im_path)
        self.shape = self.label_memmap.shape

        if self.use_im == 'frangi':
            self.im_frangi_memmap = self.im_info.get_memmap(self.im_info.pipeline_paths['im_preprocessed'])
        else:
            self.im_frangi_memmap = None

        im_marker_path = self.im_info.pipeline_paths['im_marker']
        self.im_marker_memmap = self.im_info.allocate_memory(
            im_marker_path,
            dtype='uint8',
            description='mocap marker image',
            return_memmap=True
        )

        im_distance_path = self.im_info.pipeline_paths['im_distance']
        self.im_distance_memmap = self.im_info.allocate_memory(
            im_distance_path,
            dtype='float32',  # use float32 to reduce memory
            description='distance transform image',
            return_memmap=True
            )

        im_border_path = self.im_info.pipeline_paths['im_border']
        self.im_border_memmap = self.im_info.allocate_memory(
            im_border_path,
            dtype='uint8',
            description='border image',
            return_memmap=True
        )

    def _distance_im(self, mask):
        """
        Computes the distance transform of the binary mask and identifies border pixels.

        Parameters
        ----------
        mask : array-like (bool)
            Binary mask of segmented objects.

        Returns
        -------
        tuple
            (distance_im, border_mask)
            distance_im : same shape as mask, float32 distances (in pixels) to nearest background.
            border_mask : bool mask identifying border pixels of the segmented objects.
        """
        xp_mod = self.xp
        ndi_mod = self.ndi_backend

        # Border mask: one-pixel dilation minus original mask
        border_mask = ndi_mod.binary_dilation(mask, iterations=1) ^ mask

        # Distance transform: distance from foreground to nearest background.
        # Use pixel units (no anisotropic sampling) to match original KD-tree behaviour.
        distance_im = ndi_mod.distance_transform_edt(mask)
        distance_im = distance_im.astype(xp_mod.float32, copy=False)

        # Clamp distances to 2 * max_radius_px (as original code did for inf values)
        xp_mod.minimum(distance_im, self.max_radius_px * 2.0, out=distance_im)

        return distance_im, border_mask

    def _local_max_peak(self, use_im, mask, distance_im):
        """
        Detects local maxima in the image based on multi-scale filtering.

        This implementation:
        - Streams over scales (sigmas), no 4D (scale, z, y, x) arrays.
        - For each scale, computes LoG response, local maxima, and updates a global peak mask
          where the current scale response is better than previous scales.

        Parameters
        ----------
        use_im : array-like
            Image to use for detecting peaks ('distance' or 'frangi').
        mask : array-like (bool)
            Binary mask of segmented objects.
        distance_im : array-like
            Distance transform of the binary mask.

        Returns
        -------
        np.ndarray or xp.ndarray
            Coordinates of the detected peaks, shape (N, ndim).
        """
        xp_mod = self.xp
        ndi_mod = self.ndi_backend

        # Valid pixels: inside the object mask and with positive distance
        valid_mask = mask & (distance_im > 0)

        # Initialize best response and a peak mask
        best_resp = xp_mod.zeros_like(use_im, dtype=xp_mod.float32)
        peak_mask = xp_mod.zeros_like(use_im, dtype=bool)

        for s in self.sigmas:
            sigma_val = float(s)
            sigma_vec = self._get_sigma_vec(sigma_val)

            # LoG response with scale normalization (s^2)
            log_resp = -ndi_mod.gaussian_laplace(use_im, sigma_vec)
            log_resp = (log_resp * (sigma_val ** 2)).astype(xp_mod.float32, copy=False)

            # Clamp negative values
            log_resp[log_resp < 0] = 0

            # Local maxima in image space (no scale dimension)
            local_max = log_resp == ndi_mod.maximum_filter(log_resp, size=3, mode='nearest')

            # Restrict to valid pixels (inside objects and away from border)
            local_max &= valid_mask

            # Non-max suppression across scales (keep best response)
            better = local_max & (log_resp > best_resp)
            peak_mask[better] = True
            best_resp[better] = log_resp[better]

        coords_idx = xp_mod.argwhere(peak_mask)
        return coords_idx

    def _remove_close_peaks(self, coords, intensity_im):
        """
        Removes peaks that are too close together using morphological non-max suppression.

        Parameters
        ----------
        coords : array-like, shape (N, ndim)
            Coordinates of detected peaks.
        intensity_im : array-like
            Intensity image used as a scoring function for peaks.

        Returns
        -------
        array-like
            Coordinates of the remaining peaks after filtering.
        """
        xp_mod = self.xp
        ndi_mod = self.ndi_backend

        if coords.size == 0:
            return coords

        # Build a score image with intensities only at peak coordinates
        score_img = xp_mod.zeros_like(intensity_im, dtype=xp_mod.float32)
        score_img[tuple(coords.T)] = intensity_im[tuple(coords.T)]

        # Apply maximum filter with a window corresponding to peak_min_distance
        size = 2 * int(self.peak_min_distance) + 1
        max_filtered = ndi_mod.maximum_filter(score_img, size=size, mode='nearest')

        # Keep peaks that are equal to the local max and have positive score
        keep_mask = (score_img == max_filtered) & (score_img > 0)

        kept_coords = xp_mod.argwhere(keep_mask)
        return kept_coords

    def _run_frame_impl(self, t):
        """
        Internal implementation of marker detection for a single timepoint.

        This is called by _run_frame, which wraps it in GPU OOM handling.
        """
        xp_mod = self.xp
        logger.info(f'Running motion capture marking, volume {t}/{self.num_t - 1}')

        # Load intensity and mask for this frame into the current backend
        intensity_frame = xp_mod.asarray(self.im_memmap[t])
        mask_frame = xp_mod.asarray(self.label_memmap[t] > 0)
        mask_frame = mask_frame.astype(bool, copy=False)

        # Fast path: empty mask -> no markers, zero distance and borders
        if not np.any(self._to_cpu(mask_frame)):
            # Create empty outputs with correct dtypes
            marker = np.zeros_like(self.im_memmap[t], dtype=np.uint8)
            distance_im = np.zeros_like(self.im_memmap[t], dtype=np.float32)
            border_mask = np.zeros_like(self.im_memmap[t], dtype=np.uint8)
            return marker, distance_im, border_mask

        # Distance transform and border mask
        distance_im_backend, border_mask_backend = self._distance_im(mask_frame)

        # Select the image to use for LoG-based peak detection
        if self.use_im == 'distance':
            base_im = distance_im_backend
        elif self.use_im == 'frangi':
            if self.im_frangi_memmap is None:
                raise RuntimeError("Frangi image requested for peak detection but not available.")
            base_im = xp_mod.asarray(self.im_frangi_memmap[t])
        else:
            raise ValueError(f"Unknown use_im value: {self.use_im}")

        # Multi-scale LoG peak detection
        peak_coords = self._local_max_peak(base_im, mask_frame, distance_im_backend)

        # Remove peaks that are too close together using intensity-based NMS
        peak_coords = self._remove_close_peaks(peak_coords, intensity_frame)

        # Build marker image (binary)
        marker_backend = xp_mod.zeros_like(mask_frame, dtype=xp_mod.uint8)
        if peak_coords.size > 0:
            marker_backend[tuple(peak_coords.T)] = 1

        # Convert outputs to numpy for writing to memmaps
        marker = self._to_cpu(marker_backend).astype(np.uint8, copy=False)
        distance_im = self._to_cpu(distance_im_backend).astype(np.float32, copy=False)
        border_mask = self._to_cpu(border_mask_backend).astype(np.uint8, copy=False)

        return marker, distance_im, border_mask

    def _run_frame(self, t):
        """
        Runs marker detection for a single timepoint in the image with GPU OOM fallback.
        """
        try:
            return self._run_frame_impl(t)
        except GPU_OOM_ERRORS as e:
            if self.use_gpu:
                logger.warning(
                    "GPU OOM or memory error encountered on frame %d (%s). "
                    "Switching to CPU backend for remaining frames.",
                    t, str(e)
                )
                # Switch permanently to CPU and retry this frame
                self.use_gpu = False
                return self._run_frame_impl(t)
            else:
                # Already on CPU; re-raise
                raise

    def _run_mocap_marking(self):
        """
        Runs the marker detection process for all timepoints in the image.

        This method processes each timepoint sequentially and applies motion capture marking.
        """
        for t in range(self.num_t):
            if self.viewer is not None:
                self.viewer.status = f'Mocap marking. Frame: {t + 1} of {self.num_t}.'

            marker_frame, distance_frame, border_frame = self._run_frame(t)

            if self.im_marker_memmap.shape != self.shape and self.im_info.no_t:
                self.im_marker_memmap[:] = marker_frame
                if self.im_distance_memmap is not None:
                    self.im_distance_memmap[:] = distance_frame
                if self.im_border_memmap is not None:
                    self.im_border_memmap[:] = border_frame
            else:
                self.im_marker_memmap[t] = marker_frame
                if self.im_distance_memmap is not None:
                    self.im_distance_memmap[t] = distance_frame
                if self.im_border_memmap is not None:
                    self.im_border_memmap[t] = border_frame

            self.im_marker_memmap.flush()
            if self.im_distance_memmap is not None:
                self.im_distance_memmap.flush()
            if self.im_border_memmap is not None:
                self.im_border_memmap.flush()

    def run(self):
        """
        Main method to execute the motion capture marking process over the image data.

        This method allocates memory, sets sigma values, and runs the marking process for all timepoints.
        """
        # Note: We must run mocap marking even if there is no time dimension, since we need the distance and border images for feature extraction
        self._get_t()
        self._allocate_memory()
        self._set_default_sigmas()
        self._run_mocap_marking()


if __name__ == "__main__":
    im_path = r"D:\test_files\nelly_smorgasbord\deskewed-iono_pre.ome.tif"
    im_info = ImInfo(im_path)
    num_t = 3
    markers = Markers(im_info, num_t=num_t)
    markers.run()