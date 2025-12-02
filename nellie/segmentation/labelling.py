"""
Semantic and instance segmentation for microscopy images.

This module provides the Label class for thresholding-based segmentation with
optimizations for large volumes and optional GPU acceleration.
"""
from nellie import xp, ndi, device_type
from nellie.utils.base_logger import logger
from nellie.im_info.verifier import ImInfo
from nellie.utils.gpu_functions import otsu_threshold, triangle_threshold

try:
    import cupy  # type: ignore
    CuPyOutOfMemoryError = cupy.cuda.memory.OutOfMemoryError
except Exception:  # cupy not available or not using CUDA backend
    CuPyOutOfMemoryError = MemoryError


class Label:
    """
    A class for semantic and instance segmentation of microscopy images using
    thresholding techniques, optimized for large volumes and optional GPU acceleration.
    """

    def __init__(self, im_info: ImInfo,
                 num_t=None,
                 threshold=None,
                 otsu_thresh_intensity=False,
                 viewer=None,
                 chunk_z=None,
                 flush_interval=1,
                 threshold_sampling_pixels=1_000_000,
                 histogram_nbins=256,
                 bg_min_pixels=10):
        """
        Parameters
        ----------
        im_info : ImInfo
            Image metadata and paths.
        num_t : int, optional
            Number of timepoints to process.
        threshold : float or None, optional
            Fixed intensity threshold for segmentation (if not using Otsu).
        otsu_thresh_intensity : bool, optional
            Whether to apply Otsu's method for intensity thresholding.
        viewer : object or None, optional
            Viewer object for displaying status.
        chunk_z : int or None, optional
            If not None and image has Z, process each timepoint in Z-chunks of
            this size instead of the full volume. This reduces peak memory
            at the cost of losing connectivity across chunks.
        flush_interval : int, optional
            How often (in frames) to flush the output memmap to disk.
        threshold_sampling_pixels : int, optional
            Maximum number of pixels sampled when computing global thresholds
            to reduce histogram cost for very large volumes.
        histogram_nbins : int, optional
            Number of bins to use in histogram-based thresholding.
        """
        self.im_info = im_info
        self.num_t = num_t
        if num_t is None and not self.im_info.no_t:
            self.num_t = im_info.shape[im_info.axes.index('T')]
        self.threshold = threshold
        self.otsu_thresh_intensity = otsu_thresh_intensity

        self.im_memmap = None
        self.frangi_memmap = None

        self.max_label_num = 0

        if not self.im_info.no_z:
            self.min_z_radius_um = min(self.im_info.dim_res['Z'], 0.2)

        self.semantic_mask_memmap = None
        self.instance_label_memmap = None
        self.shape = ()

        self.debug = {}

        self.viewer = viewer

        # Optimization / configuration parameters
        self.chunk_z = chunk_z if (not self.im_info.no_z and chunk_z is not None) else None
        self.flush_interval = max(1, int(flush_interval))
        self.threshold_sampling_pixels = int(threshold_sampling_pixels)
        self.histogram_nbins = int(histogram_nbins)
        self.eps = 1e-8

        # Dimensionality and structuring elements (pre-computed)
        self.ndim = 2 if self.im_info.no_z else 3
        self.footprint = ndi.generate_binary_structure(self.ndim, 1)
        self.min_area_pixels = 4

        if not self.im_info.no_z and self.im_info.dim_res['Z'] >= self.min_z_radius_um:
            self.opening_structure = xp.ones((2, 2, 2), dtype=bool)
        elif self.im_info.no_z:
            self.opening_structure = xp.ones((2, 2), dtype=bool)
        else:
            self.opening_structure = None

    def _get_t(self):
        """
        Determines the number of timepoints to process.
        """
        if self.num_t is None:
            if self.im_info.no_t:
                self.num_t = 1
            else:
                self.num_t = self.im_info.shape[self.im_info.axes.index('T')]
        else:
            return

    def _allocate_memory(self):
        """
        Allocates memory for the original image, Frangi-filtered image, and
        instance segmentation masks.
        """
        logger.debug('Allocating memory for semantic segmentation.')
        self.im_memmap = self.im_info.get_memmap(self.im_info.im_path)
        self.frangi_memmap = self.im_info.get_memmap(self.im_info.pipeline_paths['im_preprocessed'])
        self.shape = self.frangi_memmap.shape

        im_instance_label_path = self.im_info.pipeline_paths['im_instance_label']
        self.instance_label_memmap = self.im_info.allocate_memory(
            im_instance_label_path,
            dtype='int32',
            description='instance segmentation',
            return_memmap=True
        )

    # ------------------------------------------------------------------
    # Helpers for accessing per-frame views and writing to memmaps
    # ------------------------------------------------------------------

    def _get_frame_views(self, t):
        """
        Return CPU views (memmap slices) for original and Frangi frames at time t.
        Handles both 3D (Z,Y,X) and 4D (T,Z,Y,X) shapes.
        """
        if self.im_memmap.ndim == 4:
            original_view = self.im_memmap[t, ...]
            frangi_view = self.frangi_memmap[t, ...]
        elif self.im_memmap.ndim == 3:
            original_view = self.im_memmap[...]
            frangi_view = self.frangi_memmap[...]
        else:
            raise RuntimeError(f"Unsupported im_memmap ndim={self.im_memmap.ndim}")
        return original_view, frangi_view

    def _write_labels_for_frame(self, t, labels):
        """
        Write a full label volume for timepoint t into the instance_label_memmap.
        Handles both 3D (Z,Y,X) and 4D (T,Z,Y,X) shapes.
        """
        dst = self.instance_label_memmap
        if dst.ndim == 4:
            if self.num_t is not None and self.num_t > 1:
                dst[t, ...] = labels
            else:
                dst[...] = labels  # broadcast over T axis if length-1
        elif dst.ndim == 3:
            dst[...] = labels
        else:
            raise RuntimeError(f"Unsupported instance_label_memmap ndim={dst.ndim}")

    def _write_labels_chunk(self, t, z_start, z_end, labels_chunk):
        """
        Write a Z-chunk of labels for timepoint t into the instance_label_memmap.
        """
        dst = self.instance_label_memmap
        if dst.ndim == 4:
            if self.num_t is not None and self.num_t > 1:
                dst[t, z_start:z_end, ...] = labels_chunk
            else:
                dst[:, z_start:z_end, ...] = labels_chunk  # broadcast over T axis of length-1
        elif dst.ndim == 3:
            dst[z_start:z_end, ...] = labels_chunk
        else:
            raise RuntimeError(f"Unsupported instance_label_memmap ndim={dst.ndim}")

    # ------------------------------------------------------------------
    # Thresholding and labeling
    # ------------------------------------------------------------------

    def _sample_nonzero(self, frame):
        """
        Return a (possibly) downsampled 1D array of non-zero values from frame,
        used for histogram-based threshold estimation.
        """
        nonzero = frame[frame > 0]
        if nonzero.size == 0:
            return nonzero
        if nonzero.size > self.threshold_sampling_pixels:
            step = max(nonzero.size // self.threshold_sampling_pixels, 1)
            return nonzero[::step]
        return nonzero

    def _compute_frangi_threshold(self, frame):
        """
        Compute a combined triangle/Otsu threshold for a given frame (Frangi).
        """
        values = self._sample_nonzero(frame)
        if values.size == 0:
            return None

        # work in log10 domain to match original logic
        log_values = xp.log10(values)
        triangle = triangle_threshold(log_values, nbins=self.histogram_nbins)
        triangle = 10 ** triangle
        otsu, _ = otsu_threshold(log_values, nbins=self.histogram_nbins)
        otsu = 10 ** otsu
        return min(triangle, otsu)

    def _compute_intensity_otsu_threshold(self, frame):
        """
        Compute Otsu threshold on the original intensity frame using sampling.
        """
        values = self._sample_nonzero(frame)
        if values.size == 0:
            return None
        thresh, _ = otsu_threshold(values, nbins=self.histogram_nbins)
        return thresh

    def _get_labels(self, frame):
        """
        Generates binary labels for segmented objects in a single frame based
        on triangle/Otsu thresholding and connected components.
        """
        min_thresh = self._compute_frangi_threshold(frame)
        if min_thresh is None:
            mask = xp.zeros_like(frame, dtype=bool)
        else:
            mask = frame > min_thresh

        # Morphological cleanup
        if self.opening_structure is not None:
            mask = ndi.binary_opening(mask, structure=self.opening_structure)

        # Fill holes for 3D data
        if not self.im_info.no_z:
            mask = ndi.binary_fill_holes(mask)

        # Connected component labeling
        labels, _ = ndi.label(mask, structure=self.footprint)

        # Remove very small objects using bincount + lookup table
        if labels.size == 0:
            return mask, labels

        areas = xp.bincount(labels.ravel())
        if areas.size <= 1:
            return mask, labels

        areas[0] = 0  # ignore background
        keep = areas >= self.min_area_pixels  # boolean array indexed by label id
        mask = keep[labels]
        labels, _ = ndi.label(mask, structure=self.footprint)

        return mask, labels

    # ------------------------------------------------------------------
    # Per-frame execution (full-volume or chunked)
    # ------------------------------------------------------------------

    def _run_frame_full_volume(self, t):
        """
        Runs segmentation for a single timepoint as a full volume.
        """
        logger.info(f'Running semantic segmentation, volume {t}/{self.num_t - 1}')

        # Load full timepoint volume into xp array
        original_view, frangi_view = self._get_frame_views(t)
        original_in_mem = xp.asarray(original_view)
        frangi_in_mem = xp.asarray(frangi_view)

        # Optional intensity-based masking
        if self.otsu_thresh_intensity or self.threshold is not None:
            if self.otsu_thresh_intensity:
                thresh = self._compute_intensity_otsu_threshold(original_in_mem)
                if thresh is None:
                    thresh = 0
            else:
                thresh = self.threshold

            if thresh is not None:
                mask = original_in_mem > thresh
                original_in_mem *= mask
                frangi_in_mem *= mask

        # Labeling on Frangi image
        _, labels = self._get_labels(frangi_in_mem)

        # Ensure labels are positive and globally unique across frames
        if labels.size > 0:
            max_label = int(labels.max())
        else:
            max_label = 0

        if max_label > 0:
            offset = int(self.max_label_num)
            labels = labels.astype('int32', copy=False)
            labels[labels > 0] += offset
            self.max_label_num = offset + max_label

        return labels

    def _run_frame_chunked_z(self, t):
        """
        Runs segmentation for a single timepoint, processing in Z-chunks.
        Labels are not connected across chunks.
        """
        logger.info(f'Running semantic segmentation in Z-chunks, volume {t}/{self.num_t - 1}')

        original_view, frangi_view = self._get_frame_views(t)

        if self.im_info.no_z:
            # No Z dimension: fall back to full-volume 2D processing
            labels = self._run_frame_full_volume(t)
            if device_type == 'cuda':
                labels = labels.get()
            self._write_labels_for_frame(t, labels)
            return

        # Assume Z is the first axis of the per-timepoint 3D volume
        z_dim = frangi_view.shape[0]
        if self.chunk_z is None or self.chunk_z <= 0:
            # Safety fallback
            labels = self._run_frame_full_volume(t)
            if device_type == 'cuda':
                labels = labels.get()
            self._write_labels_for_frame(t, labels)
            return

        current_chunk = int(self.chunk_z)
        z_start = 0

        while z_start < z_dim:
            z_end = min(z_start + current_chunk, z_dim)

            # Extract CPU chunks from memmap
            original_chunk_cpu = original_view[z_start:z_end, ...]
            frangi_chunk_cpu = frangi_view[z_start:z_end, ...]

            try:
                original_chunk = xp.asarray(original_chunk_cpu)
                frangi_chunk = xp.asarray(frangi_chunk_cpu)
            except CuPyOutOfMemoryError:
                # Reduce chunk size and retry
                if current_chunk > 1:
                    current_chunk = max(current_chunk // 2, 1)
                    logger.warning(
                        f'CUDA OOM at Z range [{z_start}, {z_end}); '
                        f'reducing chunk_z to {current_chunk}.'
                    )
                    continue
                else:
                    logger.error('CUDA OOM even with chunk_z=1; aborting.')
                    raise

            # Optional intensity-based masking per chunk
            if self.otsu_thresh_intensity or self.threshold is not None:
                if self.otsu_thresh_intensity:
                    thresh = self._compute_intensity_otsu_threshold(original_chunk)
                    if thresh is None:
                        thresh = 0
                else:
                    thresh = self.threshold

                if thresh is not None:
                    mask = original_chunk > thresh
                    original_chunk *= mask
                    frangi_chunk *= mask

            # Labeling on Frangi chunk
            _, labels_chunk = self._get_labels(frangi_chunk)

            # Offset labels to make them unique across chunks/timepoints
            if labels_chunk.size > 0:
                max_label_chunk = int(labels_chunk.max())
            else:
                max_label_chunk = 0

            if max_label_chunk > 0:
                offset = int(self.max_label_num)
                labels_chunk = labels_chunk.astype('int32', copy=False)
                labels_chunk[labels_chunk > 0] += offset
                self.max_label_num = offset + max_label_chunk

            # Move to host if on CUDA and write to memmap
            if device_type == 'cuda':
                labels_chunk = labels_chunk.get()

            self._write_labels_chunk(t, z_start, z_end, labels_chunk)

            z_start = z_end  # advance to next chunk

    # ------------------------------------------------------------------
    # Main segmentation loop
    # ------------------------------------------------------------------

    def _run_segmentation(self):
        """
        Runs the full segmentation process for all timepoints.
        """
        for t in range(self.num_t):
            if self.viewer is not None:
                self.viewer.status = f'Extracting organelles. Frame: {t + 1} of {self.num_t}.'

            if self.chunk_z is not None and not self.im_info.no_z:
                # Chunked processing writes directly to memmap
                self._run_frame_chunked_z(t)
            else:
                # Full-volume processing
                labels = self._run_frame_full_volume(t)
                if device_type == 'cuda':
                    labels = labels.get()
                self._write_labels_for_frame(t, labels)

            if (t + 1) % self.flush_interval == 0:
                self.instance_label_memmap.flush()

        self.instance_label_memmap.flush()

    def run(self):
        """
        Main method to execute the full segmentation process over the image data.
        """
        logger.info('Running semantic segmentation.')
        self._get_t()
        self._allocate_memory()
        self._run_segmentation()


if __name__ == "__main__":
    im_path = r"F:\2024_06_26_SD_ExM_nhs_u2OS_488+578_cropped.tif"
    im_info = ImInfo(
        im_path,
        dim_res={'T': 1, 'Z': 0.2, 'Y': 0.1, 'X': 0.1},
        dimension_order='ZYX'
    )
    segment_unique = Label(im_info)
    segment_unique.run()