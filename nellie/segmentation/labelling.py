"""
Semantic and instance segmentation for microscopy images.

This module provides the Label class for thresholding-based segmentation with
optimizations for large volumes and optional GPU acceleration.
"""
import numpy as np

from nellie.utils import adaptive_run
from nellie.utils.base_logger import logger
from nellie.im_info.verifier import ImInfo
from nellie.utils.gpu_functions import otsu_threshold, triangle_threshold

_UNSET = object()


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
                 min_radius_um=0.25,
                 threshold_sampling_pixels=1_000_000,
                 histogram_nbins=256,
                 device="auto",
                 low_memory: bool = False,
                 max_chunk_voxels: int = int(1e6)):
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
            this size instead of the full volume. If None and
            low_memory is True, a chunk size is inferred from max_chunk_voxels.
        flush_interval : int, optional
            How often (in frames) to flush the output memmap to disk.
        min_radius_um : float, optional
            Minimum expected object radius in micrometers. Labels smaller than
            the area/volume of a circle/sphere with this radius are removed.
        threshold_sampling_pixels : int, optional
            Maximum number of pixels sampled when computing global thresholds
            to reduce histogram cost for very large volumes.
        histogram_nbins : int, optional
            Number of bins to use in histogram-based thresholding.
        device : {"auto", "cpu", "gpu"}, optional
            Backend selection. "auto" uses GPU if available, otherwise CPU.
        low_memory : bool, optional
            If True, prefer chunked Z processing to reduce peak memory usage.
        max_chunk_voxels : int, optional
            Target maximum number of voxels per Z-chunk when low_memory is True
            and chunk_z is not specified.
        """
        self.im_info = im_info
        self.device = device
        self.xp, self.ndi, self.device_type = self._resolve_backend(device)
        self.num_t = num_t
        if num_t is None and not self.im_info.no_t:
            self.num_t = im_info.shape[im_info.axes.index('T')]
        self.threshold = threshold
        self.otsu_thresh_intensity = otsu_thresh_intensity

        self.im_memmap = None
        self.frangi_memmap = None

        self.semantic_mask_memmap = None
        self.instance_label_memmap = None
        self.shape = ()

        self.debug = {}

        self.viewer = viewer

        # Optimization / configuration parameters
        self.chunk_z = chunk_z if (not self.im_info.no_z and chunk_z is not None) else None
        self._user_chunk_z = self.chunk_z
        self.flush_interval = max(1, int(flush_interval))
        min_radius_um = float(min_radius_um)
        x_res = self.im_info.dim_res.get("X") or 1.0
        self.min_radius_um = max(min_radius_um, float(x_res))
        self.threshold_sampling_pixels = int(threshold_sampling_pixels)
        self.histogram_nbins = int(histogram_nbins)
        self.eps = 1e-8
        self.low_memory = bool(low_memory)
        self.max_chunk_voxels = int(max_chunk_voxels)

        if self.low_memory and self.chunk_z is None and not self.im_info.no_z:
            inferred_chunk = self._infer_chunk_z()
            if inferred_chunk is not None:
                self.chunk_z = inferred_chunk

        # Dimensionality and structuring elements (pre-computed)
        self.ndim = 2 if self.im_info.no_z else 3
        self.min_area_pixels = self._compute_min_area_pixels()
        self.footprint = None
        self._set_footprint()

    def _resolve_backend(self, device):
        device = (device or "auto").lower()
        if device not in ("auto", "cpu", "gpu", "cuda"):
            raise ValueError(f"Unsupported device '{device}'. Use 'auto', 'cpu', or 'gpu'.")

        if device in ("gpu", "cuda"):
            xp, ndi = self._try_import_cupy(require=True)
            return xp, ndi, "cuda"
        if device == "cpu":
            import scipy.ndimage as ndi
            return np, ndi, "cpu"

        xp, ndi = self._try_import_cupy(require=False)
        if xp is not None:
            return xp, ndi, "cuda"
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
        import scipy.ndimage as ndi

        self.xp = np
        self.ndi = ndi
        self.device_type = "cpu"
        self._set_footprint()

    def _set_backend(self, device):
        device = adaptive_run.normalize_device(device)
        self.device = device
        self.xp, self.ndi, self.device_type = self._resolve_backend(device)
        self._set_footprint()

    def _set_low_memory(self, low_memory):
        self.low_memory = bool(low_memory)
        if self.im_info.no_z:
            self.chunk_z = None
            return
        if self._user_chunk_z is not None:
            self.chunk_z = self._user_chunk_z
            return
        if self.low_memory:
            inferred_chunk = self._infer_chunk_z()
            self.chunk_z = inferred_chunk if inferred_chunk is not None else None
        else:
            self.chunk_z = None

    def _set_footprint(self):
        if self.im_info.no_z:
            self.footprint = self.xp.ones((3, 3), dtype=bool)
        else:
            self.footprint = self.xp.ones((3, 3, 3), dtype=bool)

    def _compute_min_area_pixels(self):
        x_res = self.im_info.dim_res.get("X") or 1.0
        y_res = self.im_info.dim_res.get("Y") or x_res
        if self.im_info.no_z:
            area_um2 = np.pi * (self.min_radius_um ** 2)
            area_px = area_um2 / (float(x_res) * float(y_res))
            return max(1, int(np.ceil(area_px)))
        z_res = self.im_info.dim_res.get("Z") or x_res
        volume_um3 = (4.0 / 3.0) * np.pi * (self.min_radius_um ** 3)
        volume_px = volume_um3 / (float(x_res) * float(y_res) * float(z_res))
        return max(1, int(np.ceil(volume_px)))

    def _uf_find(self, parent, x):
        root = parent.get(x, x)
        if root != x:
            root = self._uf_find(parent, root)
            parent[x] = root
        return root

    def _uf_union(self, parent, rank, a, b):
        root_a = self._uf_find(parent, a)
        root_b = self._uf_find(parent, b)
        if root_a == root_b:
            return False
        rank_a = rank.get(root_a, 0)
        rank_b = rank.get(root_b, 0)
        if rank_a < rank_b:
            root_a, root_b = root_b, root_a
            rank_a, rank_b = rank_b, rank_a
        parent[root_b] = root_a
        if rank_a == rank_b:
            rank[root_a] = rank_a + 1
        return True

    def _boundary_label_pairs(self, prev_slice, curr_slice):
        prev = np.asarray(prev_slice)
        curr = np.asarray(curr_slice)
        mask = (prev > 0) & (curr > 0)
        if not np.any(mask):
            return None
        pairs = np.stack((prev[mask], curr[mask]), axis=1)
        if pairs.size == 0:
            return None
        return np.unique(pairs, axis=0)

    def _relabel_frame_from_unions(self, t, z_dim, chunk_z, parent):
        if chunk_z is None or chunk_z <= 0:
            chunk_z = z_dim

        label_map = {0: 0}
        next_label = 1
        z_start = 0

        while z_start < z_dim:
            z_end = min(z_start + chunk_z, z_dim)
            labels_chunk = np.asarray(self.instance_label_memmap[t, z_start:z_end, ...])
            if labels_chunk.size == 0:
                z_start = z_end
                continue

            unique = np.unique(labels_chunk)
            if unique.size == 1 and unique[0] == 0:
                z_start = z_end
                continue

            roots = np.array([self._uf_find(parent, int(lab)) for lab in unique], dtype=labels_chunk.dtype)
            for root in roots:
                root = int(root)
                if root == 0:
                    continue
                if root not in label_map:
                    label_map[root] = next_label
                    next_label += 1

            new_ids = np.array([label_map[int(root)] for root in roots], dtype=labels_chunk.dtype)
            idx = np.searchsorted(unique, labels_chunk)
            labels_chunk = new_ids[idx]

            self._write_labels_chunk(t, z_start, z_end, labels_chunk)
            z_start = z_end

    def _infer_chunk_z(self):
        if self.max_chunk_voxels is None or self.max_chunk_voxels <= 0:
            return None

        axes = list(self.im_info.axes)
        shape = tuple(self.im_info.shape)
        if "T" in axes:
            t_idx = axes.index("T")
            axes = [ax for i, ax in enumerate(axes) if i != t_idx]
            shape = tuple(dim for i, dim in enumerate(shape) if i != t_idx)

        if "Z" not in axes:
            return None

        try:
            y_dim = int(shape[axes.index("Y")])
            x_dim = int(shape[axes.index("X")])
        except (ValueError, IndexError):
            return None

        if y_dim <= 0 or x_dim <= 0:
            return None

        chunk_z = int(self.max_chunk_voxels // (y_dim * x_dim))
        return max(1, chunk_z)

    def _xp_for_array(self, arr):
        try:
            import cupy
            if isinstance(arr, cupy.ndarray):
                return cupy
        except Exception:
            pass
        return np

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
        """
        original_view = self.im_memmap[t, ...]
        frangi_view = self.frangi_memmap[t, ...]
        return original_view, frangi_view

    def _write_labels_for_frame(self, t, labels):
        """
        Write a full label volume for timepoint t into the instance_label_memmap.
        """
        dst = self.instance_label_memmap
        dst[t, ...] = labels

    def _write_labels_chunk(self, t, z_start, z_end, labels_chunk):
        """
        Write a Z-chunk of labels for timepoint t into the instance_label_memmap.
        """
        dst = self.instance_label_memmap
        dst[t, z_start:z_end, ...] = labels_chunk

    # ------------------------------------------------------------------
    # Thresholding and labeling
    # ------------------------------------------------------------------

    def _sample_nonzero(self, frame, mask=None, mask_frame=None, mask_thresh=None):
        """
        Return a (possibly) downsampled 1D array of non-zero values from frame.

        If a boolean mask is provided, values are sampled where mask is True.
        If mask_frame and mask_thresh are provided, the mask is applied to
        sampled values only to avoid allocating a full-size mask.
        """
        flat = frame.reshape(-1)
        if flat.size == 0:
            return flat

        mask_flat = None
        mask_mode = None
        if mask is not None:
            mask_flat = mask.reshape(-1)
            mask_mode = "bool"
        elif mask_frame is not None and mask_thresh is not None:
            mask_flat = mask_frame.reshape(-1)
            mask_mode = "thresh"

        max_samples = max(1, int(self.threshold_sampling_pixels))
        step = max(int(flat.size) // max_samples, 1)
        offsets = (0, step // 2) if step > 1 and step // 2 > 0 else (0,)

        values = flat[:0]
        for offset in offsets:
            sample = flat[offset::step]
            if mask_mode == "bool":
                mask_sample = mask_flat[offset::step]
                values = sample[(sample > 0) & mask_sample]
            elif mask_mode == "thresh":
                mask_sample = mask_flat[offset::step] > mask_thresh
                values = sample[(sample > 0) & mask_sample]
            else:
                values = sample[sample > 0]

            if values.size > 0 or step == 1:
                return values

        try:
            max_val = float(flat.max())
        except Exception:
            xp = self._xp_for_array(flat)
            max_val = float(xp.max(flat))

        if max_val <= 0:
            return values

        if mask_mode == "bool":
            return flat[(flat > 0) & mask_flat]
        if mask_mode == "thresh":
            return flat[(flat > 0) & (mask_flat > mask_thresh)]
        return flat[flat > 0]

    def _compute_frangi_threshold(self, frame, mask_frame=None, mask_thresh=None):
        """
        Compute a combined triangle/Otsu threshold for a given frame (Frangi).
        """
        values = self._sample_nonzero(frame, mask_frame=mask_frame, mask_thresh=mask_thresh)
        if values.size == 0:
            return None

        # work in log10 domain to match original logic
        xp = self._xp_for_array(values)
        log_values = xp.log10(values)
        triangle = triangle_threshold(log_values, nbins=self.histogram_nbins, xp=xp)
        triangle = 10 ** triangle
        otsu, _ = otsu_threshold(log_values, nbins=self.histogram_nbins, xp=xp)
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

    def _get_labels(self, frame, frangi_thresh=_UNSET):
        """
        Generates binary labels for segmented objects in a single frame based
        on triangle/Otsu thresholding and connected components.
        """
        if frangi_thresh is _UNSET:
            frangi_thresh = self._compute_frangi_threshold(frame)

        if frangi_thresh is None:
            mask = self.xp.zeros_like(frame, dtype=bool)
        else:
            mask = frame > frangi_thresh

        # # Morphological cleanup
        # if self.footprint is not None:
        #     mask = self.ndi.binary_opening(mask, structure=self.footprint)

        # Fill holes for 3D data
        if not self.im_info.no_z:
            mask = self.ndi.binary_fill_holes(mask)

        # Connected component labeling
        labels, _ = self.ndi.label(mask, structure=self.footprint)

        # Remove very small objects using bincount + lookup table
        if labels.size == 0:
            return mask, labels

        areas = self.xp.bincount(labels.ravel())
        if areas.size <= 1:
            return mask, labels

        areas[0] = 0  # ignore background
        keep = areas >= self.min_area_pixels  # boolean array indexed by label id
        mask = keep[labels]
        # Smooth mask boundaries using mean filter + threshold
        mask_float = mask.astype(self.xp.float32)
        mask_smooth = self.ndi.uniform_filter(mask_float, size=3)
        mask = mask_smooth > 0.5
        
        labels, _ = self.ndi.label(mask, structure=self.footprint)

        return mask, labels

    def _compute_frame_thresholds(self, original_view, frangi_view):
        """
        Compute per-frame intensity and Frangi thresholds using CPU views.
        """
        intensity_thresh = None
        if self.otsu_thresh_intensity:
            intensity_thresh = self._compute_intensity_otsu_threshold(original_view)
            if intensity_thresh is None:
                intensity_thresh = 0
        elif self.threshold is not None:
            intensity_thresh = self.threshold

        if intensity_thresh is not None:
            frangi_thresh = self._compute_frangi_threshold(
                frangi_view,
                mask_frame=original_view,
                mask_thresh=intensity_thresh,
            )
        else:
            frangi_thresh = self._compute_frangi_threshold(frangi_view)

        return intensity_thresh, frangi_thresh

    # ------------------------------------------------------------------
    # Per-frame execution (full-volume or chunked)
    # ------------------------------------------------------------------

    def _run_frame_full_volume(self, t, original_view, frangi_view, intensity_thresh, frangi_thresh):
        """
        Runs segmentation for a single timepoint as a full volume.
        """
        logger.info(f'Running semantic segmentation, volume {t}/{self.num_t - 1}')

        try:
            # Load full timepoint volume into xp array
            original_in_mem = self.xp.asarray(original_view)
            frangi_in_mem = self.xp.asarray(frangi_view)

            # Optional intensity-based masking (read-only)
            if intensity_thresh is not None:
                mask = original_in_mem > intensity_thresh
                frangi_in_mem = frangi_in_mem * mask

            # Labeling on Frangi image
            _, labels = self._get_labels(frangi_in_mem, frangi_thresh=frangi_thresh)
            return labels
        except Exception as exc:
            if self._is_oom_error(exc) and self.device_type == "cuda":
                self._free_gpu_memory()
                if not self.im_info.no_z:
                    logger.warning(
                        "CUDA OOM during full-volume labeling; "
                        "falling back to chunked Z processing."
                    )
                    self._run_frame_chunked_z(
                        t,
                        original_view,
                        frangi_view,
                        intensity_thresh,
                        frangi_thresh,
                        initial_chunk=frangi_view.shape[0],
                    )
                    return None
                logger.warning("CUDA OOM during full-volume labeling; switching to CPU.")
                self._switch_to_cpu()
                return self._run_frame_full_volume(
                    t,
                    original_view,
                    frangi_view,
                    intensity_thresh,
                    frangi_thresh,
                )
            raise

    def _run_frame_chunked_z(self, t, original_view, frangi_view, intensity_thresh, frangi_thresh, initial_chunk=None):
        """
        Runs segmentation for a single timepoint, processing in Z-chunks.
        Labels are merged across chunk boundaries to preserve connectivity.
        """
        logger.info(f'Running semantic segmentation in Z-chunks, volume {t}/{self.num_t - 1}')

        if self.im_info.no_z:
            # No Z dimension: fall back to full-volume 2D processing
            labels = self._run_frame_full_volume(t, original_view, frangi_view, intensity_thresh, frangi_thresh)
            if labels is not None:
                if self.device_type == 'cuda':
                    labels = labels.get()
                self._write_labels_for_frame(t, labels)
            return

        # Assume Z is the first axis of the per-timepoint 3D volume
        z_dim = frangi_view.shape[0]
        if initial_chunk is None:
            initial_chunk = self.chunk_z if self.chunk_z is not None else z_dim
        if initial_chunk is None or initial_chunk <= 0:
            initial_chunk = z_dim

        current_chunk = max(1, min(int(initial_chunk), z_dim))
        z_start = 0
        frame_label_offset = 0
        relabel_chunk_z = None
        parent = {}
        rank = {}
        prev_boundary = None
        had_merges = False

        while z_start < z_dim:
            z_end = min(z_start + current_chunk, z_dim)

            # Extract CPU chunks from memmap
            original_chunk_cpu = original_view[z_start:z_end, ...]
            frangi_chunk_cpu = frangi_view[z_start:z_end, ...]

            try:
                original_chunk = self.xp.asarray(original_chunk_cpu)
                frangi_chunk = self.xp.asarray(frangi_chunk_cpu)

                # Optional intensity-based masking per chunk (read-only)
                if intensity_thresh is not None:
                    mask = original_chunk > intensity_thresh
                    frangi_chunk = frangi_chunk * mask

                # Labeling on Frangi chunk
                _, labels_chunk = self._get_labels(frangi_chunk, frangi_thresh=frangi_thresh)

                # Offset labels to make them unique within the frame
                if labels_chunk.size > 0:
                    max_label_chunk = int(labels_chunk.max())
                else:
                    max_label_chunk = 0

                if max_label_chunk > 0:
                    labels_chunk = labels_chunk.astype('int32', copy=False)
                    labels_chunk[labels_chunk > 0] += frame_label_offset
                    frame_label_offset += max_label_chunk

                # Move to host if on CUDA and write to memmap
                if self.device_type == 'cuda':
                    labels_chunk = labels_chunk.get()

                if prev_boundary is not None and labels_chunk.size > 0:
                    curr_boundary = labels_chunk[0, ...]
                    pairs = self._boundary_label_pairs(prev_boundary, curr_boundary)
                    if pairs is not None:
                        for prev_lab, curr_lab in pairs:
                            merged = self._uf_union(
                                parent, rank, int(prev_lab), int(curr_lab)
                            )
                            had_merges = had_merges or merged

                if labels_chunk.size > 0:
                    prev_boundary = labels_chunk[-1, ...].copy()
                else:
                    prev_boundary = None

                self._write_labels_chunk(t, z_start, z_end, labels_chunk)
                relabel_chunk_z = current_chunk if relabel_chunk_z is None else min(
                    relabel_chunk_z, current_chunk
                )

                z_start = z_end  # advance to next chunk
            except Exception as exc:
                if self._is_oom_error(exc):
                    self._free_gpu_memory()
                    if current_chunk > 1:
                        current_chunk = max(current_chunk // 2, 1)
                        logger.warning(
                            f'OOM at Z range [{z_start}, {z_end}); '
                            f'reducing chunk_z to {current_chunk}.'
                        )
                        continue
                    if self.device_type == "cuda":
                        logger.warning('OOM even with chunk_z=1; switching to CPU.')
                        self._switch_to_cpu()
                        continue
                    logger.error('OOM even with chunk_z=1 on CPU; aborting.')
                    raise
                raise

        if had_merges:
            self._relabel_frame_from_unions(t, z_dim, relabel_chunk_z, parent)

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

            original_view, frangi_view = self._get_frame_views(t)
            intensity_thresh, frangi_thresh = self._compute_frame_thresholds(original_view, frangi_view)

            if self.chunk_z is not None and not self.im_info.no_z:
                # Chunked processing writes directly to memmap
                self._run_frame_chunked_z(
                    t,
                    original_view,
                    frangi_view,
                    intensity_thresh,
                    frangi_thresh,
                )
            else:
                # Full-volume processing
                labels = self._run_frame_full_volume(
                    t,
                    original_view,
                    frangi_view,
                    intensity_thresh,
                    frangi_thresh,
                )
                if labels is not None:
                    if self.device_type == 'cuda':
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
        device = adaptive_run.normalize_device(self.device)
        gpu_ok = adaptive_run.gpu_available()
        if device == "gpu" and not gpu_ok:
            logger.warning("Label: GPU requested but not available; falling back to CPU.")
        if device == "cpu" or not gpu_ok:
            device_order = ["cpu"]
        else:
            device_order = ["gpu", "cpu"]

        start_low_memory = bool(self.low_memory) or adaptive_run.should_use_low_memory(
            self.im_info, include_gpu="gpu" in device_order
        )
        if start_low_memory and not self.low_memory:
            logger.info("Label: enabling low-memory mode based on estimated usage.")

        last_exc = None
        for dev, low in adaptive_run.mode_candidates(device_order, start_low_memory):
            try:
                self._set_backend(dev)
                self._set_low_memory(low)
                self._get_t()
                self._allocate_memory()
                self._run_segmentation()
                return
            except Exception as exc:
                last_exc = exc
                if adaptive_run.is_gpu_unavailable_error(exc) and dev == "gpu":
                    logger.warning("Label: GPU backend unavailable; retrying on CPU.")
                    continue
                if adaptive_run.is_oom_error(exc):
                    logger.warning(
                        "Label: OOM on %s/%s; retrying with lower settings.",
                        dev,
                        "low-memory" if low else "high-memory",
                    )
                    continue
                raise
        raise last_exc


if __name__ == "__main__":
    im_path = r"F:\2024_06_26_SD_ExM_nhs_u2OS_488+578_cropped.tif"
    im_info = ImInfo(
        im_path,
        dim_res={'T': 1, 'Z': 0.2, 'Y': 0.1, 'X': 0.1},
        dimension_order='ZYX'
    )
    segment_unique = Label(im_info)
    segment_unique.run()
