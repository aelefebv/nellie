"""
Network skeletonization and analysis for microscopy images.

This module provides the Network class for skeletonizing network-like structures
and analyzing their topology with optimized CPU/GPU processing.
"""
import itertools
import numpy as np
import skimage.measure
import skimage.morphology as morph
from scipy.spatial import cKDTree
from scipy import ndimage as ndi_cpu

from nellie.utils import adaptive_run
from nellie.utils.base_logger import logger
from nellie.im_info.verifier import ImInfo


class Network:
    """
    Optimized class for analyzing and skeletonizing network-like structures in 3D or 4D microscopy images.

    This version focuses on:
      - Reduced CPU/GPU thrashing.
      - Vectorized neighborhood operations (no Python per-voxel loops on large arrays).
      - More memory-friendly local-max detection.
      - More efficient branch relabeling using distance transforms on per-object crops.
      - Graceful degradation when GPU memory is insufficient (CPU/chunked fallback).

    Parameters
    ----------
    im_info : ImInfo
        Image metadata and paths.
    num_t : int, optional
        Number of timepoints to process. Defaults to all timepoints.
    min_radius_um : float, optional
        Minimum radius of detected objects in micrometers.
    max_radius_um : float, optional
        Maximum radius of detected objects in micrometers.
    viewer : object or None, optional
        Viewer object for status reporting.
    device : {"auto", "cpu", "gpu"}, optional
        Backend selection for connectivity computations.
    low_memory : bool, optional
        If True, use chunked CPU fallbacks for local neighborhood operations.
    max_chunk_voxels : int, optional
        Maximum voxels per chunk for low-memory paths.
    """

    def __init__(
        self,
        im_info: ImInfo,
        num_t=None,
        min_radius_um=0.20,
        max_radius_um=1,
        viewer=None,
        device="auto",
        low_memory: bool = False,
        max_chunk_voxels: int = int(1e6),
    ):

        self.im_info = im_info
        self.device = device
        self.xp, self.ndi, self.device_type = self._resolve_backend(device)
        self.force_device = device is not None and device.lower() in ("cpu", "gpu", "cuda")
        self.low_memory = low_memory
        self.max_chunk_voxels = int(max_chunk_voxels)
        self.num_t = num_t
        if num_t is None and not self.im_info.no_t:
            self.num_t = im_info.shape[im_info.axes.index('T')]

        if not self.im_info.no_z:
            self.z_ratio = self.im_info.dim_res['Z'] / self.im_info.dim_res['X']

        # either (roughly) diffraction limit, or pixel size, whichever is larger
        self.min_radius_um = max(min_radius_um, self.im_info.dim_res['X'])
        self.max_radius_um = max_radius_um

        self.min_radius_px = self.min_radius_um / self.im_info.dim_res['X']
        self.max_radius_px = self.max_radius_um / self.im_info.dim_res['X']

        if self.im_info.no_z:
            self.scaling = (im_info.dim_res['Y'], im_info.dim_res['X'])
        else:
            self.scaling = (im_info.dim_res['Z'], im_info.dim_res['Y'], im_info.dim_res['X'])

        self.shape = ()

        self.im_memmap = None
        self.im_frangi_memmap = None
        self.label_memmap = None
        self.network_memmap = None  # kept for compatibility, not used
        self.pixel_class_memmap = None
        self.skel_memmap = None
        self.skel_relabelled_memmap = None

        self.sigmas = None

        self.debug = None

        self.viewer = viewer

    # -------------------------------------------------------------------------
    # Helper methods for device handling
    # -------------------------------------------------------------------------
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

    def _to_xp(self, arr):
        """
        Convert an array to the backend array type (xp).
        """
        # xp is numpy when running on CPU and cupy when on GPU.
        try:
            return self.xp.asarray(arr)
        except Exception as e:
            if self.device_type == "cuda":
                raise
            logger.warning(f"xp.asarray failed; falling back to numpy. Error: {e}")
            return np.asarray(arr)

    def _to_cpu(self, arr):
        """
        Convert xp array to a numpy array. If already numpy, return as-is.
        """
        if self.device_type == "cuda" and hasattr(arr, "get"):
            return arr.get()
        return np.asarray(arr)

    # -------------------------------------------------------------------------
    # Neighborhood-based skeleton cleanup
    # -------------------------------------------------------------------------
    def _remove_connected_label_pixels(self, skel_labels, force_cpu: bool = False):
        """
        Removes skeleton pixels that are connected to multiple labeled regions.

        This vectorized implementation replaces the original per-pixel Python loop
        with min/max filters over 3x3 (2D) or 3x3x3 (3D) neighborhoods.
        """
        if force_cpu:
            labels_np = np.asarray(skel_labels)
            if self.low_memory:
                return self._remove_connected_label_pixels_chunked(labels_np)
            return self._remove_connected_label_pixels_impl(labels_np, np, ndi_cpu)

        labels_xp = self._to_xp(skel_labels)
        if self.low_memory:
            labels_np = self._to_cpu(labels_xp)
            return self._remove_connected_label_pixels_chunked(labels_np)

        try:
            return self._remove_connected_label_pixels_impl(labels_xp, self.xp, self.ndi)
        except Exception as exc:
            if not self._is_oom_error(exc):
                raise
            self._free_gpu_memory()
            labels_np = self._to_cpu(labels_xp)
            return self._remove_connected_label_pixels_chunked(labels_np)

    def _remove_connected_label_pixels_impl(self, labels, xp, ndi):
        mask = labels > 0

        if self.im_info.no_z:
            size = (3, 3)
        else:
            size = (3, 3, 3)

        max_labels = ndi.maximum_filter(labels, size=size, mode="constant", cval=0)

        bg_val = int(labels.max()) + 1
        labels_no_bg = xp.where(labels == 0, bg_val, labels)
        min_labels = ndi.minimum_filter(labels_no_bg, size=size, mode="constant", cval=bg_val)
        min_labels = xp.where(min_labels == bg_val, 0, min_labels)

        ambiguous = mask & (min_labels > 0) & (max_labels > 0) & (min_labels != max_labels)

        # Preserve original behavior: do not modify boundary voxels.
        boundary = xp.zeros_like(mask, dtype=bool)
        if self.im_info.no_z:
            boundary[0, :] = True
            boundary[-1, :] = True
            boundary[:, 0] = True
            boundary[:, -1] = True
        else:
            boundary[0, :, :] = True
            boundary[-1, :, :] = True
            boundary[:, 0, :] = True
            boundary[:, -1, :] = True
            boundary[:, :, 0] = True
            boundary[:, :, -1] = True

        ambiguous = ambiguous & ~boundary

        cleaned = xp.where(ambiguous, 0, labels)
        return cleaned

    def _remove_connected_label_pixels_chunked(self, labels):
        labels_np = np.asarray(labels)
        shape = labels_np.shape
        halo = (1,) * labels_np.ndim
        chunk_shape = self._compute_chunk_shape(shape, self.max_chunk_voxels)
        cleaned = np.zeros_like(labels_np)

        for core, ext, core_in_ext in self._iter_chunks(shape, chunk_shape, halo):
            chunk = labels_np[ext]
            cleaned_chunk = self._remove_connected_label_pixels_impl(chunk, np, ndi_cpu)
            cleaned[core] = cleaned_chunk[core_in_ext]

        return cleaned

    # -------------------------------------------------------------------------
    # Ensure every object has at least one skeleton voxel
    # -------------------------------------------------------------------------
    def _add_missing_skeleton_labels(self, skel_frame, label_frame, frangi_frame):
        """
        Adds missing labels to the skeleton where the intensity is highest within a labeled region.
        """
        logger.debug("Adding missing skeleton labels.")

        labels_np = np.asarray(label_frame)
        skel_np = np.asarray(skel_frame)
        frangi_np = np.asarray(frangi_frame)

        def _normalize_pos(pos, ndim):
            if pos is None:
                return None
            pos_arr = np.asarray(pos)
            if pos_arr.ndim == 1 and pos_arr.size == ndim:
                return tuple(int(p) for p in pos_arr.tolist())
            if pos_arr.ndim == 2:
                if pos_arr.shape == (1, ndim):
                    return tuple(int(p) for p in pos_arr[0].tolist())
                if pos_arr.shape == (ndim, 1):
                    return tuple(int(p) for p in pos_arr[:, 0].tolist())
            if isinstance(pos, (list, tuple)) and len(pos) == 1:
                inner_arr = np.asarray(pos[0])
                if inner_arr.ndim == 1 and inner_arr.size == ndim:
                    return tuple(int(p) for p in inner_arr.tolist())
            return None

        unique_labels = np.unique(labels_np)
        if unique_labels.size == 0:
            return skel_np
        unique_skel_labels = np.unique(skel_np)

        missing_labels = np.setdiff1d(unique_labels, unique_skel_labels)
        missing_labels = missing_labels[missing_labels != 0]
        if missing_labels.size == 0:
            return skel_np

        try:
            positions = ndi_cpu.maximum_position(
                frangi_np, labels=labels_np, index=missing_labels
            )
        except Exception as exc:
            logger.warning(
                f"Maximum-position lookup failed; leaving {len(missing_labels)} labels without skeletons. "
                f"Error: {exc}"
            )
            return skel_np

        if len(missing_labels) == 1:
            positions = [positions]

        for lab, pos in zip(missing_labels, positions):
            pos = _normalize_pos(pos, skel_np.ndim)
            if pos is None:
                logger.warning(
                    "Skipping missing skeleton label %s due to unrecognized position: "
                    "pos=%s ndim=%s shape=%s",
                    lab,
                    pos,
                    skel_np.ndim,
                    skel_np.shape,
                )
                continue
            if any(p < 0 or p >= dim for p, dim in zip(pos, skel_np.shape)):
                logger.warning(
                    "Skipping missing skeleton label %s due to out-of-bounds position: "
                    "pos=%s shape=%s",
                    lab,
                    pos,
                    skel_np.shape,
                )
                continue
            skel_np[pos] = lab

        return skel_np

    # -------------------------------------------------------------------------
    # Skeletonization
    # -------------------------------------------------------------------------
    def _skeletonize(self, label_frame):
        """
        Skeletonizes the labeled regions on CPU.
        """
        cpu_labels = np.asarray(label_frame)
        if self.low_memory:
            return self._skeletonize_per_object(cpu_labels)

        try:
            skel_mask_cpu = morph.skeletonize(cpu_labels > 0)
        except MemoryError:
            logger.warning("Skeletonization OOM; falling back to per-object skeletonization.")
            return self._skeletonize_per_object(cpu_labels)

        skel_labels_cpu = cpu_labels * skel_mask_cpu
        return skel_labels_cpu

    def _skeletonize_per_object(self, label_frame):
        labels_np = np.asarray(label_frame)
        skel_out = np.zeros_like(labels_np)

        max_label = int(labels_np.max())
        if max_label == 0:
            return skel_out

        slices = ndi_cpu.find_objects(labels_np)
        if slices is None:
            return skel_out

        for lab in range(1, max_label + 1):
            idx = lab - 1
            if idx >= len(slices):
                break
            sl = slices[idx]
            if sl is None:
                continue

            sub_labels = labels_np[sl]
            obj_mask = sub_labels == lab
            if not obj_mask.any():
                continue

            try:
                skel_sub = morph.skeletonize(obj_mask)
            except Exception as exc:
                logger.warning(
                    f"Skeletonization failed for label {lab}; leaving object without skeleton. Error: {exc}"
                )
                continue

            skel_out_sub = skel_out[sl]
            skel_out_sub[skel_sub] = lab
            skel_out[sl] = skel_out_sub

        return skel_out

    # -------------------------------------------------------------------------
    # Sigma management for multi-scale filters
    # -------------------------------------------------------------------------
    def _get_sigma_vec(self, sigma):
        """
        Computes the sigma vector for multi-scale filtering based on image dimensions.
        """
        if self.im_info.no_z:
            sigma_vec = (sigma, sigma)
        else:
            sigma_vec = (sigma / self.z_ratio, sigma, sigma)
        return sigma_vec

    def _set_default_sigmas(self):
        """
        Sets the default sigma values for multi-scale filtering based on the
        minimum and maximum radius in pixels.
        """
        logger.debug('Setting sigma values for multi-scale filters.')
        min_sigma_step_size = 0.2
        num_sigma = 5

        self.sigma_min = self.min_radius_px / 2
        self.sigma_max = self.max_radius_px / 3

        sigma_step_size_calculated = (self.sigma_max - self.sigma_min) / num_sigma
        sigma_step_size = max(min_sigma_step_size, sigma_step_size_calculated)

        # Use numpy here; sigma values are small and do not need GPU.
        self.sigmas = np.arange(self.sigma_min, self.sigma_max, sigma_step_size).tolist()
        logger.debug(f'Calculated sigma step size = {sigma_step_size_calculated}. Sigmas = {self.sigmas}')

    # -------------------------------------------------------------------------
    # Branch relabeling using per-object distance transforms
    # -------------------------------------------------------------------------
    def _relabel_objects(self, branch_skel_labels, label_frame):
        """
        Relabels skeleton pixels by propagating labels to nearby unlabeled pixels.

        This implementation operates per object instance to reduce memory usage.
        For each object label in `label_frame`, it:

          1. Extracts a bounding-box crop containing that object.
          2. Uses the skeleton branch labels inside the crop as seeds.
          3. Runs a distance transform in the crop to find the nearest seed for
             each voxel of the object.
          4. Assigns branch labels to all voxels of the object accordingly.

        Parameters
        ----------
        branch_skel_labels : xp.ndarray or numpy.ndarray
            Branch skeleton labels (non-zero at skeleton voxels).
        label_frame : numpy.ndarray or xp.ndarray
            Instance labels in the image.

        Returns
        -------
        numpy.ndarray
            Relabeled skeleton for the entire frame.
        """
        # Work on CPU for distance transforms; SciPy's EDT is very efficient.
        labels_np = self._to_cpu(label_frame).astype(np.int32, copy=False)
        branch_np = self._to_cpu(branch_skel_labels).astype(np.int32, copy=False)

        relabelled_np = np.zeros_like(labels_np, dtype=np.uint32)

        max_label = int(labels_np.max())
        if max_label == 0:
            return relabelled_np

        # Find object bounding boxes once
        slices = ndi_cpu.find_objects(labels_np)
        if slices is None:
            return relabelled_np

        for lab in range(1, max_label + 1):
            idx = lab - 1
            if idx >= len(slices):
                break
            sl = slices[idx]
            if sl is None:
                continue

            sub_labels = labels_np[sl]
            sub_branch = branch_np[sl]

            obj_mask = (sub_labels == lab)
            if not obj_mask.any():
                continue

            # Seeds are branch labels (>0) inside this object crop
            seed_mask = (sub_branch > 0) & obj_mask
            if not (seed_mask & obj_mask).any():
                # No skeleton seeds for this object; leave unlabeled
                continue

            # For EDT, zeros are considered seeds. We invert the seed mask:
            # seed_mask True -> 0, False -> 1
            edt_input = np.logical_not(seed_mask)

            # Distance transform with indices: returns coordinates of nearest seed voxel.
            # We do not need distances, only indices.
            try:
                indices = ndi_cpu.distance_transform_edt(
                    edt_input,
                    sampling=self.scaling,
                    return_distances=False,
                    return_indices=True,
                )
            except Exception as e:
                logger.warning(
                    f"Distance transform failed for label {lab}. "
                    f"Leaving object unlabeled. Error: {e}"
                )
                continue

            # indices has shape (ndim, ...) and points into sub_branch
            nearest_labels = sub_branch[tuple(indices)]

            # Restrict to the object mask: outside the object remains zero
            nearest_labels[~obj_mask] = 0

            # Merge into global relabelled array.
            relabelled_sub = relabelled_np[sl]
            relabelled_sub[obj_mask] = nearest_labels[obj_mask].astype(np.uint32, copy=False)
            relabelled_np[sl] = relabelled_sub

        return relabelled_np

    # -------------------------------------------------------------------------
    # Multi-scale peak detection with reduced memory footprint
    # -------------------------------------------------------------------------
    def _local_max_peak(self, frame, mask):
        """
        Detects local maxima using multi-scale Laplacian of Gaussian filtering.

        This implementation is memory-friendly: instead of allocating a full
        (num_sigma, *frame.shape) array, it keeps track of the best response
        per voxel across scales.

        Parameters
        ----------
        frame : xp.ndarray or numpy.ndarray
            Input image.
        mask : xp.ndarray or numpy.ndarray
            Binary mask for regions of interest.

        Returns
        -------
        xp.ndarray
            Coordinates of detected local maxima.
        """
        if self.sigmas is None:
            self._set_default_sigmas()

        frame_xp = self._to_xp(frame)
        mask_xp = self._to_xp(mask).astype(bool)

        ndim = frame_xp.ndim
        footprint = self.xp.ones((3,) * ndim)

        best_response = self.xp.zeros_like(frame_xp, dtype=float)
        peak_mask = self.xp.zeros_like(frame_xp, dtype=bool)

        for s in self.sigmas:
            sigma_vec = self._get_sigma_vec(float(s))

            current = -self.ndi.gaussian_laplace(frame_xp, sigma_vec)
            current *= (float(s) ** 2)
            current *= mask_xp
            current = self.xp.where(current < 0, 0, current)

            max_local = self.ndi.maximum_filter(current, footprint=footprint, mode="nearest")
            is_peak = (current == max_local) & (current > best_response) & (current > 0)

            best_response = self.xp.where(is_peak, current, best_response)
            peak_mask = peak_mask | is_peak

        coords_3d = self.xp.argwhere(peak_mask)
        return coords_3d

    # -------------------------------------------------------------------------
    # Skeleton pixel classification
    # -------------------------------------------------------------------------
    def _get_pixel_class(self, skel, force_cpu: bool = False):
        """
        Classifies skeleton pixels into junctions, branches, and endpoints
        based on connectivity.

        Returns
        -------
        xp.ndarray or numpy.ndarray
            Pixel classification:
            0 = background
            1 = isolated pixels
            2 = tips
            3 = edges
            4 = junctions (clipped)
        """
        if force_cpu:
            skel_np = np.asarray(skel)
            if self.low_memory:
                return self._get_pixel_class_chunked(skel_np)
            return self._get_pixel_class_impl(skel_np, np, ndi_cpu)

        skel_xp = self._to_xp(skel)
        if self.low_memory:
            skel_np = self._to_cpu(skel_xp)
            return self._get_pixel_class_chunked(skel_np)

        try:
            return self._get_pixel_class_impl(skel_xp, self.xp, self.ndi)
        except Exception as exc:
            if not self._is_oom_error(exc):
                raise
            self._free_gpu_memory()
            skel_np = self._to_cpu(skel_xp)
            return self._get_pixel_class_chunked(skel_np)

    def _get_pixel_class_impl(self, skel, xp, ndi):
        skel_mask = (skel > 0).astype("uint8")

        if self.im_info.no_z:
            weights = xp.ones((3, 3))
        else:
            weights = xp.ones((3, 3, 3))

        skel_mask_sum = ndi.convolve(skel_mask, weights=weights, mode="constant", cval=0) * skel_mask
        skel_mask_sum[skel_mask_sum > 4] = 4

        return skel_mask_sum

    def _get_pixel_class_chunked(self, skel):
        skel_np = np.asarray(skel)
        shape = skel_np.shape
        halo = (1,) * skel_np.ndim
        chunk_shape = self._compute_chunk_shape(shape, self.max_chunk_voxels)

        if self.im_info.no_z:
            weights = np.ones((3, 3))
        else:
            weights = np.ones((3, 3, 3))

        out = np.zeros_like(skel_np, dtype=np.uint8)
        skel_mask = (skel_np > 0).astype("uint8")

        for core, ext, core_in_ext in self._iter_chunks(shape, chunk_shape, halo):
            chunk = skel_mask[ext]
            chunk_sum = ndi_cpu.convolve(chunk, weights=weights, mode="constant", cval=0)
            core_sum = chunk_sum[core_in_ext] * chunk[core_in_ext]
            core_sum[core_sum > 4] = 4
            out[core] = core_sum.astype(np.uint8, copy=False)

        return out

    # -------------------------------------------------------------------------
    # Time dimension handling
    # -------------------------------------------------------------------------
    def _get_t(self):
        """
        Determines the number of timepoints to process.
        """
        if self.num_t is None:
            if self.im_info.no_t:
                self.num_t = 1
            else:
                self.num_t = self.im_info.shape[self.im_info.axes.index('T')]

    # -------------------------------------------------------------------------
    # Memory allocation for outputs
    # -------------------------------------------------------------------------
    def _allocate_memory(self):
        """
        Allocates memory for skeleton images, pixel classification, and relabeled skeletons.
        """
        logger.debug('Allocating memory for skeletonization.')
        self.label_memmap = self.im_info.get_memmap(self.im_info.pipeline_paths['im_instance_label'])
        self.im_memmap = self.im_info.get_memmap(self.im_info.im_path)
        self.im_frangi_memmap = self.im_info.get_memmap(self.im_info.pipeline_paths['im_preprocessed'])
        self.shape = self.label_memmap.shape

        im_skel_path = self.im_info.pipeline_paths['im_skel']
        self.skel_memmap = self.im_info.allocate_memory(
            im_skel_path,
            dtype='int32',
            description='skeleton image',
            return_memmap=True
        )

        im_pixel_class = self.im_info.pipeline_paths['im_pixel_class']
        self.pixel_class_memmap = self.im_info.allocate_memory(
            im_pixel_class,
            dtype='uint8',
            description='pixel class image',
            return_memmap=True
        )

        im_skel_relabelled = self.im_info.pipeline_paths['im_skel_relabelled']
        self.skel_relabelled_memmap = self.im_info.allocate_memory(
            im_skel_relabelled,
            dtype='uint32',
            description='skeleton relabelled image',
            return_memmap=True
        )

    # -------------------------------------------------------------------------
    # Branch skeleton labels (excluding junctions)
    # -------------------------------------------------------------------------
    def _get_branch_skel_labels(self, pixel_class, force_cpu: bool = False):
        """
        Gets the branch skeleton labels, excluding junctions and background pixels.

        Parameters
        ----------
        pixel_class : xp.ndarray
            Classified skeleton pixels.

        Returns
        -------
        xp.ndarray
            Branch skeleton labels (connected components of non-junction pixels).
        """
        if force_cpu:
            pc_np = np.asarray(pixel_class)
            non_junctions = (pc_np > 0) & (pc_np != 4)
            if self.im_info.no_z:
                structure = np.ones((3, 3))
            else:
                structure = np.ones((3, 3, 3))
            non_junction_labels, _ = ndi_cpu.label(non_junctions, structure=structure)
            return non_junction_labels

        pc_xp = self._to_xp(pixel_class)
        non_junctions = (pc_xp > 0) & (pc_xp != 4)

        if self.im_info.no_z:
            structure = self.xp.ones((3, 3))
        else:
            structure = self.xp.ones((3, 3, 3))

        try:
            non_junction_labels, _ = self.ndi.label(non_junctions, structure=structure)
        except Exception as exc:
            if not self._is_oom_error(exc):
                raise
            self._free_gpu_memory()
            return self._get_branch_skel_labels(self._to_cpu(pc_xp), force_cpu=True)
        return non_junction_labels

    # -------------------------------------------------------------------------
    # Single timepoint processing
    # -------------------------------------------------------------------------
    def _run_frame(self, t):
        """
        Runs skeletonization and network analysis for a single timepoint.

        Parameters
        ----------
        t : int
            Timepoint index.

        Returns
        -------
        tuple
            (branch_skel_labels, pixel_class, branch_labels)
        """
        logger.info(f"Running network analysis, volume {t}/{self.num_t - 1}")

        try:
            return self._run_frame_backend(t)
        except Exception as exc:
            if self.device_type != "cuda" or not self._is_oom_error(exc):
                raise
            logger.warning("GPU OOM in networking; falling back to CPU for this frame.")
            self._free_gpu_memory()
            self._switch_to_cpu()
            return self._run_frame_backend(t)

    def _run_frame_backend(self, t):
        label_frame = self.label_memmap[t]
        label_frame_cpu = np.asarray(label_frame)
        frangi_frame_cpu = np.asarray(self.im_frangi_memmap[t])

        skel_frame = self._skeletonize(label_frame_cpu)
        skel_clean = self._remove_connected_label_pixels(skel_frame, force_cpu=True)
        skel_clean = self._add_missing_skeleton_labels(
            skel_clean, label_frame_cpu, frangi_frame_cpu
        )

        skel_pre_cpu = (skel_clean > 0) * label_frame_cpu

        if self.device_type == "cuda" and not self.low_memory:
            skel_pre = self._to_xp(skel_pre_cpu)
            pixel_class = self._get_pixel_class(skel_pre)
            branch_skel_labels = self._get_branch_skel_labels(pixel_class)
        else:
            pixel_class = self._get_pixel_class(skel_pre_cpu, force_cpu=True)
            branch_skel_labels = self._get_branch_skel_labels(pixel_class, force_cpu=True)

        branch_labels = self._relabel_objects(branch_skel_labels, label_frame_cpu)

        return branch_skel_labels, pixel_class, branch_labels

    # -------------------------------------------------------------------------
    # Optional junction cleanup
    # -------------------------------------------------------------------------
    def _clean_junctions(self, pixel_class):
        """
        Cleans up junctions by removing closely spaced junction pixels.

        This method uses regionprops to group junction pixels and keeps only the
        pixel closest to the junction centroid as a junction, converting the
        others to branch pixels.

        Parameters
        ----------
        pixel_class : numpy.ndarray or xp.ndarray
            Pixel classification of skeleton points.

        Returns
        -------
        numpy.ndarray
            Cleaned pixel classification with redundant junctions removed.
        """
        pc_np = self._to_cpu(pixel_class).copy()

        junctions = pc_np == 4
        if not junctions.any():
            return pc_np

        junction_labels = skimage.measure.label(junctions)
        junction_objects = skimage.measure.regionprops(junction_labels)
        junction_centroids = [obj.centroid for obj in junction_objects]

        for junction_num, junction in enumerate(junction_objects):
            coords = junction.coords
            if len(coords) < 2:
                continue
            # Use KD-tree to find closest pixel to centroid
            junction_tree = cKDTree(coords)
            _, nearest_idx = junction_tree.query(junction_centroids[junction_num], k=1, workers=-1)
            # Convert all other pixels in this junction component to branch class (3)
            coords_list = coords.tolist()
            coords_list.pop(nearest_idx)
            coords_arr = np.array(coords_list).T
            pc_np[tuple(coords_arr)] = 3

        return pc_np

    # -------------------------------------------------------------------------
    # Full networking pipeline
    # -------------------------------------------------------------------------
    def _run_networking(self):
        """
        Runs the network analysis process for all timepoints in the image.
        """
        for t in range(self.num_t):
            if self.viewer is not None:
                self.viewer.status = f'Extracting branches. Frame: {t + 1} of {self.num_t}.'

            skel, pixel_class, skel_relabelled = self._run_frame(t)

            if self.im_info.no_t or self.num_t == 1:
                # Single frame or static image
                if self.device_type == "cuda":
                    self.skel_memmap[:] = self._to_cpu(skel)
                    self.pixel_class_memmap[:] = self._to_cpu(pixel_class)
                    self.skel_relabelled_memmap[:] = self._to_cpu(skel_relabelled)
                else:
                    self.skel_memmap[:] = skel
                    self.pixel_class_memmap[:] = pixel_class
                    self.skel_relabelled_memmap[:] = skel_relabelled
            else:
                # Time series
                if self.device_type == "cuda":
                    self.skel_memmap[t] = self._to_cpu(skel)
                    self.pixel_class_memmap[t] = self._to_cpu(pixel_class)
                    self.skel_relabelled_memmap[t] = self._to_cpu(skel_relabelled)
                else:
                    self.skel_memmap[t] = skel
                    self.pixel_class_memmap[t] = pixel_class
                    self.skel_relabelled_memmap[t] = skel_relabelled

    # -------------------------------------------------------------------------
    # Public entry point
    # -------------------------------------------------------------------------
    def run(self):
        """
        Execute the full network analysis pipeline.
        """
        device = adaptive_run.normalize_device(self.device)
        gpu_ok = adaptive_run.gpu_available()
        if device == "gpu" and not gpu_ok:
            logger.warning("Network: GPU requested but not available; falling back to CPU.")
        if device == "cpu" or not gpu_ok:
            device_order = ["cpu"]
        else:
            device_order = ["gpu", "cpu"]

        start_low_memory = bool(self.low_memory) or adaptive_run.should_use_low_memory(
            self.im_info, include_gpu="gpu" in device_order
        )
        if start_low_memory and not self.low_memory:
            logger.info("Network: enabling low-memory mode based on estimated usage.")

        last_exc = None
        for dev, low in adaptive_run.mode_candidates(device_order, start_low_memory):
            try:
                self._set_backend(dev)
                self._set_low_memory(low)
                self._get_t()
                self._allocate_memory()
                self._run_networking()
                return
            except Exception as exc:
                last_exc = exc
                if adaptive_run.is_gpu_unavailable_error(exc) and dev == "gpu":
                    logger.warning("Network: GPU backend unavailable; retrying on CPU.")
                    continue
                if adaptive_run.is_oom_error(exc):
                    logger.warning(
                        "Network: OOM on %s/%s; retrying with lower settings.",
                        dev,
                        "low-memory" if low else "high-memory",
                    )
                    continue
                raise
        raise last_exc


if __name__ == "__main__":
    im_path = r"D:\\test_files\\nelly_tests\\deskewed-2023-07-13_14-58-28_000_wt_0_acquire.ome.tif"
    im_info = ImInfo(im_path)
    skel = Network(im_info, num_t=3)
    skel.run()
