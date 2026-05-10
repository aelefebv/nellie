"""
Network skeletonization and analysis for microscopy images.

This module provides the Network class for skeletonizing network-like structures
and analyzing their topology with optimized CPU/GPU processing.
"""
import itertools
from dataclasses import dataclass

import numpy as np
import skimage.morphology as morph
from scipy import ndimage as ndi_cpu

from nellie.utils import adaptive_run
from nellie.utils.base_logger import logger
from nellie.im_info.verifier import ImInfo


@dataclass(frozen=True)
class NetworkConfig:
    """Algorithm configuration for ``Network``.

    Frozen — represents the user's intent at construction time. Network
    copies these values into mutable instance attributes that the
    OOM/availability cascade can update mid-run (``device``, ``low_memory``).
    Inspect ``Network.config`` to see the original intent regardless of any
    cascade-driven runtime fallbacks.
    """

    min_radius_um: float = 0.20
    max_radius_um: float = 1
    device: str = "auto"
    low_memory: bool = False
    max_chunk_voxels: int = int(1e6)

    def __post_init__(self) -> None:
        adaptive_run.normalize_device(self.device)
        for name, value in (
            ("min_radius_um", self.min_radius_um),
            ("max_radius_um", self.max_radius_um),
            ("max_chunk_voxels", self.max_chunk_voxels),
        ):
            if value <= 0:
                raise ValueError(f"NetworkConfig.{name} must be > 0, got {value}")
        if self.min_radius_um > self.max_radius_um:
            raise ValueError(
                f"NetworkConfig.min_radius_um ({self.min_radius_um}) must be <= "
                f"max_radius_um ({self.max_radius_um})"
            )


class Network:
    """
    Optimized class for analyzing and skeletonizing network-like structures in 3D or 4D microscopy images.

    This version focuses on:
      - Reduced CPU/GPU thrashing.
      - Vectorized neighborhood operations (no Python per-voxel loops on large arrays).
      - More memory-friendly local-max detection.
      - More efficient branch relabeling using distance transforms on per-object crops.
      - Graceful degradation when GPU memory is insufficient (CPU/chunked fallback).
    """

    def __init__(
        self,
        im_info: ImInfo,
        config: NetworkConfig = NetworkConfig(),
        viewer=None,
        num_t: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        im_info : ImInfo
            Image metadata and paths.
        config : NetworkConfig
            Algorithm configuration. Defaults to ``NetworkConfig()``.
        viewer : object or None, optional
            Viewer object for status reporting.
        num_t : int, optional
            Number of timepoints to process. Defaults to all timepoints.
        """
        self.im_info = im_info
        self.config = config

        # Cascade-mutable runtime state. Initial values come from config;
        # ``_set_backend`` and ``_set_low_memory`` may update them on retry.
        self.device = config.device
        self.low_memory = bool(config.low_memory)

        self.xp, self.ndi, self.device_type = adaptive_run.resolve_backend(self.device)
        self.force_device = config.device.lower() in ("cpu", "gpu", "cuda", "mps")
        self.max_chunk_voxels = int(config.max_chunk_voxels)
        self.num_t = num_t
        if num_t is None and not self.im_info.no_t:
            self.num_t = im_info.shape[im_info.axes.index('T')]

        if not self.im_info.no_z:
            self.z_ratio = self.im_info.dim_res['Z'] / self.im_info.dim_res['X']

        # either (roughly) diffraction limit, or pixel size, whichever is larger
        self.min_radius_um = max(config.min_radius_um, self.im_info.dim_res['X'])
        self.max_radius_um = config.max_radius_um

        self.min_radius_px = self.min_radius_um / self.im_info.dim_res['X']
        self.max_radius_px = self.max_radius_um / self.im_info.dim_res['X']

        if self.im_info.no_z:
            self.scaling = (im_info.dim_res['Y'], im_info.dim_res['X'])
        else:
            self.scaling = (im_info.dim_res['Z'], im_info.dim_res['Y'], im_info.dim_res['X'])

        self.im_memmap = None
        self.im_frangi_memmap = None
        self.label_memmap = None
        self.pixel_class_memmap = None
        self.skel_memmap = None
        self.skel_relabelled_memmap = None

        self.viewer = viewer

    # -------------------------------------------------------------------------
    # Helper methods for device handling
    # -------------------------------------------------------------------------
    def _set_backend(self, device):
        device = adaptive_run.normalize_device(device)
        self.device = device
        self.xp, self.ndi, self.device_type = adaptive_run.resolve_backend(device)
        self.force_device = device in ("cpu", "gpu", "mps")

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
        # xp is numpy on CPU, cupy on CUDA, and the torch_xp shim on MPS.
        try:
            return self.xp.asarray(arr)
        except Exception as e:
            # Both GPU backends (cupy on CUDA, torch on MPS) are explicit
            # device pins by the time control reaches here — silently
            # falling back to numpy would lose GPU residency on a recoverable
            # failure (e.g. a transient dtype mismatch). Re-raise on either
            # GPU backend so the cascade in ``run`` can decide.
            if self.device_type in ("cuda", "mps"):
                raise
            logger.warning(f"xp.asarray failed; falling back to numpy. Error: {e}")
            return np.asarray(arr)

    def _to_cpu(self, arr):
        """
        Convert xp array to a numpy array. If already numpy, return as-is.
        """
        # Duck-type the GPU round-trip: ``hasattr(arr, "get")`` fires on
        # cupy.ndarray natively and on torch.Tensor via the shim's
        # ``_patch_tensor_methods``. Brings the cuda and mps paths into sync —
        # same idiom slice 2 used in ``filtering.py`` and slice 3 used in
        # ``labelling.py``.
        if hasattr(arr, "get"):
            return arr.get()
        return np.asarray(arr)

    # -------------------------------------------------------------------------
    # Neighborhood-based skeleton cleanup
    # -------------------------------------------------------------------------
    def _remove_connected_label_pixels(self, skel_labels):
        """
        Removes skeleton pixels that are connected to multiple labeled regions.

        Always runs on CPU: the sole pipeline caller (``_run_frame_backend``)
        feeds CPU arrays in, and the vectorized 3×3(×3) min/max neighborhood
        filters are inexpensive enough on CPU that the GPU branch was never
        worth exercising. The chunked low-memory variant is retained for
        peak-memory control on very large frames.
        """
        labels_np = np.asarray(skel_labels)
        if self.low_memory:
            return self._remove_connected_label_pixels_chunked(labels_np)
        return self._remove_connected_label_pixels_impl(labels_np, np, ndi_cpu)

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
            if not adaptive_run.is_oom_error(exc):
                raise
            adaptive_run.free_gpu_memory(self.xp)
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
            if not adaptive_run.is_oom_error(exc):
                raise
            adaptive_run.free_gpu_memory(self.xp)
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
            # Both GPU backends benefit from the per-frame CPU fallback on
            # OOM: cupy on CUDA can run out of VRAM, torch on MPS shares
            # the system RAM headroom and can hit the same wall. The
            # cuda-only narrowing predates the MPS onboard.
            if self.device_type not in ("cuda", "mps") or not adaptive_run.is_oom_error(exc):
                raise
            logger.warning(
                "%s OOM in networking; falling back to CPU for this frame.",
                self.device_type.upper(),
            )
            adaptive_run.free_gpu_memory(self.xp)
            self._set_backend("cpu")
            return self._run_frame_backend(t)

    def _run_frame_backend(self, t):
        label_frame = self.label_memmap[t]
        label_frame_cpu = np.asarray(label_frame)
        frangi_frame_cpu = np.asarray(self.im_frangi_memmap[t])

        skel_frame = self._skeletonize(label_frame_cpu)
        skel_clean = self._remove_connected_label_pixels(skel_frame)
        skel_clean = self._add_missing_skeleton_labels(
            skel_clean, label_frame_cpu, frangi_frame_cpu
        )

        skel_pre_cpu = (skel_clean > 0) * label_frame_cpu

        # Both GPU backends route through the convolution-based pixel-class
        # path: cupy on CUDA dispatches to its own ndimage; torch on MPS
        # dispatches to ``torch.nn.functional.conv*`` via the shim. The
        # structural ``ndi.label`` round-trips to scipy on both backends —
        # see PRD #140 § Implementation Decisions for the
        # convolutional-vs-structural split.
        if self.device_type in ("cuda", "mps") and not self.low_memory:
            skel_pre = self._to_xp(skel_pre_cpu)
            pixel_class = self._get_pixel_class(skel_pre)
            branch_skel_labels = self._get_branch_skel_labels(pixel_class)
        else:
            pixel_class = self._get_pixel_class(skel_pre_cpu, force_cpu=True)
            branch_skel_labels = self._get_branch_skel_labels(pixel_class, force_cpu=True)

        branch_labels = self._relabel_objects(branch_skel_labels, label_frame_cpu)

        return branch_skel_labels, pixel_class, branch_labels

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

            # Duck-type the GPU round-trip: ``hasattr(arr, "get")`` fires on
            # cupy.ndarray natively and on torch.Tensor via the shim's
            # ``_patch_tensor_methods``. Avoids the cuda-only narrowing
            # that was hiding ``skel`` / ``pixel_class`` torch tensors
            # behind a numpy fast-path; see ``_to_cpu`` for the same idiom.
            def _host(arr):
                return arr.get() if hasattr(arr, "get") else arr

            if self.im_info.no_t or self.num_t == 1:
                # Single frame or static image
                self.skel_memmap[:] = _host(skel)
                self.pixel_class_memmap[:] = _host(pixel_class)
                self.skel_relabelled_memmap[:] = _host(skel_relabelled)
            else:
                # Time series
                self.skel_memmap[t] = _host(skel)
                self.pixel_class_memmap[t] = _host(pixel_class)
                self.skel_relabelled_memmap[t] = _host(skel_relabelled)

    # -------------------------------------------------------------------------
    # Public entry point
    # -------------------------------------------------------------------------
    def run(self):
        """
        Execute the full network analysis pipeline.
        """
        device = adaptive_run.normalize_device(self.device)
        device_order = adaptive_run.device_cascade(self.device)
        if device == "gpu" and device_order == ["cpu"]:
            logger.warning("Network: GPU requested but not available; falling back to CPU.")

        start_low_memory = bool(self.low_memory) or adaptive_run.should_use_low_memory(
            self.im_info,
            include_gpu="gpu" in device_order,
            device_order=device_order,
        )
        if start_low_memory and not self.low_memory:
            logger.info("Network: enabling low-memory mode based on estimated usage.")

        last_exc = None
        for dev, low in adaptive_run.mode_candidates(device_order, start_low_memory):
            try:
                self._set_backend(dev)
                self._set_low_memory(low)
                self._allocate_memory()
                self._run_networking()
                return
            except Exception as exc:
                last_exc = exc
                if adaptive_run.is_gpu_unavailable_error(exc) and dev in ("gpu", "mps"):
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
