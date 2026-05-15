"""
Semantic and instance segmentation for microscopy images.

This module provides the Label class for thresholding-based segmentation with
optimizations for large volumes and optional GPU acceleration.
"""
from dataclasses import dataclass

import numpy as np

from nellie.utils import adaptive_run
from nellie.utils.base_logger import logger
from nellie.im_info.verifier import ImInfo
from nellie.utils.gpu_functions import otsu_threshold, triangle_threshold

_UNSET = object()


def _probe_torch_backend():
    """Resolve the torch+MPS backend tuple once at import time.

    Returns ``(torch_xp, torch_ndi, torch.Tensor)`` or ``None`` when
    torch is unavailable. Mirrors the cupy probe used elsewhere in the
    pipeline (see :func:`nellie.segmentation.filtering._probe_torch_backend`)
    so :func:`Label._xp_for_array` can dispatch torch tensors back to
    the ``torch_xp`` / ``torch_ndi`` shim instead of falling through to
    numpy. Without this, helpers like
    :func:`Label._compute_frangi_threshold` would call ``np.log10`` on a
    torch tensor (and worse, hand the tensor to ``otsu_threshold`` /
    ``triangle_threshold`` with the wrong ``xp=`` namespace), losing the
    GPU residency.
    """
    try:
        import torch

        from nellie.utils import torch_ndi as _torch_ndi
        from nellie.utils import torch_xp as _torch_xp

        return (_torch_xp, _torch_ndi, torch.Tensor)
    except Exception:
        return None


_TORCH_BACKEND = _probe_torch_backend()


@dataclass(frozen=True)
class LabelConfig:
    """Algorithm configuration for ``Label``.

    Frozen — represents the user's intent at construction time. Label
    copies these values into mutable instance attributes that the
    OOM/availability cascade can update mid-run (``device``, ``low_memory``).
    Inspect ``Label.config`` to see the original intent regardless of any
    cascade-driven runtime fallbacks.
    """

    threshold: float | None = None
    otsu_thresh_intensity: bool = False
    chunk_z: int | None = None
    flush_interval: int = 1
    min_radius_um: float = 0.25
    threshold_sampling_pixels: int = 1_000_000
    histogram_nbins: int = 256
    device: str = "auto"
    low_memory: bool = False
    max_chunk_voxels: int = int(1e6)

    def __post_init__(self) -> None:
        adaptive_run.normalize_device(self.device)
        for name, value in (
            ("flush_interval", self.flush_interval),
            ("min_radius_um", self.min_radius_um),
            ("threshold_sampling_pixels", self.threshold_sampling_pixels),
            ("histogram_nbins", self.histogram_nbins),
            ("max_chunk_voxels", self.max_chunk_voxels),
        ):
            if value <= 0:
                raise ValueError(f"LabelConfig.{name} must be > 0, got {value}")
        if self.chunk_z is not None and self.chunk_z <= 0:
            raise ValueError(
                f"LabelConfig.chunk_z must be > 0 if set, got {self.chunk_z}"
            )


class Label:
    """
    A class for semantic and instance segmentation of microscopy images using
    thresholding techniques, optimized for large volumes and optional GPU acceleration.
    """

    def __init__(
        self,
        im_info: ImInfo,
        config: LabelConfig = LabelConfig(),
        viewer=None,
        num_t: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        im_info : ImInfo
            Image metadata and paths.
        config : LabelConfig
            Algorithm configuration. Defaults to ``LabelConfig()``.
        viewer : object or None, optional
            Viewer object for displaying status.
        num_t : int, optional
            Number of timepoints to process.
        """
        self.im_info = im_info
        self.config = config

        # Cascade-mutable runtime state. Initial values come from config;
        # ``_set_backend`` and ``_set_low_memory`` may update them on retry.
        self.device = config.device
        self.low_memory = bool(config.low_memory)

        self.xp, self.ndi, self.device_type = adaptive_run.resolve_backend(self.device)
        self.num_t = num_t
        if self.im_info.no_t:
            self.num_t = 1
        elif num_t is None:
            self.num_t = im_info.shape[im_info.axes.index('T')]

        # Aliases for hot path readability — config remains the source of truth.
        self.threshold = config.threshold
        self.otsu_thresh_intensity = config.otsu_thresh_intensity

        self.im_memmap = None
        self.frangi_memmap = None

        self.semantic_mask_memmap = None
        self.instance_label_memmap = None
        self.shape = ()

        self.viewer = viewer

        # Optimization / configuration parameters
        self.chunk_z = config.chunk_z if (not self.im_info.no_z and config.chunk_z is not None) else None
        # Stash the *coerced* value (None for 2D), not the raw user input —
        # ``chunk_z`` has no meaning without a Z axis, so passing
        # ``chunk_z=...`` against a 2D image is intentionally a no-op.
        self._user_chunk_z = self.chunk_z
        self.flush_interval = max(1, int(config.flush_interval))
        min_radius_um = float(config.min_radius_um)
        x_res = self.im_info.dim_res.get("X") or 1.0
        self.min_radius_um = max(min_radius_um, float(x_res))
        self.threshold_sampling_pixels = int(config.threshold_sampling_pixels)
        self.histogram_nbins = int(config.histogram_nbins)
        self.eps = 1e-8
        self.max_chunk_voxels = int(config.max_chunk_voxels)

        if self.low_memory and self.chunk_z is None and not self.im_info.no_z:
            inferred_chunk = self._infer_chunk_z()
            if inferred_chunk is not None:
                self.chunk_z = inferred_chunk

        # Dimensionality and structuring elements (pre-computed)
        self.ndim = 2 if self.im_info.no_z else 3
        self.min_area_pixels = self._compute_min_area_pixels()
        self.footprint = None
        self._set_footprint()

    def _set_backend(self, device):
        device = adaptive_run.normalize_device(device)
        self.device = device
        self.xp, self.ndi, self.device_type = adaptive_run.resolve_backend(device)
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
        # Iterative two-pass path compression. The recursive form
        # (one call per chain link) overflows ``sys.recursion_limit``
        # on pathological cross-chunk merge chains — see
        # ``test_uf_find_handles_pathological_chain``.
        root = x
        while True:
            nxt = parent.get(root, root)
            if nxt == root:
                break
            root = nxt
        # Compress: point every node on the walked path directly at root.
        while x != root:
            nxt = parent[x]
            parent[x] = root
            x = nxt
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

            self.instance_label_memmap[t, z_start:z_end, ...] = labels_chunk
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
        # Check torch first since the per-call cupy import is more
        # expensive than the cached ``isinstance`` test. Both arms use
        # the module-load-time probes (see ``_TORCH_BACKEND``); per-call
        # ``import cupy`` was the prior shape and is preserved as the
        # fallback for legacy CuPy installs that the probe might miss.
        if _TORCH_BACKEND is not None and isinstance(arr, _TORCH_BACKEND[2]):
            return _TORCH_BACKEND[0]
        try:
            import cupy
            if isinstance(arr, cupy.ndarray):
                return cupy
        except Exception:
            pass
        return np

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
        # ``.size`` is an int on numpy/cupy but a method on torch.Tensor;
        # reduce via ``.shape`` for backend-agnostic element-count checks
        # (mirrors the pattern in :mod:`nellie.segmentation.filtering`).
        if int(np.prod(flat.shape)) == 0:
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
        step = max(int(np.prod(flat.shape)) // max_samples, 1)
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

            if int(np.prod(values.shape)) > 0 or step == 1:
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
        # Backend-agnostic element-count check — see ``.size`` note above.
        if int(np.prod(values.shape)) == 0:
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
        # Backend-agnostic element-count check — see ``.size`` note above.
        if int(np.prod(values.shape)) == 0:
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

        # Fill holes for 3D data
        if not self.im_info.no_z:
            mask = self.ndi.binary_fill_holes(mask)

        # Connected component labeling
        labels, _ = self.ndi.label(mask, structure=self.footprint)

        # Remove very small objects using bincount + lookup table.
        # ``.size`` is a method on torch.Tensor (an int on numpy/cupy);
        # reduce via ``.shape`` for backend-agnostic count semantics.
        if int(np.prod(labels.shape)) == 0:
            return mask, labels

        areas = self.xp.bincount(labels.ravel())
        if int(np.prod(areas.shape)) <= 1:
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
            # Load full timepoint volume into xp array. ``original_view``
            # is only consumed for the optional intensity mask, so skip
            # the host→device copy when no intensity threshold was set
            # (the common case unless ``otsu_thresh_intensity`` or an
            # explicit ``threshold`` was given).
            frangi_in_mem = self.xp.asarray(frangi_view)

            # Optional intensity-based masking (read-only)
            if intensity_thresh is not None:
                original_in_mem = self.xp.asarray(original_view)
                mask = original_in_mem > intensity_thresh
                frangi_in_mem = frangi_in_mem * mask

            # Labeling on Frangi image
            _, labels = self._get_labels(frangi_in_mem, frangi_thresh=frangi_thresh)
            return labels
        except Exception as exc:
            # Inner cascade fires for any GPU backend (cupy on CUDA, torch on
            # MPS) — both can OOM on the full-volume Frangi load and benefit
            # from the chunked-Z fallback / CPU switch below.
            if adaptive_run.is_oom_error(exc) and self.device_type in ("cuda", "mps"):
                adaptive_run.free_gpu_memory(self.xp)
                if not self.im_info.no_z:
                    logger.warning(
                        "%s OOM during full-volume labeling; "
                        "falling back to chunked Z processing.",
                        self.device_type.upper(),
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
                logger.warning(
                    "%s OOM during full-volume labeling; switching to CPU.",
                    self.device_type.upper(),
                )
                self._set_backend("cpu")
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
                # Duck-type the GPU round-trip: ``hasattr(arr, "get")``
                # fires on cupy.ndarray natively and on torch.Tensor via
                # the shim's ``_patch_tensor_methods``. Brings the cuda
                # and mps paths into sync — same idiom slice 2 used in
                # ``filtering.py`` ``_run_frame_chunked``.
                if hasattr(labels, "get"):
                    labels = labels.get()
                self.instance_label_memmap[t, ...] = labels
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

            # Extract CPU chunks from memmap. ``original_view`` is only
            # consumed for the optional intensity mask, so skip the
            # host→device copy of the original chunk when no intensity
            # threshold was set (mirrors the guard in
            # ``_run_frame_full_volume``).
            frangi_chunk_cpu = frangi_view[z_start:z_end, ...]

            try:
                frangi_chunk = self.xp.asarray(frangi_chunk_cpu)

                # Optional intensity-based masking per chunk (read-only)
                if intensity_thresh is not None:
                    original_chunk_cpu = original_view[z_start:z_end, ...]
                    original_chunk = self.xp.asarray(original_chunk_cpu)
                    mask = original_chunk > intensity_thresh
                    frangi_chunk = frangi_chunk * mask

                # Labeling on Frangi chunk
                _, labels_chunk = self._get_labels(frangi_chunk, frangi_thresh=frangi_thresh)

                # Offset labels to make them unique within the frame.
                # ``.size`` is a method on torch.Tensor — count via shape
                # for backend-agnostic semantics.
                if int(np.prod(labels_chunk.shape)) > 0:
                    max_label_chunk = int(labels_chunk.max())
                else:
                    max_label_chunk = 0

                if max_label_chunk > 0:
                    labels_chunk = labels_chunk.astype('int32', copy=False)
                    labels_chunk[labels_chunk > 0] += frame_label_offset
                    frame_label_offset += max_label_chunk

                # Move to host on any GPU backend (cupy on CUDA, torch
                # on MPS via shim) so the union-find boundary stitching
                # below operates on numpy. Same duck-type idiom slice 2
                # used in ``filtering.py``.
                if hasattr(labels_chunk, "get"):
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

                self.instance_label_memmap[t, z_start:z_end, ...] = labels_chunk
                relabel_chunk_z = current_chunk if relabel_chunk_z is None else min(
                    relabel_chunk_z, current_chunk
                )

                z_start = z_end  # advance to next chunk
            except Exception as exc:
                if adaptive_run.is_oom_error(exc):
                    adaptive_run.free_gpu_memory(self.xp)
                    if current_chunk > 1:
                        current_chunk = max(current_chunk // 2, 1)
                        logger.warning(
                            f'OOM at Z range [{z_start}, {z_end}); '
                            f'reducing chunk_z to {current_chunk}.'
                        )
                        continue
                    # Both GPU backends (cupy on CUDA, torch on MPS)
                    # benefit from the chunk_z=1 → CPU escape hatch; the
                    # cuda-only narrowing predates the MPS onboard.
                    if self.device_type in ("cuda", "mps"):
                        logger.warning('OOM even with chunk_z=1; switching to CPU.')
                        self._set_backend("cpu")
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

            original_view = self.im_memmap[t, ...]
            frangi_view = self.frangi_memmap[t, ...]
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
                    # Duck-type GPU round-trip — cupy.ndarray.get() and
                    # torch.Tensor.get() (via shim) both return numpy.
                    if hasattr(labels, "get"):
                        labels = labels.get()
                    self.instance_label_memmap[t, ...] = labels

            if (t + 1) % self.flush_interval == 0:
                self.instance_label_memmap.flush()

        self.instance_label_memmap.flush()

    def run(self):
        """
        Main method to execute the full segmentation process over the image data.
        """
        logger.info('Running semantic segmentation.')
        device = adaptive_run.normalize_device(self.device)
        device_order = adaptive_run.device_cascade(self.device)
        if device == "gpu" and device_order == ["cpu"]:
            logger.warning("Label: GPU requested but not available; falling back to CPU.")

        start_low_memory = bool(self.low_memory) or adaptive_run.should_use_low_memory(
            self.im_info,
            include_gpu="gpu" in device_order,
            device_order=device_order,
        )
        if start_low_memory and not self.low_memory:
            logger.info("Label: enabling low-memory mode based on estimated usage.")

        last_exc = None
        for dev, low in adaptive_run.mode_candidates(device_order, start_low_memory):
            try:
                self._set_backend(dev)
                self._set_low_memory(low)
                self._allocate_memory()
                self._run_segmentation()
                return
            except Exception as exc:
                last_exc = exc
                if adaptive_run.is_gpu_unavailable_error(exc) and dev in ("gpu", "mps"):
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
