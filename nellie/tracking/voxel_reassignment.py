"""
Voxel reassignment across timepoints using flow interpolation.

This module provides the VoxelReassigner class for tracking and reassigning voxel labels
across time using forward and backward flow interpolation.
"""
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from scipy.spatial import cKDTree

from nellie.utils import adaptive_run
from nellie.utils.base_logger import logger
from nellie.im_info.verifier import ImInfo
from nellie.tracking.flow_interpolation import FlowInterpolator


@dataclass
class _TreeHandle:
    backend: str
    tree: Optional[Any] = None
    coords_real_scaled: Optional[np.ndarray] = None


@dataclass(frozen=True)
class VoxelReassignerConfig:
    """Algorithm configuration for ``VoxelReassigner``.

    Frozen — represents the user's intent at construction time.
    VoxelReassigner copies these values into mutable instance attributes
    that the OOM/availability cascade can update mid-run (``device``,
    ``low_memory``, ``max_query_points``, ``max_bruteforce_pairs``).
    Inspect ``VoxelReassigner.config`` to see the original intent
    regardless of any cascade-driven runtime fallbacks.
    """

    store_running_matches: bool = True
    max_refine_iterations: int = 3
    device: str = "auto"
    low_memory: bool = False
    max_query_points: int = int(1e6)
    max_bruteforce_pairs: int = int(1e7)

    def __post_init__(self) -> None:
        adaptive_run.normalize_device(self.device)
        if self.max_refine_iterations < 0:
            raise ValueError(
                f"VoxelReassignerConfig.max_refine_iterations must be >= 0, "
                f"got {self.max_refine_iterations}"
            )
        for name, value in (
            ("max_query_points", self.max_query_points),
            ("max_bruteforce_pairs", self.max_bruteforce_pairs),
        ):
            if value <= 0:
                raise ValueError(
                    f"VoxelReassignerConfig.{name} must be > 0, got {value}"
                )


class VoxelReassigner:
    """
    A class for voxel reassignment across time points using forward and backward flow interpolation.

    This optimized version:
      - Streams over timepoints instead of holding all voxel coordinates in memory.
      - Reuses a single set of voxel matches between timepoints for all label types.
      - Avoids large intermediate dense arrays where possible.
      - Optionally stores running matches for downstream analysis.
      - Supports CPU/GPU matching with memory-aware chunking and fallbacks.
      - Assigns labels using weighted votes from forward/backward interpolations.
    """

    def __init__(
        self,
        im_info: ImInfo,
        config: VoxelReassignerConfig = VoxelReassignerConfig(),
        viewer=None,
        num_t: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        im_info : ImInfo
            Image metadata and memory-mapped data.
        config : VoxelReassignerConfig
            Algorithm configuration. Defaults to ``VoxelReassignerConfig()``.
        viewer : Any, optional
            Optional viewer for visualization / status updates.
        num_t : int, optional
            Number of timepoints in the dataset. If None, it is inferred from the image metadata.
        """
        self.im_info = im_info
        self.config = config

        # Cascade-mutable runtime state. Initial values come from config;
        # ``_set_backend`` and ``_set_low_memory`` may update them on retry.
        self.device = adaptive_run.normalize_device(config.device)
        # ``_base_max_*`` shadow attributes preserve the user's intent across
        # ``_set_low_memory`` recomputes (which clamp ``self.max_*`` based on
        # ``self.low_memory``).
        self._base_max_query_points = max(1, int(config.max_query_points))
        self._base_max_bruteforce_pairs = max(1, int(config.max_bruteforce_pairs))
        self.max_query_points = self._base_max_query_points
        self.max_bruteforce_pairs = self._base_max_bruteforce_pairs
        self.low_memory = bool(config.low_memory)
        if self.low_memory:
            self.max_query_points = min(self.max_query_points, int(2e5))
            self.max_bruteforce_pairs = min(self.max_bruteforce_pairs, int(2e6))
        self.xp, _ndi, self.device_type = adaptive_run.resolve_backend(self.device)
        self._gpu_kdtree_cls = self._get_gpu_kdtree_cls() if self.device_type == "cuda" else None
        self._warned_gpu_fallback = False

        # Aliases for hot path readability — config remains the source of truth.
        self.store_running_matches = config.store_running_matches
        self.max_refine_iterations = config.max_refine_iterations

        # handle single-timepoint data early
        if self.im_info.no_t:
            self.num_t = 1
            self.flow_interpolator_fw = None
            self.flow_interpolator_bw = None
            self.running_matches = []
            self.voxel_matches_path = None
            self.branch_label_memmap = None
            self.obj_label_memmap = None
            self.reassigned_branch_memmap = None
            self.reassigned_obj_memmap = None
            self.viewer = viewer
            return

        self.num_t = num_t
        if num_t is None and not self.im_info.no_t:
            self.num_t = im_info.shape[im_info.axes.index('T')]

        # forward and backward flow interpolators
        self.flow_interpolator_fw = FlowInterpolator(im_info)
        self.flow_interpolator_bw = FlowInterpolator(im_info, forward=False)

        # where matches between frames are optionally stored
        self.running_matches = []

        self.voxel_matches_path = None
        self.branch_label_memmap = None
        self.obj_label_memmap = None
        self.reassigned_branch_memmap = None
        self.reassigned_obj_memmap = None

        self.viewer = viewer

    # -------------------------------------------------------------------------
    # Backend helpers
    # -------------------------------------------------------------------------

    def _get_gpu_kdtree_cls(self):
        """Return ``cupyx.scipy.spatial.cKDTree`` or None if unavailable.

        Irreducible local helper: ``adaptive_run.try_import_cupy`` returns
        ``(cupy, cupyx.scipy.ndimage)`` — only this stage needs the spatial
        KDTree class, so the import lives here.
        """
        try:
            import cupyx.scipy.spatial as cupy_spatial
            return cupy_spatial.cKDTree
        except (ImportError, AttributeError):
            return None

    def _warn_gpu_fallback(self, reason):
        """Log a single GPU-fallback warning per VoxelReassigner instance.

        Replaces the warning-rate-limit role formerly held by ``_switch_to_cpu``.
        Backend-mutation duties are gone (Option A2 inner cascade — local
        rebuild only); this just preserves the once-per-instance log.
        """
        if not self._warned_gpu_fallback:
            logger.warning(reason)
            self._warned_gpu_fallback = True

    def _set_backend(self, device):
        device = adaptive_run.normalize_device(device)
        self.device = device
        self.xp, _ndi, self.device_type = adaptive_run.resolve_backend(device)
        self._gpu_kdtree_cls = self._get_gpu_kdtree_cls() if self.device_type == "cuda" else None
        self._warned_gpu_fallback = False

    def _set_low_memory(self, low_memory):
        self.low_memory = bool(low_memory)
        self.max_query_points = self._base_max_query_points
        self.max_bruteforce_pairs = self._base_max_bruteforce_pairs
        if self.low_memory:
            self.max_query_points = min(self.max_query_points, int(2e5))
            self.max_bruteforce_pairs = min(self.max_bruteforce_pairs, int(2e6))

    def _scale_coords(self, coords):
        scaling = self.flow_interpolator_fw.scaling
        return np.asarray(coords, dtype=np.float32) * scaling

    def _iter_slices(self, n_items, chunk_size):
        if n_items <= 0:
            return
        chunk_size = max(1, int(chunk_size))
        for start in range(0, n_items, chunk_size):
            yield slice(start, min(n_items, start + chunk_size))

    def _build_tree(self, coords_real_scaled):
        if coords_real_scaled.size == 0:
            return _TreeHandle(backend="cpu", tree=None, coords_real_scaled=None)

        # Local CPU rebuild only on GPU OOM — outer mode_candidates cascade
        # is the single source of truth for backend switching. Subsequent
        # _build_tree calls in later frames will retry GPU.
        gpu_kdtree_failed = False
        if self.device_type == "cuda" and self._gpu_kdtree_cls is not None:
            try:
                coords_gpu = self.xp.asarray(coords_real_scaled, dtype=self.xp.float32)
                tree = self._gpu_kdtree_cls(coords_gpu)
                return _TreeHandle(backend="gpu", tree=tree, coords_real_scaled=coords_real_scaled)
            except Exception as exc:
                if not adaptive_run.is_oom_error(exc):
                    raise
                adaptive_run.free_gpu_memory(self.xp)
                self._warn_gpu_fallback("GPU KDTree OOM; falling back to CPU within this call.")
                gpu_kdtree_failed = True
        if (
            self.device_type == "cuda"
            and self._gpu_kdtree_cls is None
            and not gpu_kdtree_failed
        ):
            # GPU is available but cupyx.scipy.spatial.cKDTree isn't installed;
            # use the GPU brute-force fallback strategy.
            return _TreeHandle(
                backend="gpu_bruteforce",
                tree=None,
                coords_real_scaled=coords_real_scaled,
            )

        # CPU path: default, or local fallback after GPU KDTree OOM in this call.
        try:
            return _TreeHandle(backend="cpu", tree=cKDTree(coords_real_scaled), coords_real_scaled=None)
        except MemoryError:
            logger.warning("KDTree allocation failed; falling back to brute-force matching.")
            return _TreeHandle(
                backend="cpu_bruteforce",
                tree=None,
                coords_real_scaled=coords_real_scaled,
            )

    def _query_tree(self, tree_handle, coords_query_scaled):
        if coords_query_scaled.size == 0:
            return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)

        if tree_handle.backend == "cpu":
            if tree_handle.tree is None:
                return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)
            dist, idx = tree_handle.tree.query(coords_query_scaled, k=1, workers=-1)
            return dist, idx

        if tree_handle.backend == "gpu":
            try:
                coords_query_gpu = self.xp.asarray(coords_query_scaled, dtype=self.xp.float32)
                dist_gpu, idx_gpu = tree_handle.tree.query(coords_query_gpu, k=1)
                dist = self.xp.asnumpy(dist_gpu).astype(np.float32, copy=False)
                idx = self.xp.asnumpy(idx_gpu).astype(np.int64, copy=False)
                return dist, idx
            except Exception as exc:
                # Local CPU rebuild only — outer mode_candidates cascade is the
                # single source of truth for backend switching. Non-OOM errors
                # propagate (no silent backend flip on dtype mismatches, etc.).
                if not adaptive_run.is_oom_error(exc):
                    raise
                adaptive_run.free_gpu_memory(self.xp)
                self._warn_gpu_fallback("GPU query failed; falling back to CPU within this call.")
                cpu_tree = cKDTree(tree_handle.coords_real_scaled)
                dist, idx = cpu_tree.query(coords_query_scaled, k=1, workers=-1)
                return dist, idx

        if tree_handle.backend == "gpu_bruteforce":
            if self.device_type != "cuda":
                cpu_tree = cKDTree(tree_handle.coords_real_scaled)
                dist, idx = cpu_tree.query(coords_query_scaled, k=1, workers=-1)
                return dist, idx
            if not self._can_use_bruteforce(tree_handle.coords_real_scaled, coords_query_scaled):
                cpu_tree = cKDTree(tree_handle.coords_real_scaled)
                dist, idx = cpu_tree.query(coords_query_scaled, k=1, workers=-1)
                return dist, idx
            try:
                return self._query_bruteforce_gpu(tree_handle.coords_real_scaled, coords_query_scaled)
            except Exception as exc:
                # Local CPU rebuild only — outer mode_candidates cascade is the
                # single source of truth for backend switching. Non-OOM errors
                # propagate.
                if not adaptive_run.is_oom_error(exc):
                    raise
                adaptive_run.free_gpu_memory(self.xp)
                self._warn_gpu_fallback("GPU brute-force OOM; falling back to CPU within this call.")
                cpu_tree = cKDTree(tree_handle.coords_real_scaled)
                dist, idx = cpu_tree.query(coords_query_scaled, k=1, workers=-1)
                return dist, idx

        if tree_handle.backend == "cpu_bruteforce":
            return self._query_bruteforce_cpu(tree_handle.coords_real_scaled, coords_query_scaled)

        raise RuntimeError(f"Unknown tree backend '{tree_handle.backend}'.")

    def _can_use_bruteforce(self, coords_real_scaled, coords_query_scaled):
        n_real = coords_real_scaled.shape[0]
        n_query = coords_query_scaled.shape[0]
        if n_real == 0 or n_query == 0:
            return False
        return (n_real * n_query) <= self.max_bruteforce_pairs

    def _query_bruteforce_gpu(self, coords_real_scaled, coords_query_scaled):
        n_real = coords_real_scaled.shape[0]
        n_query = coords_query_scaled.shape[0]
        if n_real == 0 or n_query == 0:
            return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)

        cp = self.xp
        coords_real_gpu = cp.asarray(coords_real_scaled, dtype=cp.float32)
        chunk_size = max(1, min(n_query, self.max_bruteforce_pairs // max(n_real, 1)))

        dist_out = np.empty((n_query,), dtype=np.float32)
        idx_out = np.empty((n_query,), dtype=np.int64)

        start = 0
        while start < n_query:
            end = min(n_query, start + chunk_size)
            try:
                coords_chunk_gpu = cp.asarray(coords_query_scaled[start:end], dtype=cp.float32)
                diff = coords_chunk_gpu[:, None, :] - coords_real_gpu[None, :, :]
                dist_sq = cp.sum(diff * diff, axis=2)
                idx_gpu = cp.argmin(dist_sq, axis=1)
                dist_gpu = cp.sqrt(dist_sq[cp.arange(dist_sq.shape[0]), idx_gpu])
                idx_out[start:end] = cp.asnumpy(idx_gpu).astype(np.int64, copy=False)
                dist_out[start:end] = cp.asnumpy(dist_gpu).astype(np.float32, copy=False)
                del coords_chunk_gpu, diff, dist_sq, idx_gpu, dist_gpu
                if self.low_memory:
                    adaptive_run.free_gpu_memory(self.xp)
                start = end
            except Exception as exc:
                if not adaptive_run.is_oom_error(exc):
                    raise
                adaptive_run.free_gpu_memory(self.xp)
                if chunk_size <= 1:
                    raise
                chunk_size = max(1, chunk_size // 2)

        return dist_out, idx_out

    def _query_bruteforce_cpu(self, coords_real_scaled, coords_query_scaled):
        n_real = coords_real_scaled.shape[0]
        n_query = coords_query_scaled.shape[0]
        if n_real == 0 or n_query == 0:
            return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)

        coords_real_scaled = coords_real_scaled.astype(np.float32, copy=False)
        chunk_size = max(1, min(n_query, self.max_bruteforce_pairs // max(n_real, 1)))

        dist_out = np.empty((n_query,), dtype=np.float32)
        idx_out = np.empty((n_query,), dtype=np.int64)

        start = 0
        while start < n_query:
            end = min(n_query, start + chunk_size)
            try:
                coords_chunk = coords_query_scaled[start:end].astype(np.float32, copy=False)
                diff = coords_chunk[:, None, :] - coords_real_scaled[None, :, :]
                dist_sq = np.sum(diff * diff, axis=2)
                idx = np.argmin(dist_sq, axis=1)
                dist = np.sqrt(dist_sq[np.arange(dist_sq.shape[0]), idx])
                idx_out[start:end] = idx.astype(np.int64, copy=False)
                dist_out[start:end] = dist.astype(np.float32, copy=False)
                start = end
            except MemoryError:
                if chunk_size <= 1:
                    raise
                chunk_size = max(1, chunk_size // 2)

        return dist_out, idx_out

    def _select_match_coord_dtype(self):
        if self.spatial_shape is None or len(self.spatial_shape) == 0:
            return np.uint16
        max_dim = int(max(self.spatial_shape))
        if max_dim <= (np.iinfo(np.uint16).max + 1):
            return np.uint16
        if max_dim <= (np.iinfo(np.uint32).max + 1):
            return np.uint32
        return np.uint64

    def _compute_error_distance(self, predicted_coords, matched_coords):
        if predicted_coords.size == 0 or matched_coords.size == 0:
            return np.empty((0,), dtype=np.float32)
        scaling = self.flow_interpolator_fw.scaling
        diffs = (predicted_coords - matched_coords).astype(np.float32) * scaling
        return np.linalg.norm(diffs, axis=1).astype(np.float32, copy=False)

    def _select_best_pairs(self, vox_prev, vox_next, distances):
        if vox_prev.size == 0 or vox_next.size == 0:
            dim = vox_prev.shape[1] if vox_prev.ndim == 2 else 3
            return (np.empty((0, dim), dtype=np.int64),
                    np.empty((0, dim), dtype=np.int64))
        if self.spatial_shape is None:
            raise RuntimeError("spatial_shape is not set; call _allocate_memory() before matching.")

        target_flat = np.ravel_multi_index(vox_next.T, self.spatial_shape)
        order = np.lexsort((distances, target_flat))
        target_sorted = target_flat[order]
        target_change = np.ones(len(order), dtype=bool)
        target_change[1:] = target_sorted[1:] != target_sorted[:-1]
        best_idx = order[target_change]
        return vox_prev[best_idx], vox_next[best_idx]

    def _vote_targets(self, target_coords, source_labels, distances):
        if target_coords.size == 0:
            return (np.empty((0,), dtype=np.int64),
                    np.empty((0,), dtype=source_labels.dtype),
                    np.empty((0,), dtype=np.int64))
        if self.spatial_shape is None:
            raise RuntimeError("spatial_shape is not set; call _allocate_memory() before matching.")

        target_flat = np.ravel_multi_index(target_coords.T, self.spatial_shape)
        weights = 1.0 / (distances + 1e-6)
        candidate_idx = np.arange(len(weights), dtype=np.int64)

        order = np.lexsort((-weights, source_labels, target_flat))
        target_sorted = target_flat[order]
        labels_sorted = source_labels[order]
        weights_sorted = weights[order]
        cand_idx_sorted = candidate_idx[order]

        pair_change = np.ones(len(order), dtype=bool)
        pair_change[1:] = (target_sorted[1:] != target_sorted[:-1]) | (labels_sorted[1:] != labels_sorted[:-1])
        pair_starts = np.nonzero(pair_change)[0]

        pair_targets = target_sorted[pair_change]
        pair_labels = labels_sorted[pair_change]
        pair_best_idx = cand_idx_sorted[pair_change]
        weight_sums = np.add.reduceat(weights_sorted, pair_starts)

        order2 = np.lexsort((-weight_sums, pair_targets))
        pair_targets_sorted = pair_targets[order2]
        pair_labels_sorted = pair_labels[order2]
        pair_best_idx_sorted = pair_best_idx[order2]

        target_change = np.ones(len(order2), dtype=bool)
        target_change[1:] = pair_targets_sorted[1:] != pair_targets_sorted[:-1]

        best_targets = pair_targets_sorted[target_change]
        best_labels = pair_labels_sorted[target_change]
        best_candidate_idx = pair_best_idx_sorted[target_change]
        return best_targets, best_labels, best_candidate_idx

    # -------------------------------------------------------------------------
    # Matching primitives
    # -------------------------------------------------------------------------

    def _match_forward(self, flow_interpolator, vox_prev, vox_next, t, tree_next=None):
        """
        Matches voxels forward in time using flow interpolation.

        Parameters
        ----------
        flow_interpolator : FlowInterpolator
            Flow interpolator for forward voxel matching.
        vox_prev : np.ndarray
            Voxels from the previous timepoint (N0, D).
        vox_next : np.ndarray
            Voxels from the next timepoint (N1, D).
        t : int
            Current timepoint index.
        tree_next : _TreeHandle, optional
            Prebuilt KDTree for vox_next to reuse between calls.

        Returns
        -------
        tuple of np.ndarray
            (vox_prev_matched_valid, vox_next_matched_valid, distances_valid)
            where distances are interpolation errors in physical units.
        """
        if vox_prev.size == 0 or vox_next.size == 0:
            dim = vox_prev.shape[1] if vox_prev.ndim == 2 else 3
            return (np.empty((0, dim), dtype=np.int64),
                    np.empty((0, dim), dtype=np.int64),
                    np.empty((0,), dtype=np.float64))

        vectors_interpx_prev = flow_interpolator.interpolate_coord(vox_prev, t)
        if vectors_interpx_prev is None:
            dim = vox_prev.shape[1]
            return (np.empty((0, dim), dtype=np.int64),
                    np.empty((0, dim), dtype=np.int64),
                    np.empty((0,), dtype=np.float64))

        # only keep voxels that are not nan
        kept_prev_vox_idxs = ~np.isnan(vectors_interpx_prev).any(axis=1)
        if not np.any(kept_prev_vox_idxs):
            dim = vox_prev.shape[1]
            return (np.empty((0, dim), dtype=np.int64),
                    np.empty((0, dim), dtype=np.int64),
                    np.empty((0,), dtype=np.float64))

        vectors_interpx_prev = vectors_interpx_prev[kept_prev_vox_idxs]
        vox_prev_kept = vox_prev[kept_prev_vox_idxs]

        # estimated centroids in t+1 from voxels in t and interpolated flow
        centroids_next_interpx = vox_prev_kept + vectors_interpx_prev
        if len(centroids_next_interpx) == 0:
            dim = vox_prev.shape[1]
            return (np.empty((0, dim), dtype=np.int64),
                    np.empty((0, dim), dtype=np.int64),
                    np.empty((0,), dtype=np.float64))

        # match t+1 voxels to estimated centroids
        matched_idx = self._match_voxels_to_centroids(
            vox_next,
            centroids_next_interpx,
            tree_handle=tree_next,
        )
        vox_matched_to_centroids = vox_next[matched_idx]

        distances = self._compute_error_distance(centroids_next_interpx, vox_matched_to_centroids)
        distance_mask = distances < self.flow_interpolator_fw.max_distance_um
        if not np.any(distance_mask):
            dim = vox_prev.shape[1]
            return (np.empty((0, dim), dtype=np.int64),
                    np.empty((0, dim), dtype=np.int64),
                    np.empty((0,), dtype=np.float64))

        return (vox_prev_kept[distance_mask].astype(np.int64),
                vox_matched_to_centroids[distance_mask].astype(np.int64),
                distances[distance_mask].astype(np.float64, copy=False))

    def _match_backward(self, flow_interpolator, vox_next, vox_prev, t, tree_prev=None):
        """
        Matches voxels backward in time using flow interpolation.

        Parameters
        ----------
        flow_interpolator : FlowInterpolator
            Flow interpolator for backward voxel matching.
        vox_next : np.ndarray
            Voxels from the next timepoint (N1, D).
        vox_prev : np.ndarray
            Voxels from the previous timepoint (N0, D).
        t : int
            Time index for the flow interpolation (t+1 for matching from t to t+1).
        tree_prev : _TreeHandle, optional
            Prebuilt KDTree for vox_prev to reuse between calls.

        Returns
        -------
        tuple of np.ndarray
            (vox_prev_matched_valid, vox_next_matched_valid, distances_valid)
            where distances are interpolation errors in physical units.
        """
        if vox_prev.size == 0 or vox_next.size == 0:
            dim = vox_prev.shape[1] if vox_prev.ndim == 2 else 3
            return (np.empty((0, dim), dtype=np.int64),
                    np.empty((0, dim), dtype=np.int64),
                    np.empty((0,), dtype=np.float64))

        vectors_interpx_prev = flow_interpolator.interpolate_coord(vox_next, t)
        if vectors_interpx_prev is None:
            dim = vox_prev.shape[1]
            return (np.empty((0, dim), dtype=np.int64),
                    np.empty((0, dim), dtype=np.int64),
                    np.empty((0,), dtype=np.float64))

        kept_next_vox_idxs = ~np.isnan(vectors_interpx_prev).any(axis=1)
        if not np.any(kept_next_vox_idxs):
            dim = vox_prev.shape[1]
            return (np.empty((0, dim), dtype=np.int64),
                    np.empty((0, dim), dtype=np.int64),
                    np.empty((0,), dtype=np.float64))

        vectors_interpx_prev = vectors_interpx_prev[kept_next_vox_idxs]
        vox_next_kept = vox_next[kept_next_vox_idxs]

        # estimated centroids in t from voxels in t+1 minus interpolated flow
        centroids_prev_interpx = vox_next_kept - vectors_interpx_prev
        if len(centroids_prev_interpx) == 0:
            dim = vox_prev.shape[1]
            return (np.empty((0, dim), dtype=np.int64),
                    np.empty((0, dim), dtype=np.int64),
                    np.empty((0,), dtype=np.float64))

        # match t voxels to estimated centroids
        matched_idx = self._match_voxels_to_centroids(
            vox_prev,
            centroids_prev_interpx,
            tree_handle=tree_prev,
        )
        vox_matched_to_centroids = vox_prev[matched_idx]

        distances = self._compute_error_distance(centroids_prev_interpx, vox_matched_to_centroids)
        distance_mask = distances < self.flow_interpolator_fw.max_distance_um
        if not np.any(distance_mask):
            dim = vox_prev.shape[1]
            return (np.empty((0, dim), dtype=np.int64),
                    np.empty((0, dim), dtype=np.int64),
                    np.empty((0,), dtype=np.float64))

        return (vox_matched_to_centroids[distance_mask].astype(np.int64),
                vox_next_kept[distance_mask].astype(np.int64),
                distances[distance_mask].astype(np.float64, copy=False))

    def _match_voxels_to_centroids(self, coords_real, coords_interpx, tree_handle=None):
        """
        Matches real voxel coordinates to interpolated centroids using nearest neighbor search.

        Parameters
        ----------
        coords_real : np.ndarray
            Real voxel coordinates (N_real, D).
        coords_interpx : np.ndarray
            Interpolated centroid coordinates (N_interp, D).
        tree_handle : _TreeHandle, optional
            Prebuilt tree handle for coords_real to reuse between calls.

        Returns
        -------
        np.ndarray
            Indices of nearest neighbors in coords_real for each interpolated point.
        """
        if coords_interpx.size == 0:
            return np.empty((0,), dtype=np.int64)

        if tree_handle is None:
            coords_real_scaled = self._scale_coords(coords_real)
            tree_handle = self._build_tree(coords_real_scaled)
        elif tree_handle.coords_real_scaled is None and tree_handle.backend != "cpu":
            tree_handle.coords_real_scaled = self._scale_coords(coords_real)

        idx_out = np.empty((coords_interpx.shape[0],), dtype=np.int64)
        if self.low_memory:
            for sl in self._iter_slices(coords_interpx.shape[0], self.max_query_points):
                coords_chunk_scaled = self._scale_coords(coords_interpx[sl])
                _, idx_chunk = self._query_tree(tree_handle, coords_chunk_scaled)
                idx_out[sl] = idx_chunk
        else:
            coords_interpx_scaled = self._scale_coords(coords_interpx)
            for sl in self._iter_slices(coords_interpx_scaled.shape[0], self.max_query_points):
                _, idx_chunk = self._query_tree(tree_handle, coords_interpx_scaled[sl])
                idx_out[sl] = idx_chunk
        return idx_out

    def _assign_unique_matches(self, vox_prev_matches, vox_next_matches, distances):
        """
        Assigns unique voxel matches based on minimum distance, enforcing a 1-to-1 mapping.

        Parameters
        ----------
        vox_prev_matches : np.ndarray
            Array of matched voxels from the previous timepoint (N, D).
        vox_next_matches : np.ndarray
            Array of matched voxels from the next timepoint (N, D).
        distances : np.ndarray
            Distances for each match (N,).

        Returns
        -------
        tuple
            (vox_prev_unique, vox_next_unique) with 1-to-1 matches.
        """
        if len(distances) == 0:
            dim = vox_prev_matches.shape[1] if vox_prev_matches.ndim == 2 else 3
            return (np.empty((0, dim), dtype=np.int64),
                    np.empty((0, dim), dtype=np.int64))

        # flatten voxel coordinates to scalar ids for efficient uniqueness checks
        if self.spatial_shape is None:
            raise RuntimeError("spatial_shape is not set; call _allocate_memory() before matching.")

        prev_flat = np.ravel_multi_index(vox_prev_matches.T, self.spatial_shape)
        next_flat = np.ravel_multi_index(vox_next_matches.T, self.spatial_shape)

        order = np.argsort(distances)
        prev_flat_sorted = prev_flat[order]
        next_flat_sorted = next_flat[order]

        _, prev_inv = np.unique(prev_flat_sorted, return_inverse=True)
        _, next_inv = np.unique(next_flat_sorted, return_inverse=True)

        used_prev = np.zeros(prev_inv.max() + 1, dtype=bool)
        used_next = np.zeros(next_inv.max() + 1, dtype=bool)
        keep_indices = []

        for idx_sorted in range(len(order)):
            p_idx = prev_inv[idx_sorted]
            n_idx = next_inv[idx_sorted]
            if used_prev[p_idx] or used_next[n_idx]:
                continue
            used_prev[p_idx] = True
            used_next[n_idx] = True
            keep_indices.append(order[idx_sorted])

        if not keep_indices:
            dim = vox_prev_matches.shape[1]
            return (np.empty((0, dim), dtype=np.int64),
                    np.empty((0, dim), dtype=np.int64))

        keep_indices = np.asarray(keep_indices, dtype=np.int64)
        return vox_prev_matches[keep_indices], vox_next_matches[keep_indices]

    def _distance_threshold(self, vox_prev_matched, vox_next_matched):
        """
        Filters voxel matches by applying a distance threshold in physical units.

        Parameters
        ----------
        vox_prev_matched : np.ndarray
            Array of matched voxels from the previous timepoint (N, D).
        vox_next_matched : np.ndarray
            Array of matched voxels from the next timepoint (N, D).

        Returns
        -------
        tuple
            (vox_prev_valid, vox_next_valid, distances_valid)
        """
        if vox_prev_matched.size == 0 or vox_next_matched.size == 0:
            dim = vox_prev_matched.shape[1] if vox_prev_matched.ndim == 2 else 3
            return (np.empty((0, dim), dtype=np.int64),
                    np.empty((0, dim), dtype=np.int64),
                    np.empty((0,), dtype=np.float64))

        scaling = self.flow_interpolator_fw.scaling
        diffs = (vox_prev_matched - vox_next_matched).astype(np.float32) * scaling
        distances = np.linalg.norm(diffs, axis=1)
        distance_mask = distances < self.flow_interpolator_fw.max_distance_um

        if not np.any(distance_mask):
            dim = vox_prev_matched.shape[1]
            return (np.empty((0, dim), dtype=np.int64),
                    np.empty((0, dim), dtype=np.int64),
                    np.empty((0,), dtype=np.float64))

        vox_prev_matched_valid = vox_prev_matched[distance_mask]
        vox_next_matched_valid = vox_next_matched[distance_mask]
        distances_valid = distances[distance_mask]
        return vox_prev_matched_valid, vox_next_matched_valid, distances_valid

    def match_voxels(self, vox_prev, vox_next, t):
        """
        Builds candidate voxel matches between two consecutive timepoints using
        both forward and backward interpolation.

        Parameters
        ----------
        vox_prev : np.ndarray
            Voxels from the previous timepoint (N0, D).
        vox_next : np.ndarray
            Voxels from the next timepoint (N1, D).
        t : int
            Current timepoint index.

        Returns
        -------
        tuple
            (candidate_prev, candidate_next, distances) where candidates include
            all valid forward/backward interpolated matches.
        """
        if vox_prev.size == 0 or vox_next.size == 0:
            dim = vox_prev.shape[1] if vox_prev.ndim == 2 else 3
            return (np.empty((0, dim), dtype=np.int64),
                    np.empty((0, dim), dtype=np.int64))

        if self.low_memory:
            tree_next = self._build_tree(self._scale_coords(vox_next))

            logger.debug(f'Forward voxel matching for t: {t}')
            vox_prev_fw, vox_next_fw, dist_fw = self._match_forward(
                self.flow_interpolator_fw, vox_prev, vox_next, t, tree_next=tree_next
            )

            tree_next = None
            if self.device_type == "cuda":
                adaptive_run.free_gpu_memory(self.xp)

            tree_prev = self._build_tree(self._scale_coords(vox_prev))

            logger.debug(f'Backward voxel matching for t: {t}')
            vox_prev_bw, vox_next_bw, dist_bw = self._match_backward(
                self.flow_interpolator_bw, vox_next, vox_prev, t + 1, tree_prev=tree_prev
            )

            tree_prev = None
            if self.device_type == "cuda":
                adaptive_run.free_gpu_memory(self.xp)
        else:
            tree_prev = self._build_tree(self._scale_coords(vox_prev))
            tree_next = self._build_tree(self._scale_coords(vox_next))

            logger.debug(f'Forward voxel matching for t: {t}')
            vox_prev_fw, vox_next_fw, dist_fw = self._match_forward(
                self.flow_interpolator_fw, vox_prev, vox_next, t, tree_next=tree_next
            )

            logger.debug(f'Backward voxel matching for t: {t}')
            vox_prev_bw, vox_next_bw, dist_bw = self._match_backward(
                self.flow_interpolator_bw, vox_next, vox_prev, t + 1, tree_prev=tree_prev
            )

        # combine forward and backward matches
        parts_prev = []
        parts_next = []
        parts_dist = []
        if len(vox_prev_fw):
            parts_prev.append(vox_prev_fw)
            parts_next.append(vox_next_fw)
            parts_dist.append(dist_fw)
        if len(vox_prev_bw):
            parts_prev.append(vox_prev_bw)
            parts_next.append(vox_next_bw)
            parts_dist.append(dist_bw)

        if not parts_prev:
            dim = vox_prev.shape[1]
            return (np.empty((0, dim), dtype=np.int64),
                    np.empty((0, dim), dtype=np.int64),
                    np.empty((0,), dtype=np.float64))

        vox_prev_matches = np.concatenate(parts_prev, axis=0)
        vox_next_matches = np.concatenate(parts_next, axis=0)
        distances = np.concatenate(parts_dist, axis=0)
        return (vox_prev_matches.astype(np.int64),
                vox_next_matches.astype(np.int64),
                distances.astype(np.float64, copy=False))

    # -------------------------------------------------------------------------
    # Dataset / memory helpers
    # -------------------------------------------------------------------------

    def _allocate_memory(self):
        """
        Allocates memory for voxel reassignment, including initializing memory-mapped arrays for branch and object labels.
        """
        logger.debug('Allocating memory for voxel reassignment.')
        self.voxel_matches_path = self.im_info.pipeline_paths['voxel_matches']

        self.branch_label_memmap = self.im_info.get_memmap(self.im_info.pipeline_paths['im_skel_relabelled'])
        self.obj_label_memmap = self.im_info.get_memmap(self.im_info.pipeline_paths['im_instance_label'])
        self.shape = self.branch_label_memmap.shape
        # spatial dimensions (everything except T)
        self.spatial_shape = self.shape[1:]
        self.match_coord_dtype = self._select_match_coord_dtype()

        reassigned_branch_label_path = self.im_info.pipeline_paths['im_branch_label_reassigned']
        self.reassigned_branch_memmap = self.im_info.allocate_memory(
            reassigned_branch_label_path,
            dtype='int32',
            description='branch label reassigned',
            return_memmap=True
        )

        reassigned_obj_label_path = self.im_info.pipeline_paths['im_obj_label_reassigned']
        self.reassigned_obj_memmap = self.im_info.allocate_memory(
            reassigned_obj_label_path,
            dtype='int32',
            description='object label reassigned',
            return_memmap=True
        )

    def _get_master_mask(self, t):
        """
        Returns a boolean mask of voxels that participate in matching at time t.

        Currently defined as the union of non-zero branch and object labels.
        """
        parts = []
        if self.branch_label_memmap is not None:
            parts.append(self.branch_label_memmap[t] > 0)
        if self.obj_label_memmap is not None:
            parts.append(self.obj_label_memmap[t] > 0)
        if not parts:
            return np.zeros(self.spatial_shape, dtype=bool)
        return parts[0] if len(parts) == 1 else (parts[0] | parts[1])

    def _vote_assign_labels_for_frame(
        self,
        candidate_prev,
        candidate_next,
        candidate_dist,
        label_memmap,
        reassigned_memmap,
        t,
    ):
        """
        Assign labels for a single label type from time t to t+1 using weighted votes.

        Votes are accumulated per target voxel based on all interpolations
        (forward and backward), weighted by inverse distance.
        """
        if candidate_prev.size == 0 or candidate_next.size == 0:
            dim = candidate_prev.shape[1] if candidate_prev.ndim == 2 else 3
            return (np.empty((0, dim), dtype=np.int64),
                    np.empty((0, dim), dtype=np.int64))

        prev_labels = reassigned_memmap[t][tuple(candidate_prev.T)]
        valid = prev_labels > 0
        if not np.any(valid):
            dim = candidate_prev.shape[1]
            return (np.empty((0, dim), dtype=np.int64),
                    np.empty((0, dim), dtype=np.int64))

        candidate_prev = candidate_prev[valid]
        candidate_next = candidate_next[valid]
        candidate_dist = candidate_dist[valid]
        prev_labels = prev_labels[valid]

        # only assign labels to voxels that are labeled at t+1
        target_has_label = label_memmap[t + 1][tuple(candidate_next.T)] > 0
        if not np.any(target_has_label):
            dim = candidate_prev.shape[1]
            return (np.empty((0, dim), dtype=np.int64),
                    np.empty((0, dim), dtype=np.int64))

        candidate_prev = candidate_prev[target_has_label]
        candidate_next = candidate_next[target_has_label]
        candidate_dist = candidate_dist[target_has_label]
        prev_labels = prev_labels[target_has_label]

        assigned_prev_all = []
        assigned_next_all = []
        num_iters = max(1, int(self.max_refine_iterations))

        for _ in range(num_iters):
            unassigned = reassigned_memmap[t + 1][tuple(candidate_next.T)] == 0
            if not np.any(unassigned):
                break

            cand_prev_iter = candidate_prev[unassigned]
            cand_next_iter = candidate_next[unassigned]
            cand_dist_iter = candidate_dist[unassigned]
            labels_iter = prev_labels[unassigned]

            if cand_prev_iter.size == 0:
                break

            _, best_labels, best_candidate_idx = self._vote_targets(
                cand_next_iter, labels_iter, cand_dist_iter
            )
            if len(best_candidate_idx) == 0:
                break

            best_prev = cand_prev_iter[best_candidate_idx]
            best_next = cand_next_iter[best_candidate_idx]

            reassigned_memmap[t + 1][tuple(best_next.T)] = best_labels

            assigned_prev_all.append(best_prev)
            assigned_next_all.append(best_next)

        if not assigned_prev_all:
            dim = candidate_prev.shape[1]
            return (np.empty((0, dim), dtype=np.int64),
                    np.empty((0, dim), dtype=np.int64))

        return (np.concatenate(assigned_prev_all, axis=0),
                np.concatenate(assigned_next_all, axis=0))

    # -------------------------------------------------------------------------
    # Main driver
    # -------------------------------------------------------------------------

    def _run_reassignment(self):
        self._allocate_memory()

        # initialize reassigned labels at t=0
        if self.branch_label_memmap is not None:
            vox_prev_branch = np.argwhere(self.branch_label_memmap[0] > 0)
            if len(vox_prev_branch):
                self.reassigned_branch_memmap[0][tuple(vox_prev_branch.T)] = \
                    self.branch_label_memmap[0][tuple(vox_prev_branch.T)]

        if self.obj_label_memmap is not None:
            vox_prev_obj = np.argwhere(self.obj_label_memmap[0] > 0)
            if len(vox_prev_obj):
                self.reassigned_obj_memmap[0][tuple(vox_prev_obj.T)] = \
                    self.obj_label_memmap[0][tuple(vox_prev_obj.T)]

        # clear any existing matches
        self.running_matches = []

        # Stream over timepoints, computing matches once and applying to both label types.
        # vox_next of frame t is byte-identical to vox_prev of frame t+1 (same union of label
        # memmaps at the same index), so cache and rotate to avoid re-reading both label
        # memmaps and re-running argwhere every frame.
        if self.num_t < 2:
            return
        vox_prev = np.argwhere(self._get_master_mask(0))

        for t in range(self.num_t - 1):
            if self.viewer is not None:
                self.viewer.status = f'Reassigning voxels. Frame: {t + 1} of {self.num_t}.'

            logger.info(f'Reassigning pixels between frames {t} and {t + 1}')

            vox_next = np.argwhere(self._get_master_mask(t + 1))

            if len(vox_prev) == 0 or len(vox_next) == 0:
                logger.info(f'No voxels to match between frames {t} and {t + 1}; stopping.')
                break

            candidate_prev, candidate_next, candidate_dist = self.match_voxels(vox_prev, vox_next, t)
            if len(candidate_prev) == 0:
                logger.info(f'No valid matches between frames {t} and {t + 1}; stopping.')
                break

            if self.store_running_matches:
                # store a single best source per target for downstream adjacency use
                best_prev, best_next = self._select_best_pairs(
                    candidate_prev, candidate_next, candidate_dist
                )
                match_dtype = self.match_coord_dtype or np.uint16
                self.running_matches.append([
                    best_prev.astype(match_dtype, copy=False),
                    best_next.astype(match_dtype, copy=False),
                ])

            # vote-assign for each label type separately (filtering to labeled voxels)
            if self.branch_label_memmap is not None:
                self._vote_assign_labels_for_frame(
                    candidate_prev, candidate_next, candidate_dist,
                    self.branch_label_memmap, self.reassigned_branch_memmap, t
                )

            if self.obj_label_memmap is not None:
                self._vote_assign_labels_for_frame(
                    candidate_prev, candidate_next, candidate_dist,
                    self.obj_label_memmap, self.reassigned_obj_memmap, t
                )

            vox_prev = vox_next

        # save running matches to npy if requested
        if self.store_running_matches and self.voxel_matches_path is not None:
            np.save(self.voxel_matches_path, np.array(self.running_matches, dtype=object))

    def run(self):
        """
        Main method to execute voxel reassignment for both branch and object labels.

        This implementation:
          - initializes reassigned labels at t=0 for both branch and object labels.
          - for each pair of consecutive timepoints, computes forward/backward
            interpolation candidates once (based on the union of labeled voxels).
          - assigns labels at t+1 using weighted votes from all candidates.
        """
        if self.im_info.no_t:
            logger.info("Skipping voxel reassignment for non-temporal dataset.")
            return
        device = adaptive_run.normalize_device(self.device)
        device_order = adaptive_run.device_cascade(self.device)
        if device == "gpu" and device_order == ["cpu"]:
            logger.warning("VoxelReassigner: GPU requested but not available; falling back to CPU.")

        start_low_memory = bool(self.low_memory) or adaptive_run.should_use_low_memory(
            self.im_info,
            include_gpu="gpu" in device_order,
            device_order=device_order,
        )
        if start_low_memory and not self.low_memory:
            logger.info("VoxelReassigner: enabling low-memory mode based on estimated usage.")

        last_exc = None
        for dev, low in adaptive_run.mode_candidates(device_order, start_low_memory):
            try:
                self._set_backend(dev)
                self._set_low_memory(low)
                self._run_reassignment()
                return
            except Exception as exc:
                last_exc = exc
                if adaptive_run.is_gpu_unavailable_error(exc) and dev in ("gpu", "mps"):
                    logger.warning("VoxelReassigner: GPU backend unavailable; retrying on CPU.")
                    continue
                if adaptive_run.is_oom_error(exc):
                    logger.warning(
                        "VoxelReassigner: OOM on %s/%s; retrying with lower settings.",
                        dev,
                        "low-memory" if low else "high-memory",
                    )
                    continue
                raise
        raise last_exc
