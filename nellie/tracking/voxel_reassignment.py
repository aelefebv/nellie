import heapq  # no longer used but kept for compatibility if imported elsewhere

import numpy as np
from scipy.spatial import cKDTree

from nellie import logger
from nellie.im_info.verifier import ImInfo
from nellie.tracking.flow_interpolation import FlowInterpolator


class VoxelReassigner:
    """
    A class for voxel reassignment across time points using forward and backward flow interpolation.

    This optimized version:
      - Streams over timepoints instead of holding all voxel coordinates in memory.
      - Reuses a single set of voxel matches between timepoints for all label types.
      - Avoids large intermediate dense arrays where possible.
      - Optionally stores running matches for downstream analysis.
    """

    def __init__(self, im_info: ImInfo, num_t=None, viewer=None,
                 store_running_matches: bool = True,
                 max_refine_iterations: int = 3):
        """
        Parameters
        ----------
        im_info : ImInfo
            Image metadata and memory-mapped data.
        num_t : int, optional
            Number of timepoints in the dataset. If None, it is inferred from the image metadata.
        viewer : Any, optional
            Optional viewer for visualization / status updates.
        store_running_matches : bool, optional
            If True, store per-frame voxel matches (may be large for big datasets).
        max_refine_iterations : int, optional
            Maximum number of refinement iterations to try to reassign unmatched voxels
            by nearest-neighbor propagation. Set to 0 to disable refinement for speed.
        """
        self.im_info = im_info

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
            self.debug = None
            self.viewer = viewer
            self.shape = None
            self.spatial_shape = None
            self.store_running_matches = store_running_matches
            self.max_refine_iterations = max_refine_iterations
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

        self.debug = None

        self.viewer = viewer

        # optimization / behavior controls
        self.store_running_matches = store_running_matches
        self.max_refine_iterations = max_refine_iterations

        # will be set in _allocate_memory
        self.shape = None
        self.spatial_shape = None

    # -------------------------------------------------------------------------
    # Matching primitives
    # -------------------------------------------------------------------------

    def _match_forward(self, flow_interpolator, vox_prev, vox_next, t):
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

        Returns
        -------
        tuple of np.ndarray
            (vox_prev_matched_valid, vox_next_matched_valid, distances_valid)
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
        match_dist, matched_idx = self._match_voxels_to_centroids(vox_next, centroids_next_interpx)
        vox_matched_to_centroids = vox_next[matched_idx]

        # keep matches within distance threshold
        vox_prev_matched_valid, vox_next_matched_valid, distances_valid = self._distance_threshold(
            vox_prev_kept.astype(np.int64), vox_matched_to_centroids.astype(np.int64)
        )
        return vox_prev_matched_valid, vox_next_matched_valid, distances_valid

    def _match_backward(self, flow_interpolator, vox_next, vox_prev, t):
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

        Returns
        -------
        tuple of np.ndarray
            (vox_prev_matched_valid, vox_next_matched_valid, distances_valid)
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
        match_dist, matched_idx = self._match_voxels_to_centroids(vox_prev, centroids_prev_interpx)
        vox_matched_to_centroids = vox_prev[matched_idx]

        # keep matches within distance threshold
        vox_prev_matched_valid, vox_next_matched_valid, distances_valid = self._distance_threshold(
            vox_matched_to_centroids.astype(np.int64), vox_next_kept.astype(np.int64)
        )
        return vox_prev_matched_valid, vox_next_matched_valid, distances_valid

    def _match_voxels_to_centroids(self, coords_real, coords_interpx):
        """
        Matches real voxel coordinates to interpolated centroids using nearest neighbor search.

        Parameters
        ----------
        coords_real : np.ndarray
            Real voxel coordinates (N_real, D).
        coords_interpx : np.ndarray
            Interpolated centroid coordinates (N_interp, D).

        Returns
        -------
        tuple
            (distances, indices) from cKDTree.query
        """
        scaling = self.flow_interpolator_fw.scaling
        coords_real = np.asarray(coords_real, dtype=np.float32) * scaling
        coords_interpx = np.asarray(coords_interpx, dtype=np.float32) * scaling
        tree = cKDTree(coords_real)
        dist, idx = tree.query(coords_interpx, k=1, workers=-1)
        return dist, idx

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

        used_prev = set()
        used_next = set()
        keep_indices = []

        for idx_sorted, (p, n) in enumerate(zip(prev_flat_sorted, next_flat_sorted)):
            if p in used_prev or n in used_next:
                continue
            used_prev.add(p)
            used_next.add(n)
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
        Matches voxels between two consecutive timepoints using both forward and backward interpolation.

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
            (matched_prev, matched_next) voxel coordinates with shape (N_matched, D).
        """
        if vox_prev.size == 0 or vox_next.size == 0:
            dim = vox_prev.shape[1] if vox_prev.ndim == 2 else 3
            return (np.empty((0, dim), dtype=np.int64),
                    np.empty((0, dim), dtype=np.int64))

        logger.debug(f'Forward voxel matching for t: {t}')
        vox_prev_fw, vox_next_fw, dist_fw = self._match_forward(
            self.flow_interpolator_fw, vox_prev, vox_next, t
        )

        logger.debug(f'Backward voxel matching for t: {t}')
        vox_prev_bw, vox_next_bw, dist_bw = self._match_backward(
            self.flow_interpolator_bw, vox_next, vox_prev, t + 1
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
                    np.empty((0, dim), dtype=np.int64))

        vox_prev_matches = np.concatenate(parts_prev, axis=0)
        vox_next_matches = np.concatenate(parts_next, axis=0)
        distances = np.concatenate(parts_dist, axis=0)

        logger.debug(f'Assigning unique matches for t: {t}')
        vox_prev_unique, vox_next_unique = self._assign_unique_matches(
            vox_prev_matches, vox_next_matches, distances
        )

        if len(vox_next_unique) == 0:
            dim = vox_prev.shape[1]
            return (np.empty((0, dim), dtype=np.int64),
                    np.empty((0, dim), dtype=np.int64))

        # optional refinement: try to reassign unmatched voxels
        if self.max_refine_iterations <= 0:
            return vox_prev_unique.astype(np.int64), vox_next_unique.astype(np.int64)

        # flatten all vox_next for unmatched detection
        if self.spatial_shape is None:
            raise RuntimeError("spatial_shape is not set; call _allocate_memory() before matching.")
        vox_next_flat = np.ravel_multi_index(vox_next.T, self.spatial_shape)
        matched_next_flat = np.ravel_multi_index(vox_next_unique.T, self.spatial_shape)

        scaling = self.flow_interpolator_fw.scaling

        for iteration in range(self.max_refine_iterations):
            # find unmatched voxels in t+1
            matched_mask = np.isin(vox_next_flat, matched_next_flat, assume_unique=False)
            vox_next_unmatched = vox_next[~matched_mask]
            if len(vox_next_unmatched) == 0:
                break

            tree = cKDTree(vox_next_unique.astype(np.float32) * scaling)
            dists, idxs = tree.query(vox_next_unmatched.astype(np.float32) * scaling,
                                     k=1, workers=-1)
            valid_mask = dists < self.flow_interpolator_fw.max_distance_um
            if not np.any(valid_mask):
                break

            num_unmatched = len(vox_next_unmatched)
            new_prev = vox_prev_unique[idxs[valid_mask]]
            new_next = vox_next_unmatched[valid_mask]

            vox_prev_unique = np.concatenate([vox_prev_unique, new_prev], axis=0)
            vox_next_unique = np.concatenate([vox_next_unique, new_next], axis=0)

            new_next_flat = np.ravel_multi_index(new_next.T, self.spatial_shape)
            matched_next_flat = np.concatenate([matched_next_flat, new_next_flat])

            new_num_unmatched = num_unmatched - int(valid_mask.sum())
            logger.debug(
                f'Reassigned {int(valid_mask.sum())}/{num_unmatched} unassigned voxels in iteration {iteration + 1}. '
                f'{new_num_unmatched} remain (before next iteration recomputation).'
            )

        return vox_prev_unique.astype(np.int64), vox_next_unique.astype(np.int64)

    # -------------------------------------------------------------------------
    # Dataset / memory helpers
    # -------------------------------------------------------------------------

    def _get_t(self):
        """
        Gets the number of timepoints from the image metadata or sets it if not provided.
        """
        if self.num_t is None:
            if self.im_info.no_t:
                self.num_t = 1
            else:
                self.num_t = self.im_info.shape[self.im_info.axes.index('T')]

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
        mask = None
        if self.branch_label_memmap is not None:
            mask_b = self.branch_label_memmap[t] > 0
            mask = mask_b if mask is None else (mask | mask_b)
        if self.obj_label_memmap is not None:
            mask_o = self.obj_label_memmap[t] > 0
            mask = mask_o if mask is None else (mask | mask_o)
        if mask is None:
            # no labels present; return empty mask with correct shape
            mask = np.zeros(self.spatial_shape, dtype=bool)
        return mask

    def _propagate_labels_for_frame(self, matched_prev, matched_next, label_memmap, reassigned_memmap, t):
        """
        Propagate labels for a single label type from time t to t+1, using precomputed voxel matches.

        Only voxels that are labeled (non-zero) in both t and t+1 are considered valid for propagation.
        """
        if matched_prev.size == 0 or matched_next.size == 0:
            return

        # labels at source and target
        prev_labels = label_memmap[t][tuple(matched_prev.T)]
        next_labels_nonzero = label_memmap[t + 1][tuple(matched_next.T)] > 0

        # only propagate where both source and target are labeled
        valid_prev = prev_labels > 0
        valid = valid_prev & next_labels_nonzero
        if not np.any(valid):
            return

        prev_valid = matched_prev[valid]
        next_valid = matched_next[valid]

        # ensure previous reassigned labels are initialized
        reassigned_values = reassigned_memmap[t][tuple(prev_valid.T)]
        reassigned_memmap[t + 1][tuple(next_valid.T)] = reassigned_values

    # -------------------------------------------------------------------------
    # Main driver
    # -------------------------------------------------------------------------

    def run(self):
        """
        Main method to execute voxel reassignment for both branch and object labels.

        This implementation:
          - initializes reassigned labels at t=0 for both branch and object labels.
          - for each pair of consecutive timepoints, computes voxel matches once
            (based on the union of labeled voxels), then applies those matches
            to both branch and object label volumes.
        """
        if self.im_info.no_t:
            return

        self._get_t()
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

        # stream over timepoints, computing matches once and applying to both label types
        for t in range(self.num_t - 1):
            if self.viewer is not None:
                self.viewer.status = f'Reassigning voxels. Frame: {t + 1} of {self.num_t}.'

            logger.info(f'Reassigning pixels between frames {t} and {t + 1}')

            master_mask_prev = self._get_master_mask(t)
            master_mask_next = self._get_master_mask(t + 1)

            vox_prev = np.argwhere(master_mask_prev)
            vox_next = np.argwhere(master_mask_next)

            if len(vox_prev) == 0 or len(vox_next) == 0:
                logger.info(f'No voxels to match between frames {t} and {t + 1}; stopping.')
                break

            matched_prev, matched_next = self.match_voxels(vox_prev, vox_next, t)
            if len(matched_prev) == 0:
                logger.info(f'No valid matches between frames {t} and {t + 1}; stopping.')
                break

            if self.store_running_matches:
                # store as uint16 to reduce disk footprint (safe if spatial dims < 65535)
                self.running_matches.append([
                    matched_prev.astype(np.uint16),
                    matched_next.astype(np.uint16),
                ])

            # propagate for each label type separately (filtering to labeled voxels)
            if self.branch_label_memmap is not None:
                self._propagate_labels_for_frame(
                    matched_prev, matched_next,
                    self.branch_label_memmap, self.reassigned_branch_memmap, t
                )

            if self.obj_label_memmap is not None:
                self._propagate_labels_for_frame(
                    matched_prev, matched_next,
                    self.obj_label_memmap, self.reassigned_obj_memmap, t
                )

        # save running matches to npy if requested
        if self.store_running_matches and self.voxel_matches_path is not None:
            np.save(self.voxel_matches_path, np.array(self.running_matches, dtype=object))


if __name__ == "__main__":
    # Example usage and adjacency remapping using voxel matches.
    # This section avoids allocating large dense voxel x voxel matrices and
    # optionally uses GPU (cupy) for accumulation when available.
    im_path = r"D:\test_files\nelly_smorgasbord\deskewed-iono_pre.ome.tif"
    im_info = ImInfo(im_path)
    num_t = 3
    run_obj = VoxelReassigner(im_info, num_t=num_t)
    run_obj.run()

    import pickle

    # This section allows for finding links between any level of the hierarchy to any
    # other level in the hierarchy at any time point via sparse accumulation.
    edges_loaded = pickle.load(open(im_info.pipeline_paths['adjacency_maps'], "rb"))

    # ------------------------------------------------------------------
    # Helper: optional GPU-accelerated pair-count accumulation
    # ------------------------------------------------------------------
    def accumulate_pair_counts(src_ids, dst_ids, n_src, n_dst, use_gpu=True):
        """
        Accumulate counts into an (n_src, n_dst) matrix given parallel vectors
        src_ids and dst_ids, optionally using cupy if available.

        Parameters
        ----------
        src_ids : np.ndarray
            Source indices (e.g., branch ids at t0).
        dst_ids : np.ndarray
            Destination indices (e.g., branch ids at t1).
        n_src : int
            Number of distinct source entities.
        n_dst : int
            Number of distinct destination entities.
        use_gpu : bool
            If True, attempt to use cupy for accumulation, with CPU fallback.

        Returns
        -------
        np.ndarray
            Count matrix of shape (n_src, n_dst).
        """
        src_ids = np.asarray(src_ids, dtype=np.int64)
        dst_ids = np.asarray(dst_ids, dtype=np.int64)

        if src_ids.size == 0 or dst_ids.size == 0:
            return np.zeros((n_src, n_dst), dtype=np.uint32)

        # try GPU if requested
        if use_gpu:
            try:
                import cupy as cp

                src_gpu = cp.asarray(src_ids)
                dst_gpu = cp.asarray(dst_ids)
                counts_gpu = cp.zeros((n_src, n_dst), dtype=cp.uint32)
                cp.add.at(counts_gpu, (src_gpu, dst_gpu), 1)
                counts = cp.asnumpy(counts_gpu)

                # free GPU memory
                del src_gpu, dst_gpu, counts_gpu
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass

                return counts
            except Exception:
                # cupy not available or GPU OOM; fall back to CPU
                pass

        counts = np.zeros((n_src, n_dst), dtype=np.uint32)
        np.add.at(counts, (src_ids, dst_ids), 1)
        return counts

    # ------------------------------------------------------------------
    # Example: branch adjacency between t0 and t1 without dense v_t
    # ------------------------------------------------------------------
    mask_01 = run_obj.obj_label_memmap[:2] > 0
    mask_voxels_0 = np.argwhere(mask_01[0])
    mask_voxels_1 = np.argwhere(mask_01[1])

    # coordinate -> index mappings for the object voxels at t0 and t1
    t0_coords_in_mask_0 = {tuple(coord): idx for idx, coord in enumerate(mask_voxels_0)}
    t1_coords_in_mask_1 = {tuple(coord): idx for idx, coord in enumerate(mask_voxels_1)}

    # use matches between t0 and t1 (frame index 0)
    matches_t0_t1_prev, matches_t0_t1_next = run_obj.running_matches[0]

    idx_matches_0 = []
    idx_matches_1 = []
    for coord_prev, coord_next in zip(matches_t0_t1_prev, matches_t0_t1_next):
        key_prev = tuple(int(c) for c in coord_prev)
        key_next = tuple(int(c) for c in coord_next)
        if key_prev in t0_coords_in_mask_0 and key_next in t1_coords_in_mask_1:
            idx_matches_0.append(t0_coords_in_mask_0[key_prev])
            idx_matches_1.append(t1_coords_in_mask_1[key_next])

    idx_matches_0 = np.asarray(idx_matches_0, dtype=np.int64)
    idx_matches_1 = np.asarray(idx_matches_1, dtype=np.int64)

    # branch-voxel adjacency matrices at t0 and t1
    b_v0 = edges_loaded['b_v'][0].astype(np.uint8)
    b_v1 = edges_loaded['b_v'][1].astype(np.uint8)

    # per-voxel branch labels (argmax over adjacency)
    branch_labels_0 = np.argmax(b_v0, axis=0)
    branch_labels_1 = np.argmax(b_v1, axis=0)

    # branch ids for each matched voxel pair
    branch_ids_0 = branch_labels_0[idx_matches_0]
    branch_ids_1 = branch_labels_1[idx_matches_1]

    # accumulate counts: number of matched voxels shared between each pair of branches
    b0_b1 = accumulate_pair_counts(
        branch_ids_0,
        branch_ids_1,
        n_src=b_v0.shape[0],
        n_dst=b_v1.shape[0],
        use_gpu=True,
    )

    # for each branch at t1, choose the best-matching branch at t0
    max_idx = np.argmax(b0_b1, axis=0) + 1  # +1 to keep 0 as background

    # build branch label volumes relabelled by t0 branch ids
    mask_branches = np.zeros(mask_01.shape, dtype=np.uint16)
    mask_branches[0][tuple(mask_voxels_0.T)] = branch_labels_0 + 1

    new_branch_labels_1 = max_idx[branch_labels_1]
    mask_branches[1][tuple(mask_voxels_1.T)] = new_branch_labels_1

    # ------------------------------------------------------------------
    # Example: node adjacency between t0 and t1 (analogous to branches)
    # ------------------------------------------------------------------
    n_v0 = edges_loaded['n_v'][0].astype(np.uint8)
    n_v1 = edges_loaded['n_v'][1].astype(np.uint8)

    node_labels_0 = np.argmax(n_v0, axis=0)
    node_labels_1 = np.argmax(n_v1, axis=0)

    node_ids_0 = node_labels_0[idx_matches_0]
    node_ids_1 = node_labels_1[idx_matches_1]

    n0_n1 = accumulate_pair_counts(
        node_ids_0,
        node_ids_1,
        n_src=n_v0.shape[0],
        n_dst=n_v1.shape[0],
        use_gpu=True,
    )

    max_idx_n = np.argmax(n0_n1, axis=0) + 1

    mask_nodes = np.zeros(mask_01.shape, dtype=np.uint16)
    mask_nodes[0][tuple(mask_voxels_0.T)] = node_labels_0 + 1
    new_node_labels_1 = max_idx_n[node_labels_1]
    mask_nodes[1][tuple(mask_voxels_1.T)] = new_node_labels_1