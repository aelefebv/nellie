"""
Flow vector interpolation for temporal tracking in microscopy images.

This module provides interpolation of optical flow vectors between timepoints
via distance-weighted KDTree lookup over a precomputed `flow_vector_array`.
"""
import numpy as np
from scipy.spatial import cKDTree

from nellie.utils.base_logger import logger
from nellie.im_info.verifier import ImInfo


class FlowInterpolator:
    """
    A class for interpolating flow vectors between timepoints in microscopy images using precomputed flow data.

    Attributes
    ----------
    im_info : ImInfo
        An object containing image metadata and memory-mapped image data.
    num_t : int
        Number of timepoints in the image.
    max_distance_um : float
        Maximum distance allowed for interpolation (in micrometers).
    forward : bool
        Indicates if the interpolation is performed in the forward direction (True) or backward direction (False).
    scaling : tuple
        Scaling factors for Z, Y, and X dimensions.
    shape : tuple
        Shape of the input image.
    im_memmap : np.ndarray or None
        Memory-mapped original image data.
    flow_vector_array : np.ndarray or None
        Precomputed flow vector array loaded from disk.
    current_t : int or None
        Cached timepoint for the current flow vector calculation.
    check_rows : np.ndarray or None
        Flow vector data for the current timepoint.
    check_coords : np.ndarray or None
        Coordinates corresponding to the flow vector data for the current timepoint.
    current_tree : cKDTree or None
        KDTree for fast lookup of nearby coordinates in the current timepoint.
    debug : dict or None
        Debugging information for tracking processing steps.

    Methods
    -------
    _allocate_memory()
        Allocates memory and loads the precomputed flow vector array.
    _get_t()
        Determines the number of timepoints to process.
    _get_nearby_coords(t, coords)
        Finds nearby coordinates within a defined radius from the given coordinates using a KDTree.
    _get_vector_weights(nearby_idxs, distances_all)
        Computes the weights for nearby flow vectors based on their distances and costs.
    _get_final_vector(nearby_idxs, weights_all)
        Computes the final interpolated vector for each coordinate using distance-weighted vectors.
    interpolate_coord(coords, t)
        Interpolates the flow vector at the given coordinates and timepoint.
    _initialize()
        Initializes the FlowInterpolator by allocating memory and setting the timepoints.

    """
    def __init__(self, im_info: ImInfo, num_t=None, max_distance_um=0.5, forward=True):
        """
        Initializes the FlowInterpolator with image metadata and interpolation parameters.

        Parameters
        ----------
        im_info : ImInfo
            An instance of the ImInfo class, containing metadata and paths for the image file.
        num_t : int, optional
            Number of timepoints to process. If None, defaults to the number of timepoints in the image.
        max_distance_um : float, optional
            Maximum distance allowed for interpolation (in micrometers, default is 0.5).
        forward : bool, optional
            Indicates if the interpolation is performed in the forward direction (default is True).
        """
        self.im_info = im_info

        if self.im_info.no_t:
            return

        self.num_t = num_t
        if num_t is None and not self.im_info.no_t:
            self.num_t = im_info.shape[im_info.axes.index('T')]

        if self.im_info.no_z:
            self.scaling = (im_info.dim_res['Y'], im_info.dim_res['X'])
        else:
            self.scaling = (im_info.dim_res['Z'], im_info.dim_res['Y'], im_info.dim_res['X'])

        self.max_distance_um = max(max_distance_um * im_info.dim_res['T'], 0.5)

        self.forward = forward

        self.shape = ()

        self.im_memmap = None
        self.flow_vector_array = None

        # caching
        self.current_t = None
        self.check_rows = None
        self.check_coords = None
        self.current_tree = None

        self.debug = None
        self._initialize()

    def _allocate_memory(self):
        """
        Allocates memory and loads the precomputed flow vector array.

        This method reads the flow vector data from disk and prepares it for use during interpolation.
        """
        logger.debug('Allocating memory for mocap marking.')

        self.im_memmap = self.im_info.get_memmap(self.im_info.im_path)
        self.shape = self.im_memmap.shape

        flow_vector_array_path = self.im_info.pipeline_paths['flow_vector_array']
        self.flow_vector_array = np.load(flow_vector_array_path)
        # Pre-bucket the flow_vector_array by t so per-frame `interpolate_coord`
        # lookups are O(1) dict reads instead of O(total_markers) `np.where`
        # scans. The forward path indexes by t directly; the backward path
        # indexes by t-1 (looks up markers from the previous frame). Storing
        # the index arrays (not the row slices) keeps memory minimal — the
        # full `flow_vector_array` is the source of truth.
        if self.flow_vector_array.size:
            t_col = self.flow_vector_array[:, 0]
            unique_t, inverse = np.unique(t_col, return_inverse=True)
            order = np.argsort(inverse, kind='stable')
            sorted_inverse = inverse[order]
            split_at = np.searchsorted(sorted_inverse, np.arange(1, len(unique_t)))
            grouped = np.split(order, split_at)
            self._t_to_rows = {int(t_val): rows for t_val, rows in zip(unique_t, grouped)}
        else:
            self._t_to_rows = {}

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
        else:
            return

    def _get_nearby_coords(self, t, coords):
        """
        Finds nearby coordinates within a defined radius from the given coordinates using a KDTree.

        Parameters
        ----------
        t : int
            Timepoint index.
        coords : np.ndarray
            Coordinates for which to find nearby points.

        Returns
        -------
        tuple
            Nearby indices and distances from the input coordinates.
        """
        # Tree + scaled-check-coords cache rebuild only when t changes. The
        # scaled array is needed again below for batched distance compute,
        # so cache it alongside the tree (single multiplication, single
        # buffer reuse). See ADR 0010.
        if self.current_t != t:
            self.scaled_check_coords = self.check_coords * self.scaling
            self.current_tree = cKDTree(self.scaled_check_coords)
        scaled_coords = np.asarray(coords) * self.scaling
        # NaN-coord filtering: only good coords go to the tree.
        good_coords = np.where(~np.isnan(scaled_coords[:, 0]))[0]
        if len(good_coords) == 0:
            return [], []
        scaled_query = scaled_coords[good_coords]
        # Single query_ball_point — returns within-radius neighbor indices
        # per query coord directly; no second cKDTree.query(k=max_k)
        # traversal needed. See ADR 0010.
        nearby_idxs = self.current_tree.query_ball_point(
            scaled_query, self.max_distance_um, p=2, workers=-1
        )
        counts = np.fromiter(
            (len(idx) for idx in nearby_idxs), dtype=np.intp, count=len(nearby_idxs)
        )
        total = int(counts.sum())
        # Short-circuit when no good coord has any neighbor (matches the
        # pre-rewrite max_k == 0 early-return; consumer relies on
        # `len(final_vector) == 0` to skip the whole frame).
        if total == 0:
            return [], []
        # Batched distance compute: concatenate all per-coord neighbor
        # indices into one flat array, repeat each query coord by its
        # neighbor count, and run one linalg.norm. Avoids the per-coord
        # Python loop overhead of N small linalg.norm calls.
        flat_idxs = np.concatenate(
            [np.asarray(idx, dtype=np.intp) for idx in nearby_idxs]
        )
        flat_queries = np.repeat(scaled_query, counts, axis=0)
        flat_deltas = self.scaled_check_coords[flat_idxs] - flat_queries
        flat_distances = np.linalg.norm(flat_deltas, axis=1)
        # Split back per-coord and place into original-index slots.
        # Correct positional alignment: per-pos result lands in
        # original-index slot good_coords[pos]. Fixes the pre-rewrite
        # NaN-alignment bug per ADR 0010.
        distance_return = [[] for _ in range(len(coords))]
        nearby_idxs_return = [[] for _ in range(len(coords))]
        offsets = np.concatenate(([0], np.cumsum(counts)))
        for pos, i in enumerate(good_coords):
            k = counts[pos]
            if k == 0:
                continue
            s, e = offsets[pos], offsets[pos + 1]
            nearby_idxs_return[i] = flat_idxs[s:e]
            distance_return[i] = flat_distances[s:e]
        return nearby_idxs_return, distance_return

    def _get_vector_weights(self, nearby_idxs, distances_all):
        """
        Computes the weights for nearby flow vectors based on their distances and costs.

        Parameters
        ----------
        nearby_idxs : list
            Indices of nearby coordinates.
        distances_all : list
            Distances from the input coordinates to the nearby points.

        Returns
        -------
        list
            Weights for each nearby flow vector.
        """
        weights_all = []
        for i in range(len(nearby_idxs)):
            # lowest cost should be most highly weighted
            cost_weights = -self.check_rows[nearby_idxs[i], -1]

            if len(distances_all[i]) == 0:
                weights_all.append(None)
                continue

            if np.min(distances_all[i]) == 0:
                distance_weights = (distances_all[i] == 0) * 1.0
            else:
                distance_weights = 1 / distances_all[i]

            weights = cost_weights * distance_weights
            weights -= np.min(weights) - 1
            weights /= np.sum(weights)
            weights_all.append(weights)
        return weights_all

    def _get_final_vector(self, nearby_idxs, weights_all):
        """
        Computes the final interpolated vector for each coordinate using distance-weighted vectors.

        Parameters
        ----------
        nearby_idxs : list
            Indices of nearby coordinates.
        weights_all : list
            Weights for the flow vectors.

        Returns
        -------
        np.ndarray
            Final interpolated vectors for each input coordinate.
        """
        if self.im_info.no_z:
            final_vectors = np.zeros((len(nearby_idxs), 2))
        else:
            final_vectors = np.zeros((len(nearby_idxs), 3))
        for i in range(len(nearby_idxs)):
            if weights_all[i] is None:
                final_vectors[i] = np.nan
                continue
            if self.im_info.no_z:
                vectors = self.check_rows[nearby_idxs[i], 3:5]
            else:
                vectors = self.check_rows[nearby_idxs[i], 4:7]
            if len(weights_all[i].shape) == 0:
                final_vectors[i] = vectors[0]
            else:
                weighted_vectors = vectors * weights_all[i][:, None]
                final_vectors[i] = np.sum(weighted_vectors, axis=0)  # already normalized by weights
        return final_vectors

    def interpolate_coord(self, coords, t):
        """
        Interpolates the flow vector at the given coordinates and timepoint.

        Parameters
        ----------
        coords : np.ndarray
            Input coordinates for interpolation.
        t : int
            Timepoint index.

        Returns
        -------
        np.ndarray
            Interpolated flow vectors at the given coordinates and timepoint.
        """
        # interpolate the flow vector at the coordinate at time t, either forward in time or backward in time.
        # For forward, simply find nearby LMPs, interpolate based on distance-weighted vectors
        # For backward, get coords from t-1 + vector, then find nearby coords from that, and interpolate based on distance-weighted vectors
        if self.current_t != t:
            # O(1) dict lookup over the pre-bucketed `_t_to_rows` index built
            # in `_allocate_memory` — replaces the per-frame
            # `np.where(self.flow_vector_array[:, 0] == t)` scan.
            lookup_t = t if self.forward else t - 1
            row_indices = self._t_to_rows.get(int(lookup_t))
            if row_indices is None or len(row_indices) == 0:
                self.check_rows = self.flow_vector_array[:0]
            else:
                self.check_rows = self.flow_vector_array[row_indices, :]
            if self.forward:
                if self.im_info.no_z:
                    self.check_coords = self.check_rows[:, 1:3]
                else:
                    self.check_coords = self.check_rows[:, 1:4]
            else:
                # Backward: check coords are pre-coord + forward vector
                # (the marker's destination at t).
                if self.im_info.no_z:
                    self.check_coords = self.check_rows[:, 1:3] + self.check_rows[:, 3:5]
                else:
                    self.check_coords = self.check_rows[:, 1:4] + self.check_rows[:, 4:7]

        nearby_idxs, distances_all = self._get_nearby_coords(t, coords)
        self.current_t = t

        weights_all = self._get_vector_weights(nearby_idxs, distances_all)
        final_vectors = self._get_final_vector(nearby_idxs, weights_all)

        return final_vectors

    def _initialize(self):
        """
        Initializes the FlowInterpolator by allocating memory and setting the timepoints.

        This method prepares the internal state of the object, including reading the flow vector array.
        """
        if self.im_info.no_t:
            return
        self._get_t()
        self._allocate_memory()


def _interpolate_all_directional(
    coords, start_t, end_t, im_info, min_track_num, max_distance_um, *, forward
):
    """Shared driver for ``interpolate_all_forward`` / ``_backward``.

    Parameters mirror the public functions; ``forward`` controls the
    direction of traversal and the sign of the per-frame coord update.

    Per-frame work:
      1. ``flow_interpx.interpolate_coord(coords, t)`` returns per-coord
         flow vectors (NaN row when no neighbor was found).
      2. NaN-mask: ``valid_mask = ~np.isnan(final_vector).any(axis=1)``.
      3. Pre-update coords for valid rows are captured (used only for the
         init row at ``t_idx == 0``).
      4. In-place update for valid rows (``+=`` for forward, ``-=`` for
         backward); invalid rows clobbered with NaN (terminal NaN
         propagation per the wiki contract).
      5. Per-frame block built as one ndarray; at ``t_idx == 0`` the
         block has shape ``(2 * n_valid, n_dim + 2)`` with init/post
         rows interleaved per valid coord (``block[0::2] = init_rows``,
         ``block[1::2] = post_rows``) — preserves the bit-identical
         per-coord row ordering of the pre-vectorize implementation.
    """
    flow_interpx = FlowInterpolator(im_info, forward=forward, max_distance_um=max_distance_um)
    n_coords = len(coords)
    base_ids = np.arange(n_coords, dtype=np.float64) + min_track_num
    track_blocks: list[np.ndarray] = []
    frame_num_blocks: list[np.ndarray] = []
    if forward:
        frame_range = np.arange(start_t, end_t)
    else:
        frame_range = np.array(list(np.arange(end_t, start_t + 1))[::-1])
    if len(frame_range) == 0:
        return [], {'frame_num': []}
    initial_frame = float(frame_range[0])
    for t_idx, t in enumerate(frame_range):
        final_vector = flow_interpx.interpolate_coord(coords, t)
        if final_vector is None or len(final_vector) == 0:
            continue
        valid_mask = ~np.isnan(final_vector).any(axis=1)
        n_valid = int(valid_mask.sum())
        if n_valid == 0:
            # Even with no valid coords, the original loop walks every
            # coord and marks invalids as NaN — preserve that here.
            coords[~valid_mask] = np.nan
            continue
        # `coords[valid_mask]` is advanced indexing → returns a copy that
        # is unaffected by the in-place update below.
        valid_coords_pre = coords[valid_mask]
        valid_ids = base_ids[valid_mask]
        if forward:
            coords[valid_mask] += final_vector[valid_mask]
        else:
            coords[valid_mask] -= final_vector[valid_mask]
        coords[~valid_mask] = np.nan
        valid_coords_post = coords[valid_mask]
        post_frame = float(t + 1) if forward else float(t - 1)
        n_dim = valid_coords_pre.shape[1]
        if t_idx == 0:
            # Interleave init + post rows per valid coord:
            # [init_0, post_0, init_1, post_1, ...].
            block = np.empty((2 * n_valid, n_dim + 2))
            block[0::2, 0] = valid_ids
            block[0::2, 1] = initial_frame
            block[0::2, 2:] = valid_coords_pre
            block[1::2, 0] = valid_ids
            block[1::2, 1] = post_frame
            block[1::2, 2:] = valid_coords_post
            frames_for_block = np.empty(2 * n_valid)
            frames_for_block[0::2] = initial_frame
            frames_for_block[1::2] = post_frame
        else:
            block = np.empty((n_valid, n_dim + 2))
            block[:, 0] = valid_ids
            block[:, 1] = post_frame
            block[:, 2:] = valid_coords_post
            frames_for_block = np.full(n_valid, post_frame)
        track_blocks.append(block)
        frame_num_blocks.append(frames_for_block)

    if track_blocks:
        tracks = np.concatenate(track_blocks, axis=0).tolist()
    else:
        tracks = []
    if frame_num_blocks:
        frame_num_list = np.concatenate(frame_num_blocks).tolist()
    else:
        frame_num_list = []
    return tracks, {'frame_num': frame_num_list}


def interpolate_all_forward(coords, start_t, end_t, im_info, min_track_num=0, max_distance_um=0.5):
    """
    Interpolates coordinates forward in time across multiple timepoints using flow vectors.

    Parameters
    ----------
    coords : np.ndarray
        Array of input coordinates to track.
    start_t : int
        Starting timepoint.
    end_t : int
        Ending timepoint.
    im_info : ImInfo
        An instance of the ImInfo class containing image metadata and paths.
    min_track_num : int, optional
        Minimum track number to assign to coordinates (default is 0).
    max_distance_um : float, optional
        Maximum distance allowed for interpolation (in micrometers, default is 0.5).

    Returns
    -------
    tuple
        List of tracks and associated track properties.
    """
    return _interpolate_all_directional(
        coords, start_t, end_t, im_info, min_track_num, max_distance_um, forward=True,
    )


def interpolate_all_backward(coords, start_t, end_t, im_info, min_track_num=0, max_distance_um=0.5):
    """
    Interpolates coordinates backward in time across multiple timepoints using flow vectors.

    Parameters
    ----------
    coords : np.ndarray
        Array of input coordinates to track.
    start_t : int
        Starting timepoint.
    end_t : int
        Ending timepoint.
    im_info : ImInfo
        An instance of the ImInfo class containing image metadata and paths.
    min_track_num : int, optional
        Minimum track number to assign to coordinates (default is 0).
    max_distance_um : float, optional
        Maximum distance allowed for interpolation (in micrometers, default is 0.5).

    Returns
    -------
    tuple
        List of tracks and associated track properties.
    """
    return _interpolate_all_directional(
        coords, start_t, end_t, im_info, min_track_num, max_distance_um, forward=False,
    )

