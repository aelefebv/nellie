"""
Flow vector interpolation for temporal tracking in microscopy images.

This module provides interpolation of optical flow vectors between timepoints with
optimizations for large datasets and optional GPU acceleration.
"""
import numpy as np
from scipy.spatial import cKDTree

from nellie.utils.base_logger import logger
from nellie.im_info.verifier import ImInfo

# Optional GPU support (CuPy). If unavailable, everything runs on CPU.
try:
    import cupy as cp  # type: ignore
    _HAS_CUPY = True
except Exception:  # pragma: no cover - environment without CuPy
    cp = None
    _HAS_CUPY = False


class FlowStore:
    """
    Thin wrapper around the flow vector array on disk.

    Responsibilities
    ----------------
    - Load the flow array with configurable memory usage (memmap vs in-memory).
    - Build an index from timepoint t -> rows for that t, so we avoid O(N) scans per t.
    """

    def __init__(self, flow_array_path: str, memory_mode: str = "auto"):
        """
        Parameters
        ----------
        flow_array_path : str
            Path to the .npy file with flow vectors.
        memory_mode : {"auto", "memmap", "in_memory"}
            How to load the flow vector array.
        """
        self.flow_array_path = flow_array_path
        self.memory_mode = memory_mode
        self.array = self._load_array()
        self._build_time_index()

    def _load_array(self) -> np.ndarray:
        import os

        if self.memory_mode == "in_memory":
            logger.debug("Loading flow vector array fully into memory.")
            return np.load(self.flow_array_path)

        if self.memory_mode == "memmap":
            logger.debug("Memory-mapping flow vector array.")
            return np.load(self.flow_array_path, mmap_mode="r")

        # auto: decide based on file size
        size_gb = os.path.getsize(self.flow_array_path) / (1024 ** 3)
        if size_gb < 2.0:
            logger.debug(
                f"Flow vector array ~{size_gb:.2f} GB; loading fully into memory (auto mode)."
            )
            return np.load(self.flow_array_path)
        else:
            logger.debug(
                f"Flow vector array ~{size_gb:.2f} GB; using memmap (auto mode)."
            )
            return np.load(self.flow_array_path, mmap_mode="r")

    def _build_time_index(self):
        """
        Build an index mapping t -> slice into a sorted-by-time view of the array.

        This avoids scanning the full array for each timepoint.
        """
        logger.debug("Building time index for flow vectors.")
        t_col = np.asarray(self.array[:, 0], dtype=np.int64)
        order = np.argsort(t_col, kind="mergesort")  # stable sort
        t_sorted = t_col[order]

        unique_t, start_idx, counts = np.unique(
            t_sorted, return_index=True, return_counts=True
        )

        self._order = order
        self._t_to_slice = {
            int(t): (int(s), int(s + c))
            for t, s, c in zip(unique_t, start_idx, counts)
        }

    def get_rows_for_t(self, t: int) -> np.ndarray | None:
        """
        Return a view of rows belonging to timepoint t.

        Parameters
        ----------
        t : int
            Timepoint index.

        Returns
        -------
        np.ndarray or None
            Rows for this timepoint, or None if t is not present.
        """
        t = int(t)
        if t not in self._t_to_slice:
            return None
        s, e = self._t_to_slice[t]
        idx = self._order[s:e]
        return self.array[idx]


def _compute_vectors_generic(
    neighbor_indices,
    neighbor_distances,
    check_rows,
    no_z: bool,
    max_distance_um: float,
    xp,
):
    """
    Core vector-weighting logic, shared by CPU and GPU paths.

    Parameters
    ----------
    neighbor_indices : xp.ndarray, shape (G, K)
        Indices of neighbor flow landmarks for each query point.
    neighbor_distances : xp.ndarray, shape (G, K)
        Distances from query points to each neighbor.
    check_rows : xp.ndarray
        Flow rows for the current timepoint.
    no_z : bool
        Whether data is 2D (Y,X) or 3D (Z,Y,X).
    max_distance_um : float
        Maximum radius for valid neighbors (in microns).
    xp : module
        Either numpy or cupy.

    Returns
    -------
    xp.ndarray, shape (G, 2 or 3)
        Final interpolated vectors for each query point.
    """
    neighbor_distances = xp.asarray(neighbor_distances)
    neighbor_indices = xp.asarray(neighbor_indices)

    if neighbor_distances.ndim == 1:
        neighbor_distances = neighbor_distances[:, None]
        neighbor_indices = neighbor_indices[:, None]

    G, K = neighbor_distances.shape

    # Valid neighbors: finite and within radius.
    valid = xp.isfinite(neighbor_distances) & (neighbor_distances <= max_distance_um)
    has_valid = valid.any(axis=1)

    dim = 2 if no_z else 3
    if check_rows.shape[0] == 0:
        # No landmarks for this t.
        return xp.full((G, dim), xp.nan, dtype=xp.float32)

    # Set invalid neighbor indices to 0 (safe index); we'll zero their weights later.
    safe_indices = neighbor_indices.copy()
    safe_indices[~valid] = 0

    # Extract vectors and costs.
    vec_slice = slice(3, 5) if no_z else slice(4, 7)
    vectors = check_rows[safe_indices, vec_slice]  # (G, K, dim)
    costs = check_rows[safe_indices, -1]          # (G, K)

    # Cost weights: lower cost => higher weight.
    cost_weights = xp.where(valid, -costs, 0.0)

    # Distance weights: 1/d for d > 0, handled safely.
    d = xp.where(valid, neighbor_distances, xp.nan)
    dist_weights = xp.zeros_like(d)
    nonzero = (d > 0) & valid
    dist_weights[nonzero] = 1.0 / d[nonzero]

    # Initial weights.
    weights = cost_weights * dist_weights

    # Handle zero-distance neighbors specially: equal weights across exact matches.
    zero_mask = (d == 0) & valid
    weights_zero = xp.where(zero_mask, 1.0, 0.0)
    zero_counts = weights_zero.sum(axis=1, keepdims=True)
    has_zero = zero_counts > 0
    # Uniform weights over zero-distance neighbors.
    weights_zero = xp.where(
        zero_mask, 1.0 / xp.maximum(zero_counts, 1), 0.0
    )

    # Use zero-distance weights where applicable; otherwise use cost/distance weights.
    weights = xp.where(has_zero, weights_zero, weights)

    # Normalize weights for each query.
    sum_w = weights.sum(axis=1, keepdims=True)
    # Handle negative weights (from positive costs) by checking absolute sum
    weights = xp.where(xp.abs(sum_w) > 1e-12, weights / sum_w, 0.0)

    # Weighted average of neighbor vectors.
    weighted_vectors = vectors * weights[..., None]
    final_vectors = weighted_vectors.sum(axis=1)

    # Rows with no valid neighbors get NaN vectors.
    final_vectors = xp.where(has_valid[:, None], final_vectors, xp.nan)

    return final_vectors.astype(xp.float32, copy=False)


class FlowInterpolator:
    """
    Interpolates flow vectors between timepoints for microscopy images, with
    optimizations for large datasets and optional GPU acceleration.

    Key features
    ------------
    - Flow vectors are loaded via a FlowStore with memmap / in-memory control.
    - Time indexing is precomputed: O(N) upfront, O(n_t) per timepoint.
    - Neighbor search:
        * CPU: cKDTree with configurable max_neighbors and radius.
        * GPU: optional brute-force nearest neighbor with batching.
    - Vector weighting fully vectorized (no per-point Python loops).
    """

    def __init__(
        self,
        im_info: ImInfo,
        num_t: int | None = None,
        max_distance_um: float = 0.5,
        forward: bool = True,
        memory_mode: str = "auto",
        backend: str = "auto",
        max_neighbors: int = 32,
        coord_batch_size: int = 8192,
        gpu_batch_size: int = 2048,
    ):
        """
        Parameters
        ----------
        im_info : ImInfo
            Image metadata and paths.
        num_t : int or None
            Number of timepoints to process. If None, inferred from image.
        max_distance_um : float
            Maximum spatial distance (in micrometers) for neighbor consideration.
            Enforced as at least 0.5 µm.
        forward : bool
            If True, interpolate forward in time. If False, interpolate backward.
        memory_mode : {"auto", "memmap", "in_memory"}
            Controls how the flow array is loaded.
        backend : {"auto", "cpu", "gpu"}
            Preferred compute backend. "gpu" requires CuPy; "auto" tries GPU then
            falls back to CPU.
        max_neighbors : int
            Maximum number of nearest neighbors to use per interpolation query.
        coord_batch_size : int
            Number of coordinates per batch when interpolating (for memory control).
        gpu_batch_size : int
            Batch size for GPU neighbor search (controls GPU memory usage).
        """
        self.im_info = im_info

        # If no time dimension, nothing to do here.
        if self.im_info.no_t:
            logger.debug("Image has no time dimension; FlowInterpolator inactive.")
            return

        # Timepoints
        self.num_t = num_t
        if num_t is None and not self.im_info.no_t:
            self.num_t = im_info.shape[im_info.axes.index("T")]

        # Spatial scaling in microns per pixel.
        if self.im_info.no_z:
            self.scaling = np.asarray(
                [im_info.dim_res["Y"], im_info.dim_res["X"]], dtype=float
            )
        else:
            self.scaling = np.asarray(
                [im_info.dim_res["Z"], im_info.dim_res["Y"], im_info.dim_res["X"]],
                dtype=float,
            )

        # Spatial radius in microns (min 0.5).
        self.max_distance_um = float(max(max_distance_um, 0.5))

        self.forward = forward

        # Performance / memory configuration.
        self.memory_mode = memory_mode
        self.max_neighbors = int(max_neighbors)
        self.coord_batch_size = int(coord_batch_size)
        self.gpu_batch_size = int(gpu_batch_size)

        # Backend selection (CPU vs GPU).
        if backend == "gpu" and _HAS_CUPY:
            self.backend = "gpu"
            self._gpu_usable = True
        elif backend == "auto" and _HAS_CUPY:
            self.backend = "gpu"
            self._gpu_usable = True
        else:
            self.backend = "cpu"
            self._gpu_usable = False

        # Flow storage (set in _allocate_memory).
        self.flow_store: FlowStore | None = None
        self.flow_vector_array: np.ndarray | None = None

        # Per-timepoint cache.
        self.current_t: int | None = None
        self.check_rows_np: np.ndarray | None = None
        self.check_rows_gpu = None  # CuPy array, lazily allocated.
        self.check_coords: np.ndarray | None = None
        self.current_tree: cKDTree | None = None

        self.debug = None  # placeholder if you want to attach debug info externally

        self._initialize()

    def _allocate_memory(self):
        """
        Allocate and index the flow vector array.
        """
        logger.debug("Allocating memory / building index for flow vectors.")
        flow_vector_array_path = self.im_info.pipeline_paths["flow_vector_array"]
        self.flow_store = FlowStore(flow_vector_array_path, memory_mode=self.memory_mode)
        self.flow_vector_array = self.flow_store.array

    def _get_t(self):
        """
        Determine the number of timepoints if not explicitly set.
        """
        if self.num_t is None:
            if self.im_info.no_t:
                self.num_t = 1
            else:
                self.num_t = self.im_info.shape[self.im_info.axes.index("T")]

    def _ensure_tree_for_t(self, t: int):
        """
        Ensure we have per-timepoint flow data and a KDTree (for CPU backend).

        For forward interpolation:
            - Use rows with time == t; coordinates = (z,y,x) or (y,x).
        For backward interpolation:
            - Use rows with time == t-1; coordinates = (coords + vector) at t.
        """
        if self.current_t == t:
            return

        if self.flow_store is None:
            raise RuntimeError("FlowStore not initialized; call _initialize() first.")

        if self.forward:
            rows = self.flow_store.get_rows_for_t(t)
        else:
            rows = self.flow_store.get_rows_for_t(t - 1)

        if rows is None or len(rows) == 0:
            # No flow vectors for this t.
            self.check_rows_np = np.empty(
                (0, self.flow_vector_array.shape[1]), dtype=self.flow_vector_array.dtype
            )
            self.check_coords = np.empty((0, self.scaling.size), dtype=float)
            self.current_tree = None
            self.check_rows_gpu = None
            self.current_t = t
            return

        self.check_rows_np = np.asarray(rows)

        if self.im_info.no_z:
            if self.forward:
                coords_np = self.check_rows_np[:, 1:3]
            else:
                coords_np = self.check_rows_np[:, 1:3] + self.check_rows_np[:, 3:5]
        else:
            if self.forward:
                coords_np = self.check_rows_np[:, 1:4]
            else:
                coords_np = self.check_rows_np[:, 1:4] + self.check_rows_np[:, 4:7]

        self.check_coords = coords_np
        self.check_rows_gpu = None  # reset GPU copy for new t

        # Build KDTree for CPU backend (or when GPU is disabled).
        if self.backend == "cpu" or not self._gpu_usable or not _HAS_CUPY:
            if coords_np.shape[0] > 0:
                scaled = coords_np * self.scaling
                self.current_tree = cKDTree(scaled)
            else:
                self.current_tree = None
        else:
            # GPU backend: we don't need a KDTree; neighbor search is brute-force.
            self.current_tree = None

        self.current_t = t

    def _compute_vectors_from_neighbors(self, neighbor_indices, neighbor_distances, xp):
        """
        Wrapper around the generic vector-weighting function, providing
        the appropriate check_rows array (CPU or GPU).
        """
        if xp is np:
            check_rows = self.check_rows_np
        else:  # CuPy
            if self.check_rows_gpu is None:
                self.check_rows_gpu = xp.asarray(self.check_rows_np)
            check_rows = self.check_rows_gpu

        return _compute_vectors_generic(
            neighbor_indices,
            neighbor_distances,
            check_rows,
            self.im_info.no_z,
            self.max_distance_um,
            xp,
        )

    def _interpolate_cpu(self, coords_good: np.ndarray) -> np.ndarray:
        """
        CPU interpolation for a batch of valid coordinates (no NaNs in first column).
        """
        if (
            self.check_coords is None
            or self.check_coords.shape[0] == 0
            or self.current_tree is None
        ):
            # No source landmarks for this t.
            G, d = coords_good.shape
            return np.full((G, d), np.nan, dtype=np.float32)

        scaled_coords = coords_good * self.scaling

        distances, indices = self.current_tree.query(
            scaled_coords,
            k=self.max_neighbors,
            distance_upper_bound=self.max_distance_um,
            workers=-1,
        )

        distances = np.asarray(distances, dtype=float)
        indices = np.asarray(indices, dtype=np.int64)

        if distances.ndim == 1:
            distances = distances[:, None]
            indices = indices[:, None]

        final_vectors = self._compute_vectors_from_neighbors(indices, distances, np)
        return final_vectors

    def _interpolate_gpu(self, coords_good: np.ndarray):
        """
        GPU interpolation for a batch of valid coordinates using brute-force
        neighbor search with batching.

        Returns
        -------
        cp.ndarray
            Final vectors on GPU (float32).
        """
        if not (_HAS_CUPY and self._gpu_usable):
            raise RuntimeError("GPU backend not available.")

        xp = cp

        if self.check_coords is None or self.check_coords.shape[0] == 0:
            G, d = coords_good.shape
            return xp.full((G, d), xp.nan, dtype=xp.float32)

        try:
            scaling = xp.asarray(self.scaling, dtype=xp.float32)
            check_coords = xp.asarray(self.check_coords, dtype=xp.float32) * scaling
            coords_gpu = xp.asarray(coords_good, dtype=xp.float32) * scaling

            G = coords_gpu.shape[0]
            N = check_coords.shape[0]
            d = coords_gpu.shape[1]

            if N == 0:
                return xp.full((G, d), xp.nan, dtype=xp.float32)

            K = min(self.max_neighbors, N)

            indices = xp.empty((G, self.max_neighbors), dtype=xp.int32)
            distances = xp.empty((G, self.max_neighbors), dtype=xp.float32)

            batch_size = max(1, self.gpu_batch_size)

            for start in range(0, G, batch_size):
                end = min(start + batch_size, G)
                qb = coords_gpu[start:end]  # (B, d)

                # Pairwise distances: (B, N)
                diff = check_coords[None, :, :] - qb[:, None, :]
                d2 = xp.sum(diff * diff, axis=-1)

                k_use = min(K, N)
                idx_part = xp.argpartition(d2, k_use - 1, axis=1)[:, :k_use]
                part_vals = xp.take_along_axis(d2, idx_part, axis=1)

                order = xp.argsort(part_vals, axis=1)
                idx_sorted = xp.take_along_axis(idx_part, order, axis=1)
                dist_sorted = xp.sqrt(xp.take_along_axis(part_vals, order, axis=1))

                # Fill primary neighbors.
                indices[start:end, :k_use] = idx_sorted
                distances[start:end, :k_use] = dist_sorted

                # Filter out neighbors that are too far
                too_far = distances[start:end, :k_use] > self.max_distance_um
                if too_far.any():
                    # We need to be careful with array assignment in CuPy/NumPy
                    # Create a view for the current batch's valid part
                    dist_view = distances[start:end, :k_use]
                    idx_view = indices[start:end, :k_use]
                    
                    # Set too far distances to inf and indices to 0
                    dist_view[too_far] = xp.inf
                    idx_view[too_far] = 0

                # Mark remaining neighbors as invalid.
                if k_use < self.max_neighbors:
                    indices[start:end, k_use:] = 0
                    distances[start:end, k_use:] = xp.inf

            final_vectors_gpu = self._compute_vectors_from_neighbors(
                indices, distances, xp
            )
            return final_vectors_gpu

        except cp.cuda.memory.OutOfMemoryError:
            logger.warning(
                "GPU out-of-memory during flow interpolation; falling back to CPU."
            )
            cp.get_default_memory_pool().free_all_blocks()
            raise

    def _interpolate_coord_batch(self, coords_batch: np.ndarray) -> np.ndarray:
        """
        Interpolate vectors for a batch of coordinates (single timepoint), with
        optional GPU acceleration and robust fallback to CPU.
        """
        coords_batch = np.asarray(coords_batch, dtype=np.float32)
        B, d = coords_batch.shape

        final_vectors = np.full((B, d), np.nan, dtype=np.float32)

        # Good coords: first dimension not NaN.
        mask_good = ~np.isnan(coords_batch[:, 0])
        if not np.any(mask_good):
            return final_vectors

        coords_good = coords_batch[mask_good]

        # Try GPU (if enabled).
        if self.backend == "gpu" and self._gpu_usable and _HAS_CUPY:
            try:
                gpu_vecs = self._interpolate_gpu(coords_good)
                final_vectors[mask_good] = cp.asnumpy(gpu_vecs)
                return final_vectors
            except Exception:
                # Any GPU error: disable GPU for the rest of this run.
                logger.warning(
                    "Disabling GPU backend for flow interpolation due to error.",
                    exc_info=True,
                )
                self._gpu_usable = False

        # CPU fallback / default.
        cpu_vecs = self._interpolate_cpu(coords_good)
        final_vectors[mask_good] = cpu_vecs
        return final_vectors

    def interpolate_coord(self, coords: np.ndarray, t: int) -> np.ndarray:
        """
        Interpolate the flow vector at the given coordinates and timepoint.

        Parameters
        ----------
        coords : np.ndarray, shape (N, 2) or (N, 3)
            Input coordinates for interpolation (in pixel units).
        t : int
            Timepoint index.

        Returns
        -------
        np.ndarray, shape (N, 2) or (N, 3)
            Interpolated flow vectors at the given coordinates and timepoint.
        """
        if self.im_info.no_t:
            raise RuntimeError("FlowInterpolator is inactive for images without time.")

        self._ensure_tree_for_t(t)

        coords = np.asarray(coords, dtype=np.float32)
        N = coords.shape[0]
        if N == 0:
            return np.empty_like(coords)

        d = coords.shape[1]
        final_vectors = np.full((N, d), np.nan, dtype=np.float32)

        # Process coordinates in batches for memory control.
        for start in range(0, N, self.coord_batch_size):
            end = min(start + self.coord_batch_size, N)
            batch = coords[start:end]
            batch_vecs = self._interpolate_coord_batch(batch)
            final_vectors[start:end] = batch_vecs

        return final_vectors

    def _initialize(self):
        """
        Initialize the interpolator by setting timepoints and loading flow data.
        """
        if self.im_info.no_t:
            return
        self._get_t()
        self._allocate_memory()


def interpolate_all_forward(
    coords: np.ndarray,
    start_t: int,
    end_t: int,
    im_info: ImInfo,
    min_track_num: int = 0,
    max_distance_um: float = 0.5,
    memory_mode: str = "auto",
    backend: str = "auto",
    max_neighbors: int = 32,
):
    """
    Interpolate coordinates forward in time across multiple timepoints.

    Parameters
    ----------
    coords : np.ndarray, shape (N, 2) or (N, 3)
        Initial coordinates at time `start_t`.
    start_t : int
        Starting timepoint (inclusive).
    end_t : int
        Ending timepoint (exclusive). Frames [start_t, end_t) are processed.
    im_info : ImInfo
        Image metadata.
    min_track_num : int, optional
        Minimum track ID to assign.
    max_distance_um : float, optional
        Maximum spatial distance for interpolation (micrometers).
    memory_mode : {"auto", "memmap", "in_memory"}
        How to load the flow array.
    backend : {"auto", "cpu", "gpu"}
        Preferred compute backend.
    max_neighbors : int
        Maximum number of neighbors per interpolation.

    Returns
    -------
    tracks : list[list[float]]
        Track rows. Each row is [track_id, t, y, x] or [track_id, t, z, y, x].
    track_properties : dict
        Properties dict with at least 'frame_num'.
    """
    flow_interpx = FlowInterpolator(
        im_info,
        forward=True,
        max_distance_um=max_distance_um,
        memory_mode=memory_mode,
        backend=backend,
        max_neighbors=max_neighbors,
    )

    coords = np.asarray(coords, dtype=np.float32)
    tracks: list[list[float]] = []
    track_properties = {"frame_num": []}

    frame_range = np.arange(start_t, end_t)  # t, t+1, ..., end_t-1

    for t in frame_range:
        final_vector = flow_interpx.interpolate_coord(coords, int(t))
        if final_vector is None or len(final_vector) == 0:
            continue

        for coord_num, coord in enumerate(coords):
            vec = final_vector[coord_num]
            if np.all(np.isnan(vec)):
                coords[coord_num] = np.nan
                continue

            # Current coordinate at time t.
            if np.any(np.isnan(coord)):
                continue

            # Record initial point at first frame.
            if t == frame_range[0]:
                if im_info.no_z:
                    tracks.append(
                        [
                            coord_num + min_track_num,
                            int(t),
                            float(coord[0]),
                            float(coord[1]),
                        ]
                    )
                else:
                    tracks.append(
                        [
                            coord_num + min_track_num,
                            int(t),
                            float(coord[0]),
                            float(coord[1]),
                            float(coord[2]),
                        ]
                    )
                track_properties["frame_num"].append(int(t))

            # Advance coordinate to t+1.
            new_coord = coord + vec
            coords[coord_num] = new_coord

            if im_info.no_z:
                tracks.append(
                    [
                        coord_num + min_track_num,
                        int(t) + 1,
                        float(new_coord[0]),
                        float(new_coord[1]),
                    ]
                )
            else:
                tracks.append(
                    [
                        coord_num + min_track_num,
                        int(t) + 1,
                        float(new_coord[0]),
                        float(new_coord[1]),
                        float(new_coord[2]),
                    ]
                )
            track_properties["frame_num"].append(int(t) + 1)

    return tracks, track_properties


def interpolate_all_backward(
    coords: np.ndarray,
    start_t: int,
    end_t: int,
    im_info: ImInfo,
    min_track_num: int = 0,
    max_distance_um: float = 0.5,
    memory_mode: str = "auto",
    backend: str = "auto",
    max_neighbors: int = 32,
):
    """
    Interpolate coordinates backward in time across multiple timepoints.

    Parameters
    ----------
    coords : np.ndarray, shape (N, 2) or (N, 3)
        Initial coordinates at time `end_t`.
    start_t : int
        Earliest timepoint to reach (inclusive).
    end_t : int
        Latest timepoint (exclusive for the forward sense). Coordinates are assumed
        to be at frame `end_t`.
    im_info : ImInfo
        Image metadata.
    min_track_num : int, optional
        Minimum track ID to assign.
    max_distance_um : float, optional
        Maximum spatial distance for interpolation (micrometers).
    memory_mode : {"auto", "memmap", "in_memory"}
        How to load the flow array.
    backend : {"auto", "cpu", "gpu"}
        Preferred compute backend.
    max_neighbors : int
        Maximum number of neighbors per interpolation.

    Returns
    -------
    tracks : list[list[float]]
        Track rows. Each row is [track_id, t, y, x] or [track_id, t, z, y, x].
    track_properties : dict
        Properties dict with at least 'frame_num'.
    """
    flow_interpx = FlowInterpolator(
        im_info,
        forward=False,
        max_distance_um=max_distance_um,
        memory_mode=memory_mode,
        backend=backend,
        max_neighbors=max_neighbors,
    )

    coords = np.asarray(coords, dtype=np.float32)
    tracks: list[list[float]] = []
    track_properties = {"frame_num": []}

    # Iterate t from end_t down to start_t + 1.
    frame_range = list(range(int(end_t), int(start_t), -1))

    for t in frame_range:
        final_vector = flow_interpx.interpolate_coord(coords, int(t))
        if final_vector is None or len(final_vector) == 0:
            continue

        for coord_num, coord in enumerate(coords):
            vec = final_vector[coord_num]
            if np.all(np.isnan(vec)):
                coords[coord_num] = np.nan
                continue

            if np.any(np.isnan(coord)):
                continue

            # Record initial point at the first (latest) frame.
            if t == frame_range[0]:
                if im_info.no_z:
                    tracks.append(
                        [
                            coord_num + min_track_num,
                            int(t),
                            float(coord[0]),
                            float(coord[1]),
                        ]
                    )
                else:
                    tracks.append(
                        [
                            coord_num + min_track_num,
                            int(t),
                            float(coord[0]),
                            float(coord[1]),
                            float(coord[2]),
                        ]
                    )
                track_properties["frame_num"].append(int(t))

            # Move coordinate back to t-1.
            new_coord = coord - vec
            coords[coord_num] = new_coord

            if im_info.no_z:
                tracks.append(
                    [
                        coord_num + min_track_num,
                        int(t) - 1,
                        float(new_coord[0]),
                        float(new_coord[1]),
                    ]
                )
            else:
                tracks.append(
                    [
                        coord_num + min_track_num,
                        int(t) - 1,
                        float(new_coord[0]),
                        float(new_coord[1]),
                        float(new_coord[2]),
                    ]
                )
            track_properties["frame_num"].append(int(t) - 1)

    return tracks, track_properties


if __name__ == "__main__":
    # Example usage / visualization with napari.
    im_path = r"D:\test_files\nelly_smorgasbord\deskewed-iono_pre.ome.tif"
    im_info = ImInfo(im_path)

    label_memmap = im_info.get_memmap(im_info.pipeline_paths["im_instance_label"])
    im_memmap = im_info.get_memmap(im_info.im_path)

    import napari

    viewer = napari.Viewer()
    start_frame = 0

    # Example: select coordinates from first label frame.
    coords = np.argwhere(label_memmap[start_frame] > 0).astype(float)

    # Restrict to a spatial ROI for interactive speed.
    new_coords = []
    for coord in coords:
        # coord[-2] = y, coord[-1] = x for 2D
        if 450 < coord[-1] < 650 and 600 < coord[-2] < 750:
            new_coords.append(coord)
    coords = np.asarray(new_coords, dtype=float)

    tracks, track_properties = interpolate_all_forward(
        coords,
        start_frame,
        3,
        im_info,
        max_distance_um=0.5,
        backend="auto",
        max_neighbors=32,
    )

    viewer.add_image(im_memmap)
    viewer.add_tracks(tracks, properties=track_properties, name="tracks")