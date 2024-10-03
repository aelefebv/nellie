import numpy as np
from scipy.spatial import cKDTree

from nellie import logger
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

        self.max_distance_um = max_distance_um * im_info.dim_res['T']
        self.max_distance_um = np.max(np.array([self.max_distance_um, 0.5]))

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
        # using a ckdtree, check for any nearby coords from coord
        if self.current_t != t:
            self.current_tree = cKDTree(self.check_coords * self.scaling)
        scaled_coords = np.array(coords) * self.scaling
        # get all coords and distances within the radius of the coord
        # good coords are non-nan
        good_coords = np.where(~np.isnan(scaled_coords[:, 0]))[0]
        nearby_idxs = self.current_tree.query_ball_point(scaled_coords[good_coords], self.max_distance_um, p=2)
        if len(nearby_idxs) == 0:
            return [], []
        k_all = [len(nearby_idxs[i]) for i in range(len(nearby_idxs))]
        max_k = np.max(k_all)
        if max_k == 0:
            return [], []
        distances, nearby_idxs = self.current_tree.query(scaled_coords[good_coords], k=max_k, p=2, workers=-1)
        # if the first index is scalar, wrap the whole list in another list
        if len(distances.shape) == 1:
            distances = [distances]
            nearby_idxs = [nearby_idxs]
        distance_return = [[] for _ in range(len(coords))]
        nearby_idxs_return = [[] for _ in range(len(coords))]
        pos = 0
        for i in range(len(distances)):
            if i not in good_coords:
                continue
            distance_return[i] = distances[pos][:k_all[pos]]
            nearby_idxs_return[i] = nearby_idxs[pos][:k_all[pos]]
            pos += 1
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
            if self.forward:
                # check_rows will be all rows where the self.flow_vector_array's 0th column is equal to t
                self.check_rows = self.flow_vector_array[np.where(self.flow_vector_array[:, 0] == t)[0], :]
                if self.im_info.no_z:
                    self.check_coords = self.check_rows[:, 1:3]
                else:
                    self.check_coords = self.check_rows[:, 1:4]
            else:
                # check_rows will be all rows where the self.flow_vector_array's 0th columns is equal to t-1
                self.check_rows = self.flow_vector_array[np.where(self.flow_vector_array[:, 0] == t - 1)[0], :]
                # check coords will be the coords + vector
                if self.im_info.no_z:
                    self.check_coords = self.check_rows[:, 1:3] + self.check_rows[:, 3:5]
                else:
                    self.check_coords = self.check_rows[:, 1:4] + self.check_rows[:, 4:7]

        nearby_idxs, distances_all = self._get_nearby_coords(t, coords)
        self.current_t = t

        if nearby_idxs is None:
            return None

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
    flow_interpx = FlowInterpolator(im_info, forward=True, max_distance_um=max_distance_um)
    tracks = []
    track_properties = {'frame_num': []}
    frame_range = np.arange(start_t, end_t)
    for t in frame_range:
        final_vector = flow_interpx.interpolate_coord(coords, t)
        if final_vector is None or len(final_vector) == 0:
            continue
        for coord_num, coord in enumerate(coords):
            if np.all(np.isnan(final_vector[coord_num])):
                coords[coord_num] = np.nan
                continue
            if t == frame_range[0]:
                if im_info.no_z:
                    tracks.append([coord_num + min_track_num, frame_range[0], coord[0], coord[1]])
                else:
                    tracks.append([coord_num + min_track_num, frame_range[0], coord[0], coord[1], coord[2]])
                track_properties['frame_num'].append(frame_range[0])

            track_properties['frame_num'].append(t + 1)
            if im_info.no_z:
                coords[coord_num] = np.array([coord[0] + final_vector[coord_num][0],
                                              coord[1] + final_vector[coord_num][1]])
                tracks.append([coord_num + min_track_num, t + 1, coord[0], coord[1]])
            else:
                coords[coord_num] = np.array([coord[0] + final_vector[coord_num][0],
                                              coord[1] + final_vector[coord_num][1],
                                              coord[2] + final_vector[coord_num][2]])
                tracks.append([coord_num + min_track_num, t + 1, coord[0], coord[1], coord[2]])
    return tracks, track_properties


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
    flow_interpx = FlowInterpolator(im_info, forward=False, max_distance_um=max_distance_um)
    tracks = []
    track_properties = {'frame_num': []}
    frame_range = list(np.arange(end_t, start_t + 1))[::-1]
    for t in frame_range:
        final_vector = flow_interpx.interpolate_coord(coords, t)
        if final_vector is None or len(final_vector) == 0:
            continue
        for coord_num, coord in enumerate(coords):
            # if final_vector[coord_num] is all nan, skip
            if np.all(np.isnan(final_vector[coord_num])):
                coords[coord_num] = np.nan
                continue
            if t == frame_range[0]:
                if im_info.no_z:
                    tracks.append([coord_num + min_track_num, frame_range[0], coord[0], coord[1]])
                else:
                    tracks.append([coord_num + min_track_num, frame_range[0], coord[0], coord[1], coord[2]])
                track_properties['frame_num'].append(frame_range[0])
            if im_info.no_z:
                coords[coord_num] = np.array([coord[0] - final_vector[coord_num][0],
                                              coord[1] - final_vector[coord_num][1]])
                tracks.append([coord_num + min_track_num, t - 1, coord[0], coord[1]])
            else:
                coords[coord_num] = np.array([coord[0] - final_vector[coord_num][0],
                                              coord[1] - final_vector[coord_num][1],
                                              coord[2] - final_vector[coord_num][2]])
                tracks.append([coord_num + min_track_num, t - 1, coord[0], coord[1], coord[2]])
            track_properties['frame_num'].append(t - 1)
    return tracks, track_properties


if __name__ == "__main__":
    im_path = r"D:\test_files\nelly_smorgasbord\deskewed-iono_pre.ome.tif"
    im_info = ImInfo(im_path)
    label_memmap = self.im_info.get_memmap(im_info.pipeline_paths['im_instance_label'])
    im_memmap = self.im_info.get_memmap(im_info.im_path)

    import napari
    viewer = napari.Viewer()
    start_frame = 0
    # going backwards
    coords = np.argwhere(label_memmap[0] > 0).astype(float)
    # get 100 random coords
    # np.random.seed(0)
    # coords = coords[np.random.choice(coords.shape[0], 10000, replace=False), :].astype(float)
    # x in range 450-650
    # y in range 600-750
    new_coords = []
    for coord in coords:
        if 450 < coord[-1] < 650 and 600 < coord[-2] < 750:
            new_coords.append(coord)
    coords = np.array(new_coords[::1])
    tracks, track_properties = interpolate_all_forward(coords, start_frame, 3, im_info)

    viewer.add_image(im_memmap)
    viewer.add_tracks(tracks, properties=track_properties, name='tracks')
