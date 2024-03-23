import numpy as np
from scipy.spatial import cKDTree

from nellie import logger
from nellie.im_info.im_info import ImInfo
from nellie.utils.general import get_reshaped_image


class FlowInterpolator:
    def __init__(self, im_info: ImInfo, num_t=None, max_distance_um=0.5, forward=True):
        self.im_info = im_info
        self.num_t = num_t
        if num_t is None and not self.im_info.no_t:
            self.num_t = im_info.shape[im_info.axes.index('T')]

        if self.im_info.no_z:
            self.scaling = (im_info.dim_sizes['Y'], im_info.dim_sizes['X'])
        else:
            self.scaling = (im_info.dim_sizes['Z'], im_info.dim_sizes['Y'], im_info.dim_sizes['X'])

        self.max_distance_um = max_distance_um
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
        logger.debug('Allocating memory for mocap marking.')

        im_memmap = self.im_info.get_im_memmap(self.im_info.im_path)
        self.im_memmap = get_reshaped_image(im_memmap, self.num_t, self.im_info)
        self.shape = self.im_memmap.shape

        flow_vector_array_path = self.im_info.pipeline_paths['flow_vector_array']
        self.flow_vector_array = np.load(flow_vector_array_path)

    def _get_t(self):
        if self.num_t is None:
            if self.im_info.no_t:
                self.num_t = 1
            else:
                self.num_t = self.im_info.shape[self.im_info.axes.index('T')]
        else:
            return

    def _get_nearby_coords(self, t, coords):
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
        if self.im_info.no_t:
            return
        self._get_t()
        self._allocate_memory()


def interpolate_all_forward(coords, start_t, end_t, im_info, min_track_num=0, max_distance_um=0.5):
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
    label_memmap = im_info.get_im_memmap(im_info.pipeline_paths['im_instance_label'])
    label_memmap = get_reshaped_image(label_memmap, im_info=im_info)
    im_memmap = im_info.get_im_memmap(im_info.im_path)
    im_memmap = get_reshaped_image(im_memmap, im_info=im_info)

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
