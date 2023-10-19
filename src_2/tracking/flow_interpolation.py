import tifffile

from src_2.io.im_info import ImInfo
from src import xp, ndi, logger
from src_2.utils.general import get_reshaped_image
import numpy as np
from scipy.spatial import cKDTree


class FlowInterpolation:
    def __init__(self, im_info: ImInfo, num_t=None, max_distance_um=0.5):
        self.im_info = im_info
        self.num_t = num_t
        if num_t is None:
            self.num_t = im_info.shape[im_info.axes.index('T')]
        self.scaling = (im_info.dim_sizes['Z'], im_info.dim_sizes['Y'], im_info.dim_sizes['X'])

        self.max_distance_um = max_distance_um

        # self.vector_start_coords = []
        # self.vectors = []
        # self.vector_magnitudes = []

        self.shape = ()

        self.im_memmap = None
        self.im_distance_memmap = None
        self.flow_vector_array = None

        self.debug = None

    def _allocate_memory(self):
        logger.debug('Allocating memory for mocap marking.')

        im_memmap = self.im_info.get_im_memmap(self.im_info.im_path)
        self.im_memmap = get_reshaped_image(im_memmap, self.num_t, self.im_info)

        im_distance_memmap = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_distance'])
        self.im_distance_memmap = get_reshaped_image(im_distance_memmap, self.num_t, self.im_info)
        self.shape = self.im_distance_memmap.shape

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

    def _get_nearby_coords(self, t, coord, check_coords):
        # coord_radius = self.im_distance_memmap[t, coord[0], coord[1], coord[2]]
        # if coord_radius == 0:
        #     logger.error(f'Selected voxel {coord} is not within the mask in temporal frame {t}.')
        #     return None
        # using a ckdtree, check for any nearby coords from coord
        tree = cKDTree(check_coords * self.scaling)
        scaled_coord = np.array(coord) * self.scaling
        # get all coords and distances within the radius of the coord
        nearby_idxs = tree.query_ball_point(scaled_coord, self.max_distance_um, p=2)
        if len(nearby_idxs) == 0:
            logger.error(f'No nearby flow vectors found for {coord} in temporal frame {t}.')
            return None, None
        distances = tree.query(scaled_coord, k=len(nearby_idxs), p=2)[0]
        return nearby_idxs, distances

    def _get_vector_weights(self, check_rows, nearby_idxs, distances):
        # lowest cost should be most highly weighted
        cost_weights = -check_rows[nearby_idxs, -1]

        # closest distance should be most highly weighted
        if np.min(distances) == 0:
            distance_weights = (distances == 0) * 1.0
        else:
            distance_weights = 1 / distances

        weights = cost_weights * distance_weights
        # weights = distance_weights#cost_weights * distance_weights
        weights -= np.min(weights) - 1
        weights /= np.sum(weights)
        return weights

    def _get_final_vector(self, coord, check_rows, nearby_idxs, weights):
        vectors = check_rows[nearby_idxs, 4:7]# + check_rows[nearby_idxs, 1:4] - coord
        if len(weights.shape) == 0:
            final_vector = vectors[0]
            return final_vector
        weighted_vectors = vectors * weights[:, None]
        final_vector = np.sum(weighted_vectors, axis=0)  # already normalized by weights
        return final_vector

    def interpolate_coord(self, coord, t, forward=True):
        # interpolate the flow vector at the coordinate at time t, either forward in time or backward in time.
        # For forward, simply find nearby LMPs, interpolate based on distance-weighted vectors
        # For backward, get coords from t-1 + vector, then find nearby coords from that, and interpolate based on distance-weighted vectors
        if forward:
            # check_rows will be all rows where the self.flow_vector_array's 0th column is equal to t+1
            check_rows = self.flow_vector_array[np.where(self.flow_vector_array[:, 0] == t)[0], :]
            check_coords = check_rows[:, 1:4]
        else:
            # check_rows will be all rows where the self.flow_vector_array's 0th columns is equal to t
            check_rows = self.flow_vector_array[np.where(self.flow_vector_array[:, 0] == t-1)[0], :]
            # check coords will be the coords + vector
            check_coords = check_rows[:, 1:4] + check_rows[:, 4:7]

        nearby_idxs, distances = self._get_nearby_coords(t, coord, check_coords)

        if nearby_idxs is None:
            logger.error(f'No nearby flow vectors found for {coord} in temporal frame {t}.')
            return None

        weights = self._get_vector_weights(check_rows, nearby_idxs, distances)
        final_vector = self._get_final_vector(coord, check_rows, nearby_idxs, weights)

        return final_vector


    def run(self):
        self._get_t()
        self._allocate_memory()


if __name__ == "__main__":
    import os
    import napari
    viewer = napari.Viewer()
    test_folder = r"D:\test_files\nelly_tests"
    all_files = os.listdir(test_folder)
    all_files = [file for file in all_files if not os.path.isdir(os.path.join(test_folder, file))]
    im_infos = []
    for file in all_files:
        im_path = os.path.join(test_folder, file)
        im_info = ImInfo(im_path)
        im_info.create_output_path('im_distance')
        im_info.create_output_path('flow_vector_array', ext='.npy')
        im_infos.append(im_info)

    flow_interpx = FlowInterpolation(im_infos[0])
    flow_interpx.run()
    viewer.add_image(flow_interpx.im_memmap[:3])

    test_label = tifffile.memmap(r"D:\test_files\nelly_tests\output\deskewed-2023-07-13_14-58-28_000_wt_0_acquire.ome-ch0-im_skel.ome.tif", mode='r')
    test_skel = tifffile.memmap(r"D:\test_files\nelly_tests\output\deskewed-2023-07-13_14-58-28_000_wt_0_acquire.ome-ch0-im_instance_label.ome.tif", mode='r')
    coords_skel = np.argwhere(test_skel[0] == 108)
    coords_label = np.argwhere(test_label[0] == 137)
    # coords = np.argwhere(test_im[0] > 0)
    # # get 100 random coords
    # np.random.seed(0)
    # coords = coords[np.random.choice(coords.shape[0], 100, replace=False), :]

    tracks = []
    track_properties = {'frame_num': []}
    for track_num, coord in enumerate(coords_skel):
        tracks.append([track_num, 0, coord[0], coord[1], coord[2]])
        track_properties['frame_num'].append(0)
        for t in range(4):
            final_vector = flow_interpx.interpolate_coord(coord, t, forward=True)
            if final_vector is None:
                break
            track_properties['frame_num'].append(t+1)
            coord = (coord[0] + final_vector[0], coord[1] + final_vector[1], coord[2] + final_vector[2])
            tracks.append([track_num, t+1, coord[0], coord[1], coord[2]])
    viewer.add_tracks(tracks, properties = track_properties, name='tracks')
    tracks = []
    track_properties = {'frame_num': []}
    for track_num, coord in enumerate(coords_label):
        tracks.append([track_num, 0, coord[0], coord[1], coord[2]])
        track_properties['frame_num'].append(0)
        for t in range(4):
            final_vector = flow_interpx.interpolate_coord(coord, t, forward=True)
            if final_vector is None:
                break
            track_properties['frame_num'].append(t+1)
            coord = (coord[0] + final_vector[0], coord[1] + final_vector[1], coord[2] + final_vector[2])
            tracks.append([track_num, t+1, coord[0], coord[1], coord[2]])
    viewer.add_tracks(tracks, properties = track_properties, name='tracks')

    print('done')

