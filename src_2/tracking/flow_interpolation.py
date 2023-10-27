import tifffile

from src_2.im_info.im_info import ImInfo
from src import xp, ndi, logger
from src_2.utils.general import get_reshaped_image
import numpy as np
from scipy.spatial import cKDTree


class FlowInterpolator:
    def __init__(self, im_info: ImInfo, num_t=None, max_distance_um=0.5, forward=True):
        self.im_info = im_info
        self.num_t = num_t
        if num_t is None:
            self.num_t = im_info.shape[im_info.axes.index('T')]
        self.scaling = (im_info.dim_sizes['Z'], im_info.dim_sizes['Y'], im_info.dim_sizes['X'])

        self.max_distance_um = max_distance_um
        self.forward = forward

        # self.vector_start_coords = []
        # self.vectors = []
        # self.vector_magnitudes = []

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
        # coord_radius = self.im_distance_memmap[t, coord[0], coord[1], coord[2]]
        # if coord_radius == 0:
        #     logger.error(f'Selected voxel {coord} is not within the mask in temporal frame {t}.')
        #     return None
        # using a ckdtree, check for any nearby coords from coord
        if self.current_t != t:
            self.current_tree = cKDTree(self.check_coords * self.scaling)
        scaled_coords = np.array(coords) * self.scaling
        # get all coords and distances within the radius of the coord
        nearby_idxs = self.current_tree.query_ball_point(scaled_coords, self.max_distance_um, p=2)
        # if len(nearby_idxs) == 0:
        #     logger.debug(f'No nearby flow vectors found for {coord} in temporal frame {t}.')
        #     return None, None
        k_all = [len(nearby_idxs[i]) for i in range(len(nearby_idxs))]
        max_k = np.max(k_all)
        if max_k == 0:
            return None, None
        distances, nearby_idxs = self.current_tree.query(scaled_coords, k=max_k, p=2, workers=-1)
        distances = [distances[i][:k_all[i]] for i in range(len(distances))]
        nearby_idxs = [nearby_idxs[i][:k_all[i]] for i in range(len(nearby_idxs))]
        return nearby_idxs, distances

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
            # weights = distance_weights#cost_weights * distance_weights
            weights -= np.min(weights) - 1
            weights /= np.sum(weights)
            weights_all.append(weights)
        return weights_all

    def _get_final_vector(self, nearby_idxs, weights_all):
        final_vectors = np.zeros((len(nearby_idxs), 3))
        for i in range(len(nearby_idxs)):
            if weights_all[i] is None:
                final_vectors[i] = np.nan
                continue
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
                self.check_coords = self.check_rows[:, 1:4]
            else:
                # check_rows will be all rows where the self.flow_vector_array's 0th columns is equal to t-1
                self.check_rows = self.flow_vector_array[np.where(self.flow_vector_array[:, 0] == t-1)[0], :]
                # check coords will be the coords + vector
                self.check_coords = self.check_rows[:, 1:4] + self.check_rows[:, 4:7]

        nearby_idxs, distances_all = self._get_nearby_coords(t, coords)
        self.current_t = t

        if nearby_idxs is None:
            return None

        weights_all = self._get_vector_weights(nearby_idxs, distances_all)
        final_vectors = self._get_final_vector(nearby_idxs, weights_all)

        return final_vectors

    def _initialize(self):
        self._get_t()
        self._allocate_memory()


if __name__ == "__main__":
    import os
    import napari
    viewer = napari.Viewer()
    # test_folder = r"D:\test_files\nelly_tests"
    # test_skel = tifffile.memmap(r"D:\test_files\nelly_tests\output\deskewed-2023-07-13_14-58-28_000_wt_0_acquire.ome-ch0-im_skel.ome.tif", mode='r')
    # test_label = tifffile.memmap(r"D:\test_files\nelly_tests\output\deskewed-2023-07-13_14-58-28_000_wt_0_acquire.ome-ch0-im_instance_label.ome.tif", mode='r')
    # test_folder = r"D:\test_files\beading"
    # test_skel = tifffile.memmap(r"D:\test_files\beading\output\deskewed-single.ome-ch0-im_skel.ome.tif", mode='r')
    # test_label = tifffile.memmap(r"D:\test_files\beading\output\deskewed-single.ome-ch0-im_instance_label.ome.tif", mode='r')

    im_path = r"D:\test_files\stress_granules\deskewed-2023-04-13_17-34-08_000_AELxES-stress_granules-dmr_perk-activate_deactivate-1nM-activate.ome.tif"
    im_info = ImInfo(im_path)
    im_info.create_output_path('im_instance_label')
    im_info.create_output_path('flow_vector_array', ext='.npy')
    label_memmap = im_info.get_im_memmap(im_info.pipeline_paths['im_instance_label'])
    label_memmap = get_reshaped_image(label_memmap, im_info=im_info)

    # all_files = os.listdir(test_folder)
    # all_files = [file for file in all_files if not os.path.isdir(os.path.join(test_folder, file))]
    # im_infos = []
    # for file in all_files:
    #     im_path = os.path.join(test_folder, file)
    #     im_info = ImInfo(im_path)
    #     im_info.create_output_path('flow_vector_array', ext='.npy')
    #     im_infos.append(im_info)

    flow_interpx = FlowInterpolator(im_info, forward=True)
    # flow_interpx.run()
    num_frames = flow_interpx.im_memmap.shape[0]

    # going backwards
    coords = np.argwhere(label_memmap[num_frames-1] > 0).astype(float)
    # get 100 random coords
    # np.random.seed(0)
    # coords = coords[np.random.choice(coords.shape[0], 10000, replace=False), :].astype(float)
    tracks = []
    track_properties = {'frame_num': []}
    frame_range = np.arange(num_frames)[:-1]
    for t in frame_range:
        print(f'Interpolating frame {t} of {num_frames-1}')
        final_vector = flow_interpx.interpolate_coord(coords, t)
        for coord_num, coord in enumerate(coords):
            # if final_vector[coord_num] is all nan, skip
            if np.all(np.isnan(final_vector[coord_num])):
                coords[coord_num] = np.nan
                continue
            if t == frame_range[0]:
                tracks.append([coord_num, frame_range[0], coord[0], coord[1], coord[2]])
                track_properties['frame_num'].append(frame_range[0])
            track_properties['frame_num'].append(t + 1)
            coords[coord_num] = np.array([coord[0] + final_vector[coord_num][0],
                                          coord[1] + final_vector[coord_num][1],
                                          coord[2] + final_vector[coord_num][2]])
            tracks.append([coord_num, t + 1, coord[0], coord[1], coord[2]])

    viewer.add_image(flow_interpx.im_memmap)
    viewer.add_tracks(tracks, properties=track_properties, name='tracks')

    # flow_interpx = FlowInterpolator(im_infos[0])
    # # going forwards
    # coords_skel = np.argwhere(test_skel[0] == 80)
    # coords_label = np.argwhere(test_label[0] == 651)
    # coords = np.argwhere(test_label[0] > 0)
    # # get 100 random coords
    # np.random.seed(0)
    # coords = coords[np.random.choice(coords.shape[0], 100, replace=False), :]
    #
    # tracks = []
    # track_properties = {'frame_num': []}
    # for track_num, coord in enumerate(coords):
    #     tracks.append([track_num, 0, coord[0], coord[1], coord[2]])
    #     track_properties['frame_num'].append(0)
    #     for t in range(num_frames):
    #         final_vector = flow_interpx.interpolate_coord(coord, t)
    #         if final_vector is None:
    #             break
    #         track_properties['frame_num'].append(t+1)
    #         coord = (coord[0] + final_vector[0], coord[1] + final_vector[1], coord[2] + final_vector[2])
    #         tracks.append([track_num, t+1, coord[0], coord[1], coord[2]])
    # viewer.add_tracks(tracks, properties = track_properties, name='tracks')
    # tracks = []
    # track_properties = {'frame_num': []}
    # for track_num, coord in enumerate(coords_skel):
    #     tracks.append([track_num, 0, coord[0], coord[1], coord[2]])
    #     track_properties['frame_num'].append(0)
    #     for t in range(num_frames):
    #         final_vector = flow_interpx.interpolate_coord(coord, t)
    #         if final_vector is None:
    #             break
    #         track_properties['frame_num'].append(t+1)
    #         coord = (coord[0] + final_vector[0], coord[1] + final_vector[1], coord[2] + final_vector[2])
    #         tracks.append([track_num, t+1, coord[0], coord[1], coord[2]])
    # viewer.add_tracks(tracks, properties = track_properties, name='tracks')
    # tracks = []
    # track_properties = {'frame_num': []}
    # for track_num, coord in enumerate(coords_label):
    #     tracks.append([track_num, 0, coord[0], coord[1], coord[2]])
    #     track_properties['frame_num'].append(0)
    #     for t in range(num_frames):
    #         final_vector = flow_interpx.interpolate_coord(coord, t)
    #         if final_vector is None:
    #             break
    #         track_properties['frame_num'].append(t+1)
    #         coord = (coord[0] + final_vector[0], coord[1] + final_vector[1], coord[2] + final_vector[2])
    #         tracks.append([track_num, t+1, coord[0], coord[1], coord[2]])
    # viewer.add_tracks(tracks, properties = track_properties, name='tracks')
    #
    # print('done')

