# features I want:
# for each skeleton node:
# - local intensity CoV
# - local frangi CoV
# - thickness at that point
# - mean intensity?
# - mean frangi?
# - median, max, std, min of 'rel_ang_vel_mag_12', 'rel_ang_acc_mag',
#                                'rel_lin_vel_mag_12', 'rel_lin_acc_mag',
from src import logger
from src.im_info.im_info import ImInfo
from src.utils.general import get_reshaped_image
import numpy as np
import networkx as nx
import scipy.ndimage as ndi
from scipy.spatial import cKDTree

class Tree:
    def __init__(self, label: int, voxel_idxs: np.ndarray):
        self.label = label
        self.voxel_idxs = voxel_idxs
        self.neighbors = []
        self.start_node = None
        self.jump_distances = None
        self.nodelists = None
        self.direct_connections = None

    def get_neighbors(self):
        ckdtree = cKDTree(self.voxel_idxs)
        self.neighbors = ckdtree.query_ball_point(self.voxel_idxs, r=1.74)  # a little over sqrt(3)
        self.neighbors = [np.array(neighbor) for neighbor in self.neighbors]
        self.neighbors = [neighbor[neighbor != i] for i, neighbor in enumerate(self.neighbors)]

    def get_start_node(self):
        # pick the first node with only one neighbor. If none exists, pick the first node
        for i, neighbor in enumerate(self.neighbors):
            if len(neighbor) == 1:
                self.start_node = i
                return
        self.start_node = 0

    def calculate_jump_distances(self):
        self.jump_distances = np.full(len(self.voxel_idxs), np.inf)
        self.jump_distances[self.start_node] = 0

        stack = [(self.start_node, 0)]
        while stack:
            node, current_distance = stack.pop()
            for neighbor in self.neighbors[node]:
                if self.jump_distances[neighbor] == np.inf:  # Unvisited
                    self.jump_distances[neighbor] = current_distance + 1
                    stack.append((neighbor, current_distance + 1))

    def generate_scale_nodelists(self, max_scale):
        self.nodelists = [list(range(len(self.jump_distances)))]  # All nodes for scale 0
        for scale in range(2, max_scale + 1):
            skip = 2 ** (scale - 1)
            valid_nodes = [i for i, dist in enumerate(self.jump_distances) if dist % skip == 0]
            self.nodelists.append(valid_nodes)

    def generate_direct_connections(self):
        self.direct_connections = []

        for scale_nodelist in self.nodelists:
            connections = set()
            for node in scale_nodelist:
                for direction in [-1, 1]:  # Backward and forward
                    current_node = node
                    while True:
                        neighbors = [n for n in self.neighbors[current_node] if n in scale_nodelist]
                        if not neighbors:
                            break
                        next_node = neighbors[0] if direction == 1 else neighbors[-1]
                        if next_node == node or next_node in scale_nodelist:
                            connections.add((min(node, next_node), max(node, next_node)))
                            break
                        current_node = next_node
            self.direct_connections.append(connections)


class GraphBuilder:
    def __init__(self, im_info: ImInfo,
                 t=1):
        self.im_info = im_info
        self.t = t
        self.shape = None
        if self.im_info.no_z:
            self.spacing = (self.im_info.dim_sizes['Y'], self.im_info.dim_sizes['X'])
        else:
            self.spacing = (self.im_info.dim_sizes['Z'], self.im_info.dim_sizes['Y'], self.im_info.dim_sizes['X'])

        self.im_memmap = None
        self.label_memmap = None
        self.pixel_class = None
        self.distance_memmap = None
        self.preproc_memmap = None
        self.flow_vector_array = None

        self.features = {}
        self.trees = []

    def _get_memmaps(self):
        logger.debug('Allocating memory for spatial feature extraction.')

        num_t = self.im_info.shape[self.im_info.axes.index('T')]
        if num_t == 1:
            self.t = 0

        im_memmap = self.im_info.get_im_memmap(self.im_info.im_path)
        self.im_memmap = get_reshaped_image(im_memmap, num_t, self.im_info)

        pixel_class = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_pixel_class'])
        self.pixel_class = get_reshaped_image(pixel_class, num_t, self.im_info)

        label_memmap = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_instance_label'])
        self.label_memmap = get_reshaped_image(label_memmap, num_t, self.im_info)

        distance_memmap = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_distance'])
        self.distance_memmap = get_reshaped_image(distance_memmap, num_t, self.im_info)

        preproc_memmap = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_frangi'])
        self.preproc_memmap = get_reshaped_image(preproc_memmap, num_t, self.im_info)

        rel_ang_vel_mag_12 = self.im_info.get_im_memmap(self.im_info.pipeline_paths['rel_ang_vel_mag_12'])
        self.rel_ang_vel_mag_12 = get_reshaped_image(rel_ang_vel_mag_12, num_t, self.im_info)
        rel_lin_vel_mag_12 = self.im_info.get_im_memmap(self.im_info.pipeline_paths['rel_lin_vel_mag_12'])
        self.rel_lin_vel_mag_12 = get_reshaped_image(rel_lin_vel_mag_12, num_t, self.im_info)
        rel_ang_acc_mag = self.im_info.get_im_memmap(self.im_info.pipeline_paths['rel_ang_acc_mag'])
        self.rel_ang_acc_mag = get_reshaped_image(rel_ang_acc_mag, num_t, self.im_info)
        rel_lin_acc_mag = self.im_info.get_im_memmap(self.im_info.pipeline_paths['rel_lin_acc_mag'])
        self.rel_lin_acc_mag = get_reshaped_image(rel_lin_acc_mag, num_t, self.im_info)

        flow_vector_array_path = self.im_info.pipeline_paths['flow_vector_array']
        self.flow_vector_array = np.load(flow_vector_array_path)

        if not self.im_info.no_t:
            self.im_memmap = self.im_memmap[self.t]
            self.pixel_class = self.pixel_class[self.t]
            self.label_memmap = self.label_memmap[self.t]
            self.distance_memmap = self.distance_memmap[self.t]
            self.preproc_memmap = self.preproc_memmap[self.t]
            self.rel_ang_vel_mag_12 = self.rel_ang_vel_mag_12[self.t]
            self.rel_lin_vel_mag_12 = self.rel_lin_vel_mag_12[self.t]
            self.rel_ang_acc_mag = self.rel_ang_acc_mag[self.t]
            self.rel_lin_acc_mag = self.rel_lin_acc_mag[self.t]

        self.shape = self.pixel_class.shape

    def _build_graph(self):
        mask = self.label_memmap > 0
        masked_raw = self.im_memmap * mask
        masked_preproc = self.preproc_memmap * mask
        # get all network pixels
        skeleton_pixels = np.argwhere(self.pixel_class > 0)
        skeleton_radius = self.distance_memmap[skeleton_pixels[:, 0], skeleton_pixels[:, 1], skeleton_pixels[:, 2]]
        largest_thickness = int(np.ceil(np.max(skeleton_radius * 2))) + 1
        # create bounding boxes of size largest_thickness around each skeleton pixel
        vals_raw = np.zeros((len(skeleton_pixels), largest_thickness, largest_thickness, largest_thickness))
        vals_preproc = np.zeros((len(skeleton_pixels), largest_thickness, largest_thickness, largest_thickness))
        vals_rel_ang_acc = np.zeros((len(skeleton_pixels), largest_thickness, largest_thickness, largest_thickness))
        vals_rel_lin_acc = np.zeros((len(skeleton_pixels), largest_thickness, largest_thickness, largest_thickness))
        vals_rel_ang_vel = np.zeros((len(skeleton_pixels), largest_thickness, largest_thickness, largest_thickness))
        vals_rel_lin_vel = np.zeros((len(skeleton_pixels), largest_thickness, largest_thickness, largest_thickness))
        # for each pixel, get the surrounding im_memmap values based on its specific radius, then add to bounding_boxes
        z_lims = (skeleton_radius[:, np.newaxis] * np.array([-1, 1]) + skeleton_pixels[:, 0, np.newaxis]).astype(int)
        z_lims[:, 1] += 1
        y_lims = (skeleton_radius[:, np.newaxis] * np.array([-1, 1]) + skeleton_pixels[:, 1, np.newaxis]).astype(int)
        y_lims[:, 1] += 1
        x_lims = (skeleton_radius[:, np.newaxis] * np.array([-1, 1]) + skeleton_pixels[:, 2, np.newaxis]).astype(int)
        x_lims[:, 1] += 1

        z_lims[z_lims < 0] = 0
        y_lims[y_lims < 0] = 0
        x_lims[x_lims < 0] = 0

        z_lims[z_lims > self.shape[0]] = self.shape[0]
        y_lims[y_lims > self.shape[1]] = self.shape[1]
        x_lims[x_lims > self.shape[2]] = self.shape[2]

        for i, (z_lim, y_lim, x_lim) in enumerate(zip(z_lims, y_lims, x_lims)):
            z_range = z_lim[1] - z_lim[0]
            y_range = y_lim[1] - y_lim[0]
            x_range = x_lim[1] - x_lim[0]
            vals_raw[i, :z_range, :y_range, :x_range] = masked_raw[z_lim[0]:z_lim[1], y_lim[0]:y_lim[1], x_lim[0]:x_lim[1]]
            vals_preproc[i, :z_range, :y_range, :x_range] = masked_preproc[z_lim[0]:z_lim[1], y_lim[0]:y_lim[1], x_lim[0]:x_lim[1]]
            vals_rel_ang_acc[i, :z_range, :y_range, :x_range] = self.rel_ang_acc_mag[z_lim[0]:z_lim[1], y_lim[0]:y_lim[1], x_lim[0]:x_lim[1]]
            vals_rel_lin_acc[i, :z_range, :y_range, :x_range] = self.rel_lin_acc_mag[z_lim[0]:z_lim[1], y_lim[0]:y_lim[1], x_lim[0]:x_lim[1]]
            vals_rel_ang_vel[i, :z_range, :y_range, :x_range] = self.rel_ang_vel_mag_12[z_lim[0]:z_lim[1], y_lim[0]:y_lim[1], x_lim[0]:x_lim[1]]
            vals_rel_lin_vel[i, :z_range, :y_range, :x_range] = self.rel_lin_vel_mag_12[z_lim[0]:z_lim[1], y_lim[0]:y_lim[1], x_lim[0]:x_lim[1]]

        vals_preproc_log10 = np.log10(vals_preproc)
        vals_preproc_log10[vals_preproc_log10 == -np.inf] = np.nan
        vals_preproc_log10[vals_preproc_log10 == np.inf] = np.nan

        vals_preproc_log10[vals_preproc_log10 == 0] = np.nan
        vals_raw[vals_raw == 0] = np.nan
        vals_rel_ang_acc[vals_rel_ang_acc == 0] = np.nan
        vals_rel_lin_acc[vals_rel_lin_acc == 0] = np.nan
        vals_rel_ang_vel[vals_rel_ang_vel == 0] = np.nan
        vals_rel_lin_vel[vals_rel_lin_vel == 0] = np.nan

        self.features['thickness'] = skeleton_radius * 2
        self.features['raw_median'] = np.nanmedian(vals_raw, axis=(1, 2, 3))
        self.features['raw_max'] = np.nanmax(vals_raw, axis=(1, 2, 3))
        self.features['raw_min'] = np.nanmin(vals_raw, axis=(1, 2, 3))
        self.features['raw_CoV'] = np.nanstd(vals_raw, axis=(1, 2, 3)) / np.nanmean(vals_raw, axis=(1, 2, 3))
        self.features['struc_median'] = np.nanmedian(vals_preproc_log10, axis=(1, 2, 3))
        self.features['struc_max'] = np.nanmax(vals_preproc_log10, axis=(1, 2, 3))
        self.features['struc_min'] = np.nanmin(vals_preproc_log10, axis=(1, 2, 3))
        self.features['struc_CoV'] = np.nanstd(vals_preproc_log10, axis=(1, 2, 3)) / np.nanmean(vals_preproc_log10, axis=(1, 2, 3))
        self.features['rel_ang_acc_median'] = np.nanmedian(vals_rel_ang_acc, axis=(1, 2, 3))
        self.features['rel_ang_acc_max'] = np.nanmax(vals_rel_ang_acc, axis=(1, 2, 3))
        self.features['rel_ang_acc_min'] = np.nanmin(vals_rel_ang_acc, axis=(1, 2, 3))
        self.features['rel_ang_acc_CoV'] = np.nanstd(vals_rel_ang_acc, axis=(1, 2, 3)) / np.nanmean(vals_rel_ang_acc, axis=(1, 2, 3))
        self.features['rel_lin_acc_median'] = np.nanmedian(vals_rel_lin_acc, axis=(1, 2, 3))
        self.features['rel_lin_acc_max'] = np.nanmax(vals_rel_lin_acc, axis=(1, 2, 3))
        self.features['rel_lin_acc_min'] = np.nanmin(vals_rel_lin_acc, axis=(1, 2, 3))
        self.features['rel_lin_acc_CoV'] = np.nanstd(vals_rel_lin_acc, axis=(1, 2, 3)) / np.nanmean(vals_rel_lin_acc, axis=(1, 2, 3))
        self.features['rel_ang_vel_median'] = np.nanmedian(vals_rel_ang_vel, axis=(1, 2, 3))
        self.features['rel_ang_vel_max'] = np.nanmax(vals_rel_ang_vel, axis=(1, 2, 3))
        self.features['rel_ang_vel_min'] = np.nanmin(vals_rel_ang_vel, axis=(1, 2, 3))
        self.features['rel_ang_vel_CoV'] = np.nanstd(vals_rel_ang_vel, axis=(1, 2, 3)) / np.nanmean(vals_rel_ang_vel, axis=(1, 2, 3))
        self.features['rel_lin_vel_median'] = np.nanmedian(vals_rel_lin_vel, axis=(1, 2, 3))
        self.features['rel_lin_vel_max'] = np.nanmax(vals_rel_lin_vel, axis=(1, 2, 3))
        self.features['rel_lin_vel_min'] = np.nanmin(vals_rel_lin_vel, axis=(1, 2, 3))
        self.features['rel_lin_vel_CoV'] = np.nanstd(vals_rel_lin_vel, axis=(1, 2, 3)) / np.nanmean(vals_rel_lin_vel, axis=(1, 2, 3))

    def _build_jump_map(self):
        tree_labels, _ = ndi.label(self.pixel_class, structure=np.ones((3, 3, 3)))

        # for big speed boost
        valid_coords = np.argwhere(tree_labels > 0)
        valid_coord_labels = tree_labels[valid_coords[:, 0], valid_coords[:, 1], valid_coords[:, 2]]

        unique_labels = np.unique(tree_labels)
        longest_jump_dist = 0
        for label_num, label in enumerate(unique_labels):
            print(f'Processing label {label_num + 1} of {len(unique_labels)}')
            if label == 0:
                continue
            tree = Tree(label, valid_coords[valid_coord_labels == label])
            tree.get_neighbors()
            tree.get_start_node()
            tree.calculate_jump_distances()
            longest_jump_dist = max(longest_jump_dist, np.max(tree.jump_distances))
            self.trees.append(tree)

        max_scale = int(np.ceil(np.log2(longest_jump_dist)))

        for tree in self.trees:
            tree.generate_scale_nodelists(max_scale)
            tree.generate_direct_connections()

        print('hi')

    def run(self):
        self._get_memmaps()
        # self._build_graph()
        self._build_jump_map()
        print('hi')


if __name__ == "__main__":
    # im_path = r"D:\test_files\nelly_gav_tests\fibro_7.nd2"
    im_path = r"D:\test_files\nelly_tests\deskewed-2023-07-13_14-58-28_000_wt_0_acquire.ome.tif"
    im_info = ImInfo(im_path)
    graph_builder = GraphBuilder(im_info)
    graph_builder.run()
