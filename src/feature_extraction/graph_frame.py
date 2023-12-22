from src import logger
from src.im_info.im_info import ImInfo
from src.utils.general import get_reshaped_image
import numpy as np
import scipy.ndimage as ndi
from scipy.spatial import cKDTree
import pandas as pd


class Tree:
    def __init__(self, label: int, voxel_idxs: np.ndarray, global_idxs):
        self.label = label
        self.voxel_idxs = voxel_idxs
        self.global_idxs = global_idxs
        self.neighbors = []
        self.start_node = None
        self.jump_distances = None
        self.nodelists = None
        self.multiscale_edge_list = None

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

    def generate_direct_accessibility(self):
        self.multiscale_edge_list = set()

        for scale_num, scale_nodelist in enumerate(self.nodelists):
            scale_nodelist_set = set(scale_nodelist)

            for node in scale_nodelist:
                visited = set()
                queue = [(node, 0)]  # (current_node, distance)

                while queue:
                    current_node, dist = queue.pop(0)
                    if dist > (scale_num**2 + 1):
                        continue
                    if current_node != node and current_node in scale_nodelist_set:
                        self.multiscale_edge_list.add((self.global_idxs[node], self.global_idxs[current_node]))
                        # break  # Stop after reaching the first node in the nodelist

                    for neighbor in self.neighbors[current_node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, dist + 1))

class GraphBuilder:
    def __init__(self, im_info: ImInfo,
                 num_t=None):
        self.im_info = im_info
        self.num_t = num_t
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

        self.features = {'t': [], 'thickness': [],
                         'raw_mean': [], 'raw_max': [], 'raw_min': [], 'raw_median': [], 'raw_CoV': [],
                         'struc_mean': [], 'struc_max': [], 'struc_min': [], 'struc_median': [], 'struc_CoV': [],
                         'rel_ang_acc_mean': [], 'rel_ang_acc_max': [], 'rel_ang_acc_min': [], 'rel_ang_acc_median': [], 'rel_ang_acc_CoV': [],
                         'rel_lin_acc_mean': [], 'rel_lin_acc_max': [], 'rel_lin_acc_min': [], 'rel_lin_acc_median': [], 'rel_lin_acc_CoV': [],
                         'rel_ang_vel_mean': [], 'rel_ang_vel_max': [], 'rel_ang_vel_min': [], 'rel_ang_vel_median': [], 'rel_ang_vel_CoV': [],
                         'rel_lin_vel_mean': [], 'rel_lin_vel_max': [], 'rel_lin_vel_min': [], 'rel_lin_vel_median': [], 'rel_lin_vel_CoV': []}
        self.edges = {'t': [], 'edge_0': [], 'edge_1': []}

    def _get_t(self):
        if self.num_t is None:
            if self.im_info.no_t:
                self.num_t = 1
            else:
                self.num_t = self.im_info.shape[self.im_info.axes.index('T')]
        else:
            return

    def _get_memmaps(self):
        logger.debug('Allocating memory for spatial feature extraction.')

        im_memmap = self.im_info.get_im_memmap(self.im_info.im_path)
        self.im_memmap = get_reshaped_image(im_memmap, self.num_t, self.im_info)

        pixel_class = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_pixel_class'])
        self.pixel_class = get_reshaped_image(pixel_class, self.num_t, self.im_info)

        label_memmap = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_instance_label'])
        self.label_memmap = get_reshaped_image(label_memmap, self.num_t, self.im_info)

        distance_memmap = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_distance'])
        self.distance_memmap = get_reshaped_image(distance_memmap, self.num_t, self.im_info)

        preproc_memmap = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_frangi'])
        self.preproc_memmap = get_reshaped_image(preproc_memmap, self.num_t, self.im_info)

        rel_ang_vel_mag_12 = self.im_info.get_im_memmap(self.im_info.pipeline_paths['rel_ang_vel_mag_12'])
        self.rel_ang_vel_mag_12 = get_reshaped_image(rel_ang_vel_mag_12, self.num_t, self.im_info)
        rel_lin_vel_mag_12 = self.im_info.get_im_memmap(self.im_info.pipeline_paths['rel_lin_vel_mag_12'])
        self.rel_lin_vel_mag_12 = get_reshaped_image(rel_lin_vel_mag_12, self.num_t, self.im_info)
        rel_ang_acc_mag = self.im_info.get_im_memmap(self.im_info.pipeline_paths['rel_ang_acc_mag'])
        self.rel_ang_acc_mag = get_reshaped_image(rel_ang_acc_mag, self.num_t, self.im_info)
        rel_lin_acc_mag = self.im_info.get_im_memmap(self.im_info.pipeline_paths['rel_lin_acc_mag'])
        self.rel_lin_acc_mag = get_reshaped_image(rel_lin_acc_mag, self.num_t, self.im_info)

        flow_vector_array_path = self.im_info.pipeline_paths['flow_vector_array']
        self.flow_vector_array = np.load(flow_vector_array_path)

        self.shape = self.pixel_class.shape

    def _get_features(self, t):
        mask = self.label_memmap[t] > 0
        masked_raw = self.im_memmap[t] * mask
        masked_preproc = self.preproc_memmap[t] * mask
        # get all network pixels
        skeleton_pixels = np.argwhere(self.pixel_class[t] > 0)
        skeleton_radius = self.distance_memmap[t][skeleton_pixels[:, 0], skeleton_pixels[:, 1], skeleton_pixels[:, 2]]
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

        z_max = self.im_info.shape[self.im_info.axes.index('Z')]
        y_max = self.im_info.shape[self.im_info.axes.index('Y')]
        x_max = self.im_info.shape[self.im_info.axes.index('X')]

        z_lims[z_lims > z_max] = z_max
        y_lims[y_lims > y_max] = y_max
        x_lims[x_lims > x_max] = x_max

        for i, (z_lim, y_lim, x_lim) in enumerate(zip(z_lims, y_lims, x_lims)):
            z_range = z_lim[1] - z_lim[0]
            y_range = y_lim[1] - y_lim[0]
            x_range = x_lim[1] - x_lim[0]
            vals_raw[i, :z_range, :y_range, :x_range] = masked_raw[z_lim[0]:z_lim[1], y_lim[0]:y_lim[1], x_lim[0]:x_lim[1]]
            vals_preproc[i, :z_range, :y_range, :x_range] = masked_preproc[z_lim[0]:z_lim[1], y_lim[0]:y_lim[1], x_lim[0]:x_lim[1]]
            vals_rel_ang_acc[i, :z_range, :y_range, :x_range] = self.rel_ang_acc_mag[t][z_lim[0]:z_lim[1], y_lim[0]:y_lim[1], x_lim[0]:x_lim[1]]
            vals_rel_lin_acc[i, :z_range, :y_range, :x_range] = self.rel_lin_acc_mag[t][z_lim[0]:z_lim[1], y_lim[0]:y_lim[1], x_lim[0]:x_lim[1]]
            vals_rel_ang_vel[i, :z_range, :y_range, :x_range] = self.rel_ang_vel_mag_12[t][z_lim[0]:z_lim[1], y_lim[0]:y_lim[1], x_lim[0]:x_lim[1]]
            vals_rel_lin_vel[i, :z_range, :y_range, :x_range] = self.rel_lin_vel_mag_12[t][z_lim[0]:z_lim[1], y_lim[0]:y_lim[1], x_lim[0]:x_lim[1]]

        vals_preproc_log10 = np.log10(vals_preproc)
        vals_preproc_log10[vals_preproc_log10 == -np.inf] = np.nan
        vals_preproc_log10[vals_preproc_log10 == np.inf] = np.nan

        vals_preproc_log10[vals_preproc_log10 == 0] = np.nan
        vals_raw[vals_raw == 0] = np.nan
        vals_rel_ang_acc[vals_rel_ang_acc == 0] = np.nan
        vals_rel_lin_acc[vals_rel_lin_acc == 0] = np.nan
        vals_rel_ang_vel[vals_rel_ang_vel == 0] = np.nan
        vals_rel_lin_vel[vals_rel_lin_vel == 0] = np.nan

        # todo, add features for translation of nodes? that way we can predict where the node goes next.
        #  should direction be wrt pivot point?

        self.features['thickness'].extend(skeleton_radius * 2)

        self.features['raw_mean'].extend(np.nanmean(vals_raw, axis=(1, 2, 3)))
        self.features['raw_max'].extend(np.nanmax(vals_raw, axis=(1, 2, 3)))
        self.features['raw_min'].extend(np.nanmin(vals_raw, axis=(1, 2, 3)))
        self.features['raw_median'].extend(np.nanmedian(vals_raw, axis=(1, 2, 3)))
        self.features['raw_CoV'].extend(np.nanstd(vals_raw, axis=(1, 2, 3)) / np.nanmean(vals_raw, axis=(1, 2, 3)))

        self.features['struc_mean'].extend(np.nanmean(vals_preproc_log10, axis=(1, 2, 3)))
        self.features['struc_max'].extend(np.nanmax(vals_preproc_log10, axis=(1, 2, 3)))
        self.features['struc_min'].extend(np.nanmin(vals_preproc_log10, axis=(1, 2, 3)))
        self.features['struc_median'].extend(np.nanmedian(vals_preproc_log10, axis=(1, 2, 3)))
        self.features['struc_CoV'].extend(np.nanstd(vals_preproc_log10, axis=(1, 2, 3)) / np.nanmean(vals_preproc_log10, axis=(1, 2, 3)))

        self.features['rel_ang_acc_mean'].extend(np.nanmean(vals_rel_ang_acc, axis=(1, 2, 3)))
        self.features['rel_ang_acc_max'].extend(np.nanmax(vals_rel_ang_acc, axis=(1, 2, 3)))
        self.features['rel_ang_acc_min'].extend(np.nanmin(vals_rel_ang_acc, axis=(1, 2, 3)))
        self.features['rel_ang_acc_median'].extend(np.nanmedian(vals_rel_ang_acc, axis=(1, 2, 3)))
        self.features['rel_ang_acc_CoV'].extend(np.nanstd(vals_rel_ang_acc, axis=(1, 2, 3)) / np.nanmean(vals_rel_ang_acc, axis=(1, 2, 3)))

        self.features['rel_lin_acc_mean'].extend(np.nanmean(vals_rel_lin_acc, axis=(1, 2, 3)))
        self.features['rel_lin_acc_max'].extend(np.nanmax(vals_rel_lin_acc, axis=(1, 2, 3)))
        self.features['rel_lin_acc_min'].extend(np.nanmin(vals_rel_lin_acc, axis=(1, 2, 3)))
        self.features['rel_lin_acc_median'].extend(np.nanmedian(vals_rel_lin_acc, axis=(1, 2, 3)))
        self.features['rel_lin_acc_CoV'].extend(np.nanstd(vals_rel_lin_acc, axis=(1, 2, 3)) / np.nanmean(vals_rel_lin_acc, axis=(1, 2, 3)))

        self.features['rel_ang_vel_mean'].extend(np.nanmean(vals_rel_ang_vel, axis=(1, 2, 3)))
        self.features['rel_ang_vel_max'].extend(np.nanmax(vals_rel_ang_vel, axis=(1, 2, 3)))
        self.features['rel_ang_vel_min'].extend(np.nanmin(vals_rel_ang_vel, axis=(1, 2, 3)))
        self.features['rel_ang_vel_median'].extend(np.nanmedian(vals_rel_ang_vel, axis=(1, 2, 3)))
        self.features['rel_ang_vel_CoV'].extend(np.nanstd(vals_rel_ang_vel, axis=(1, 2, 3)) / np.nanmean(vals_rel_ang_vel, axis=(1, 2, 3)))

        self.features['rel_lin_vel_mean'].extend(np.nanmean(vals_rel_lin_vel, axis=(1, 2, 3)))
        self.features['rel_lin_vel_max'].extend(np.nanmax(vals_rel_lin_vel, axis=(1, 2, 3)))
        self.features['rel_lin_vel_min'].extend(np.nanmin(vals_rel_lin_vel, axis=(1, 2, 3)))
        self.features['rel_lin_vel_median'].extend(np.nanmedian(vals_rel_lin_vel, axis=(1, 2, 3)))
        self.features['rel_lin_vel_CoV'].extend(np.nanstd(vals_rel_lin_vel, axis=(1, 2, 3)) / np.nanmean(vals_rel_lin_vel, axis=(1, 2, 3)))

        self.features['t'].extend([t] * len(skeleton_radius))

    def _build_jump_map(self, t):
        trees = []

        tree_labels, _ = ndi.label(self.pixel_class[t], structure=np.ones((3, 3, 3)))

        # for big speed boost
        valid_coords = np.argwhere(tree_labels > 0)
        valid_coord_labels = tree_labels[valid_coords[:, 0], valid_coords[:, 1], valid_coords[:, 2]]

        unique_labels = np.unique(tree_labels)
        longest_jump_dist = 0
        for label_num, label in enumerate(unique_labels):
            print(f'Processing label {label_num + 1} of {len(unique_labels)}')
            if label == 0:
                continue
            global_idxs = np.argwhere(valid_coord_labels == label).flatten().tolist()
            tree = Tree(label, valid_coords[valid_coord_labels == label], global_idxs)
            tree.get_neighbors()
            tree.get_start_node()
            tree.calculate_jump_distances()
            longest_jump_dist = max(longest_jump_dist, np.max(tree.jump_distances))
            trees.append(tree)

        max_scale = int(np.ceil(np.log2(longest_jump_dist)))

        for tree_num, tree in enumerate(trees):
            print(f'Generating scale nodelists for tree {tree_num + 1} of {len(trees)}')
            tree.generate_scale_nodelists(max_scale)
            tree.generate_direct_accessibility()  # these are the edges

        return trees

    def run(self):
        self._get_t()
        self._get_memmaps()
        for t in range(1, self.num_t-1):
            self._get_features(t)
            trees = self._build_jump_map(t)
            num_edges = sum([len(tree.multiscale_edge_list) for tree in trees])
            edge_0, edge_1 = zip(*[edge for tree in trees for edge in tree.multiscale_edge_list])
            self.edges['edge_0'].extend(edge_0)
            self.edges['edge_1'].extend(edge_1)
            self.edges['t'].extend([t] * num_edges)
        # for any feature in self.features, if it is nan, set it to 0
        for feature in self.features:
            self.features[feature] = np.array(self.features[feature])
            self.features[feature][np.isnan(self.features[feature])] = 0

        # save features and edges as csv
        features_df = pd.DataFrame.from_dict(self.features)
        features_df.to_csv(self.im_info.pipeline_paths['graph_features'], index=False)
        edges_df = pd.DataFrame.from_dict(self.edges)
        edges_df.to_csv(self.im_info.pipeline_paths['graph_edges'], index=False)

        # self.node_features = torch.tensor(np.vstack([v for v in self.features.values()]).T, dtype=torch.float)
        # # get every edge in every tree in one big list
        # self.edges = []
        # for tree in self.trees:
        #     self.edges.extend(list(tree.multiscale_edge_list))
        # self.edges = torch.tensor(self.edges, dtype=torch.long).T
        # # save the node features and edges
        # torch.save(self.node_features, 'node_features.pt')
        # load the node features and edges


if __name__ == "__main__":
    # im_path = r"D:\test_files\nelly_gav_tests\fibro_7.nd2"
    im_path = r"D:\test_files\nelly_tests\deskewed-2023-07-13_14-58-28_000_wt_0_acquire.ome.tif"
    im_info = ImInfo(im_path)
    graph_builder = GraphBuilder(im_info)
    graph_builder.run()
