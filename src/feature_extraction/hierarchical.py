import numpy as np

from src.im_info.im_info import ImInfo
from src.utils.general import get_reshaped_image


class Hierarchy:
    def __init__(self, im_info: ImInfo, num_t: int):
        self.im_info = im_info
        self.num_t = num_t
        if self.im_info.no_z:
            self.spacing = (self.im_info.dim_sizes['Y'], self.im_info.dim_sizes['X'])
        else:
            self.spacing = (self.im_info.dim_sizes['Z'], self.im_info.dim_sizes['Y'], self.im_info.dim_sizes['X'])

        self.im_raw = None
        self.im_struct = None
        self.im_distance = None
        self.im_skel = None
        self.label_components = None
        self.label_branches = None

        self.shape = None

    def _get_t(self):
        if self.num_t is None:
            if self.im_info.no_t:
                raise ValueError("No time dimension in image.")
            self.num_t = self.im_info.shape[self.im_info.axes.index('T')]
        if self.num_t < 3:
            raise ValueError("num_t must be at least 3")
        return self.num_t

    def _allocate_memory(self):
        im_raw = self.im_info.get_im_memmap(self.im_info.im_path)
        self.im_raw = get_reshaped_image(im_raw, self.num_t, self.im_info)
        im_struct = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_frangi'])
        self.im_struct = get_reshaped_image(im_struct, self.num_t, self.im_info)
        label_components = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_instance_label'])
        self.label_components = get_reshaped_image(label_components, self.num_t, self.im_info)
        label_branches = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_skel_relabelled'])
        self.label_branches = get_reshaped_image(label_branches, self.num_t, self.im_info)
        im_distance = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_distance'])
        self.im_distance = get_reshaped_image(label_branches, self.num_t, self.im_info)
        im_skel = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_skel'])
        self.im_skel = get_reshaped_image(im_skel, self.num_t, self.im_info)

        self.shape = self.im_raw.shape

    def run(self):
        self._get_t()
        self._allocate_memory()


class Voxels:
    def __init__(self, hierarchy: Hierarchy):
        self.hierarchy = hierarchy

        self.time = []
        self.coords = []
        # add voxel metrics
        # self.vel_lin = []
        # self.vel_ang = []
        # self.acc_lin = []
        # self.acc_ang = []
        self.node_labels = []
        self.branch_labels = []
        self.component_labels = []
        self.image_name = []
        self.intensity = []
        self.structure = []

        self.node_z_lims = []
        self.node_y_lims = []
        self.node_x_lims = []
        self.node_voxel_idxs = []

    def _get_node_info(self, t, frame_coords):
        # get all network pixels
        skeleton_pixels = np.argwhere(self.hierarchy.im_skel[t] > 0)
        skeleton_radius = self.hierarchy.im_distance[t][skeleton_pixels[:, 0], skeleton_pixels[:, 1], skeleton_pixels[:, 2]]
        largest_thickness = int(np.ceil(np.max(skeleton_radius * 2))) + 1
        # create bounding boxes of size largest_thickness around each skeleton pixel
        z_lims = (skeleton_radius[:, np.newaxis] * np.array([-1, 1]) + skeleton_pixels[:, 0, np.newaxis]).astype(int)
        z_lims[:, 1] += 1
        y_lims = (skeleton_radius[:, np.newaxis] * np.array([-1, 1]) + skeleton_pixels[:, 1, np.newaxis]).astype(int)
        y_lims[:, 1] += 1
        x_lims = (skeleton_radius[:, np.newaxis] * np.array([-1, 1]) + skeleton_pixels[:, 2, np.newaxis]).astype(int)
        x_lims[:, 1] += 1

        z_lims[z_lims < 0] = 0
        y_lims[y_lims < 0] = 0
        x_lims[x_lims < 0] = 0

        z_max = self.hierarchy.im_info.shape[self.hierarchy.im_info.axes.index('Z')]
        y_max = self.hierarchy.im_info.shape[self.hierarchy.im_info.axes.index('Y')]
        x_max = self.hierarchy.im_info.shape[self.hierarchy.im_info.axes.index('X')]

        z_lims[z_lims > z_max] = z_max
        y_lims[y_lims > y_max] = y_max
        x_lims[x_lims > x_max] = x_max

        self.node_z_lims.append(z_lims)
        self.node_y_lims.append(y_lims)
        self.node_x_lims.append(x_lims)

        frame_coord_nodes_idxs = []
        # if a frame coord is within the bounding box of a skeleton pixel, add the skeleton pixel's idx to the list for that frame coord
        for i, frame_coord in enumerate(frame_coords):
            z, y, x = frame_coord
            z_mask = (z_lims[:, 0] <= z) & (z_lims[:, 1] >= z)
            y_mask = (y_lims[:, 0] <= y) & (y_lims[:, 1] >= y)
            x_mask = (x_lims[:, 0] <= x) & (x_lims[:, 1] >= x)
            mask = z_mask & y_mask & x_mask
            frame_coord_nodes_idxs.append(np.argwhere(mask).flatten())

        self.node_labels.append(frame_coord_nodes_idxs)

        # Initialize an empty dictionary for frame_coord_nodes_idxs
        node_voxel_idxs = []

        for i, skeleton_px in enumerate(skeleton_pixels):
            bbox_voxels = np.argwhere((z_lims[i, 0] <= frame_coords[:, 0]) & (z_lims[i, 1] >= frame_coords[:, 0]) &
                                        (y_lims[i, 0] <= frame_coords[:, 1]) & (y_lims[i, 1] >= frame_coords[:, 1]) &
                                        (x_lims[i, 0] <= frame_coords[:, 2]) & (x_lims[i, 1] >= frame_coords[:, 2]))
            node_voxel_idxs.append(bbox_voxels.flatten())

        self.node_voxel_idxs.append(node_voxel_idxs)

    def _run_frame(self, t):
        frame_coords = np.argwhere(self.hierarchy.label_components[t] > 0)
        self.coords.append(frame_coords)

        frame_component_labels = self.hierarchy.label_components[t][frame_coords[:, 0], frame_coords[:, 1], frame_coords[:, 2]]
        self.component_labels.append(frame_component_labels)

        frame_branch_labels = self.hierarchy.label_branches[t][frame_coords[:, 0], frame_coords[:, 1], frame_coords[:, 2]]
        self.branch_labels.append(frame_branch_labels)

        frame_intensity_vals = self.hierarchy.im_raw[t][frame_coords[:, 0], frame_coords[:, 1], frame_coords[:, 2]]
        self.intensity.append(frame_intensity_vals)

        frame_structure_vals = self.hierarchy.im_struct[t][frame_coords[:, 0], frame_coords[:, 1], frame_coords[:, 2]]
        self.structure.append(frame_structure_vals)

        frame_t = np.ones(frame_coords.shape[0], dtype=int) * t
        self.time.append(frame_t)

        self._get_node_info(t, frame_coords)

    def run(self):
        for t in range(self.hierarchy.num_t):
            self._run_frame(t)

        print('hi')


class Nodes:
    def __init__(self, hierarchy, voxels):
        self.hierarchy = hierarchy
        self.voxels = voxels

        self.time = []
        self.nodes = []
        # add voxel aggregate metrics
        # add node metrics
        self.voxel_idxs = voxels.node_voxel_idxs
        self.branch_label = []
        self.component_label = []
        self.image_name = []

        self.node_z_lims = voxels.node_z_lims
        self.node_y_lims = voxels.node_y_lims
        self.node_x_lims = voxels.node_x_lims

    def _run_frame(self, t):
        frame_skel_coords = np.argwhere(self.hierarchy.im_skel[t] > 0)
        frame_t = np.ones(frame_skel_coords.shape[0], dtype=int) * t
        frame_component_label = self.hierarchy.label_components[t][frame_skel_coords[:, 0], frame_skel_coords[:, 1], frame_skel_coords[:, 2]]
        frame_branch_label = self.hierarchy.label_branches[t][frame_skel_coords[:, 0], frame_skel_coords[:, 1], frame_skel_coords[:, 2]]

        self.time.append(frame_t)
        self.nodes.append(frame_skel_coords)
        self.component_label.append(frame_component_label)
        self.branch_label.append(frame_branch_label)
        print('hi')

    def run(self):
        for t in range(self.hierarchy.num_t):
            self._run_frame(t)
        print('hi')


class Branches:
    def __init__(self):
        pass


class Components:
    def __init__(self):
        pass


class Image:
    def __init__(self):
        pass


if __name__ == "__main__":
    im_path = r"D:\test_files\nelly_smorgasbord\deskewed-iono_pre.ome.tif"
    im_info = ImInfo(im_path)
    num_t = 3

    hierarchy = Hierarchy(im_info, num_t)
    hierarchy.run()

    voxels = Voxels(hierarchy)
    voxels.run()

    nodes = Nodes(hierarchy, voxels)
    nodes.run()

    # branches = Branches(hierarchy)
    # components = Components(hierarchy)
    # image = Image(hierarchy)


