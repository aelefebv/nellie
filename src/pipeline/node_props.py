import tifffile

from src.io.im_info import ImInfo
from src import logger #xp, ndi, measure, is_gpu  faster on cpu
is_gpu = False
import numpy as xp
import scipy.ndimage as ndi
import skimage.measure as measure


class Node:
    """
    A class representing a node in a network.

    Attributes
    ----------
    node_type : str
        The type of the node, which should be either 'tip' or 'junction'.
    instance_label : int
        The instance label of the node, which is used to identify it in the network.
    centroid : tuple of float
        The centroid coordinates of the node.
    coords : tuple of tuple of int
        The coordinates of all points in the node.
    """

    def __init__(self, node_type: str, node_region, time_point: float, spacing: tuple, dummy_region: dict = None):
        """
        Constructs a Node object.

        Parameters
        ----------
        node_type : str
            The type of the node, which should be either 'tip' or 'junction'.
        node_region : RegionProperties
            The region properties of the node as computed by `measure.regionprops()`.
        """
        self.node_type = node_type  # should be 'tip' or 'junction'
        self.connected_branches = []
        self.connected_nodes = []
        self.skeleton_label = None
        self.time_point_sec = time_point
        self.assigned_track = None
        if node_region is not None:
            self.centroid_um = tuple(x * y for x, y in zip(node_region.centroid, spacing))
            self.instance_label = node_region.label
            self.coords = node_region.coords
        else:
            self.centroid_um = tuple(x * y for x, y in zip(dummy_region['centroid'], spacing))
            self.instance_label = dummy_region['instance_label']
            self.coords = dummy_region['coords']


class NodeConstructor:
    """
    A class for constructing nodes in a network.

    Attributes
    ----------
    im_info : ImInfo
        An object containing information about the input image.
    spacing : tuple of float
        The voxel spacing of the image.
    nodes : list of Node objects
        The list of nodes in the network.
    """
    # todo, again, min_radius should probably default to something based off of a specific organelle. LUT for size?
    def __init__(self, im_info: ImInfo,
                 min_radius_um: float = 0.25):
        """
        Constructs a NodeConstructor object.

        Parameters
        ----------
        im_info : ImInfo
            An object containing information about the input image.
        """
        self.im_info = im_info
        self.node_type_memmap = None
        self.tip_label_memmap = None
        self.junction_label_memmap = None
        self.edge_label_memmap = None
        if self.im_info.is_3d:
            self.spacing = self.im_info.dim_sizes['Z'], self.im_info.dim_sizes['Y'], self.im_info.dim_sizes['X']
        else:
            self.spacing = self.im_info.dim_sizes['Y'], self.im_info.dim_sizes['X']
        self.time_spacing = self.im_info.dim_sizes['T']
        self.nodes = {}
        self.shape = ()

    def clean_labels(self, frame, frame_num, skeleton_frame):
        """
        Cleans up and labels branches in a 3D image stack. Assumes branch points are labeled as 3,
        edges are labeled as 2, and tips are labeled as 1.

        Parameters:
        -----------
        frame : np.ndarray
            The input 3D image stack. Assumes branch points are labeled as 3, edges as 2, and tips as 1.

        Returns:
        --------
        np.ndarray
            A labeled image where each branch is labeled with a unique integer.
        """
        self.nodes[frame_num] = []

        time_point_sec = frame_num * self.time_spacing

        # Find edge points in image
        edge_points = frame == 2
        # Find edge points in image
        edge_labels, num_edge_labels = ndi.label(edge_points, structure=xp.ones((3, 3, 3)))
        edge_label_set = list(range(1, num_edge_labels+1))
        edge_regions = measure.regionprops(edge_labels)
        # Label individual branch points
        junction_labels, _ = ndi.label(frame == 3, structure=xp.ones((3, 3, 3)))
        junction_regions = measure.regionprops(junction_labels)

        # Loop over each branch point region
        for junction_num, junction_region in enumerate(junction_regions):

            # Get the coords of the neighboring pixels
            coords = []
            for coord in junction_region.coords:
                z, y, x = coord
                coords.extend([(z + i, y + j, x + k)
                               for i in [-1, 0, 1] for j in [-1, 0, 1] for k in [-1, 0, 1]
                               if (i != 0 or j != 0 or k != 0)])

            # Get tip neighbors
            tip_neigh = xp.asarray([idx for idx in coords if frame[idx] == 1])

            # Order matters:

            # if tip is a neighbor, tip disappears
            for neigh in tip_neigh:
                frame[tuple(neigh)] = 0
            # get junction's edge neighbors
            edge_neighbors = xp.asarray([idx for idx in coords if frame[idx] == 2])
            # if only 1 neighbor, junction becomes a tip
            if len(edge_neighbors) == 1:
                frame[junction_labels == junction_region.label] = 1
                junction_labels[junction_labels == junction_region.label] = 0
            # if 2 neighbors only, label junction becomes label 1, label 2 becomes label 1
            elif len(edge_neighbors) == 2:
                frame[junction_labels == junction_region.label] = 2
                edge_1_label = edge_labels[tuple(edge_neighbors[0])]
                edge_2_label = edge_labels[tuple(edge_neighbors[-1])]
                edge_labels[edge_labels == edge_2_label] = edge_1_label
                edge_labels[junction_labels == junction_region.label] = edge_1_label
                junction_labels[junction_labels == junction_region.label] = 0
                if edge_1_label in edge_label_set:
                    edge_label_set.remove(edge_1_label)
                if edge_2_label in edge_label_set:
                    edge_label_set.remove(edge_2_label)
            else:  # node is a valid junction
                new_node = Node('junction', junction_region, time_point_sec, self.spacing)
                new_node.skeleton_label = skeleton_frame[tuple(junction_region.coords[0])]
                for edge_neighbor in edge_neighbors:
                    edge_label = edge_labels[tuple(edge_neighbor)]
                    new_node.connected_branches.append(edge_label)
                    if edge_label in edge_label_set:
                        edge_label_set.remove(edge_label)
                self.nodes[frame_num].append(new_node)

        # Label individual tips
        tip_labels, num_tip_labels = ndi.label((frame == 1) | (frame == 11), structure=xp.ones((3, 3, 3)))
        tip_regions = measure.regionprops(tip_labels)
        # if no neighbors, tip becomes lone tip type
        for tip_num, tip_region in enumerate(tip_regions):

            # Get the coords of the neighboring pixels
            coords = []
            for coord in tip_region.coords:
                z, y, x = coord
                coords.extend([(z + i, y + j, x + k)
                               for i in [-1, 0, 1] for j in [-1, 0, 1] for k in [-1, 0, 1]
                               if (i != 0 or j != 0 or k != 0)])

            # Get non-tip or background neighbors
            tip_neighbors = xp.asarray([idx for idx in coords if frame[idx] > 1])
            has_neighbors = len(tip_neighbors)
            if not has_neighbors:
                # set that tip to a "lone tip"
                frame[tip_labels == tip_region.label] = 11
                new_node = Node('lone tip', tip_region, time_point_sec, self.spacing)
                new_node.skeleton_label = skeleton_frame[tuple(tip_region.coords[0])]
                self.nodes[frame_num].append(new_node)
            elif has_neighbors == 1:  # node is a valid tip
                new_node = Node('tip', tip_region, time_point_sec, self.spacing)
                edge_label = edge_labels[tuple(tip_neighbors[0])]
                new_node.connected_branches.append(edge_label)
                new_node.skeleton_label = skeleton_frame[tuple(tip_region.coords[0])]
                self.nodes[frame_num].append(new_node)
                if edge_label in edge_label_set:
                    edge_label_set.remove(edge_label)
            else:
                logger.warning("Tip has more than 1 neighbor... This should not be the case. Definitely Austin's fault")

        # if edge has not been assigned, it has no tip or junction. Set its centroid as a lone tip
        for edge_label in edge_label_set:
            edge_label_idx = edge_label-1
            edge_region = edge_regions[edge_label_idx]
            rounded_centroid = tuple(round(x) for x in edge_region.centroid)
            frame[rounded_centroid] = 11
            node_coords = xp.zeros_like(edge_region.coords, shape=(1, edge_region.coords.shape[-1]))
            node_coords[..., :] = rounded_centroid
            num_tip_labels += 1
            dummy_region = {'instance_label': num_tip_labels, 'coords': node_coords, 'centroid': edge_region.centroid}
            new_node = Node('lone tip', None, time_point_sec, self.spacing, dummy_region=dummy_region)
            new_node.connected_branches.append(edge_label)  # could be useful for later?
            new_node.skeleton_label = skeleton_frame[tuple(edge_region.coords[0])]
            self.nodes[frame_num].append(new_node)
            tip_labels[rounded_centroid] = num_tip_labels
        return tip_labels, junction_labels, edge_labels

    def _find_node_connections(self, frame_num):
        """
        Find the connections of a node to other nodes in a given frame.

        Parameters
        ----------
        frame_num : int
            The frame number in which to search for node connections.
        """
        branch_labels = {}
        for node_num, node in enumerate(self.nodes[frame_num]):
            for branch_num in node.connected_branches:
                if branch_num not in branch_labels.keys():
                    branch_labels[branch_num] = [node_num]
                else:
                    branch_labels[branch_num].append(node_num)
        for branch_label, connected_nodes in branch_labels.items():
            for node_num in connected_nodes:
                self.nodes[frame_num][node_num].connected_nodes += connected_nodes
        for node_num, node in enumerate(self.nodes[frame_num]):
            node.connected_nodes = [x for x in set(node.connected_nodes) if x != node_num]

    # def _find_all_neighboring_objects(self, tip_labels, junction_labels):
    #     # Rescale the labeled image according to the scales
    #     # could do this to scale for x y and z... but the zoom function is slow
    #     # labelled_image = ndi.zoom(labelled_image, self.spacing, order=0)
    #     combo_labels = tip_labels + junction_labels
    #
    #     # Calculate the Euclidean distance transform for the entire labeled image
    #     distance_transform = ndi.distance_transform_edt(combo_labels == 0)
    #
    #     # Dilate the labeled image using a structuring element with a radius equal to the distance threshold
    #     structuring_element = ndi.generate_binary_structure(3, 1)
    #     structuring_element = ndi.iterate_structure(structuring_element, self.min_radius_px)
    #     dilated_image = ndi.grey_dilation(labelled_image, structure=structuring_element)
    #
    #     # Identify where the dilated image and the original labeled image intersect
    #     intersection_mask = (dilated_image != 0) & (labelled_image != 0) & (dilated_image != labelled_image)
    #
    #     # Create a tuple containing coordinates for the intersection points
    #     intersection_coordinates = xp.where(intersection_mask)
    #
    #     # Extract the labels of the original image and the dilated image at the intersection points
    #     original_labels = labelled_image[intersection_coordinates]
    #     dilated_labels = dilated_image[intersection_coordinates]
    #
    #     # Create a dictionary to store the neighboring labels for each object
    #     neighbors = {}
    #
    #     # Iterate over the labels and add them to the neighbors dictionary
    #     for original_label, dilated_label in zip(original_labels, dilated_labels):
    #         if original_label not in neighbors:
    #             neighbors[original_label] = set()
    #         neighbors[original_label].add(dilated_label)
    #
    #     return neighbors


    def get_node_properties(self, num_t: int = None, dtype='uint32'):
        """
        Computes the properties of nodes in the network and stores them in `self.nodes`.

        Parameters
        ----------
        num_t : int or None, optional
            The number of timepoints to process. If None, all timepoints are processed.
        """
        network_im = tifffile.memmap(self.im_info.path_im_network, mode='r')


        if num_t is not None:
            num_t = min(num_t, network_im.shape[0])
            network_im = network_im[:num_t, ...]
        self.shape = network_im.shape

        # Allocate memory for the node type and label, and node segment volumes and load it as a memory-mapped file
        self.im_info.allocate_memory(
            self.im_info.path_im_node_types, shape=self.shape, dtype='uint8', description='Node type image'
        )
        self.im_info.allocate_memory(
            self.im_info.path_im_label_tips, shape=self.shape, dtype=dtype, description='Tip label image'
        )
        self.im_info.allocate_memory(
            self.im_info.path_im_label_junctions, shape=self.shape, dtype=dtype, description='Junction label image'
        )
        self.im_info.allocate_memory(
            self.im_info.path_im_label_seg, shape=self.shape, dtype=dtype, description='Branch segments image'
        )
        node_type_memmap = tifffile.memmap(self.im_info.path_im_node_types, mode='r+')
        tip_label_memmap = tifffile.memmap(self.im_info.path_im_label_tips, mode='r+')
        junction_label_memmap = tifffile.memmap(self.im_info.path_im_label_junctions, mode='r+')
        edge_label_memmap = tifffile.memmap(self.im_info.path_im_label_seg, mode='r+')
        skeleton_memmap = tifffile.memmap(self.im_info.path_im_skeleton, mode='r')

        for frame_num, frame in enumerate(network_im):
            logger.info(f'Running branch point analysis, volume {frame_num}/{len(network_im) - 1}')
            # self.node_type_memmap[frame_num] = xp.asarray(frame)
            node_type_memmap[frame_num] = frame

            tip_labels, junction_labels, edge_labels = self.clean_labels(
                xp.array(node_type_memmap[frame_num]), frame_num, skeleton_memmap[frame_num])

            self._find_node_connections(frame_num)
            # self._find_all_neighboring_objects(tip_labels, junction_labels)

            if is_gpu:
                tip_label_memmap[frame_num] = tip_labels.get()
                junction_label_memmap[frame_num] = junction_labels.get()
                edge_label_memmap[frame_num] = edge_labels.get()
            else:
                tip_label_memmap[frame_num] = tip_labels
                junction_label_memmap[frame_num] = junction_labels
                edge_label_memmap[frame_num] = edge_labels


if __name__ == "__main__":
    from src.io.pickle_jar import pickle_object, unpickle_object
    import os
    filepath = r"D:\test_files\nelly\deskewed-single.ome.tif"
    if not os.path.isfile(filepath):
        filepath = "/Users/austin/Documents/Transferred/deskewed-single.ome.tif"
    try:
        test = ImInfo(filepath, ch=0)
    except FileNotFoundError:
        logger.error("File not found.")
        exit(1)
    node_props = NodeConstructor(test)
    node_props.get_node_properties(5)
    pickle_object(test.path_pickle_node, node_props)
    node_props_unpickled = unpickle_object(test.path_pickle_node)
    print('hi')
