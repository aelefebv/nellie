import tifffile

from src.io.im_info import ImInfo
from src import logger, xp, ndi, measure, is_gpu


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

    def __init__(self, node_type: str, node_region, time_point: float, spacing: tuple):
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
        self.instance_label = node_region.label
        self.centroid_um = tuple(x * y for x, y in zip(node_region.centroid, spacing))
        self.coords = node_region.coords
        self.time_point_sec = time_point
        self.assigned_track = None


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
    def __init__(self, im_info: ImInfo):
        """
        Constructs a NodeConstructor object.

        Parameters
        ----------
        im_info : ImInfo
            An object containing information about the input image.
        """
        self.im_info = im_info
        self.segment_memmap = None
        if self.im_info.is_3d:
            self.spacing = self.im_info.dim_sizes['Z'], self.im_info.dim_sizes['Y'], self.im_info.dim_sizes['X']
        else:
            self.spacing = self.im_info.dim_sizes['Y'], self.im_info.dim_sizes['X']
        self.time_spacing = self.im_info.dim_sizes['T']
        self.nodes = {}
        self.shape = ()

    def clean_labels(self, frame, frame_num):
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
        edge_label_set = list(range(1, num_edge_labels))
        print(edge_label_set)
        # Label individual branch points
        junction_label, _ = ndi.label(frame == 3, structure=xp.ones((3, 3, 3)))
        junction_regions = measure.regionprops(junction_label)

        # Loop over each branch point region
        for junction_num, junction_region in enumerate(junction_regions):
            logger.debug(f'Cleaning junction {junction_num}/{len(junction_regions) - 1}')

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
                frame[junction_label == junction_region.label] = 1
            # if 2 neighbors only, label junction becomes label 1, label 2 becomes label 1
            elif len(edge_neighbors) == 2:
                frame[junction_label == junction_region.label] = 2
                edge_1_label = edge_labels[tuple(edge_neighbors[0])]
                edge_2_label = edge_labels[tuple(edge_neighbors[-1])]
                edge_labels[edge_labels == edge_2_label] = edge_1_label
                edge_labels[junction_label == junction_region.label] = edge_1_label
                if edge_1_label in edge_label_set:
                    edge_label_set.remove(edge_1_label)
                if edge_2_label in edge_label_set:
                    edge_label_set.remove(edge_2_label)
            else:  # node is a valid junction
                new_node = Node('junction', junction_region, time_point_sec, self.spacing)
                for edge_neighbor in edge_neighbors:
                    edge_label = edge_labels[tuple(edge_neighbor)]
                    new_node.connected_branches.append(edge_label)
                    if edge_label in edge_label_set:
                        edge_label_set.remove(edge_label)
                self.nodes[frame_num].append(new_node)

        # todo here, if edge has no attached tip or junction, set centroid as a lone tip.
        # Label individual tips
        tip_labels, _ = ndi.label((frame == 1) | (frame == 11), structure=xp.ones((3, 3, 3)))
        tip_regions = measure.regionprops(tip_labels)
        # if no neighbors, tip becomes lone tip type
        # Loop over each branch point region
        for tip_num, tip_region in enumerate(tip_regions):
            logger.debug(f'Checking for lone tips {tip_num}/{len(tip_regions) - 1}')

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
                self.nodes[frame_num].append(new_node)
            elif has_neighbors == 1:  # node is a valid tip
                new_node = Node('tip', tip_region, time_point_sec, self.spacing)
                new_node.connected_branches.append(edge_labels[tuple(tip_neighbors[0])])
                self.nodes[frame_num].append(new_node)
            else:
                logger.warning("Tip has more than 1 neighbor... This should not be the case. Definitely Austin's fault")


        return edge_labels

    def get_node_properties(self, num_t: int = None, dtype='uint32'):
        """
        Computes the properties of nodes in the network and stores them in `self.nodes`.

        Parameters
        ----------
        num_t : int or None, optional
            The number of timepoints to process. If None, all timepoints are processed.
        """
        network_im = tifffile.memmap(self.im_info.path_im_neighbors, mode='r+')

        if num_t is not None:
            num_t = min(num_t, network_im.shape[0])
            network_im = network_im[:num_t, ...]
        self.shape = network_im.shape

        # Allocate memory for the branch segment volume and load it as a memory-mapped file
        self.im_info.allocate_memory(
            self.im_info.path_im_label_seg, shape=self.shape, dtype=dtype, description='Branch segments image'
        )
        self.segment_memmap = tifffile.memmap(self.im_info.path_im_label_seg, mode='r+')

        for frame_num, frame in enumerate(network_im):
            logger.info(f'Running branch point analysis, volume {frame_num}/{len(network_im) - 1}')
            frame_neighbor = xp.asarray(frame)

            edge_labels = self.clean_labels(frame_neighbor, frame_num)

            if is_gpu:
                self.segment_memmap[frame_num] = edge_labels.get()
            else:
                self.segment_memmap[frame_num] = edge_labels

            # logger.info(f'Getting node properties, volume {frame_num}/{len(network_im)-1}')
            # time_point_sec = frame_num * self.time_spacing
            # frame_mem = xp.asarray(frame)
            #
            # tips, _ = ndi.label(frame_mem == 1, structure=xp.ones((3, 3, 3)))
            # tip_regions = measure.regionprops(tips, spacing=self.spacing)
            # lone_tips, _ = ndi.label(frame_mem == 11, structure=xp.ones((3, 3, 3)))
            # lone_tip_regions = measure.regionprops(lone_tips, spacing = self.spacing)
            # junctions, _ = ndi.label(frame_mem == 3, structure=xp.ones((3, 3, 3)))
            # junction_regions = measure.regionprops(junctions, spacing=self.spacing)
            #
            # nodes_frame = []
            # for tip in tip_regions:
            #     nodes_frame.append(Node('tip', tip, time_point_sec))
            # for lone_tip in lone_tip_regions:
            #     nodes_frame.append(Node('lone tip', lone_tip, time_point_sec))
            # for junction in junction_regions:
            #     nodes_frame.append(Node('junction', junction, time_point_sec))
            #
            # self.nodes.append(nodes_frame)


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
    node_props.get_node_properties(2)
    pickle_object(test.path_pickle_node, node_props)
    node_props_unpickled = unpickle_object(test.path_pickle_node)
    print('hi')
