import tifffile

from src.io.im_info import ImInfo
from src import logger, xp, ndi, measure


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

    def __init__(self, node_type: str, node_region, time_point: float):
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
        self.instance_label = node_region.label
        self.centroid_um = node_region.centroid
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
        if self.im_info.is_3d:
            self.spacing = self.im_info.dim_sizes['Z'], self.im_info.dim_sizes['Y'], self.im_info.dim_sizes['X']
        else:
            self.spacing = self.im_info.dim_sizes['Y'], self.im_info.dim_sizes['X']
        self.time_spacing = self.im_info.dim_sizes['T']
        self.nodes = []

    def get_node_properties(self, num_t: int = None):
        """
        Computes the properties of nodes in the network and stores them in `self.nodes`.

        Parameters
        ----------
        num_t : int or None, optional
            The number of timepoints to process. If None, all timepoints are processed.
        """
        network_im = tifffile.memmap(self.im_info.path_im_neighbors, mode='r')

        if num_t is not None:
            num_t = min(num_t, network_im.shape[0])
            network_im = network_im[:num_t, ...]

        for frame_num, frame in enumerate(network_im):
            logger.info(f'Getting node properties, volume {frame_num}/{len(network_im)-1}')
            time_point_sec = frame_num * self.time_spacing
            frame_mem = xp.asarray(frame)

            tips, _ = ndi.label(frame_mem == 1, structure=xp.ones((3, 3, 3)))
            tip_regions = measure.regionprops(tips, spacing=self.spacing)
            lone_tips, _ = ndi.label(frame_mem == 11, structure=xp.ones((3, 3, 3)))
            lone_tip_regions = measure.regionprops(lone_tips, spacing = self.spacing)
            junctions, _ = ndi.label(frame_mem == 3, structure=xp.ones((3, 3, 3)))
            junction_regions = measure.regionprops(junctions, spacing=self.spacing)

            nodes_frame = []
            for tip in tip_regions:
                nodes_frame.append(Node('tip', tip, time_point_sec))
            for lone_tip in lone_tip_regions:
                nodes_frame.append(Node('lone tip', lone_tip, time_point_sec))
            for junction in junction_regions:
                nodes_frame.append(Node('junction', junction, time_point_sec))

            self.nodes.append(nodes_frame)


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
