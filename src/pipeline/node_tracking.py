import tifffile

from src.io.im_info import ImInfo
from src import logger, xp, ndi, measure


class Node:
    def __init__(self, node_type: str, node_region):
        self.node_type = node_type  # should be 'tip' or 'junction'
        self.instance_label = node_region.label
        self.centroid = node_region.centroid
        self.coords = node_region.coords


class NodeConstructor:
    def __init__(self, im_info: ImInfo):
        self.im_info = im_info
        if self.im_info.is_3d:
            self.spacing = self.im_info.dim_sizes['Z'], self.im_info.dim_sizes['Y'], self.im_info.dim_sizes['X']
        else:
            self.spacing = self.im_info.dim_sizes['Y'], self.im_info.dim_sizes['X']
        self.nodes = []

    def get_node_properties(self, num_t: int = None):
        network_im = tifffile.memmap(self.im_info.path_im_neighbors, mode='r')

        if num_t is not None:
            num_t = min(num_t, network_im.shape[0])
            network_im = network_im[:num_t, ...]

        for frame_num, frame in enumerate(network_im):
            logger.info(f'Getting node properties, volume {frame_num}/{len(network_im)-1}')
            frame_mem = xp.asarray(frame)

            tips, _ = ndi.label(frame_mem == 1, structure=xp.ones((3, 3, 3)))
            tip_regions = measure.regionprops(tips, spacing=self.spacing)
            junctions, _ = ndi.label(frame_mem == 3, structure=xp.ones((3, 3, 3)))
            junction_regions = measure.regionprops(junctions, spacing=self.spacing)

            for tip in tip_regions:
                self.nodes.append(Node('tip', tip))
            for junction in junction_regions:
                self.nodes.append(Node('junction', junction))


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
    pickle_object(test.path_pickle_obj, node_props)
    node_props_unpickled = unpickle_object(test.path_pickle_obj)
    print('hi')
