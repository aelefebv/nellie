from src.io.im_info import ImInfo
from src import logger


class Node:
    def __init__(self, node_type: str, node_region):
        self.node_type = node_type  # should be 'tip' or 'junction'


class NodeConstructor:
    def __init__(self, im_info: ImInfo):
        self.im_info = im_info
        if self.im_info.is_3d:
            self.spacing = self.im_info.dim_sizes['Z'], self.im_info.dim_sizes['Y'], self.im_info.dim_sizes['X']
        else:
            self.spacing = self.im_info.dim_sizes['Y'], self.im_info.dim_sizes['X']
        self.nodes = []


if __name__ == "__main__":
    from src.io.pickle_jar import pickle_object
    import os
    filepath = r"D:\test_files\nelly\deskewed-single.ome.tif"
    if not os.path.isfile(filepath):
        filepath = "/Users/austin/Documents/Transferred/deskewed-single.ome.tif"
    try:
        test = ImInfo(filepath, ch=0)
    except FileNotFoundError:
        logger.error("File not found.")
        exit(1)
