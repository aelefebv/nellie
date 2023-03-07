import tifffile
import os
from src.pipeline.node_props import NodeConstructor, Node
from src.io.im_info import ImInfo
from src import logger, xp, ndi, measure
from src.io.pickle_jar import unpickle_object


class NodeTrack:
    def __init__(self, im_info: ImInfo):
        self.im_info = im_info
        self.nodes = unpickle_object(self.im_info.path_pickle_node)


if __name__ == "__main__":
    import os
    filepath = r"D:\test_files\nelly\deskewed-single.ome.tif"
    if not os.path.isfile(filepath):
        filepath = "/Users/austin/Documents/Transferred/deskewed-single.ome.tif"
    try:
        test = ImInfo(filepath, ch=0)
    except FileNotFoundError:
        logger.error("File not found.")
        exit(1)
    nodes_test = NodeTrack(test)
