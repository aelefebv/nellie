from src.pipeline.node_props import Node, NodeConstructor
from src.io.pickle_jar import unpickle_object
from src.io.im_info import ImInfo
from src import logger
import numpy as xp

class NodeTrack:
    def __init__(self, node):
        # stores information about how nodes link to one another
        # the confidence of those linkages
        #
        pass

class NodeTrackConstructor:
    def __init__(self, im_info: ImInfo,
                 distance_thresh_um_per_sec: float = 2):
        # will basically be in charge of making node tracks, and keeping them organized by frame.
        # Also in charge of connecting nodes between frames
        # assigning merge and unmerge events
        self.im_info = im_info
        self.node_constructor: NodeConstructor = unpickle_object(self.im_info.path_pickle_node)

        self.nodes: list[list[Node]] = self.node_constructor.nodes
        self.tracks: dict[list[NodeTrack]] = {}

        self.num_frames = len(self.nodes)
        self.current_frame_num = None

        self.distance_thresh_um_per_sec = distance_thresh_um_per_sec

    def populate_tracks(self, num_t: int = None):
        if num_t is not None:
            num_t = min(num_t, self.num_frames)
            self.num_frames = num_t
        for frame_num in range(self.num_frames):
            logger.debug(f'Tracking frame {frame_num}/{self.num_frames - 1}')
            self.current_frame_num = frame_num
            self._initialize_tracks()
            if frame_num == 0:
                continue

    def _initialize_tracks(self):
        node_list = []
        for node_num, node in enumerate(self.nodes[self.current_frame_num]):
            node_list.append(NodeTrack(node))
        self.tracks[self.current_frame_num] = node_list


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
    nodes_test = NodeTrackConstructor(test, distance_thresh_um_per_sec=1)
    nodes_test.populate_tracks(5)
    print('hi')