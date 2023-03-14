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
        self.node = node
        pass

class NodeTrackConstructor:
    def __init__(self, im_info: ImInfo,
                 distance_thresh_um_per_sec: float = 2):
        # will basically be in charge of making node tracks, and keeping them organized by frame.
        # Also in charge of connecting nodes between frames
        # assigning merge and unmerge events
        self.im_info = im_info

        node_constructor = unpickle_object(self.im_info.path_pickle_node)
        self.nodes: list[list[Node]] = node_constructor.nodes
        self.tracks: dict[list[NodeTrack]] = {}

        self.num_frames = len(self.nodes)
        self.current_frame_num = None

        self.distance_thresh_um_per_sec = distance_thresh_um_per_sec

    def populate_tracks(self, num_t: int = None):
        if num_t is not None:
            num_t = min(num_t, self.num_frames)
            self.num_frames = num_t
        self._initialize_tracks()
        for frame_num in range(self.num_frames):
            logger.debug(f'Tracking frame {frame_num}/{self.num_frames - 1}')
            self.current_frame_num = frame_num
            if frame_num == 0:
                continue
            self._get_assignment_matrix()

    def _initialize_tracks(self):
        for frame_num in range(self.num_frames):
            node_list = []
            for node_num, node in enumerate(self.nodes[frame_num]):
                node_list.append(NodeTrack(node))
            self.tracks[frame_num] = node_list

    def _get_assignment_matrix(self):
        tracks_t1 = self.tracks[self.current_frame_num-1]
        tracks_t2 = self.tracks[self.current_frame_num]
        num_tracks_t1 = len(tracks_t1)
        num_tracks_t2 = len(tracks_t2)
        num_dimensions = len(tracks_t1[0].node.centroid_um)

        t1_centroids = xp.empty((num_dimensions, num_tracks_t1, 1))
        t2_centroids = xp.empty((num_dimensions, 1, num_tracks_t2))

        time_difference = tracks_t2[0].node.time_point_sec - tracks_t1[0].node.time_point_sec

        for track_num, track in enumerate(tracks_t1):
            t1_centroids[:, track_num, 0] = track.node.centroid_um
        for track_num, track in enumerate(tracks_t2):
            t2_centroids[:, 0, track_num] = track.node.centroid_um

        distance_matrix = xp.sqrt(xp.sum((t2_centroids - t1_centroids) ** 2, axis=0))
        distance_matrix /= time_difference
        distance_matrix[distance_matrix > self.distance_thresh_um_per_sec] = xp.inf

        cost_matrix = self._append_unassignment_costs(distance_matrix)
        return cost_matrix

    def _append_unassignment_costs(self, pre_cost_matrix):
        rows, cols = pre_cost_matrix.shape
        cost_matrix = xp.ones(
            (rows+cols, rows+cols)
        ) * self.distance_thresh_um_per_sec
        cost_matrix[:rows, :cols] = pre_cost_matrix
        return cost_matrix


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