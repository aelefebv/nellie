import tifffile
import os
from src.pipeline.node_props import Node, NodeConstructor
from src.io.im_info import ImInfo
from src import logger, xp, ndi, measure
from src.io.pickle_jar import unpickle_object


class NodeTrack:
    def __init__(self, node):
        self.nodes = [node]
        self.time_points = [node.time_point_sec]
        self.centroids_um = [node.centroid_um]
        self.instance_labels = [node.instance_label]
        self.node_types = [node.node_type]

    def add_node(self, node):
        self.nodes.append(node)
        self.time_points.append(node.time_point_sec)
        self.centroids_um.append(node.centroid_um)
        self.instance_labels.append(node.instance_label)
        self.node_types.append(node.node_type)


class NodeTrackConstructor:
    def __init__(self, im_info: ImInfo, distance_thresh_um: float = xp.inf, time_thresh_s: float = xp.inf):
        self.im_info = im_info
        self.node_constructor: NodeConstructor = unpickle_object(self.im_info.path_pickle_node)
        self.nodes: [[Node]] = self.node_constructor.nodes
        self.tracks: [NodeTrack] = []
        self.num_frames = len(self.nodes)
        self.distance_thresh_um = distance_thresh_um
        self.time_thresh_s = time_thresh_s

    def initialize_tracks(self):
        for node in self.nodes[0]:
            self.tracks.append(NodeTrack(node))

    def assign_nodes_to_tracks(self, frame_num):
        frame_nodes = self.nodes[frame_num]
        num_nodes = len(frame_nodes)
        num_tracks = len(self.tracks)
        num_dimensions = len(frame_nodes[0].centroid_um)

        node_centroids = xp.empty((num_dimensions, 1, num_nodes))
        track_centroids = xp.empty((num_dimensions, num_tracks, 1))

        for node_num, node in enumerate(frame_nodes):
            node_centroids[:, 0, node_num] = node.centroid_um
        for track_num, track in enumerate(self.tracks):
            track_centroids[:, track_num, 0] = track.centroids_um[-1]

        distance_matrix = xp.sqrt(xp.sum((node_centroids - track_centroids) ** 2, axis=0))
        distance_matrix[distance_matrix > self.distance_thresh_um] = xp.inf
        # todo do the same for time

    def populate_tracks(self, num_t: int = None):
        if num_t is not None:
            num_t = min(num_t, self.num_frames)
        for frame_num in range(self.num_frames):
            test_dist = self.assign_nodes_to_tracks(frame_num)
        return test_dist


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
    nodes_test = NodeTrackConstructor(test)
    nodes_test.initialize_tracks()
    dist_matrix = nodes_test.populate_tracks()
