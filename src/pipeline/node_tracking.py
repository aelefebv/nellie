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
    def __init__(self, im_info: ImInfo,
                 distance_thresh_um_per_sec: float = xp.inf,
                 time_thresh_sec: float = xp.inf):
        self.im_info = im_info
        self.node_constructor: NodeConstructor = unpickle_object(self.im_info.path_pickle_node)
        self.nodes: list[list[Node]] = self.node_constructor.nodes
        self.tracks: list[NodeTrack] = []
        self.num_frames = len(self.nodes)
        self.distance_thresh_um_per_sec = distance_thresh_um_per_sec
        self.time_thresh_sec = time_thresh_sec

    def initialize_tracks(self):
        for node in self.nodes[0]:
            self.tracks.append(NodeTrack(node))

    def get_assignment_matrix(self, frame_num):
        frame_nodes = self.nodes[frame_num]
        num_nodes = len(frame_nodes)
        num_tracks = len(self.tracks)
        num_dimensions = len(frame_nodes[0].centroid_um)
        frame_time_s = frame_nodes[0].time_point_sec

        node_centroids = xp.empty((num_dimensions, 1, num_nodes))
        track_centroids = xp.empty((num_dimensions, num_tracks, 1))

        time_matrix = xp.empty((num_tracks, num_nodes))

        for node_num, node in enumerate(frame_nodes):
            node_centroids[:, 0, node_num] = node.centroid_um
        for track_num, track in enumerate(self.tracks):
            time_check = frame_time_s - track.time_points[-1]
            if time_check > self.time_thresh_sec:
                track_centroids[:, track_num, 0] = xp.nan
                continue
            track_centroids[:, track_num, 0] = track.centroids_um[-1]
            time_matrix[track_num, :] = time_check

        distance_matrix = xp.sqrt(xp.sum((node_centroids - track_centroids) ** 2, axis=0))
        distance_matrix /= time_matrix  # this is now a distance/sec matrix
        distance_matrix[distance_matrix > self.distance_thresh_um_per_sec] = xp.nan

        return distance_matrix

    def assign_confident_tracks(self, assignment_matrix, frame_num):
        # rows are tracks, columns are frame nodes
        # Find indices of non-nan values
        indices = xp.argwhere(~xp.isnan(assignment_matrix))
        # Count the number of non-nan values in each row and column
        row_counts = xp.bincount(indices[:, 0])
        col_counts = xp.bincount(indices[:, 1])

        # Find the rows and columns with exactly one non-nan value
        unique_rows = xp.where(row_counts == 1)[0]
        unique_cols = xp.where(col_counts == 1)[0]

        # Any t1 tip can go to 1 single t2 node (not multiple)
        unique_tips = [tip for tip in unique_rows if self.tracks[tip].node_types[-1] == 'tip']

        # Assign unique t1 tip matches to t2 nodes
        num_assigned = 0
        for tip in unique_tips:
            tip_idx = xp.argwhere(indices[:, 0] == tip)
            node_match_num = indices[tip_idx[0], 1][0]

            # Find all matches that this node could be assigned to
            other_possible_matches = xp.argwhere(indices[:, 1] == node_match_num)

            # T2 junctions but not T2 tips can be assigned to multiple T1 tips
            # If there is only 1 T2 tip matched to 1 T1 tip, assign them
            if (self.tracks[tip].node_types[-1] == 'junction') or (len(other_possible_matches) == 1):
                num_assigned += 1
                self.tracks[tip].add_node(self.nodes[frame_num][node_match_num])

        # Find the common elements between unique_rows and unique_cols
        common_elements = xp.intersect1d(unique_rows, unique_cols)

        # Get the pairs as a list of tuple
        confident_pairs = [(row, col) for row, col in zip(unique_rows, unique_cols)
                           if row in common_elements and col in common_elements]

        # Assign the nodes to the tracks, and sets row and column of assignment matrix to nan:
        for track_num, node_num in confident_pairs:
            if track_num in unique_tips:
                continue
            self.tracks[track_num].add_node(self.nodes[frame_num][node_num])
            assignment_matrix[track_num, :] = xp.nan
            assignment_matrix[:, node_num] = xp.nan

    def populate_tracks(self, num_t: int = None):
        if num_t is not None:
            num_t = min(num_t, self.num_frames)
            self.num_frames = num_t
        for frame_num in range(self.num_frames):
            if frame_num == 0:
                self.initialize_tracks()
                continue
            assignment_matrix = self.get_assignment_matrix(frame_num)
            self.assign_confident_tracks(assignment_matrix, frame_num)
        self.assignment_matrix = assignment_matrix


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
    nodes_test = NodeTrackConstructor(test, distance_thresh_um_per_sec=0.5)
    nodes_test.populate_tracks()
