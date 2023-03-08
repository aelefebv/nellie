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
        # rows are tracks (t1 nodes), columns are frame nodes (t2 nodes)
        # Find indices of non-nan values
        indices = xp.argwhere(~xp.isnan(assignment_matrix))

        # Count the number of non-nan values in each row and column
        t1_counts = xp.bincount(indices[:, 0])
        t2_counts = xp.bincount(indices[:, 1])

        # Find the rows and columns with exactly one non-nan value
        unique_t1 = xp.where(t1_counts == 1)[0]
        unique_t2 = xp.where(t2_counts == 1)[0]

        # Any t1 nodes that have only one match to a t2 node that has only one match are assigned
        assignment_matrix = self.assign_t1_t2_unique_matches(unique_t1, unique_t2, assignment_matrix, frame_num)

        # get a node_type-node_type connection look up table
        t1_type_lut = xp.array([[0 if node.node_type == 'tip' else 1 for node in self.nodes[frame_num]]])
        t2_type_lut = xp.array([[0 if track.node_types[-1] == 'tip' else 2 for track in self.tracks]])
        combo_lut = t1_type_lut + t2_type_lut.T  # 0 is t-t, 1 is j-t, 2 is t-j, 3 is j-j
        combo_lut[xp.isnan(assignment_matrix)] = -1  # -1 if no assignment possible
        # self.combo_lut = combo_lut

        min_t1 = xp.argmin(
            xp.nan_to_num(assignment_matrix, nan=xp.inf, posinf=xp.inf, neginf=xp.inf),
            axis=1).reshape(-1, 1)
        min_t2 = xp.argmin(
            xp.nan_to_num(assignment_matrix, nan=xp.inf, posinf=xp.inf, neginf=xp.inf),
            axis=0).reshape(-1, 1)
        t1_match_types = combo_lut[xp.arange(min_t1.shape[0]).reshape(-1, 1), min_t1[:]]
        t2_match_types = combo_lut[xp.arange(min_t1.shape[0]).reshape(-1, 1), min_t1[:]]

        # a t1 junction can be assigned to any number of t2 junctions and tips:
        t1_junction_idxs = xp.argwhere((t1_match_types == 1) ^ (t1_match_types == 3))[:, 0]
        t2_node_matches = min_t1[t1_junction_idxs]
        t2_t1_junction_matches = {}
        for i, t2_node in enumerate(t2_node_matches):
            if t2_node[0] not in t2_t1_junction_matches.keys():
                t2_t1_junction_matches[t2_node[0]] = [t1_junction_idxs[i]]
            else:
                t2_t1_junction_matches[t2_node[0]].append(t1_junction_idxs[i])

        # a t2 junction can have come from any number of t1 junctions and tips:
        t2_junction_idxs = xp.argwhere((t1_match_types == 2) ^ (t1_match_types == 3))[:, 0]
        t1_node_matches = min_t2[t2_junction_idxs]
        t1_t2_junction_matches = {}
        for i, t1_node in enumerate(t1_node_matches):
            if t1_node[0] not in t1_t2_junction_matches.keys():
                t1_t2_junction_matches[t1_node[0]] = [t2_junction_idxs[i]]
            else:
                t1_t2_junction_matches[t1_node[0]].append(t2_junction_idxs[i])
        print(t1_t2_junction_matches)

        # # Any t1 tip can go to 1 single t2 node (not multiple)
        # unique_t1_tips = [tip for tip in unique_t1 if self.tracks[tip].node_types[-1] == 'tip']
        # unique_t2_tips = [tip for tip in unique_t2 if self.nodes[frame_num][tip].node_type == 'tip']
        # self.unique_t1_tips = unique_t1_tips
        # self.unique_t2_tips = unique_t2_tips
        # self.unique_rows = unique_t1
        #
        # # Assign unique t1 tip matches to t2 nodes
        # num_assigned = 0
        # for tip in unique_t1_tips:
        #     tip_idx = xp.argwhere(indices[:, 0] == tip)
        #     node_match_num = indices[tip_idx[0], 1][0]
        #
        #     # Find all matches that this node could be assigned to
        #     other_possible_matches = xp.argwhere(indices[:, 1] == node_match_num)
        #
        #     # T2 junctions but not t2 tips can be assigned to multiple t1 tips
        #     # this is essentially a T shaped fission event
        #     if self.nodes[frame_num][node_match_num].node_type == 'junction':
        #         num_assigned += 1
        #         self.tracks[tip].add_node(self.nodes[frame_num][node_match_num])
        #         assignment_matrix[tip, :] = xp.nan  # remove this t1 tip from other possible matches
        #         # do not remove t2 junction from other possible matches
        #
        #     # If there is only 1 t2 tip matched to 1 t1 tip, assign them, and remove it from possible other matches
        #     # this is essentially a simple translation of a tip.
        #     elif len(other_possible_matches) == 1:
        #         num_assigned += 1
        #         self.tracks[tip].add_node(self.nodes[frame_num][node_match_num])
        #         assignment_matrix[tip, :] = xp.nan  # remove this t1 tip from other possible matches
        #         assignment_matrix[:, node_match_num] = xp.nan  # remove this t2 tip node from other possible matches
        #
        #     # At this point, all the t1 tips with 1 possible match have been assigned to
        #     #   t2 tips with a 1 possible assignment or t2 junctions with 1+ possible assignments
        #     # Any t1 tips with 1 possible match that remain either disappeared in t2
        #     #   or can possibly be assigned to t2 tips with 2+ possible assignments
        #     else:  # construct a cost matrix of assignments for the rest of the tips?
        #         pass
        # print(len(unique_t1_tips), num_assigned)


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

    def assign_t1_t2_unique_matches(self, unique_rows, unique_cols, assignment_matrix, frame_num):
        # Find the common elements between unique_rows and unique_cols
        common_elements = xp.intersect1d(unique_rows, unique_cols)

        # Get the pairs as a list of tuple
        confident_pairs = [(row, col) for row, col in zip(unique_rows, unique_cols)
                           if row in common_elements and col in common_elements]

        # Assign the nodes to the tracks, and sets row and column of assignment matrix to nan:
        for track_num, node_num in confident_pairs:
            self.tracks[track_num].add_node(self.nodes[frame_num][node_num])
            assignment_matrix[track_num, :] = xp.nan
            assignment_matrix[:, node_num] = xp.nan

        return assignment_matrix

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
