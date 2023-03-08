from src.pipeline.node_props import Node, NodeConstructor
from src.io.im_info import ImInfo
from src import logger
from src.io.pickle_jar import unpickle_object
from scipy.optimize import linear_sum_assignment
import numpy as xp


class NodeTrack:
    def __init__(self, node, frame_num):
        self.nodes = [node]
        self.time_points = [node.time_point_sec]
        self.frame_nums = [frame_num]
        self.centroids_um = [node.centroid_um]
        self.instance_labels = [node.instance_label]
        self.node_types = [node.node_type]
        self.assignment_cost = [0]

        self.possible_merges_to = {}
        self.possible_emerges_from = {}

    def add_node(self, node, frame_num, assignment_cost):
        self.nodes.append(node)
        self.time_points.append(node.time_point_sec)
        self.frame_nums.append(frame_num)
        self.centroids_um.append(node.centroid_um)
        self.instance_labels.append(node.instance_label)
        self.node_types.append(node.node_type)
        self.assignment_cost.append(assignment_cost)

    def possibly_merged_to(self, node, frame_num, assignment_cost):
        if frame_num not in self.possible_merges_to.keys():
            self.possible_merges_to[frame_num] = [(node, assignment_cost)]
        else:
            self.possible_merges_to[frame_num].append((node, assignment_cost))

    def possibly_emerged_from(self, track, frame_num, assignment_cost):
        if frame_num not in self.possible_merges_to.keys():
            self.possible_merges_to[frame_num] = [(track, assignment_cost)]
        else:
            self.possible_merges_to[frame_num].append((track, assignment_cost))


class NodeTrackConstructor:
    def __init__(self, im_info: ImInfo,
                 distance_thresh_um_per_sec: float = 0.5,
                 time_thresh_sec: float = xp.inf):
        self.im_info = im_info
        self.node_constructor: NodeConstructor = unpickle_object(self.im_info.path_pickle_node)
        self.nodes: list[list[Node]] = self.node_constructor.nodes
        self.tracks: list[NodeTrack] = []
        self.num_frames = len(self.nodes)
        self.distance_thresh_um_per_sec = distance_thresh_um_per_sec
        self.time_thresh_sec = time_thresh_sec
        self.average_assignment_cost = {}
        self.num_nodes = 0
        self.num_tracks = 0

    def populate_tracks(self, num_t: int = None):
        if num_t is not None:
            num_t = min(num_t, self.num_frames)
            self.num_frames = num_t
        for frame_num in range(self.num_frames):
            if frame_num == 0:
                self._initialize_tracks(frame_num)
                continue
            cost_matrix = self._get_assignment_matrix(frame_num)
            # self.assignment_matrix = assignment_matrix
            track_nums, node_nums = linear_sum_assignment(cost_matrix)
            self.average_assignment_cost[frame_num] = cost_matrix[track_nums, node_nums].sum()/len(track_nums)
            self._assign_nodes_to_tracks(track_nums, node_nums, cost_matrix, frame_num)
            self._check_unassigned_tracks(track_nums, node_nums, cost_matrix, frame_num)
            self._check_new_tracks(track_nums, node_nums, cost_matrix, frame_num)
            # go through nodes. if unassigned, find lowest assignment (if not the unassigned one)
            #   and assign it as the fission point
            # if the fusion or fission point has too many nodes fusing / fissioning from it, keep the N lowest cost
            #   the rest should be rightfully unassigned.
        # return assignments

    def _initialize_tracks(self, frame_num):
        for node in self.nodes[0]:
            self.tracks.append(NodeTrack(node, frame_num))

    def _get_assignment_matrix(self, frame_num):
        frame_nodes = self.nodes[frame_num]
        self.num_nodes = len(frame_nodes)
        self.num_tracks = len(self.tracks)
        num_dimensions = len(frame_nodes[0].centroid_um)
        frame_time_s = frame_nodes[0].time_point_sec

        node_centroids = xp.empty((num_dimensions, 1, self.num_nodes))
        track_centroids = xp.empty((num_dimensions, self.num_tracks, 1))

        time_matrix = xp.empty((self.num_tracks, self.num_nodes))

        for node_num, node in enumerate(frame_nodes):
            node_centroids[:, 0, node_num] = node.centroid_um
        for track_num, track in enumerate(self.tracks):
            time_check = frame_time_s - track.time_points[-1]
            if time_check > self.time_thresh_sec:
                track_centroids[:, track_num, 0] = xp.inf
                continue
            track_centroids[:, track_num, 0] = track.centroids_um[-1]
            time_matrix[track_num, :] = time_check

        distance_matrix = xp.sqrt(xp.sum((node_centroids - track_centroids) ** 2, axis=0))
        distance_matrix /= time_matrix  # this is now a distance/sec matrix
        distance_matrix[distance_matrix > self.distance_thresh_um_per_sec] = xp.inf

        cost_matrix = xp.ones(
            (self.num_tracks+self.num_nodes, self.num_tracks+self.num_nodes)
        ) * self.distance_thresh_um_per_sec
        cost_matrix[:self.num_tracks, :self.num_nodes] = distance_matrix

        return cost_matrix

    def _assign_nodes_to_tracks(self, track_nums, node_nums, cost_matrix, frame_num):
        # Get all pairs of existing tracks and nodes that are assigned to each other
        valid_idx = xp.where(node_nums[:self.num_tracks] < self.num_nodes)
        valid_nodes = node_nums[valid_idx]
        valid_tracks = track_nums[valid_idx]

        # Assign each node to its corresponding track
        for node_idx, track in enumerate(valid_tracks):
            assignment_cost = cost_matrix[track, valid_nodes[node_idx]]
            node_to_assign = self.nodes[frame_num][valid_nodes[node_idx]]
            self.tracks[track].add_node(node_to_assign, frame_num, assignment_cost)

    def _check_unassigned_tracks(self, track_nums, node_nums, cost_matrix, frame_num):
        # Get a list of all the unassigned tracks
        unassigned_tracks_all = track_nums[xp.where(node_nums > self.num_nodes)]
        unassigned_tracks_all = unassigned_tracks_all[xp.where(unassigned_tracks_all < self.num_tracks)]

        # Get the cost matrix only of unassigned tracks and all nodes
        unassigned_track_cost_matrix = cost_matrix[unassigned_tracks_all, :]

        # Get coordinates of all possible nodes where the track could have merged to and save those
        unassigned_track_idx, nearby_nodes = xp.where(unassigned_track_cost_matrix < 0.5)
        for idx in range(len(unassigned_track_idx)):
            unassigned_track_num = unassigned_tracks_all[unassigned_track_idx[idx]]
            assignment_cost = cost_matrix[unassigned_track_num, nearby_nodes[idx]]
            self.tracks[unassigned_track_num].possibly_merged_to(nearby_nodes[idx], frame_num, assignment_cost)

    def _check_new_tracks(self, track_nums, node_nums, cost_matrix, frame_num):
        # Get a list of all the new tracks
        new_tracks_all = node_nums[xp.where(track_nums > self.num_tracks)]
        new_tracks_all = new_tracks_all[xp.where(new_tracks_all < self.num_nodes)]

        # Get the cost matrix only of all existing tracks and nodes that will form new tracks
        new_track_cost_matrix = cost_matrix[:, new_tracks_all]

        # Get coordinates of all possible existing tracks where the new track could have emerged from and save those
        nearby_tracks, new_track_node_idx = xp.where(new_track_cost_matrix < 0.5)
        new_track_nodes = new_tracks_all[new_track_node_idx]

        for idx, new_track_node in enumerate(new_tracks_all):
            possibly_emerged_from = nearby_tracks[xp.where(new_track_nodes == new_track_node)]
            print(possibly_emerged_from)
            new_track = NodeTrack(self.nodes[frame_num][new_track_node], frame_num)

            # new_track_node_num = new_tracks_all[idx]
            # node_locs = xp.where(new_track_node_idx == new_track_node_num)
            # possible_tracks = nearby_tracks[node_locs]
            # # print(possible_tracks)
            #
            # for loc_idx, possible_track in enumerate(possible_tracks):
            #     # print(possible_track, node_locs[loc_idx])
            #     assignment_cost = cost_matrix[possible_track, node_locs[loc_idx]]
            #     # print(assignment_cost)
            #
            # new_track.possibly_emerged_from(nearby_tracks[idx], frame_num, assignment_cost)
            # self.tracks.append(NodeTrack(node, frame_num))


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
    assignments = nodes_test.populate_tracks()
    print('hi')
    # todo visualize whats going on and make sure it's correct

