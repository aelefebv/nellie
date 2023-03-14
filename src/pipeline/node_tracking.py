from src.pipeline.node_props import Node, NodeConstructor
from src.io.im_info import ImInfo
from src import logger
from src.io.pickle_jar import unpickle_object
from scipy.optimize import linear_sum_assignment
import numpy as xp

# todo keep track of those mito that are well separated from others (some kind of # nearest neighbors)
# todo reassign unconfident tracks that are pretty sure to be valid, maybe based on # nearest neighbors?

class NodeTrack:
    def __init__(self, node, frame_num, node_num):
        self.nodes = [node]
        self.time_points = [node.time_point_sec]
        self.frame_nums = [frame_num]
        self.centroids_um = [node.centroid_um]
        self.instance_labels = [node.instance_label]
        self.node_num = [node_num]
        self.node_types = [node.node_type]
        self.assignment_cost = [0]
        self.confidence = [0]

        self.possibly_consumed_by = {}
        self.possibly_produced_by = {}
        self.possibly_consumed = {}
        self.possibly_produced = {}

        self.blocked = False

    def add_node(self, node, frame_num, assignment_cost, confident, node_num):
        self.nodes.append(node)
        self.time_points.append(node.time_point_sec)
        self.frame_nums.append(frame_num)
        self.centroids_um.append(node.centroid_um)
        self.instance_labels.append(node.instance_label)
        self.node_num.append(node_num)
        self.node_types.append(node.node_type)
        self.assignment_cost.append(assignment_cost)
        self.confidence.append(confident)


class NodeTrackConstructor:
    def __init__(self, im_info: ImInfo,
                 distance_thresh_um_per_sec: float = 2,
                 time_thresh_sec: float = xp.inf):

        self.im_info = im_info
        self.node_constructor: NodeConstructor = unpickle_object(self.im_info.path_pickle_node)
        self.nodes: list[list[Node]] = self.node_constructor.nodes
        self.tracks: list[NodeTrack] = []

        self.distance_thresh_um_per_sec = distance_thresh_um_per_sec
        self.time_thresh_sec = time_thresh_sec

        self.average_assignment_cost = {}
        self.average_std_assignment_cost_unconfident = {}
        self.average_std_assignment_cost_confident = {}

        self.num_frames = len(self.nodes)
        self.num_nodes = 0
        self.num_tracks = 0

        self.assigned_unassigned = {}

        self.tracks_to_assign = []
        self.nodes_to_assign = []

        self.new_tracks = None

    def _possibly_produced(self, producing_track, produced_track, production_cost):
        producing_track_num, production_frame = producing_track
        produced_track_num, produced_frame = produced_track
        # producing track num possibly produced produced track num in produced frame with a production cost
        if produced_frame not in self.tracks[producing_track_num].possibly_produced.keys():
            self.tracks[producing_track_num].possibly_produced[produced_frame] = [(
                produced_track_num, production_cost)]
        else:
            self.tracks[producing_track_num].possibly_produced[produced_frame].append(
                (produced_track_num, production_cost))
        # produced track num possible produced by producing track in production frame with a production cost
        if production_frame not in self.tracks[produced_track_num].possibly_produced_by.keys():
            self.tracks[produced_track_num].possibly_produced_by[production_frame] = [(
                producing_track_num, production_cost)]
        else:
            self.tracks[produced_track_num].possibly_produced_by[production_frame].append(
                (producing_track_num, production_cost))

    def _possibly_consumed(self, consuming_track, consumed_track, consumption_cost):
        consuming_track_num, consumption_frame = consuming_track
        consumed_track_num, consumed_frame = consumed_track
        # consuming track num possibly consumed consumed track num in consumed frame with a consumption cost
        if consumed_frame not in self.tracks[consuming_track_num].possibly_consumed.keys():
            self.tracks[consuming_track_num].possibly_consumed[consumed_frame] = [(
                consumed_track_num, consumption_cost)]
        else:
            self.tracks[consuming_track_num].possibly_consumed[consumed_frame].append(
                (consumed_track_num, consumption_cost))
        # consumed track num possible consumed by consuming track in consumption frame with a consumption cost
        if consumption_frame not in self.tracks[consumed_track_num].possibly_consumed_by.keys():
            self.tracks[consumed_track_num].possibly_consumed_by[consumption_frame] = [(
                consuming_track_num, consumption_cost)]
        else:
            self.tracks[consumed_track_num].possibly_consumed_by[consumption_frame].append(
                (consuming_track_num, consumption_cost))

    def populate_tracks(self, num_t: int = None):
        if num_t is not None:
            num_t = min(num_t, self.num_frames)
            self.num_frames = num_t
        for frame_num in range(self.num_frames):
            logger.debug(f'Tracking frame {frame_num}/{self.num_frames - 1}')
            if frame_num == 0:
                self._initialize_tracks(frame_num)
                continue
            self.new_tracks = []
            cost_matrix = self._get_assignment_matrix(frame_num)
            # self.assignment_matrix = assignment_matrix
            track_nums, node_nums = linear_sum_assignment(cost_matrix)
            self.track_nums = track_nums
            self.node_nums = node_nums
            self.average_assignment_cost[frame_num] = cost_matrix[track_nums, node_nums].sum() / len(track_nums)
            self._assign_confidence_1_matches(track_nums, node_nums, cost_matrix, frame_num)
            self._assign_confidence_2_matches(track_nums, node_nums, cost_matrix, frame_num)
            self._assign_confidence_3_matches(cost_matrix, frame_num)
            # this should be very last measure...
            # self._assign_confidence_4_matches(cost_matrix, frame_num)
            self._check_new_tracks(cost_matrix, frame_num)
            self._back_assign_track_to_nodes(frame_num)
            self._check_unassigned_tracks(cost_matrix, frame_num)
            # go through nodes. if unassigned, find lowest assignment (if not the unassigned one)
            #   and assign it as the fission point
            # if the fusion or fission point has too many nodes fusing / fissioning from it, keep the N lowest cost
            #   the rest should be rightfully unassigned.
        # return assignments

    def _initialize_tracks(self, frame_num):
        for node_num, node in enumerate(self.nodes[0]):
            node.assigned_track = node_num
            self.tracks.append(NodeTrack(node, frame_num, node_num))

    def _append_unassignment_costs(self, pre_cost_matrix):
        rows, cols = pre_cost_matrix.shape
        cost_matrix = xp.ones(
            (rows+cols, rows+cols)
        ) * self.distance_thresh_um_per_sec
        cost_matrix[:rows, :cols] = pre_cost_matrix
        return cost_matrix

    def _get_assignment_matrix(self, frame_num):
        frame_nodes = self.nodes[frame_num]
        self.num_nodes = len(frame_nodes)
        self.num_tracks = len(self.tracks)
        self.tracks_to_assign = list(range(self.num_tracks))
        self.nodes_to_assign = list(range(self.num_nodes))

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

        cost_matrix = self._append_unassignment_costs(distance_matrix)
        for track in range(distance_matrix.shape[0]):
            if self.tracks[track].blocked:
                cost_matrix[track, :] = xp.inf
                self.tracks_to_assign.remove(track)
        self.cost_matrix = cost_matrix
        return cost_matrix

    def _assign_confidence_1_matches(self, track_nums, node_nums, cost_matrix, frame_num):

        # First for nodes
        for check_node_num in self.nodes_to_assign:
            check_match_idx = xp.where(node_nums == check_node_num)[0][0]
            check_track_num = track_nums[check_match_idx]
            # Skip if this node was assigned to start a new track
            if check_track_num >= self.num_tracks:
                continue
            node_to_assign = self.nodes[frame_num][check_node_num]
            assignment_cost = cost_matrix[check_track_num, check_node_num]

            possible_assignments = xp.array(
                cost_matrix[cost_matrix[:, check_node_num] < self.distance_thresh_um_per_sec, check_node_num]
            )
            sorted_possible = xp.sort(possible_assignments)


            # Assign nodes to their matched tracks if that match is the lowest possible match cost
            if assignment_cost == sorted_possible[0]:
                # check the track's costs. If cost of assignment is higher than its min match cost, skip.
                track_possible_assignments = xp.array(
                    cost_matrix[check_track_num, cost_matrix[check_track_num, :] < self.distance_thresh_um_per_sec]
                )
                min_track_cost = xp.min(track_possible_assignments)
                if assignment_cost-min_track_cost > 0:
                    continue
                self.tracks[check_track_num].add_node(node_to_assign, frame_num, assignment_cost,
                                                      confident=1, node_num=check_node_num)
                node_to_assign.assigned_track = check_track_num
                self.tracks_to_assign.remove(check_track_num)
                self.nodes_to_assign.remove(check_node_num)

        # Then for tracks
        for check_track_num in self.tracks_to_assign:
            check_match_idx = xp.where(track_nums == check_track_num)[0][0]
            check_node_num = node_nums[check_match_idx]
            # Skip if this track was assigned to be lost
            if check_node_num >= self.num_nodes:
                continue
            node_to_assign = self.nodes[frame_num][check_node_num]
            assignment_cost = cost_matrix[check_track_num, check_node_num]

            possible_assignments = xp.array(
                cost_matrix[check_track_num, cost_matrix[check_track_num, :] < self.distance_thresh_um_per_sec]
            )
            sorted_possible = xp.sort(possible_assignments)

            # Assign tracks to their matched nodes if that match is the lowest possible match cost
            if assignment_cost == sorted_possible[0]:
                # check the node's costs. If cost of assignment is higher than its min match cost, skip.
                node_possible_assignments = xp.array(
                    cost_matrix[cost_matrix[:, check_node_num] < self.distance_thresh_um_per_sec, check_node_num]
                )
                min_node_cost = xp.min(node_possible_assignments)
                if assignment_cost - min_node_cost > 0:
                    continue
                self.tracks[check_track_num].add_node(node_to_assign, frame_num, assignment_cost,
                                                      confident=1, node_num=check_node_num)
                node_to_assign.assigned_track = check_track_num
                self.tracks_to_assign.remove(check_track_num)
                self.nodes_to_assign.remove(check_node_num)

    def _assign_confidence_2_matches(self, track_nums, node_nums, cost_matrix, frame_num):
        # First for nodes
        for check_node_num in self.nodes_to_assign:
            check_match_idx = xp.where(node_nums == check_node_num)[0][0]
            check_track_num = track_nums[check_match_idx]
            # todo could make this faster by only feeding in list of nodes/tracks that are valid.
            # Skip if this node was assigned to start a new track
            if check_track_num >= self.num_tracks:
                continue
            node_to_assign = self.nodes[frame_num][check_node_num]
            assignment_cost = cost_matrix[check_track_num, check_node_num]

            possible_assignments = xp.array(
                cost_matrix[cost_matrix[:, check_node_num] < self.distance_thresh_um_per_sec, check_node_num]
            )
            possible_assignments = xp.concatenate([possible_assignments,
                                    xp.array(cost_matrix[
                                        check_track_num,
                                        cost_matrix[check_track_num, :] < self.distance_thresh_um_per_sec]
                                    )])
            # If it has only one other possible match, and it's assigned to it, assign it.
            if len(possible_assignments) == 2:
                self.tracks[check_track_num].add_node(node_to_assign, frame_num, assignment_cost,
                                                      confident=2, node_num=check_node_num)
                node_to_assign.assigned_track = check_track_num
                self.tracks_to_assign.remove(check_track_num)
                self.nodes_to_assign.remove(check_node_num)

        # Then for tracks
        for check_track_num in self.tracks_to_assign:
            check_match_idx = xp.where(track_nums == check_track_num)[0][0]
            check_node_num = node_nums[check_match_idx]
            # Skip if this track was assigned to be lost
            if check_node_num >= self.num_nodes:
                continue
            node_to_assign = self.nodes[frame_num][check_node_num]
            assignment_cost = cost_matrix[check_track_num, check_node_num]

            possible_assignments = xp.array(
                cost_matrix[check_track_num, cost_matrix[check_track_num, :] < self.distance_thresh_um_per_sec]
            )
            possible_assignments = xp.concatenate([possible_assignments,
                                    xp.array(cost_matrix[
                                        cost_matrix[:, check_node_num] < self.distance_thresh_um_per_sec,
                                        check_node_num]
                                    )])
            # If it has only one other possible match, and it's assigned to it, assign it.
            if len(possible_assignments) == 2:
                self.tracks[check_track_num].add_node(node_to_assign, frame_num, assignment_cost,
                                                      confident=2, node_num=check_node_num)
                node_to_assign.assigned_track = check_track_num
                self.tracks_to_assign.remove(check_track_num)
                self.nodes_to_assign.remove(check_node_num)

    def _assign_confidence_3_matches(self, cost_matrix, frame_num):
        valid_cost_matrix = cost_matrix[self.tracks_to_assign, :][:, self.nodes_to_assign]
        valid_costs = valid_cost_matrix[valid_cost_matrix<self.distance_thresh_um_per_sec]
        valid_costs.sort()
        hold_tracks_to_assign = self.tracks_to_assign.copy()
        hold_nodes_to_assign = self.nodes_to_assign.copy()
        for valid_cost in valid_costs:
            min_each_col = xp.min(valid_cost_matrix, axis=0)
            min_col = xp.argmin(min_each_col)
            min_each_row = xp.min(valid_cost_matrix, axis=1)
            min_row = xp.argmin(min_each_row)
            min_all = xp.argmin([min_each_col[min_col], min_each_row[min_row]])
            if min_all == 0:
                col = min_col
                row = xp.argmin(valid_cost_matrix[:, col])
            else:
                row = min_row
                col = xp.argmin(valid_cost_matrix[row, :])
            valid_cost_matrix[row, col] = xp.inf
            track_to_check = hold_tracks_to_assign[row]
            node_to_check = hold_nodes_to_assign[col]
            if (track_to_check not in self.tracks_to_assign) or (node_to_check not in self.nodes_to_assign):
                continue
            lowest_costs = cost_matrix[track_to_check, :]
            lowest_costs = xp.concatenate([lowest_costs, cost_matrix[:, node_to_check]])
            if xp.min(lowest_costs) == valid_cost:
                node_to_assign = self.nodes[frame_num][node_to_check]
                self.tracks[track_to_check].add_node(node_to_assign, frame_num, valid_cost,
                                                     confident=3, node_num=node_to_check)
                node_to_assign.assigned_track = track_to_check
                self.tracks_to_assign.remove(track_to_check)
                self.nodes_to_assign.remove(node_to_check)
        return

    def _assign_confidence_4_matches(self, full_cost_matrix, frame_num):
        # do a cost minimization on whatever tracks and nodes are left.
        valid_pre_cost_matrix = full_cost_matrix[self.tracks_to_assign, :][:, self.nodes_to_assign]
        valid_cost_matrix = self._append_unassignment_costs(valid_pre_cost_matrix)
        track_nums, node_nums = linear_sum_assignment(valid_cost_matrix)
        tracks_to_remove = []
        nodes_to_remove = []
        for idx in range(len(track_nums)):
            if track_nums[idx] > len(self.tracks_to_assign)-1 or node_nums[idx] > len(self.nodes_to_assign)-1:
                continue
            track_num = self.tracks_to_assign[track_nums[idx]]
            node_num = self.nodes_to_assign[node_nums[idx]]
            node_to_assign = self.nodes[frame_num][node_num]
            assignment_cost = valid_pre_cost_matrix[track_nums[idx], node_nums[idx]]
            self.tracks[track_num].add_node(node_to_assign, frame_num, assignment_cost,
                                            confident=4, node_num=node_num)
            node_to_assign.assigned_track = track_num
            tracks_to_remove.append(track_num)
            nodes_to_remove.append(node_num)
        for track in tracks_to_remove:
            self.tracks_to_assign.remove(track)
        for node in nodes_to_remove:
            self.nodes_to_assign.remove(node)

    def _make_new_tracks(self, frame_num):
        current_track_idx = len(self.tracks)
        for node_num in self.nodes_to_assign:
            node = self.nodes[frame_num][node_num]
            self.tracks.append(NodeTrack(node, frame_num, node_num))
            self.new_tracks.append(current_track_idx)
            node.assigned_track = current_track_idx
            current_track_idx += 1

    def _check_new_tracks(self, cost_matrix, frame_num):
        # check if any two new tracks are associated
        self._make_new_tracks(frame_num)
        nearby_cost_matrix = self._check_nearby_new_tracks()
        self.dist_mat = nearby_cost_matrix
        for i in range(nearby_cost_matrix.shape[0]):
            track_1_num = self.new_tracks[i]
            possible_track_combos = xp.where(nearby_cost_matrix[i] < self.distance_thresh_um_per_sec)[0]
            possible_costs = nearby_cost_matrix[i][possible_track_combos]
            for j in range(len(possible_track_combos)):
                # if possible_track_combos[j] == i:
                #     break
                track_2_num = self.new_tracks[possible_track_combos[j]]
                assignment_cost = possible_costs[j]
                self._possibly_produced((track_1_num, frame_num), (track_2_num, frame_num), assignment_cost)

        valid_cost_matrix = cost_matrix[:, self.nodes_to_assign].T
        for idx in range(nearby_cost_matrix.shape[0]):
            new_track_num = self.new_tracks[idx]
            possible_tracks = xp.where(valid_cost_matrix[idx] < self.distance_thresh_um_per_sec)[0]
            possible_costs = valid_cost_matrix[idx][possible_tracks]
            for producing_track_idx, producing_track_num in enumerate(possible_tracks):
                producing_track = self.tracks[producing_track_num]
                # if track has not been assigned this frame, allow it
                if producing_track_num in self.tracks_to_assign:
                    self._possibly_produced(
                        (producing_track_num, producing_track.frame_nums[-1]),
                        (new_track_num, frame_num),
                        possible_costs[producing_track_idx])
                    continue
                if len(producing_track.frame_nums) < 2:
                    logger.debug("!!! This should never appear.")
                track_previous_node = producing_track.nodes[-2].node_type
                # if track was a tip last frame and already assigned, not possible, unless not yet assigned
                if (producing_track.nodes[-2].node_type == 'tip' and
                        producing_track.nodes[-1].node_type == 'tip' and
                        producing_track_num not in self.tracks_to_assign):
                    continue
                # if it was a lone tip, it could've elongated into two tips or tip and junction
                # if it was a junction, it could've grown a new tip, or 6x junction to 3x + 3x junction
                # if another tip appeared near it in the same frame, it could have been a midpoint fission.
                self._possibly_produced(
                    (producing_track_num, producing_track.frame_nums[-2]),
                    (new_track_num, frame_num),
                    possible_costs[producing_track_idx])

    def _check_nearby_new_tracks(self):
        num_new_tracks = len(self.new_tracks)
        num_dimensions = len(self.tracks[self.new_tracks[0]].centroids_um[-1])

        node_centroids = xp.empty((num_dimensions, 1, num_new_tracks))

        for new_track_idx, new_track_num in enumerate(self.new_tracks):
            node_centroids[:, 0, new_track_idx] = self.tracks[new_track_num].centroids_um[-1]

        distance_matrix = xp.sqrt(xp.sum((node_centroids - xp.swapaxes(node_centroids, -1, -2)) ** 2, axis=0))
        distance_matrix[distance_matrix > self.distance_thresh_um_per_sec] = xp.inf
        xp.fill_diagonal(distance_matrix, xp.inf)

        return distance_matrix

    def _back_assign_track_to_nodes(self, frame_num):
        for track_num, track in enumerate(self.tracks):
            if frame_num not in track.frame_nums:
                continue
            track.nodes[track.frame_nums==frame_num].assigned_track = track_num

    def _check_unassigned_tracks(self, cost_matrix, frame_num):
        nearby_cost_matrix = self._check_nearby_lost_tracks()
        self.dist_mat = nearby_cost_matrix
        for i in range(nearby_cost_matrix.shape[0]):
            track_1_num = self.tracks_to_assign[i]
            possible_track_combos = xp.where(nearby_cost_matrix[i] < self.distance_thresh_um_per_sec)[0]
            possible_costs = nearby_cost_matrix[i][possible_track_combos]
            for j in range(len(possible_track_combos)):
                # if possible_track_combos[j] == i:
                #     break
                track_2_num = self.tracks_to_assign[possible_track_combos[j]]
                if self.tracks[track_1_num].frame_nums[-1] != self.tracks[track_2_num].frame_nums[-1]:
                    continue
                assignment_cost = possible_costs[j]
                self._possibly_consumed((track_1_num, self.tracks[track_1_num].frame_nums[-1]),
                                        (track_2_num, self.tracks[track_2_num].frame_nums[-1]),
                                        assignment_cost)

        valid_cost_matrix = cost_matrix[self.tracks_to_assign, :]
        unassigned_track_nums, possible_consuming_nodes = xp.where(valid_cost_matrix < self.distance_thresh_um_per_sec)
        for idx, unassigned_track_idx in enumerate(unassigned_track_nums):
            lost_track_num = self.tracks_to_assign[unassigned_track_idx]
            node_num = possible_consuming_nodes[idx]
            node_object = self.nodes[frame_num][node_num]
            consuming_track_num = node_object.assigned_track
            consuming_track = self.tracks[consuming_track_num]
            assignment_cost = cost_matrix[lost_track_num, node_num]
            if consuming_track.frame_nums[-1] < frame_num:
                continue
            # if node has not been assigned this frame, allow it
            if node_num in self.nodes_to_assign:
                self._possibly_consumed(
                    (consuming_track_num, consuming_track.frame_nums[-1]),
                    (lost_track_num, self.tracks[lost_track_num].frame_nums[-1]),
                    assignment_cost)
                continue
            # Nothing can be consumed by tip unless it hasn't been assigned yet:
            if (node_object.node_type == 'tip'
                    and self.tracks[lost_track_num].node_types[-1] == 'tip'
                    and node_num not in self.nodes_to_assign):
                continue
            self._possibly_consumed((consuming_track_num, consuming_track.frame_nums[-1]),
                                    (lost_track_num, self.tracks[lost_track_num].frame_nums[-1]),
                                    assignment_cost)

    def _check_nearby_lost_tracks(self):
        num_lost_tracks = len(self.tracks_to_assign)
        num_dimensions = len(self.tracks[self.tracks_to_assign[0]].centroids_um[-1])

        node_centroids = xp.empty((num_dimensions, 1, num_lost_tracks))

        for lost_track_idx, lost_track_num in enumerate(self.tracks_to_assign):
            node_centroids[:, 0, lost_track_idx] = self.tracks[lost_track_num].centroids_um[-1]

        distance_matrix = xp.sqrt(xp.sum((node_centroids - xp.swapaxes(node_centroids, -1, -2)) ** 2, axis=0))
        distance_matrix[distance_matrix > self.distance_thresh_um_per_sec] = xp.inf
        xp.fill_diagonal(distance_matrix, xp.inf)

        return distance_matrix


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
    nodes_test.populate_tracks(3)
    print('hi')

    visualize = False

    if visualize:
        from src.utils.visualize import track_list_to_napari_track
        import napari
        import tifffile

        # graph eg: graph = { 782: [1280] }  track 782 merges into track 1280
        napari_tracks, properties = track_list_to_napari_track(nodes_test.tracks)
        viewer = napari.Viewer(ndisplay=3)
        viewer.add_image(tifffile.memmap(test.path_im_mask),
                         scale=[test.dim_sizes['Z'], test.dim_sizes['Y'], test.dim_sizes['X']],
                         rendering='iso', iso_threshold=0, opacity=0.2, contrast_limits=[0, 1])
        viewer.add_tracks(napari_tracks, properties=properties, color_by='confidence')
        neighbor_layer = viewer.add_image(tifffile.memmap(test.path_im_network),
                         scale=[test.dim_sizes['Z'], test.dim_sizes['Y'], test.dim_sizes['X']],
                         contrast_limits=[0, 3], colormap='turbo', interpolation='nearest', opacity=0.2)
        neighbor_layer.interpolation = 'nearest'
