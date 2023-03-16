from src.pipeline.node_props import Node, NodeConstructor
from src.io.pickle_jar import unpickle_object
from src.io.im_info import ImInfo
from src import logger
import numpy as xp
from scipy.optimize import linear_sum_assignment

class NodeTrack:
    def __init__(self, node, frame_num, track_id):
        # stores information about how nodes link to one another
        # the confidence of those linkages
        #
        self.node = node
        self.parents = []
        self.children = []
        self.splits = []
        self.joins = []
        self.frame_num = frame_num
        self.track_id = track_id
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

        self.num_tracks_t1 = None
        self.num_tracks_t2 = None
        self.t1_remaining = None
        self.t2_remaining = None

        self.t1_t2_cost_matrix = None
        self.t1_t2_assignment = None

        self.t1_cost_matrix = None
        self.t2_cost_matrix = None

        self.track_id = 0

        self.joins = {}
        self.splits = {}
        self.fissions = {}
        self.fusions = {}

        self.confidence_1_linkages = None
        self.confidence_1_linkage_mean_std = None

        self.possible_connections = None

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
            self._get_t1_t2_cost_matrix()
            self.t1_t2_assignment = linear_sum_assignment(self.t1_t2_cost_matrix)
            self._assign_confidence_1_linkages()
            self._get_tn_cost_matrix()
            self._assign_tn_tn_linkages()
            self._check_consumptions()
            self._check_productions()
            self._check_connection_linkages()
            self._assign_remainders()
            # self._confidence_2_assignment()

    def _initialize_tracks(self):
        for frame_num in range(self.num_frames):
            node_list = []
            for node_num, node in enumerate(self.nodes[frame_num]):
                node_list.append(NodeTrack(node, frame_num, self.track_id))
                self.track_id += 1
            self.tracks[frame_num] = node_list

    def _get_t1_t2_cost_matrix(self):
        tracks_t1 = self.tracks[self.current_frame_num-1]
        tracks_t2 = self.tracks[self.current_frame_num]
        self.num_tracks_t1 = len(tracks_t1)
        self.num_tracks_t2 = len(tracks_t2)
        num_dimensions = len(tracks_t1[0].node.centroid_um)

        self.t1_remaining = list(range(self.num_tracks_t1))
        self.t2_remaining = list(range(self.num_tracks_t2))

        t1_centroids = xp.empty((num_dimensions, self.num_tracks_t1, 1))
        t2_centroids = xp.empty((num_dimensions, 1, self.num_tracks_t2))

        time_difference = tracks_t2[0].node.time_point_sec - tracks_t1[0].node.time_point_sec

        for track_num, track in enumerate(tracks_t1):
            t1_centroids[:, track_num, 0] = track.node.centroid_um
        for track_num, track in enumerate(tracks_t2):
            t2_centroids[:, 0, track_num] = track.node.centroid_um

        distance_matrix = xp.sqrt(xp.sum((t2_centroids - t1_centroids) ** 2, axis=0))
        distance_matrix /= time_difference
        distance_matrix[distance_matrix > self.distance_thresh_um_per_sec] = xp.inf

        self.t1_t2_cost_matrix = self._append_unassignment_costs(distance_matrix)

    def _append_unassignment_costs(self, pre_cost_matrix):
        rows, cols = pre_cost_matrix.shape
        cost_matrix = xp.ones(
            (rows+cols, rows+cols)
        ) * self.distance_thresh_um_per_sec
        cost_matrix[:rows, :cols] = pre_cost_matrix
        return cost_matrix

    def _assign_confidence_1_linkages(self):
        self.possible_connections = []
        confidence_1_linkages = []
        confidence_1_linkage_costs = []
        for match_num in range(len(self.t1_t2_assignment[0])):
            t1_match = self.t1_t2_assignment[0][match_num]
            t2_match = self.t1_t2_assignment[1][match_num]
            # if assigned to be unmatched, skip
            if (t1_match > self.num_tracks_t1-1) or (t2_match > self.num_tracks_t2-1):
                continue

            t1_min = xp.min(self.t1_t2_cost_matrix[t1_match, :])
            t2_min = xp.min(self.t1_t2_cost_matrix[:, t2_match])
            # if min costs don't match, skip
            if t1_min != t2_min:
                continue

            assignment_cost = self.t1_t2_cost_matrix[t1_match, t2_match]
            # if min cost is not assignment cost, skip
            if assignment_cost != t1_min:
                continue

            # otherwise, match them
            self._match_tracks(t1_match, t2_match, assignment_cost, 1)
            self.t1_remaining.remove(t1_match)
            self.t2_remaining.remove(t2_match)
            confidence_1_linkages.append((t1_match, t2_match))
            confidence_1_linkage_costs.append(assignment_cost)
        self.confidence_1_linkages = confidence_1_linkages
        self.confidence_1_linkage_mean_std = (xp.mean(confidence_1_linkage_costs), xp.std(confidence_1_linkage_costs))

    def _get_tn_cost_matrix(self):
        for frame in ['t1', 't2']:
            if frame == 't1':
                tracks_tn = [self.tracks[self.current_frame_num-1][track] for track in self.t1_remaining]

            else:
                tracks_tn = [self.tracks[self.current_frame_num][track] for track in self.t2_remaining]

            time_difference = self.tracks[self.current_frame_num][0].node.time_point_sec - \
                              self.tracks[self.current_frame_num-1][0].node.time_point_sec

            self.num_tracks_tn = len(tracks_tn)
            num_dimensions = len(tracks_tn[0].node.centroid_um)

            tn_centroids = xp.empty((num_dimensions, 1, self.num_tracks_tn))

            for track_idx, track in enumerate(tracks_tn):
                tn_centroids[:, 0, track_idx] = track.node.centroid_um

            distance_matrix = xp.sqrt(xp.sum((tn_centroids - xp.swapaxes(tn_centroids, -1, -2)) ** 2, axis=0))
            distance_matrix /= time_difference
            distance_matrix[distance_matrix > self.distance_thresh_um_per_sec] = xp.inf
            xp.fill_diagonal(distance_matrix, xp.inf)
            # remaining_cost_matrix = self.t1_t2_cost_matrix[self.t1_remaining, :][:, self.t2_remaining]

            if frame == 't1':
                self.t1_cost_matrix = xp.concatenate(
                    [self.t1_t2_cost_matrix[self.t1_remaining, :self.num_tracks_t2], distance_matrix], axis=1
                )
            else:
                self.t2_cost_matrix = xp.concatenate(
                    [self.t1_t2_cost_matrix[:self.num_tracks_t1, self.t2_remaining], distance_matrix], axis=0
                )

    def _assign_tn_tn_linkages(self):
        # check t1 for joins
        # check the smallest values for each remaining t1 track
        t1_check = xp.argmin(self.t1_cost_matrix, axis=1)
        # check where smallest value links to another t1 track
        matches_1 = xp.argwhere(t1_check >= self.num_tracks_t2)
        # pair up those links
        matches_2 = t1_check[matches_1]-self.num_tracks_t2
        # keep only those that have matches in both matches_1 and matches_2
        matches = [(matches_1[i][0], matches_2[i][0]) for i in range(len(matches_1)) if matches_1[i] in matches_2]
        # keep only those that match with each other
        kept = [match for match in matches if tuple(reversed(match)) in matches]
        remove_t1_tracks = []
        self.joins[self.current_frame_num - 1] = []
        for match in kept:
            track_1_num = self.t1_remaining[match[0]]
            track_2_num = self.t1_remaining[match[1]]
            assignment_cost = self.t1_cost_matrix[match[0], match[1]+self.num_tracks_t2]
            confidence = 2
            self._match_joined_tracks(track_1_num, track_2_num, assignment_cost, confidence)
            remove_t1_tracks.append(track_1_num)

        for track in remove_t1_tracks:
            self.t1_remaining.remove(track)

        remove_t2_tracks = []
        # check t2 for splits
        t2_check = xp.argmin(self.t2_cost_matrix, axis=0)
        matches_1 = xp.argwhere(t2_check >= self.num_tracks_t1)
        matches_2 = t2_check[matches_1] - self.num_tracks_t1
        matches = [(matches_1[i][0], matches_2[i][0]) for i in range(len(matches_1)) if matches_1[i] in matches_2]
        kept = [match for match in matches if tuple(reversed(match)) in matches]
        self.splits[self.current_frame_num] = []
        for match in kept:
            track_1_num = self.t2_remaining[match[0]]
            track_2_num = self.t2_remaining[match[1]]
            assignment_cost = self.t2_cost_matrix[match[0] + self.num_tracks_t1, match[1]]
            confidence = 2
            self._match_split_tracks(track_1_num, track_2_num, assignment_cost, confidence)
            remove_t2_tracks.append(track_1_num)

        for track in remove_t2_tracks:
            self.t2_remaining.remove(track)

    def _check_connection_linkages(self):
        # Check all of t1's connected nodes that are unassigned
        # Check all of t2's connected nodes that are unassigned
        # If possible, connect them back via cost minimization
        for t1_track, t2_track in self.confidence_1_linkages:
            t1_all_connections = self.tracks[self.current_frame_num-1][t1_track].node.connected_nodes
            t2_all_connections = self.tracks[self.current_frame_num][t2_track].node.connected_nodes
            t1_unassigned_connections = [t1_all_connection for t1_all_connection in t1_all_connections
                                         if t1_all_connection in self.t1_remaining]
            t2_unassigned_connections = [t2_all_connection for t2_all_connection in t2_all_connections
                                         if t2_all_connection in self.t2_remaining]
            if (len(t1_unassigned_connections) == 0) or (len(t2_unassigned_connections) == 0):
                continue
            pre_cost_submatrix = self.t1_t2_cost_matrix[t1_unassigned_connections, :][:, t2_unassigned_connections]
            cost_submatrix = self._append_unassignment_costs(pre_cost_submatrix)
            matches = linear_sum_assignment(cost_submatrix)
            for match_num in range(len(matches[0])):
                t1_match = matches[0][match_num]
                t2_match = matches[1][match_num]
                # if assigned to be unmatched, skip
                if (t1_match > len(t1_unassigned_connections) - 1) or (t2_match > len(t2_unassigned_connections) - 1):
                    continue

                # otherwise, match them if assignment cost is lowest of valid assignments.
                # todo, need to also test against consumption/production costs, but on the right track
                # todo maybe just make a running list of all possible connections, then pick the smallest
                assignment_cost = cost_submatrix[t1_match, t2_match]
                # if assignment_cost > self.confidence_1_linkage_mean_std[0]:
                #     continue
                t1_track_num = t1_unassigned_connections[t1_match]
                t2_track_num = t2_unassigned_connections[t2_match]
                self.possible_connections.append([t1_track_num, t2_track_num, assignment_cost, 0])
                # valid_connections_t1 = self.t1_t2_cost_matrix[t1_track_num, :][self.t2_remaining]
                # valid_connections_t1 = valid_connections_t1[valid_connections_t1 < self.distance_thresh_um_per_sec]
                # if assignment_cost != xp.min(valid_connections_t1):
                #     continue
                # self._match_tracks(t1_track_num, t2_track_num, assignment_cost, 2)
                # self.t1_remaining.remove(t1_track_num)
                # self.t2_remaining.remove(t2_track_num)

    def _confidence_2_assignment(self):
        # if a t1 track only has 2 possible assignments,
        # and it's assigned to a t2 track's lowest cost possibility, assign it
        t1_removals = []
        t2_removals = []
        for t1_track in self.t1_remaining:
            possible_assignments = xp.array(
                self.t1_t2_cost_matrix[t1_track, self.t1_t2_cost_matrix[t1_track, :] < self.distance_thresh_um_per_sec]
            )
            if len(possible_assignments) > 2:
                continue

            assigned_t1_idx = self.t1_t2_assignment[0] == t1_track
            assigned_t2_track = self.t1_t2_assignment[1][assigned_t1_idx][0]
            min_t2_cost = xp.min(self.t1_t2_cost_matrix[:, assigned_t2_track])
            assignment_cost = self.t1_t2_cost_matrix[t1_track, assigned_t2_track]
            if (min_t2_cost != assignment_cost) or (assignment_cost >= self.distance_thresh_um_per_sec):
                continue

            # otherwise, match them
            self._match_tracks(t1_track, assigned_t2_track, assignment_cost, 2)
            t1_removals.append(t1_track)
            t2_removals.append(assigned_t2_track)
        for t1_removal in t1_removals:
            self.t1_remaining.remove(t1_removal)
        for t2_removal in t2_removals:
            self.t2_remaining.remove(t2_removal)

        # if a t2 track only has 2 possible assignments,
        # and it's assigned to a t1 track's lowest cost possibility, assign it
        t1_removals = []
        t2_removals = []
        for t2_track in self.t2_remaining:
            possible_assignments = xp.array(
                self.t1_t2_cost_matrix[self.t1_t2_cost_matrix[:, t2_track] < self.distance_thresh_um_per_sec, t2_track]
            )
            if len(possible_assignments) > 2:
                continue

            assigned_t2_idx = self.t1_t2_assignment[1] == t2_track
            assigned_t1_track = self.t1_t2_assignment[0][assigned_t2_idx][0]
            min_t2_cost = xp.min(self.t1_t2_cost_matrix[assigned_t1_track, :])
            assignment_cost = self.t1_t2_cost_matrix[assigned_t1_track, t2_track]
            if (min_t2_cost != assignment_cost) or (assignment_cost >= self.distance_thresh_um_per_sec):
                continue

            # otherwise, match them
            self._match_tracks(assigned_t1_track, t2_track, assignment_cost, 2)
            t1_removals.append(assigned_t1_track)
            t2_removals.append(t2_track)
        for t1_removal in t1_removals:
            self.t1_remaining.remove(t1_removal)
        for t2_removal in t2_removals:
            self.t2_remaining.remove(t2_removal)

    def _confidence_3_assignment(self):
        pass

    def _check_consumptions(self):
        # self.possible_connections type 1
        # for all unassigned t1 nodes, check for any nearby junction it could have merged into
        # keep track of merge cost in self.possible_connections
        for t1_track in self.t1_remaining:
            possible_matches = xp.where(self.t1_t2_cost_matrix[t1_track]<self.distance_thresh_um_per_sec)[0]
            for possible_match in possible_matches:
                if self.tracks[self.current_frame_num][possible_match].node.node_type == 'tip':
                    continue
                assignment_cost = self.t1_t2_cost_matrix[t1_track, possible_match]
                self.possible_connections.append([t1_track, possible_match, assignment_cost, 1])

    def _check_productions(self):
        # self.possible_connections type 2
        # for all unassigned t2 nodes, check for any nearby junction it could have popped off of
        # keep track of merge cost in self.possible_connections
        for t2_track in self.t2_remaining:
            possible_matches = xp.where(self.t1_t2_cost_matrix[:, t2_track]<self.distance_thresh_um_per_sec)[0]
            for possible_match in possible_matches:
                if self.tracks[self.current_frame_num-1][possible_match].node.node_type == 'tip':
                    continue
                assignment_cost = self.t1_t2_cost_matrix[possible_match, t2_track]
                self.possible_connections.append([possible_match, t2_track, assignment_cost, 2])
                print(possible_match, t2_track, assignment_cost)

    def _assign_remainders(self):
        # for all self.possible_connections, sort by lowest to highest assignment cost
        # assign based on order, while removing any other corresponding rows after assignment
        # if it's a consumption assigned (1), remove t1_track from running
        # if it's a production assigned (2), remove t2_track from running
        # if it's a node_link assigned (0), remove t1_track and/or t2_track if they are tips only
        # make sure lone tips are properly accounted for.
        pass

    def _match_tracks(self,
                      track_t1_num: int,
                      track_t2_num: int,
                      assignment_cost: float,
                      confidence: int):
        track_t1 = self.tracks[self.current_frame_num - 1][track_t1_num]
        track_t2 = self.tracks[self.current_frame_num][track_t2_num]
        t1_assignment = {'frame': self.current_frame_num,
                         'track': track_t1_num,
                         'cost': assignment_cost,
                         'confidence': confidence,
                         'track_id': track_t2.track_id}
        t2_assignment = {'frame': self.current_frame_num - 1,
                         'track': track_t2_num,
                         'cost': assignment_cost,
                         'confidence': confidence,
                         'track_id': track_t1.track_id}
        track_t1.children.append(t1_assignment)
        track_t2.parents.append(t2_assignment)

    def _match_joined_tracks(self,
                             track_1_num: int,
                             track_2_num: int,
                             assignment_cost: float,
                             confidence: int):
        track_t1 = self.tracks[self.current_frame_num - 1][track_1_num]
        track_t2 = self.tracks[self.current_frame_num - 1][track_2_num]
        if 'junction' in [track_t1.node.node_type, track_t2.node.node_type]:
            join_type = 'retraction'
        else:
            join_type = 'fusion'
        t1_assignment = {'frame': self.current_frame_num - 1,
                         'track': track_2_num,
                         'cost': assignment_cost,
                         'confidence': confidence,
                         'track_id': track_t2.track_id,
                         'join_type': join_type}
        track_t1.joins.append(t1_assignment)
        self.joins[self.current_frame_num-1].append((join_type, track_t1.node.centroid_um, track_t2.node.centroid_um))

    def _match_split_tracks(self,
                            track_1_num: int,
                            track_2_num: int,
                            assignment_cost: float,
                            confidence: int):
        track_t1 = self.tracks[self.current_frame_num][track_1_num]
        track_t2 = self.tracks[self.current_frame_num][track_2_num]
        if 'junction' in [track_t1.node.node_type, track_t2.node.node_type]:
            split_type = 'protrusion'
        else:
            split_type = 'fission'
        t1_assignment = {'frame': self.current_frame_num,
                         'track': track_2_num,
                         'cost': assignment_cost,
                         'confidence': confidence,
                         'track_id': track_t2.track_id,
                         'split_type': split_type}
        track_t1.splits.append(t1_assignment)
        self.splits[self.current_frame_num].append((split_type, track_t1.node.centroid_um, track_t2.node.centroid_um))


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

    visualize = False

    if visualize:
        from src.utils.visualize import node_to_node_to_napari
        import napari
        import tifffile

        napari_tracks, napari_props, napari_graph = node_to_node_to_napari(nodes_test.tracks)
        viewer = napari.Viewer(ndisplay=3)
        viewer.add_image(tifffile.memmap(test.path_im_mask),
                         scale=[test.dim_sizes['Z'], test.dim_sizes['Y'], test.dim_sizes['X']],
                         rendering='iso', iso_threshold=0, opacity=0.2, contrast_limits=[0, 1])
        viewer.add_tracks(napari_tracks, properties=napari_props, color_by='confidence')
        neighbor_layer = viewer.add_image(tifffile.memmap(test.path_im_network),
                                          scale=[test.dim_sizes['Z'], test.dim_sizes['Y'], test.dim_sizes['X']],
                                          contrast_limits=[0, 3], colormap='turbo', interpolation='nearest',
                                          opacity=0.2)
        neighbor_layer.interpolation = 'nearest'
        fissions = []
        for frame_num, splits in nodes_test.splits.items():
            for split in splits:
                if split[0] == 'protrusion':
                    continue
                point_1 = [frame_num, split[1][0], split[1][1], split[1][2]]
                point_2 = [frame_num, split[2][0], split[2][1], split[2][2]]
                fissions.append(xp.array([point_1, point_2]))
        shapes_layer = viewer.add_shapes(fissions, ndim=4, shape_type='line',
                                         edge_width=nodes_test.im_info.dim_sizes['X'],
                                         edge_color='magenta', opacity=0.1)
        fusions = []
        for frame_num, joins in nodes_test.joins.items():
            for join in joins:
                if join[0] == 'retraction':
                    continue
                point_1 = [frame_num, join[1][0], join[1][1], join[1][2]]
                point_2 = [frame_num, join[2][0], join[2][1], join[2][2]]
                fusions.append(xp.array([point_1, point_2]))
        shapes_layer = viewer.add_shapes(fusions, ndim=4, shape_type='line',
                                         edge_width=nodes_test.im_info.dim_sizes['X'],
                                         edge_color='lime', opacity=0.1)

    print('hi')