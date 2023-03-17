from src.pipeline.node_props import Node, NodeConstructor
from src.io.pickle_jar import unpickle_object
from src.io.im_info import ImInfo
from src import logger
import numpy as xp
from scipy.optimize import linear_sum_assignment

class NodeTrack:
    """
    A class that stores information about how nodes link to one another.

    Attributes:
        node (Node): the node object associated with the track
        parents (list): a list of NodeTrack objects that are parents of the current NodeTrack
        children (list): a list of NodeTrack objects that are children of the current NodeTrack
        splits (list): a list of NodeTrack objects that are split from the current NodeTrack
        joins (list): a list of NodeTrack objects that are joined with the current NodeTrack
        frame_num (int): the frame number associated with the NodeTrack
        track_id (int): the track ID associated with the NodeTrack
    """

    def __init__(self, node, frame_num, track_id):
        """
         Initializes a new NodeTrack object.

         Args:
             node (Node): the node object associated with the track
             frame_num (int): the frame number associated with the NodeTrack
             track_id (int): the track ID associated with the NodeTrack
         """
        self.node = node
        self.parents = []
        self.children = []
        self.splits = []
        self.joins = []
        self.frame_num = frame_num
        self.track_id = track_id

class NodeTrackConstructor:
    """
    A class for constructing node tracks and connecting them between frames.

    Attributes:
    -----------
    im_info: ImInfo
        An ImInfo object containing metadata about the images to be analyzed.
    distance_thresh_um_per_sec: float
        The maximum distance in micrometers per second between two nodes that can be connected.
        Default value is 2.

    nodes: list[list[Node]]
        A list of lists of nodes, where each sublist contains the nodes from one frame.
    tracks: dict[int: list[NodeTrack]]
        A dictionary where keys are the frame numbers and values are lists of NodeTrack objects
        containing the nodes that belong to each track in that frame.
    num_frames: int
        The number of frames in the image data.
    current_frame_num: int
        The current frame number being processed.
    num_tracks_t1: int
        The number of tracks in the previous frame.
    num_tracks_t2: int
        The number of tracks in the current frame.
    t1_remaining: list[int]
        A list of indices of tracks in the previous frame that have not yet been assigned to a track
        in the current frame.
    t2_remaining: list[int]
        A list of indices of tracks in the current frame that have not yet been assigned to a track
        in the previous frame.
    t1_t2_cost_matrix: numpy.ndarray
        A 2D numpy array where the element at row i and column j represents the cost of connecting
        track i in the previous frame to track j in the current frame.
    t1_t2_assignment: Tuple[numpy.ndarray]
        A tuple containing two 1D numpy arrays: the first array contains the indices of tracks in the
        previous frame that were assigned to tracks in the current frame, and the second array contains
        the indices of the tracks in the current frame that they were assigned to.
    t1_cost_matrix: numpy.ndarray
        A 2D numpy array where the element at row i and column j represents the cost of continuing track i
        in the previous frame to track j in the current frame.
    t2_cost_matrix: numpy.ndarray
        A 2D numpy array where the element at row i and column j represents the cost of continuing track j
        in the current frame to track i in the next frame.
    track_id: int
        An integer used to assign unique IDs to tracks.
    joins: dict
        A dictionary where keys are tuples of track IDs and values are the frame numbers in which the tracks
        were joined together.
    splits: dict
        A dictionary where keys are tuples of track IDs and values are the frame numbers in which the tracks
        were split apart.
    fissions: dict
        A dictionary where keys are tuples of track IDs and values are the frame numbers in which the tracks
        underwent fission events.
    fusions: dict
        A dictionary where keys are tuples of track IDs and values are the frame numbers in which the tracks
        underwent fusion events.
    confidence_1_linkages: dict
        A dictionary where keys are tuples of track IDs and values are the frame numbers in which they were
        linked together with high confidence.
    confidence_1_linkage_mean_std: Tuple[float, float]
        A tuple containing the mean and standard deviation of the costs of high-confidence links between tracks.
    possible_connections: list
        A list of the possible connections between tracks in the current frame based on a cost matrix.
        Each element in the list is a tuple of the form (t1_idx, t2_idx, cost), where t1_idx and t2_idx are the
        indices of the tracks in the previous and current frames, respectively, and cost is the associated cost of
        the connection.
    """
    def __init__(self, im_info: ImInfo,
                 distance_thresh_um_per_sec: float = 2):
        self.im_info = im_info

        node_constructor: NodeConstructor = unpickle_object(self.im_info.path_pickle_node)
        self.nodes: list[list[Node]] = node_constructor.nodes
        self.tracks: dict[int: list[NodeTrack]] = {}

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

        self.track_id = 1

        self.joins = {}
        self.splits = {}
        self.fissions = {}
        self.fusions = {}

        self.confidence_1_linkages = None
        self.confidence_1_linkage_mean_std = None

        self.possible_connections = None

    def populate_tracks(self, num_t: int = None) -> None:
        """
        Populate the tracks for each frame by assigning them to the nearest track in the next frame.

        Parameters:
        -----------
        num_t : int, optional
            The number of frames to track. If None, then all frames are tracked.

        Returns:
        --------
        None
        """
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
            self._reverse_confidence_assignment()
            self._check_consumptions()
            self._check_productions()
            self._check_connection_linkages()
            self._assign_remainders()

    def _initialize_tracks(self) -> None:
        """
        Initialize the tracks for each frame with the nodes.

        Returns:
        --------
        None
        """
        for frame_num in range(self.num_frames):
            node_list = []
            for node_num, node in enumerate(self.nodes[frame_num]):
                node_list.append(NodeTrack(node, frame_num, self.track_id))
                self.track_id += 1
            self.tracks[frame_num] = node_list

    def _get_t1_t2_cost_matrix(self) -> None:
        """Calculate the cost matrix for the assignment of tracks from frame T1 to frame T2."""
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
        self.min_t1 = xp.min(self.t1_t2_cost_matrix[:self.num_tracks_t1, :], axis=1)
        self.min_t2 = xp.min(self.t1_t2_cost_matrix[:, :self.num_tracks_t2], axis=0)

    def _append_unassignment_costs(self, pre_cost_matrix):
        """
        Append the unassignment costs to the cost matrix for the Hungarian algorithm.

        Parameters:
        -----------
        pre_cost_matrix : numpy.ndarray
            The pre-assignment cost matrix.

        Returns:
        --------
        cost_matrix : numpy.ndarray
            The cost matrix with unassignment costs appended to it.
        """
        rows, cols = pre_cost_matrix.shape
        cost_matrix = xp.ones(
            (rows+cols, rows+cols)
        ) * self.distance_thresh_um_per_sec
        cost_matrix[:rows, :cols] = pre_cost_matrix
        return cost_matrix

    def _assign_confidence_1_linkages(self):
        """
        Assigns confidence 1 linkages to tracks and updates the remaining tracks accordingly.
        A confidence 1 linkage is an assignment which both globally minizes the cost, and locally minimizes the cost
            for both its t1 and t2 assignment.
        """
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
        """Computes the cost matrix between tracks in the current frame and the next frame."""
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
        """
        Checks t1 for joins, identifies smallest values for each remaining t1 track, checks where the smallest value
            links to another t1 track, pairs up those links, keeps only those that have matches in both matches_1 and
            matches_2, keeps only those that match with each other, and then removes t1 tracks.

        Checks t2 for splits, identifies smallest values for each remaining t2 track, checks where the smallest value
            links to another t2 track, pairs up those links, keeps only those that have matches in both matches_1 and
            matches_2, keeps only those that match with each other, and then removes t2 tracks.
        """
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
        """
        Checks unassigned connections of t1 and t2 tracks, and connects them back via cost minimization if possible,
        but only if cost is less than the mean + std of the average c1 cost.
        """
        # Check all of t1's connected nodes that are unassigned
        # Check all of t2's connected nodes that are unassigned
        # If possible, connect them back via cost minimization but only if cost is less than mean+std of average c1 cost
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
                assignment_cost = cost_submatrix[t1_match, t2_match]
                if assignment_cost >= (self.confidence_1_linkage_mean_std[0] + 2*self.confidence_1_linkage_mean_std[1]):
                    continue
                # if assignment_cost > self.confidence_1_linkage_mean_std[0]:
                #     continue
                t1_track_num = t1_unassigned_connections[t1_match]
                t2_track_num = t2_unassigned_connections[t2_match]
                self.possible_connections.append([t1_track_num, t2_track_num, assignment_cost, 0])

    def _reverse_confidence_assignment(self):
        """
        For non-confidence 1 assignments, if the assignment is locally cost minimizing for either t1 or t2, match the
        tracks.
        """
        t1_removals = []
        t2_removals = []
        for t1_track in self.t1_remaining:

            assigned_t1_idx = self.t1_t2_assignment[0] == t1_track
            assigned_t2_track = self.t1_t2_assignment[1][assigned_t1_idx][0]
            if (assigned_t2_track >= self.num_tracks_t2) or (assigned_t2_track not in self.t2_remaining):
                continue
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

        t1_removals = []
        t2_removals = []
        for t2_track in self.t2_remaining:

            assigned_t2_idx = self.t1_t2_assignment[1] == t2_track
            assigned_t1_track = self.t1_t2_assignment[0][assigned_t2_idx][0]
            if (assigned_t1_track >= self.num_tracks_t1) or (assigned_t1_track not in self.t1_remaining):
                continue
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

    def _check_consumptions(self):
        """
        Check all unassigned t1 nodes for possible junction merge candidates within a standard deviation of
        the mean confidence 1 cost, adding them to self.possible_connections.
        """
        # self.possible_connections type 1
        # for all unassigned t1 nodes, check for any nearby junction it could have merged into
        # but only if merge cost is within a standard deviation of the mean confidence 1 cost
        # keep track of merge cost in self.possible_connections
        for t1_track in self.t1_remaining:
            possible_matches = xp.where(self.t1_t2_cost_matrix[t1_track]<self.distance_thresh_um_per_sec)[0]
            for possible_match in possible_matches:
                # if self.tracks[self.current_frame_num][possible_match].node.node_type == 'tip':
                #     continue
                assignment_cost = self.t1_t2_cost_matrix[t1_track, possible_match]
                if assignment_cost >= (self.confidence_1_linkage_mean_std[0] + 2*self.confidence_1_linkage_mean_std[1]):
                    continue
                self.possible_connections.append([t1_track, possible_match, assignment_cost, 1])

    def _check_productions(self):
        """
        Check all unassigned t2 nodes for possible junction pop-off candidates within a standard deviation
        of the mean confidence 1 cost, adding them to self.possible_connections.
        """
        # self.possible_connections type 2
        # for all unassigned t2 nodes, check for any nearby junction it could have popped off of
        # but only if merge cost is within a standard deviation of the mean confidence 1 cost
        # keep track of merge cost in self.possible_connections
        for t2_track in self.t2_remaining:
            possible_matches = xp.where(self.t1_t2_cost_matrix[:, t2_track]<self.distance_thresh_um_per_sec)[0]
            for possible_match in possible_matches:
                # if self.tracks[self.current_frame_num-1][possible_match].node.node_type == 'tip':
                #     continue
                assignment_cost = self.t1_t2_cost_matrix[possible_match, t2_track]
                if assignment_cost >= (self.confidence_1_linkage_mean_std[0] + 2*self.confidence_1_linkage_mean_std[1]):
                    continue
                self.possible_connections.append([possible_match, t2_track, assignment_cost, 2])

    def _assign_remainders(self):
        """
        Assign all nodes in self.possible_connections to each other based on lowest to highest assignment cost,
        removing any other corresponding rows after assignment. Remove any t1_track and/or t2_track if they
        are tips only.
        """
        # for all self.possible_connections, sort by lowest to highest assignment cost
        # assign based on order, while removing any other corresponding rows after assignment
        # if it's a consumption assigned (1), remove t1_track from running
        # if it's a production assigned (2), remove t2_track from running
        # if it's a node_link assigned (0), remove t1_track and/or t2_track if they are tips only
        possible_connections = xp.array(self.possible_connections)
        # sort first by cost, then by assignment type
        sort_idx = xp.lexsort((possible_connections[:, 3], possible_connections[:, 2]))
        sorted_connections = possible_connections[sort_idx]
        remove_t1 = []
        remove_t2 = []
        last_match = xp.array([])
        for row, connection in enumerate(sorted_connections):
            track_t1_num = int(connection[0])
            track_t2_num = int(connection[1])
            # if the combination was already checked, skip to the next
            if xp.array_equal(last_match, connection[:3]):
                continue
            else:
                last_match = connection[:3]

            # if the match is no longer possible due to a previous assignment, skip to the next
            if (track_t1_num in remove_t1) or (track_t2_num in remove_t2):
                continue

            track_t1 = self.tracks[self.current_frame_num-1][track_t1_num]
            track_t2 = self.tracks[self.current_frame_num][track_t2_num]
            # check for a 1-1 connection
            if connection[-1] == 0:
                self._match_tracks(track_t1_num, track_t2_num, connection[2], 2)
                if track_t1.node.node_type == 'tip': remove_t1.append(track_t1_num)
                if track_t2.node.node_type == 'tip': remove_t2.append(track_t2_num)
                continue

            # check for a 1-1 connection via consumption/production match
            i = 1
            same_match = True
            match_types = [connection[-1]]
            while same_match:
                if row+i >= len(sorted_connections):
                    same_match = False
                    continue
                if xp.array_equal(sorted_connections[row+i, :3], sorted_connections[row, :3]):
                    match_types.append(sorted_connections[row, -1])
                    i += 1
                else:
                    same_match = False

            # assign nodes to each other
            self._match_tracks(track_t1_num, track_t2_num, connection[2], 2)
            if 1 in match_types:
                if track_t2.node.node_type == 'tip' and track_t2_num not in self.t2_remaining:
                    continue
                remove_t1.append(track_t1_num)
                if track_t2.node.node_type == 'tip': remove_t2.append(track_t2_num)
            if 2 in match_types:
                if track_t1.node.node_type == 'tip' and track_t1_num not in self.t1_remaining:
                    continue
                remove_t2.append(track_t2_num)
                if track_t1.node.node_type == 'tip': remove_t1.append(track_t1_num)
        for t1_removal in set(remove_t1):
            self.t1_remaining.remove(t1_removal)
        for t2_removal in set(remove_t2):
            self.t2_remaining.remove(t2_removal)

    def _match_tracks(self,
                      track_t1_num: int,
                      track_t2_num: int,
                      assignment_cost: float,
                      confidence: int):
        """
        Matches two tracks and appends assignment information to their parents and children.

        Args:
            track_t1_num (int): Index of the first track.
            track_t2_num (int): Index of the second track.
            assignment_cost (float): Cost of the assignment.
            confidence (int): Confidence in the assignment.

        Returns:
            None
        """
        track_t1 = self.tracks[self.current_frame_num - 1][track_t1_num]
        track_t2 = self.tracks[self.current_frame_num][track_t2_num]
        t1_assignment = {'frame': self.current_frame_num,
                         'track': track_t2_num,
                         'cost': assignment_cost,
                         'confidence': confidence,
                         'track_id': track_t2.track_id}
        t2_assignment = {'frame': self.current_frame_num - 1,
                         'track': track_t1_num,
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
        """
        Matches two joined tracks and appends assignment information to their joins.

        Args:
            track_1_num (int): Index of the first track.
            track_2_num (int): Index of the second track.
            assignment_cost (float): Cost of the assignment.
            confidence (int): Confidence in the assignment.

        Returns:
            None
        """
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
        """
        Matches two split tracks and appends assignment information to their splits.

        Args:
            track_1_num (int): Index of the first track.
            track_2_num (int): Index of the second track.
            assignment_cost (float): Cost of the assignment.
            confidence (int): Confidence in the assignment.

        Returns:
            None
        """
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

    visualize = True

    if visualize:
        from src.utils import visualize
        import napari
        import tifffile

        napari_tracks, napari_props, napari_graph = visualize.node_to_node_to_napari_graph(nodes_test.tracks)
        # napari_tracks, napari_props, napari_graph = visualize.node_to_node_to_napari(nodes_test.tracks)
        viewer = napari.Viewer(ndisplay=3)
        viewer.add_image(tifffile.memmap(test.path_im_mask),
                         scale=[test.dim_sizes['Z'], test.dim_sizes['Y'], test.dim_sizes['X']],
                         rendering='iso', iso_threshold=0, opacity=0.2, contrast_limits=[0, 1])
        viewer.add_tracks(napari_tracks, graph=napari_graph, properties=napari_props)
        neighbor_layer = viewer.add_image(tifffile.memmap(test.path_im_network),
                                          scale=[test.dim_sizes['Z'], test.dim_sizes['Y'], test.dim_sizes['X']],
                                          contrast_limits=[0, 3], colormap='turbo', interpolation='nearest',
                                          opacity=0.2)
        neighbor_layer.interpolation = 'nearest'
        fissions = []
        for frame_number, splits in nodes_test.splits.items():
            for split in splits:
                if split[0] == 'protrusion':
                    continue
                point_1 = [frame_number, split[1][0], split[1][1], split[1][2]]
                point_2 = [frame_number, split[2][0], split[2][1], split[2][2]]
                fissions.append(xp.array([point_1, point_2]))
        shapes_layer = viewer.add_shapes(fissions, ndim=4, shape_type='line',
                                         edge_width=nodes_test.im_info.dim_sizes['X'],
                                         edge_color='magenta', opacity=0.1)
        fusions = []
        for frame_number, joins in nodes_test.joins.items():
            for join in joins:
                if join[0] == 'retraction':
                    continue
                point_1 = [frame_number, join[1][0], join[1][1], join[1][2]]
                point_2 = [frame_number, join[2][0], join[2][1], join[2][2]]
                fusions.append(xp.array([point_1, point_2]))
        shapes_layer = viewer.add_shapes(fusions, ndim=4, shape_type='line',
                                         edge_width=nodes_test.im_info.dim_sizes['X'],
                                         edge_color='lime', opacity=0.1)

    print('hi')


