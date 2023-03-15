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

        self.cost_matrix = None
        self.t1_t2_assignment = None

        self.track_id = 0

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
            self.t1_t2_assignment = linear_sum_assignment(self.cost_matrix)
            self._confidence_1_assignment()

    def _initialize_tracks(self):
        for frame_num in range(self.num_frames):
            node_list = []
            for node_num, node in enumerate(self.nodes[frame_num]):
                node_list.append(NodeTrack(node, frame_num, self.track_id))
                self.track_id += 1
            self.tracks[frame_num] = node_list

    def _get_assignment_matrix(self):
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

        self.cost_matrix = self._append_unassignment_costs(distance_matrix)

    def _append_unassignment_costs(self, pre_cost_matrix):
        rows, cols = pre_cost_matrix.shape
        cost_matrix = xp.ones(
            (rows+cols, rows+cols)
        ) * self.distance_thresh_um_per_sec
        cost_matrix[:rows, :cols] = pre_cost_matrix
        return cost_matrix

    def _confidence_1_assignment(self):
        for match_num in range(len(self.t1_t2_assignment[0])):
            t1_match = self.t1_t2_assignment[0][match_num]
            t2_match = self.t1_t2_assignment[1][match_num]
            # if assigned to be unmatched, skip
            if (t1_match > self.num_tracks_t1-1) or (t2_match > self.num_tracks_t2-1):
                continue

            t1_min = xp.min(self.cost_matrix[t1_match, :])
            t2_min = xp.min(self.cost_matrix[:, t2_match])
            # if min costs don't match, skip
            if t1_min != t2_min:
                continue

            assignment_cost = self.cost_matrix[t1_match, t2_match]
            # if min cost is not assignment cost, skip
            if assignment_cost != t1_min:
                continue

            # otherwise, match them
            self._match_tracks(t1_match, t2_match, assignment_cost, 1)

    def _confidence_2_assignment(self):

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

    print('hi')