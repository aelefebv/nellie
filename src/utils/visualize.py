from src.pipeline.tracking.node_to_track import NodeTrack as ntNT
from src.pipeline.tracking.node_to_node import NodeTrack as nnNT


def node_to_track_to_napari(track_list: list[ntNT]) -> (list[list[int, int, float, float, float]], dict):
    """
    Convert a list of NodeTrack objects to a format compatible with napari tracks.

    Parameters:
    -----------
    track_list : list[NodeTrack]
        A list of NodeTrack objects containing information about tracks.

    Returns:
    --------
    list
        A list of track information compatible with napari tracks.
        Each element in the list represents a single time point in a track and contains the following information:
        - track_id : int
        - frame_num : int
        - z : float
        - y : float
        - x : float
    """
    # Create an empty list to hold napari track data
    napari_track = []
    properties = {'confidence': []}
    # Loop through each track in the list of tracks
    for track_id, track in enumerate(track_list):
        # Loop through each time point in the current track and extract the centroid and frame number
        for idx, track_centroid in enumerate(track.centroids_um):
            z, y, x = track_centroid
            frame_num = track.frame_nums[idx]
            properties['confidence'].append(track.confidence[idx])
            # todo could also construct other arrays of properties like tip v junction
            # todo could also construct a graph of the node consumptions and productions

            # Append the track data for this time point to the napari_track list
            napari_track.append([track_id, frame_num, z, y, x])

    return napari_track, properties

def node_to_node_to_napari_graph(track_dict: dict[int: list[nnNT]]) -> (
list[list[int, int, float, float, float]], dict):
    """
    Adapted from https://napari.org/stable/tutorials/tracking/cell_tracking.html
    Convert a dictionary of tracks into a list and graph that can be visualized in napari.

    Parameters
    ----------
    track_dict : dict[int: list[nnNT]]
        A dictionary of tracks where keys are integers representing frame numbers
        and values are lists of objects of type `NodeTrack`, representing individual tracks.

    Returns
    -------
    data : list[list[int, int, float, float, float]]
        A list of lists containing the napari_track_id, frame number, z coordinate,
        y coordinate, and x coordinate for each track in `track_dict`.

    properties : dict
        A dictionary containing properties of the graph.

    graph : dict
        A dictionary representing the graph. The keys are the napari_track_id values
        from `data` and the values are the parents of each node.
    """

    import numpy as xp
    for frame_num, track_frames in track_dict.items():
        for track in track_frames:
            track.checked = False
    data = []
    unique_id = 0
    lp = []
    assignment_costs = []
    assignment_confidence = []
    for frame_num, track_frames in track_dict.items():
        for track in track_frames:
            parents = 0
            if track.checked:
                continue
            track.checked = True
            if len(track.parents) < 1:
                unique_id += 1
                track.napari_track_id = unique_id
                parents = 0
                assignment_costs.append(0)
                assignment_confidence.append(0)
            elif len(track.parents) > 1:
                unique_id += 1
                track.napari_track_id = unique_id
                parents = []
                costs = []
                confidence = []
                for parent in track.parents:
                    parent_track_id = track_dict[parent['frame']][parent['track']].napari_track_id
                    parents.append(parent_track_id)
                    costs.append(parent['cost'])
                    confidence.append(parent['confidence'])
                assignment_costs.append(xp.mean(costs))
                assignment_confidence.append(xp.max(confidence))
            elif len(track.parents) == 1:
                # if the parent track has >1 child, assign this track a new id, otherwise, same id.
                parent_track = track_dict[track.parents[0]['frame']][track.parents[0]['track']]
                parent_track_id = parent_track.napari_track_id
                if len(parent_track.children) > 1:
                    unique_id += 1
                    track.napari_track_id = unique_id
                    parents = [parent_track_id]
                else:
                    track.napari_track_id = parent_track_id
                    parents = parent_track.napari_parents
                assignment_costs.append(track.parents[0]['cost'])
                assignment_confidence.append(track.parents[0]['confidence'])
            track.napari_parents = parents
            z, y, x = track.node.centroid_um
            data.append([track.napari_track_id, frame_num, z, y, x])
            lp.append([track.napari_track_id, parents])

    data = xp.array(data)
    full_graph = {lp_single[0]: lp_single[1] for lp_single in lp}
    graph = {k: v for k, v in full_graph.items() if v != 0}

    def root(node: int):
        """Recursive function to determine the root node of each subgraph.

        Parameters
        ----------
        node : int
            the track_id of the starting graph node.

        Returns
        -------
        root_id : int
           The track_id of the root of the track specified by node.
        """
        if isinstance(node, list):  # we did not find the root
            return root(full_graph[node])
        return node

    roots = {k: root(k) for k in full_graph.keys()}
    properties = {'root_id': [roots[idx] for idx in data[:, 0]],
                  'costs': assignment_costs,
                  'confidence': assignment_confidence}

    return data, properties, graph
