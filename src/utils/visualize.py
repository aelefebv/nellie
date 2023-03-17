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

def check_child_tracks(check_track, all_tracks, napari_track_id):
    for child in check_track.children:
        child_track = all_tracks[child['track_id']]
        if child_track.checked:
            continue
        child_track.checked = True
        if len(child_track.parents) > 1:
            napari_track_id += 1
            continue
        child_track.napari_track_id = napari_track_id
        child_track.cost = child_track.parents[0]['cost']
        child_track.confidence = child_track.parents[0]['confidence']
        if len(child_track.children) > 0:
            all_tracks, napari_track_id = check_child_tracks(child_track, all_tracks, napari_track_id)
    return all_tracks, napari_track_id


def node_to_node_to_napari(track_dict: dict[int: list[nnNT]]) -> (list[list[int, int, float, float, float]], dict):
    napari_track = []
    napari_graph = {}
    napari_props = {'confidence': [], 'cost': []}
    napari_track_id = 0

    all_tracks = {}
    for frame_num, track_frames in track_dict.items():
        for track in track_frames:
            track.checked = False
            all_tracks[track.track_id] = track

    for track_id, track in all_tracks.items():
        if track.checked:
            continue
        track.checked = True
        if len(track.parents) > 1:
            napari_track_id += 1
            continue
        track.napari_track_id = napari_track_id
        track.cost = 0
        track.confidence = 0
        if len(track.children) > 0:
            all_tracks, napari_track_id = check_child_tracks(track, all_tracks, napari_track_id)
        napari_track_id += 1

    for track_id, track in all_tracks.items():
        z, y, x = track.node.centroid_um
        frame_num = track.frame_num
        napari_track.append([track.napari_track_id, frame_num, z, y, x])
        napari_props['cost'].append(track.cost)
        napari_props['confidence'].append(track.confidence)

    return napari_track, napari_props, napari_graph


def nodes_to_napari_graph(track_dict: dict[int: list[nnNT]]) -> (list[list[int, int, float, float, float]], dict):
    """Adapted from https://napari.org/stable/tutorials/tracking/cell_tracking.html"""
    import numpy as xp
    data = []
    lbep = []
    for frame_num, track_frames in track_dict.items():
        for track in track_frames:
            z, y, x = track.node.centroid_um
            label = track.track_id
            begins = frame_num
            ends = frame_num + 1
            if len(track.parents) < 1:
                parent_id = 0
            else:
                closest_parent_cost = None
                closest_parent_id = None
                for parent in track.parents:
                    if closest_parent_cost is None or closest_parent_id is None:
                        closest_parent_cost = parent['cost']
                        closest_parent_id = parent['track_id']
                        continue
                    if parent['cost'] < closest_parent_cost:
                        closest_parent_cost = parent['cost']
                        closest_parent_id = parent['track_id']
                parent_id = closest_parent_id
            lbep.append([label, begins, ends, parent_id])
            data.append([label, frame_num, z, y, x])
    data = xp.array(data)
    lbep = xp.array(lbep, dtype=xp.uint)
    full_graph = dict(lbep[:, [0, 3]])
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
        if full_graph[node] == 0:  # we found the root
            return node
        return root(full_graph[node])

    roots = {k: root(k) for k in full_graph.keys()}
    properties = {'root_id': [roots[idx] for idx in data[:, 0]]}

    return data, properties, graph

def check_children_2(tracks, child_info, label, data):
    child_track = tracks[child_info['frame']][child_info['track']]
    child_track.checked = True
    z, y, x = child_track.node.centroid_um
    data.append([label, child_info['frame'], z, y, x])
    if len(child_track.children) == 1:
        data, end_frame = check_children_2(tracks, child_track.children[0], label, data)
    else:
        end_frame = child_info['frame']
    return data, end_frame


def nodes_to_napari_graph_2(track_dict: dict[int: list[nnNT]]) -> (
list[list[int, int, float, float, float]], dict):
    """Adapted from https://napari.org/stable/tutorials/tracking/cell_tracking.html"""
    import numpy as xp
    for frame_num, track_frames in track_dict.items():
        for track in track_frames:
            track.checked = False
    lbep = []
    data = []
    unique_id = 0
    for frame_num, track_frames in track_dict.items():
        for track in track_frames:
            if track.checked:
                continue
            unique_id += 1
            track.checked = True
            label = unique_id
            begins = frame_num
            if len(track.parents) < 1:
                parents = 0
            else:
                parents = []
                for parent in track.parents:
                    parents.append(parent['track_id'])
            if len(track.children) == 1:
                data, ends = check_children_2(track_dict, track.children[0], label, data)
            else:
                ends = frame_num

            lbep.append([label, begins, ends, parents])

    data = xp.array(data)
    full_graph = {lbep_single[0]: lbep_single[3] for lbep_single in lbep}
    # full_graph = dict(lbep[:, [0, 3]])
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
    properties = {'root_id': [roots[idx] for idx in data[:, 0]]}

    return data, properties, graph
