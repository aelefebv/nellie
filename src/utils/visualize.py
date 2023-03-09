from src.pipeline.node_tracking import NodeTrack


def track_list_to_napari_track(track_list: list[NodeTrack]) -> (list[list[int, int, float, float, float]], dict):
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
            properties['confidence'].append(track.confidence[idx]+1)
            # todo could also construct other arrays of properties like tip v junction
            # todo could also construct a graph of the node consumptions and productions

            # Append the track data for this time point to the napari_track list
            napari_track.append([track_id, frame_num, z, y, x])

    return napari_track, properties
