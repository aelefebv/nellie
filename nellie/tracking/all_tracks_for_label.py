import numpy as np

from nellie.im_info.verifier import ImInfo
from nellie.tracking.flow_interpolation import interpolate_all_forward, interpolate_all_backward


class LabelTracks:
    """
    A class to track labeled objects over multiple timepoints in a microscopy image using flow interpolation.

    Attributes
    ----------
    im_info : ImInfo
        An object containing image metadata and memory-mapped image data.
    num_t : int
        Number of timepoints in the image.
    label_im_path : str
        Path to the labeled instance image.
    im_memmap : np.ndarray or None
        Memory-mapped original image data.
    label_memmap : np.ndarray or None
        Memory-mapped labeled instance image data.

    Methods
    -------
    initialize()
        Initializes memory-mapped data for both the raw image and the labeled instance image.
    run(label_num=None, start_frame=0, end_frame=None, min_track_num=0, skip_coords=1, max_distance_um=0.5)
        Runs the tracking process for labeled objects across timepoints, both forward and backward.
    """
    def __init__(self, im_info: ImInfo, num_t: int = None, label_im_path: str = None):
        """
        Initializes the LabelTracks class with image metadata, timepoints, and label image path.

        Parameters
        ----------
        im_info : ImInfo
            An instance of the ImInfo class containing image metadata and paths.
        num_t : int, optional
            Number of timepoints in the image (default is None, in which case it is inferred from the image metadata).
        label_im_path : str, optional
            Path to the labeled instance image. If not provided, defaults to the 'im_instance_label' path in `im_info`.
        """
        self.im_info = im_info
        self.num_t = num_t
        if label_im_path is None:
            label_im_path = self.im_info.pipeline_paths['im_instance_label']
        self.label_im_path = label_im_path

        if num_t is None:
            self.num_t = im_info.shape[im_info.axes.index('T')]

        self.im_memmap = None
        self.label_memmap = None

    def initialize(self):
        """
        Initializes memory-mapped data for both the raw image and the labeled instance image.

        This method prepares the image data and the labeled data for processing, mapping them into memory.
        """
        self.label_memmap = self.im_info.get_memmap(self.label_im_path)
        self.im_memmap = self.im_info.get_memmap(self.im_info.im_path)

    def run(self, label_num=None, start_frame=0, end_frame=None, min_track_num=0, skip_coords=1, max_distance_um=0.5):
        """
        Runs the tracking process for labeled objects across timepoints, using flow interpolation.

        This method uses forward and backward interpolation to track objects across multiple frames, starting from a given
        frame. It can also track specific labels or all labels in the image.

        Parameters
        ----------
        label_num : int, optional
            Label number to track. If None, all labels are tracked (default is None).
        start_frame : int, optional
            The starting frame from which to begin tracking (default is 0).
        end_frame : int, optional
            The ending frame for the tracking. If None, tracks until the last frame (default is None).
        min_track_num : int, optional
            Minimum track number to assign to the coordinates (default is 0).
        skip_coords : int, optional
            The interval at which coordinates are sampled (default is 1).
        max_distance_um : float, optional
            Maximum distance allowed for interpolation (in micrometers, default is 0.5).

        Returns
        -------
        tuple
            A list of tracks and a dictionary of track properties.
        """
        if end_frame is None:
            end_frame = self.num_t
        num_frames = self.label_memmap.shape[0] - 1
        if start_frame > num_frames:
            return [], {}
        if label_num is None:
            coords = np.argwhere(self.label_memmap[start_frame] > 0).astype(float)
        else:
            coords = np.argwhere(self.label_memmap[start_frame] == label_num).astype(float)
        if coords.shape[0] == 0:
            return [], {}
        coords = np.array(coords[::skip_coords])
        coords_copy = coords.copy()
        tracks = []
        track_properties = {}
        if start_frame < end_frame:
            tracks, track_properties = interpolate_all_forward(coords, start_frame, end_frame, self.im_info,
                                                               min_track_num, max_distance_um=max_distance_um)
        new_end_frame = 0  # max(0, end_frame - start_frame)
        if start_frame > 0:
            tracks_bw, track_properties_bw = interpolate_all_backward(coords_copy, start_frame, new_end_frame,
                                                                      self.im_info, min_track_num,
                                                                      max_distance_um=max_distance_um)
            tracks_bw = tracks_bw[::-1]
            for property in track_properties_bw.keys():
                track_properties_bw[property] = track_properties_bw[property][::-1]
            sort_idx = np.argsort([track[0] for track in tracks_bw])
            tracks_bw = [tracks_bw[i] for i in sort_idx]
            tracks = tracks_bw + tracks
            for property in track_properties_bw.keys():
                track_properties_bw[property] = [track_properties_bw[property][i] for i in sort_idx]
            if not track_properties:
                track_properties = track_properties_bw
            else:
                for property in track_properties_bw.keys():
                    track_properties[property] = track_properties_bw[property] + track_properties[property]
        return tracks, track_properties


if __name__ == "__main__":
    im_path = r"D:\test_files\nellie_longer_smorgasbord\deskewed-peroxisome.ome.tif"
    im_info = ImInfo(im_path, ch=0)
    num_t = 20
    label_tracks = LabelTracks(im_info, num_t=num_t)
    label_tracks.initialize()
    # tracks, track_properties = label_tracks.run(label_num=None, skip_coords=1)

    all_tracks = []
    all_props = {}
    max_track_num = 0
    for frame in range(num_t):
        tracks, track_properties = label_tracks.run(label_num=None, start_frame=frame, end_frame=None,
                                                    min_track_num=max_track_num,
                                                    skip_coords=100)
        all_tracks += tracks
        for property in track_properties.keys():
            if property not in all_props.keys():
                all_props[property] = []
            all_props[property] += track_properties[property]
        if len(tracks) == 0:
            break
        max_track_num = max([track[0] for track in tracks]) + 1

    # pickle tracks and track properties
    import pickle
    import datetime
    dt = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(f'{dt}-mt_tracks.pkl', 'wb') as f:
        pickle.dump(all_tracks, f)
    with open(f'{dt}-mt_props.pkl', 'wb') as f:
        pickle.dump(all_props, f)

    # import napari
    # viewer = napari.Viewer()
    #
    # raw_im = im_info.get_im_memmap(im_info.im_path)[:num_t]
    # viewer.add_image(raw_im, name='raw_im')
    # viewer.add_tracks(all_tracks, properties=all_props, name='tracks')
    # napari.run()
