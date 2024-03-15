import numpy as np

from src.im_info.im_info import ImInfo
from src.tracking.flow_interpolation import interpolate_all_forward, interpolate_all_backward
from src.utils.general import get_reshaped_image


class LabelTracks:
    def __init__(self, im_info: ImInfo, num_t: int = None, label_im_path: str = None):
        self.im_info = im_info
        self.num_t = num_t
        if label_im_path is None:
            label_im_path = self.im_info.pipeline_paths['im_instance_label_reassigned']
        self.label_im_path = label_im_path

        if num_t is None:
            self.num_t = im_info.shape[im_info.axes.index('T')]

    def initialize(self):
        self.label_memmap = self.im_info.get_im_memmap(self.label_im_path)
        self.label_memmap = get_reshaped_image(self.label_memmap, im_info=self.im_info)
        self.im_memmap = self.im_info.get_im_memmap(self.im_info.im_path)
        self.im_memmap = get_reshaped_image(self.im_memmap, im_info=self.im_info)

    def run(self, label_num=11, start_frame=0, end_frame=None, min_track_num=0, skip_coords=1, max_distance_um=0.5):
        if end_frame is None:
            end_frame = self.num_t
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
        new_end_frame = max(0, end_frame - start_frame)
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
