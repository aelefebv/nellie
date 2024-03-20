import numpy as np

from nellie.im_info.im_info import ImInfo
from nellie.tracking.flow_interpolation import interpolate_all_forward, interpolate_all_backward
from nellie.utils.general import get_reshaped_image


class LabelTracks:
    def __init__(self, im_info: ImInfo, num_t: int = None, label_im_path: str = None):
        self.im_info = im_info
        self.num_t = num_t
        if label_im_path is None:
            label_im_path = self.im_info.pipeline_paths['im_instance_label']
        self.label_im_path = label_im_path

        if num_t is None:
            self.num_t = im_info.shape[im_info.axes.index('T')]

    def initialize(self):
        self.label_memmap = self.im_info.get_im_memmap(self.label_im_path)
        self.label_memmap = get_reshaped_image(self.label_memmap, im_info=self.im_info)
        self.im_memmap = self.im_info.get_im_memmap(self.im_info.im_path)
        self.im_memmap = get_reshaped_image(self.im_memmap, im_info=self.im_info)

    def run(self, label_num=None, start_frame=0, end_frame=None, min_track_num=0, skip_coords=1, max_distance_um=0.5):
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
