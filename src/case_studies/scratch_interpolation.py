from src.im_info.im_info import ImInfo
import napari
import tifffile
import numpy as np
import scipy

from src.segmentation.filtering import Filter
from src.segmentation.labelling import Label
from src.segmentation.mocap_marking import Markers
from src.segmentation.networking import Network
from src.tracking.flow_interpolation import interpolate_all_forward, interpolate_all_backward
from src.tracking.hu_tracking import HuMomentTracking
from src.utils.general import get_reshaped_image

ch = 0
im_path = r"D:\test_files\nelly_iono\partial_for_interp\deskewed-pre_1.ome.tif"
lowres_path = r"D:\test_files\nelly_iono\partial_for_interp\deskewed-pre_1_lowres.tif"
downsample_t = 3
# downsample_z = 2
im_info = ImInfo(im_path, ch=ch)
im_memmap = im_info.get_im_memmap(im_info.im_path)
# lowres_memmap = im_memmap[::downsample_t]
# save lowres memmap to disk
# tifffile.imwrite(lowres_path, lowres_memmap)
im_info.im_path = lowres_path
im_info.dim_sizes['T'] = im_info.dim_sizes['T']*downsample_t
# im_info.dim_sizes['Z'] = im_info.dim_sizes['Z']*3
im_info.shape = (im_info.shape[0]//downsample_t+1, im_info.shape[1], im_info.shape[2], im_info.shape[3])

num_t = None
# preprocessing = Filter(im_info, num_t, remove_edges=True)
# preprocessing.run()
#
# segmenting = Label(im_info, num_t)
# segmenting.run()
#
# networking = Network(im_info, num_t)
# networking.run()
#
# mocap_marking = Markers(im_info, num_t)
# mocap_marking.run()

# hu_tracking = HuMomentTracking(im_info, num_t)
# hu_tracking.run()

label_memmap = im_info.get_im_memmap(im_info.pipeline_paths['im_instance_label'])
label_memmap = get_reshaped_image(label_memmap, im_info=im_info)
im_memmap = im_info.get_im_memmap(im_info.im_path)
im_memmap = get_reshaped_image(im_memmap, im_info=im_info)

tracks_all = []
for start_frame in range(3, 5):
    coords = np.argwhere(label_memmap[start_frame] > 0).astype(float)
    np.random.seed(0)
    coords = coords[np.random.choice(coords.shape[0], 10000, replace=False), :].astype(float).copy()
    tracks_fw, _ = interpolate_all_forward(coords.copy(), start_frame, im_info.shape[0], im_info)
    if start_frame > 0:
        tracks_bw, _ = interpolate_all_backward(coords.copy(), start_frame, 0, im_info)
        tracks_bw = tracks_bw[::-1]
        # sort by track id
        tracks_bw = sorted(tracks_bw, key=lambda x: x[0])
    else:
        tracks_bw = []
    tracks = tracks_fw + tracks_bw

    track_dict = {}
    for track in tracks:
        if track[0] in track_dict.keys():
            track_dict[track[0]] += [track[1:]]
        else:
            track_dict[track[0]] = [track[1:]]

    for track, track_list in track_dict.items():
        track_dict[track] = list(set(tuple(x) for x in track_list))
        track_dict[track] = sorted(track_dict[track], key=lambda x: x[0])
        track_dict[track] = np.array(track_dict[track])

    new_track_dict = {}
    step_size = 1/3
    for track, track_list in track_dict.items():
        # run interpolation and create new track dict
        t, z, y, x = track_list[:, 0], track_list[:, 1], track_list[:, 2], track_list[:, 3]
        interpolated_t = np.arange(t[0], t[-1]+step_size, step_size)
        # 3rd order spline interpolation
        if len(t) <= 1:
            continue
        k = len(t) - 1
        # if k is even, it must be reduced by 1
        if k % 2 == 0:
            k -= 1
        # choosing k=1 because movement is much more twitchy than smooth in general
        interpolated_z = scipy.interpolate.make_interp_spline(t, z, k=k)(interpolated_t)
        interpolated_y = scipy.interpolate.make_interp_spline(t, y, k=k)(interpolated_t)
        interpolated_x = scipy.interpolate.make_interp_spline(t, x, k=k)(interpolated_t)
        interpolated_coords = np.stack([interpolated_t/step_size, interpolated_z, interpolated_y, interpolated_x], axis=1)
        new_track_dict[track] = interpolated_coords
        # todo note, interpolated coords can be visualized here.

    all_new_tracks = []
    for track, track_list in new_track_dict.items():
        for i in range(len(track_list)):
            all_new_tracks.append(np.concatenate([[track], track_list[i]]))
    tracks_all.append(all_new_tracks)

    # tracks_all.append(tracks_fw)
    # tracks_all.append(tracks_bw)
    # load_vecs = np.load(r"D:\test_files\nelly_iono\partial_for_interp\output\deskewed-pre_1.ome-ch0-flow_vector_array.npy")
# start_frame = 0
# coords = np.argwhere(label_memmap[start_frame] > 0).astype(float)
# np.random.seed(0)
# coords = coords[np.random.choice(coords.shape[0], 10000, replace=False), :].astype(float)
# tracks, track_properties = interpolate_all_forward(coords, start_frame, im_info.shape[0], im_info)
# load_vecs = np.load(r"D:\test_files\nelly_iono\partial_for_interp\output\deskewed-pre_1.ome-ch0-flow_vector_array.npy")

viewer = napari.Viewer()
for tracks in tracks_all:
    viewer.add_tracks(tracks)
viewer.add_image(im_memmap, name='im')
# napari.run()
# viewer.add_image(preprocessing.frangi_memmap)
# viewer.add_labels(segmenting.instance_label_memmap)
