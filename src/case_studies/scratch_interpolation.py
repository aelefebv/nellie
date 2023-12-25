from src.im_info.im_info import ImInfo
import napari
import tifffile
import numpy as np

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
    tracks_all.append(tracks)
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
