
output_dir = r"H:\My Drive\Projects\Collaboration-based\Nelly-AEL\walking\tif_frames"


### Convert pngs to tifs
import cv2
import os
import tifffile
import numpy as np

# png frames of the walking video
frame_dir = r"H:\My Drive\Projects\Collaboration-based\Nelly-AEL\walking\frames_output"
# get all pngs in the frame_dir, convert to bw, and save as tif
pngs = [f for f in os.listdir(frame_dir) if f.endswith('.png')]
pngs.sort()
tif_list = []
for png_num, png in enumerate(pngs):
    print(f"Processing {png_num + 1}/{len(pngs)}")
    im_path = os.path.join(frame_dir, png)
    im = cv2.imread(im_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    bw_im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    tif_list.append(bw_im)
tif_stack = np.array(tif_list)
# save as tif
output_path = os.path.join(output_dir, f"bw.tif")
# axes should be TYX
tifffile.imwrite(output_path, tif_stack)


### Preprocess tifs, extract out wanted segmentation
from src.im_info.im_info import ImInfo
import os
import tifffile
import numpy as np

tifs = [f for f in os.listdir(output_dir) if f.endswith('.tif')]
im_path = os.path.join(output_dir, tifs[0])
im_info = ImInfo(im_path, dim_sizes={'X': 0.2, 'Y': 0.2, 'Z': None, 'T': 0.1})
mask_dir = r"H:\My Drive\Projects\Collaboration-based\Nelly-AEL\walking\segmentations"
# get all tifs and sort
tifs = [f for f in os.listdir(mask_dir) if f.endswith('.tif')]
tifs.sort()
#
good_chs = [3, 5, 5, 1, 3, 3, 2, 3, 5, 5, 5, 4, 5, 6, 6, 4, 4, 5, 8, 6, 5, 7, 7, 4, 5, 3, 4, 5, 5, 5, 6, 6, 7, 6, 7,
            4, 5, 6, 4, 4, 4, 4, 4, 4, 2, 2, 2, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 2, 4, 4, 5, 4, 5]
time_mask = []
for im_num, good_ch in enumerate(good_chs):
    mask_path = os.path.join(mask_dir, tifs[im_num])
    time_mask.append(tifffile.imread(mask_path)[good_ch])
    # save mask to the im_info.pipeline_paths['im_instance_label']
time_mask = np.array(time_mask)
tifffile.imwrite(im_info.pipeline_paths['im_instance_label'], time_mask)


### Run pipeline
from src.im_info.im_info import ImInfo
from src.segmentation.filtering import Filter
from src.segmentation.mocap_marking import Markers
from src.tracking.hu_tracking import HuMomentTracking
from src.utils.general import get_reshaped_image
import os

tifs = [f for f in os.listdir(output_dir) if f.endswith('.tif')]
im_path = os.path.join(output_dir, tifs[0])
im_info = ImInfo(im_path, dim_sizes={'X': 0.2, 'Y': 0.2, 'Z': None, 'T': 0.1})
num_t = None
preprocessing = Filter(im_info, num_t=num_t, remove_edges=False)
preprocessing.run(mask=False)
mocap_marking = Markers(im_info, num_t, use_im='distance')
mocap_marking.run()
hu_tracking = HuMomentTracking(im_info, num_t, max_distance_um=5)
hu_tracking.run()
im_memmap = im_info.get_im_memmap(im_info.im_path)
im_memmap = get_reshaped_image(im_memmap, im_info=im_info)


### Run interpolation
from src.tracking.all_tracks_for_label import LabelTracks
from src.im_info.im_info import ImInfo
import os

output_dir = r"H:\My Drive\Projects\Collaboration-based\Nelly-AEL\walking\tif_frames"
tifs = [f for f in os.listdir(output_dir) if f.endswith('.tif')]
im_path = os.path.join(output_dir, tifs[0])
im_info = ImInfo(im_path, dim_sizes={'X': 0.2, 'Y': 0.2, 'Z': None, 'T': 0.1})
num_t = None
label_tracks = LabelTracks(im_info=im_info, num_t=num_t, label_im_path=im_info.pipeline_paths['im_instance_label'])
label_tracks.initialize()
all_tracks = []
all_props = {}
track_by_frame = {}
props_by_frame = {}
max_track_num = 0
run_num_t = im_info.shape[im_info.axes.index('T')]
# run_num_t = 10
interp_range = 5
# for frame in range(num_t):
for frame in range(run_num_t):
    print(f"Processing frame {frame + 1}/{run_num_t}")
    end_frame = min(frame + interp_range, run_num_t)
    tracks, track_properties = label_tracks.run(label_num=1, start_frame=frame, end_frame=end_frame,
                                                min_track_num=max_track_num, skip_coords=20, max_distance_um=2)
    all_tracks += tracks
    track_by_frame[frame] = tracks
    props_by_frame[frame] = track_properties
    for property in track_properties.keys():
        if property not in all_props.keys():
            all_props[property] = []
        all_props[property] += track_properties[property]
    if not tracks:
        max_track_num = 0
    else:
        max_track_num = max([track[0] for track in tracks])+1


### Save tracks and properties
import pickle
output_dir = r"H:\My Drive\Projects\Collaboration-based\Nelly-AEL\walking"
output_path = os.path.join(output_dir, "all_tracks.pkl")
with open(output_path, 'wb') as f:
    pickle.dump(all_tracks, f)
output_path = os.path.join(output_dir, "all_props.pkl")
with open(output_path, 'wb') as f:
    pickle.dump(all_props, f)


### Visualize
import napari
import pickle
import os
from src.im_info.im_info import ImInfo
import tifffile
frame_dir = r"H:\My Drive\Projects\Collaboration-based\Nelly-AEL\walking\tif_frames"
output_dir = r"H:\My Drive\Projects\Collaboration-based\Nelly-AEL\walking"
tifs = [f for f in os.listdir(frame_dir) if f.endswith('.tif')]
im_path = os.path.join(frame_dir, tifs[0])
im_info = ImInfo(im_path, dim_sizes={'X': 0.2, 'Y': 0.2, 'Z': None, 'T': 0.1})
im_memmap = im_info.get_im_memmap(im_info.im_path)
im_mask = tifffile.imread(r"H:\My Drive\Projects\Collaboration-based\Nelly-AEL\walking\tif_frames\output\bw-ch0-im_instance_label.ome.tif")
all_tracks_path = os.path.join(output_dir, "all_tracks.pkl")
all_props_path = os.path.join(output_dir, "all_props.pkl")
with open(all_tracks_path, 'rb') as f:
    all_tracks = pickle.load(f)
with open(all_props_path, 'rb') as f:
    all_props = pickle.load(f)
viewer = napari.Viewer()
viewer.add_image(im_memmap, name='im', opacity=0.3)
viewer.add_labels(im_mask, name='mask')
viewer.add_tracks(all_tracks, properties=all_props, name='all_tracks')
# for frame_num, track_by in enumerate(track_by_frame):
#     viewer.add_tracks(track_by_frame[track_by], properties=props_by_frame[track_by], name=f'tracks_{frame_num}')
