import cv2
import tifffile
import os

from src.im_info.im_info import ImInfo
from src.segmentation.filtering import Filter

output_dir = r"H:\My Drive\Projects\Collaboration-based\Nelly-AEL\walking\tif_frames"

frame_dir = r"H:\My Drive\Projects\Collaboration-based\Nelly-AEL\walking\frames_output"
# get all pngs in the frame_dir, convert to bw, and save as tif
pngs = [f for f in os.listdir(frame_dir) if f.endswith('.png')]
pngs.sort()
for png_num, png in enumerate(pngs):
    print(f"Processing {png_num + 1}/{len(pngs)}")
    im_path = os.path.join(frame_dir, png)
    im = cv2.imread(im_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    bw_im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    output_path = os.path.join(output_dir, f"{png}-bw.tif")
    # save as tif
    tifffile.imwrite(output_path, bw_im)

tifs = [f for f in os.listdir(output_dir) if f.endswith('.tif')]
im_path = os.path.join(output_dir, tifs[0])
test_im_info = ImInfo(im_path, dim_sizes={'X': 0.2, 'Y': 0.2, 'T': 0.1})
preprocessing = Filter(test_im_info, 1, remove_edges=False)
preprocessing.run()

segmentation_path = r"H:\My Drive\Projects\Collaboration-based\Nelly-AEL\walking\segmentations\frame_0000.png-segmentations.tif"

frame_im = cv2.imread(frame_path)
frame_im = cv2.cvtColor(frame_im, cv2.COLOR_BGR2RGB)
bw_im = cv2.cvtColor(frame_im, cv2.COLOR_RGB2GRAY)
output_path = r"H:\My Drive\Projects\Collaboration-based\Nelly-AEL\walking\tif_frames\frame_0000_bw.png"
# save as tif
tifffile.imwrite(output_path, frame_im)
segmentations = tifffile.imread(segmentation_path)

im_info = ImInfo(im_path, ch=ch, dim_sizes={'X': 0.2, 'Y': 0.2, 'T': 0.1})

import napari
viewer = napari.Viewer()
viewer.add_image(frame_im, name='frame')
viewer.add_labels(segmentations, name='segmentations')
