import os
import pandas as pd
import napari
import tifffile
import numpy as np

from src.io.pickle_jar import unpickle_object
from src.utils.general import get_reshaped_image

top_dir = r'D:\test_files\nelly\20230330-AELxZL-A549-TMRE_mtG\output\pickles'
pkl_file = r'ch0-node-deskewed-2023-03-30_15-11-11_000_20230330-AELxZL-A549-TMRE_mtG-ctrl.ome.pkl'
full_pkl_path = os.path.join(top_dir, pkl_file)

node_props = unpickle_object(full_pkl_path)

# open up stats and tsne file
stats = pd.read_csv(r'C:\Users\austin\GitHub\wool-sweater\nelly_analysis\mtg_tmre_ssat\data\20230410_110959_tsne_dbascan_data.csv')

full_file_path = rf'D:\\test_files\\nelly\\20230330-AELxZL-A549-TMRE_mtG\\deskewed-2023-03-30_15-11-11_000_20230330-AELxZL-A549-TMRE_mtG-ctrl.ome.tif'
# get the file name without extension
file_name_no_ext = os.path.splitext(os.path.basename(full_file_path))[0]
#open up image tiffile memmap
mask_im = tifffile.memmap(node_props.im_info.path_im_mask, mode='r')
mask_im = get_reshaped_image(mask_im, 2, node_props.im_info)
# make a new im with all zeros, the same size as mask im but uint16
new_im = np.zeros(mask_im.shape, dtype=np.uint16)

# get the dataframe rows where 'filename' is in the file name no ext
df = stats[stats['filename'].str.contains(file_name_no_ext)]
frame_to_color = 1
frame_nodes = node_props.nodes[frame_to_color]
# go through each row and set the pixel value to the cluster number
for i, row in df.iterrows():
    node = frame_nodes[row['node_id']]
    # make the node.coords new_im coordinates the same value as the cluster number
    for coord in node.coords:
        new_im[frame_to_color, coord[0], coord[1], coord[2]] = row['cluster']+2

viewer = napari.Viewer()
viewer.add_labels(new_im, name='tip_labels')
# add the original image
im_og = tifffile.memmap(node_props.im_info.im_path, mode='r')
im_og = im_og[:2, :, 1, ...]
viewer.add_image(im_og*mask_im, name='original image')
viewer.add_image(mask_im, name='mask image')
