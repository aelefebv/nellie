import os
import pandas as pd
import napari
import tifffile
import numpy as np

from src.io.pickle_jar import unpickle_object
from src.utils.general import get_reshaped_image
csv_top_dir = r'D:\test_files\nelly\20230406-AELxKL-dmr_lipid_droplets_mtDR\output\csv'
csv_file = '20230410_165121_umap_pca_data.csv'
full_csv_path = os.path.join(csv_top_dir, csv_file)

pkl_top_dir = r'D:\test_files\nelly\20230406-AELxKL-dmr_lipid_droplets_mtDR\output\pickles'
pkl_file = 'ch1-obj-deskewed-2023-04-06_17-01-43_000_AELxKL-dmr_PERK-lipid_droplets_mtDR-5000-4h.ome.pkl'
full_pkl_path = os.path.join(pkl_top_dir, pkl_file)

region_props = unpickle_object(full_pkl_path)

# open up stats and tsne file
stats = pd.read_csv(full_csv_path)

full_file_path = rf'D:\\test_files\\nelly\\20230330-AELxZL-A549-TMRE_mtG\\deskewed-2023-03-30_15-11-11_000_20230330-AELxZL-A549-TMRE_mtG-ctrl.ome.tif'
# get the file name without extension
# file_name_no_ext = os.path.splitext(os.path.basename(full_file_path))[0]
#open up image tiffile memmap
mask_im = tifffile.memmap(region_props.im_info.path_im_mask, mode='r')
mask_im = get_reshaped_image(mask_im, 2, region_props.im_info)
label_im = tifffile.memmap(region_props.im_info.path_im_label_obj, mode='r')
label_im = get_reshaped_image(label_im, 2, region_props.im_info)
# make a new im with all zeros, the same size as mask im but uint16
new_im = np.zeros(mask_im.shape, dtype=np.uint16)

# get the dataframe rows where 'filename' is in the file name no ext
# df = stats[stats['filename'].str.contains(file_name_no_ext)]
df = stats
frame_to_color = 1
frame_regions = region_props.organelles[frame_to_color]
# go through each row and set the pixel value to the cluster number
for i, row in df.iterrows():
    region = frame_regions[row['region_id']-1]
    print(region.instance_label)
    # make the node.coords new_im coordinates the same value as the cluster number
    for coord in region.coords:
        new_im[frame_to_color, coord[0], coord[1], coord[2]] = row['cluster_umap']+2

viewer = napari.Viewer()
# add the original image
im_og = tifffile.memmap(region_props.im_info.im_path, mode='r')
im_og = im_og[:2, 1, ...]
scaling = [region_props.im_info.dim_sizes['Z'], region_props.im_info.dim_sizes['Y'], region_props.im_info.dim_sizes['X']]
viewer.add_image(im_og, name='original image', scale=scaling)
# viewer.add_labels(label_im, name='mask image')
viewer.add_labels(new_im, name='tip_labels', scale=scaling)
