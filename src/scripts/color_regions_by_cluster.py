import os
import pandas as pd
import napari
import tifffile
import numpy as np

from src.analysis.preprocessing.clustering import ClusteringAnalysis
from src.analysis.preprocessing.csv_preprocessing import fill_na_in_csv
from src.io.pickle_jar import unpickle_object
from src.utils.general import get_reshaped_image
top_output_dir = r'D:\test_files\nelly\20230406-AELxKL-dmr_lipid_droplets_mtDR\output'
im_name = 'deskewed-2023-04-06_17-01-43_000_AELxKL-dmr_PERK-lipid_droplets_mtDR-5000-4h.ome'
ch = 1

csv_dir = os.path.join(top_output_dir, 'csv')
analysis_output_dir = os.path.join(top_output_dir, 'analysis')
file_path = f'summary_stats_regions-{im_name}-ch{ch}.csv'
df_preprocessed = fill_na_in_csv(csv_dir, file_path)

ignore_cols = ['group', 'filename', 'frame_number',
               'branch_ids', 'node_id', 'region_id', 'branch_id', 'node_ids']
remove_cols = None
# remove_cols = ['qqq']
clustering_analysis = ClusteringAnalysis(df_preprocessed, analysis_output_dir, file_path,
                                         corr_threshold=0.8, frames_to_keep=1,
                                         ignore_cols=ignore_cols,
                                         remove_cols=remove_cols,)
df_out = clustering_analysis.run()

pkl_dir = os.path.join(top_output_dir, 'pickles')
pkl_file = f'ch1-obj-{im_name}.pkl'
full_pkl_file = os.path.join(pkl_dir, pkl_file)
region_props = unpickle_object(full_pkl_file)


mask_im = tifffile.memmap(region_props.im_info.path_im_mask, mode='r')
mask_im = get_reshaped_image(mask_im, 2, region_props.im_info)
label_im = tifffile.memmap(region_props.im_info.path_im_label_obj, mode='r')
label_im = get_reshaped_image(label_im, 2, region_props.im_info)
new_im = np.zeros(mask_im.shape, dtype=np.uint16)

frame_to_color = 1
frame_regions = region_props.organelles[frame_to_color]
# go through each row and set the pixel value to the cluster number
for i, row in df_out.iterrows():
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
