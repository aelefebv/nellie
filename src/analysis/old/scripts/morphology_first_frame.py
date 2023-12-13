import pandas as pd

from src import logger
from src import ImInfo
from src import unpickle_object
import tifffile
import os
import numpy as np
import skimage.measure
import scipy.ndimage


top_dir = r"D:\test_files\20230713-AELxZL-coated_DENSPM_wt_ko_A549"

# find all .ome.tif files in the top_dir
ome_tif_files = [os.path.join(top_dir, file) for file in os.listdir(top_dir) if file.endswith('.ome.tif')]

# create a df for all the data
df = pd.DataFrame(columns=[
    'file_num', 'condition', 'concentration', 'region_num', 'mt_green_mean', 'tmrm_red_mean', 'mt_green_std',
    'tmrm_red_std', 'mt_green_median', 'tmrm_red_median', 'tmrm_mtg_mean_ratio',
    'tmrm_mtg_median_ratio', 'num_pixels', 'length', 'thickness_mean', 'thickness_median',
    'thickness_std', 'aspect_ratio_mean', 'aspect_ratio_median'
])

for file_num, filepath in enumerate(ome_tif_files):
    print(f'Processing file {file_num + 1} of {len(ome_tif_files)}')
    try:
        test = ImInfo(filepath, ch=0, dimension_order='')
    except FileNotFoundError:
        logger.error("File not found.")
        continue

    file_info = filepath.split('_acquire')[0].split('_')
    scale = [test.dim_sizes['Z'], test.dim_sizes['Y'], test.dim_sizes['X']]
    condition = file_info[-2]
    concentration = file_info[-1]

    nodes = unpickle_object(test.path_pickle_node)
    nodes_0 = nodes.nodes[0]
    branches = tifffile.imread(test.path_im_label_seg)[0]
    og_im = tifffile.imread(test.im_path)[0]
    mask_im = tifffile.imread(test.path_im_mask)[0]
    distance_transform_mask = scipy.ndimage.distance_transform_edt(mask_im)

    regions = skimage.measure.regionprops(branches, spacing=scale)
    for region_num, region in enumerate(regions):
        mt_green_mean = np.nanmean(og_im[0][tuple(region.coords.T)])
        tmrm_red_mean = np.nanmean(og_im[1][tuple(region.coords.T)])
        mt_green_std = np.nanstd(og_im[0][tuple(region.coords.T)])
        tmrm_red_std = np.nanstd(og_im[1][tuple(region.coords.T)])
        mt_green_median = np.nanmedian(og_im[0][tuple(region.coords.T)])
        tmrm_red_median = np.nanmedian(og_im[1][tuple(region.coords.T)])
        tmrm_mtg_mean_ratio = tmrm_red_mean / mt_green_mean
        tmrm_mtg_median_ratio = tmrm_red_median / mt_green_median
        num_pixels = region.coords.shape[0]
        length = len(region.coords)
        thickness_mean = np.nanmean(distance_transform_mask[tuple(region.coords.T)])*2
        thickness_median = np.nanmedian(distance_transform_mask[tuple(region.coords.T)])*2
        thickness_std = np.nanstd(distance_transform_mask[tuple(region.coords.T)])*2
        aspect_ratio_mean = length / thickness_mean
        aspect_ratio_median = length / thickness_median
        region_df = pd.DataFrame([
            [file_num, condition, concentration, region_num, mt_green_mean, tmrm_red_mean,
                mt_green_std, tmrm_red_std, mt_green_median, tmrm_red_median,
                tmrm_mtg_mean_ratio, tmrm_mtg_median_ratio, num_pixels, length,
                thickness_mean, thickness_median, thickness_std, aspect_ratio_mean,
                aspect_ratio_median]],
            columns=['file_num', 'condition', 'concentration', 'region_num', 'mt_green_mean', 'tmrm_red_mean',
                'mt_green_std', 'tmrm_red_std', 'mt_green_median', 'tmrm_red_median',
                'tmrm_mtg_mean_ratio', 'tmrm_mtg_median_ratio', 'num_pixels', 'length',
                'thickness_mean', 'thickness_median', 'thickness_std', 'aspect_ratio_mean',
                'aspect_ratio_median'])
        # concat the region_df to the df
        df = pd.concat([df, region_df], ignore_index=True)

# save df as csv in the top dir with top dir name
df.to_csv(os.path.join(top_dir, f'{top_dir.split(os.sep)[-1]}_first_frame_morphology.csv'), index=False)

# import napari
# viewer = napari.Viewer()
# viewer.add_labels(branches, name='branches')
# viewer.add_image(og_im[0], name='mt_green')
# viewer.add_image(mask_im, name='mask')
# viewer.add_image(distance_transform_mask, name='distance_transform_mask')
