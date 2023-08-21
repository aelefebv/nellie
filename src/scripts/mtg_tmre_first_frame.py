import pandas as pd

from src import logger
from src.io.im_info import ImInfo
from src.io.pickle_jar import unpickle_object
import tifffile
import os
import numpy as np

top_dir = r"D:\test_files\20230713-AELxZL-coated_DENSPM_wt_ko_A549"

# find all .ome.tif files in the top_dir
ome_tif_files = [os.path.join(top_dir, file) for file in os.listdir(top_dir) if file.endswith('.ome.tif')]

# create a df for all the data
df = pd.DataFrame(columns=['file_num', 'condition', 'concentration', 'region_num', 'mt_green_mean', 'tmrm_red_mean', 'mt_green_std',
                           'tmrm_red_std', 'mt_green_median', 'tmrm_red_median', 'tmrm_mtg_mean_ratio',
                           'tmrm_mtg_median_ratio', 'num_pixels'])

for file_num, filepath in enumerate(ome_tif_files):
    print(f'Processing file {file_num + 1} of {len(ome_tif_files)}')
    try:
        test = ImInfo(filepath, ch=0, dimension_order='')
    except FileNotFoundError:
        logger.error("File not found.")
        continue

    file_info = filepath.split('_acquire')[0].split('_')
    condition = file_info[-2]
    concentration = file_info[-1]

    just_regions = unpickle_object(test.path_pickle_obj)
    first_frame_regions = just_regions.organelles[0]

    mt_green = tifffile.imread(test.im_path)[0][0]
    tmrm_red = tifffile.imread(test.im_path)[0][1]

    # for each region in first_frame_regions, get the mean intensity of the mt_green and tmrm_red channels via the coords attr
    # then add those values to a dataframe

    for region_num, region in enumerate(first_frame_regions):
        mt_green_mean = np.nanmean(mt_green[region.coords[:, 0], region.coords[:, 1]])
        tmrm_red_mean = np.nanmean(tmrm_red[region.coords[:, 0], region.coords[:, 1]])
        mt_green_std = np.nanstd(mt_green[region.coords[:, 0], region.coords[:, 1]])
        tmrm_red_std = np.nanstd(tmrm_red[region.coords[:, 0], region.coords[:, 1]])
        mt_green_median = np.nanmedian(mt_green[region.coords[:, 0], region.coords[:, 1]])
        tmrm_red_median = np.nanmedian(tmrm_red[region.coords[:, 0], region.coords[:, 1]])
        tmrm_mtg_mean_ratio = tmrm_red_mean / mt_green_mean
        tmrm_mtg_median_ratio = tmrm_red_median / mt_green_median
        num_pixels = region.coords.shape[0]
        region_df = pd.DataFrame([[file_num, condition, concentration, region_num, mt_green_mean, tmrm_red_mean,
                                      mt_green_std, tmrm_red_std, mt_green_median, tmrm_red_median,
                                      tmrm_mtg_mean_ratio, tmrm_mtg_median_ratio, num_pixels]],
                                    columns=['file_num', 'condition', 'concentration', 'region_num', 'mt_green_mean', 'tmrm_red_mean',
                                            'mt_green_std', 'tmrm_red_std', 'mt_green_median', 'tmrm_red_median',
                                            'tmrm_mtg_mean_ratio', 'tmrm_mtg_median_ratio', 'num_pixels'])
        # concat the region_df to the df
        df = pd.concat([df, region_df], ignore_index=True)

# save df as csv in the top dir with top dir name
df.to_csv(os.path.join(top_dir, f'{top_dir.split(os.sep)[-1]}_first_frame.csv'), index=False)
