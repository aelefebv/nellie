import numpy as np
import tifffile

top_path = r'\\zfsdata02\SOLS_v1.1-ro\Austin_Leafbeaver\20230413_AELxES-good-dmr_lipid_droplets_mt_DR-activate_deactivate'
file_name_contains = '-1nM'
endswith = 'activate'

# get all directories in top_path that contain file_name_contains
import os
import glob

# get all directories in top_path that ends with endswith and contains string file_name_contains

# Get all directories in top_path that contain file_name_contains
dir_list = [d for d in os.listdir(top_path) if os.path.isdir(os.path.join(top_path, d)) and file_name_contains in d]

# Get all directories in top_path that end with endswith and contain string file_name_contains
dir_list = [d for d in dir_list if d.endswith(endswith)]

print(dir_list)


timelapse_images = []
for num_t, timelapse in enumerate(dir_list):
    print(num_t, len(dir_list), timelapse)
    # get all tif files in timelapse
    tif_files = glob.glob(os.path.join(top_path, timelapse, 'preview', '*.tif*'))
    tif_files.sort()
    # store all tif files in timelapse
    for tif_file in tif_files:
        timelapse_images.append(tifffile.memmap(tif_file, mode='r')[0])

# concatenate images along a new dimension
concatenated_images = np.stack(timelapse_images, axis=0)

save_path = r'D:\test_files\nelly\lipid_droplets_timelapses'
# Save the concatenated image as a new TIFF file
tifffile.imwrite(os.path.join(save_path, 'concatenated_1nM.tif'), concatenated_images)
