# get the label image from the multichannel file
# get the label mask from the single channel files
# if a branch has more overlap with mitochondria mask, its a mitochondria (ch0)
# if a branch has more overlap with golgi mask, its a golgi (ch1)
# this is our ground truth
# train a model on all but test file with combined branch data from all files
# train a morphology only model, a motility only model, and a combined model
# test the model on the test file, have it generate a vector assigning branches to the golgi channel or mito channel (0 or 1)
# compare the vector to the ground truth for metrics
# do this for every leave one out test file --> new model for each combination
# aggregate the metrics for final results.
import os
import pandas as pd
import numpy as np
import skimage.measure

from src_2.im_info.im_info import ImInfo

top_dir = r"D:\test_files\nelly_multichannel"
# get all non-folder files
all_files = os.listdir(top_dir)
all_files = [os.path.join(top_dir, file) for file in all_files if not os.path.isdir(os.path.join(top_dir, file))]

motility_path_ext = '.ome-ch0-branch_motility_features.csv'
morphology_branch_path_ext = '.ome-ch0-branch_label_features.csv'
morphology_skeleton_path_ext = '.ome-ch0-branch_skeleton_features.csv'

file_to_leave_out = 0
test_file = all_files[file_to_leave_out]

test_iminfo = ImInfo(test_file, ch=-1)
test_motility_df = pd.read_csv(test_iminfo.pipeline_paths['branch_motility_features'])
test_motility_df.reset_index(drop=True, inplace=True)
test_label_morphology_df = pd.read_csv(test_iminfo.pipeline_paths['branch_label_features'])
test_label_morphology_df.reset_index(drop=True, inplace=True)
test_branch_morphology_df = pd.concat([test_label_morphology_df, pd.read_csv(test_iminfo.pipeline_paths['branch_skeleton_features'])], axis=1)
test_branch_morphology_df.reset_index(drop=True, inplace=True)

test_combo_morphology_df = pd.concat([test_label_morphology_df, test_branch_morphology_df], axis=1)

test_combo_df = pd.concat([test_combo_morphology_df, test_motility_df], axis=1)

test_ch0_gt = ImInfo(test_file, ch=0)
test_ch1_gt = ImInfo(test_file, ch=1)

# get label image, frame 1 from multichannel file
test_all_labels = test_iminfo.get_im_memmap(test_iminfo.pipeline_paths['im_skel_relabelled'])[1]
test_ch0_gt_mask = test_ch0_gt.get_im_memmap(test_ch0_gt.pipeline_paths['im_skel_relabelled'])[1]>0
test_ch1_gt_mask = test_ch1_gt.get_im_memmap(test_ch1_gt.pipeline_paths['im_skel_relabelled'])[1]>0

gt_ch = []

non_0_all_regions = skimage.measure.regionprops(test_all_labels)
for region_num, region in enumerate(non_0_all_regions):
    ch0_sum_px = np.sum(test_ch0_gt_mask[region.coords[:, 0], region.coords[:, 1], region.coords[:, 2]])
    ch1_sum_px = np.sum(test_ch1_gt_mask[region.coords[:, 0], region.coords[:, 1], region.coords[:, 2]])
    if ch0_sum_px > ch1_sum_px:
        print(f'Label {region.label}: \t ch0')
        gt_ch.append(0)
    else:
        print(f'Label {region.label}: \t ch1')
        gt_ch.append(1)



