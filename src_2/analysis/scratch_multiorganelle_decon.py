
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

df_type_to_use = 'combo'  # combo, motility, or motility

file_to_leave_out = 0
test_file = all_files[file_to_leave_out]

test_iminfo = ImInfo(test_file, ch=-1)
test_motility_df = pd.read_csv(test_iminfo.pipeline_paths['branch_motility_features'])
test_motility_df.reset_index(drop=True, inplace=True)
test_label_morphology_df = pd.read_csv(test_iminfo.pipeline_paths['branch_label_features'])
test_label_morphology_df.reset_index(drop=True, inplace=True)
test_branch_morphology_df = pd.read_csv(test_iminfo.pipeline_paths['branch_skeleton_features'])
test_branch_morphology_df.reset_index(drop=True, inplace=True)

test_combo_morphology_df = pd.concat([test_label_morphology_df, test_branch_morphology_df], axis=1)

test_combo_df = pd.concat([test_combo_morphology_df, test_motility_df], axis=1)

test_ch0_gt = ImInfo(test_file, ch=0)
test_ch1_gt = ImInfo(test_file, ch=1)

# get label image, frame 1 from multichannel file
# get the label image from the multichannel file
test_all_labels = test_iminfo.get_im_memmap(test_iminfo.pipeline_paths['im_skel_relabelled'])[1]
# get the label mask from the single channel files
test_ch0_gt_mask = test_ch0_gt.get_im_memmap(test_ch0_gt.pipeline_paths['im_skel_relabelled'])[1]>0
test_ch1_gt_mask = test_ch1_gt.get_im_memmap(test_ch1_gt.pipeline_paths['im_skel_relabelled'])[1]>0

gt_ch = []

non_0_all_regions = skimage.measure.regionprops(test_all_labels)
# if a branch has more overlap with mitochondria mask, its a mitochondria (ch0)
# if a branch has more overlap with golgi mask, its a golgi (ch1)
# this is our ground truth
for region_num, region in enumerate(non_0_all_regions):
    ch0_sum_px = np.sum(test_ch0_gt_mask[region.coords[:, 0], region.coords[:, 1], region.coords[:, 2]])
    ch1_sum_px = np.sum(test_ch1_gt_mask[region.coords[:, 0], region.coords[:, 1], region.coords[:, 2]])
    if ch0_sum_px > ch1_sum_px:
        print(f'Label {region.label}: \t ch0')
        gt_ch.append(0)
    else:
        print(f'Label {region.label}: \t ch1')
        gt_ch.append(1)

# add the groundtruth channel numbers to the dataframes for the test file
test_combo_df['gt_ch'] = gt_ch
test_motility_df['gt_ch'] = gt_ch
test_combo_morphology_df['gt_ch'] = gt_ch

if df_type_to_use == 'combo':
    test_df_to_use = test_combo_df.copy()
elif df_type_to_use == 'motility':
    test_df_to_use = test_motility_df.copy()
elif df_type_to_use == 'morphology':
    test_df_to_use = test_combo_morphology_df.copy()
else:
    raise ValueError(f'Invalid df_type_to_use: {df_type_to_use}')

test_df_to_use = test_df_to_use.replace([np.inf, -np.inf], np.nan)
test_df_to_use = test_df_to_use.dropna()
test_df_to_use.reset_index(drop=True, inplace=True)
og_test_df = test_df_to_use.copy()

train_motility_df = pd.DataFrame()
train_label_morphology_df = pd.DataFrame()
train_branch_morphology_df = pd.DataFrame()
ch_array = []
for ch in range(2):
    for train_file in all_files:
        if train_file == test_file:
            continue
        train_iminfo = ImInfo(train_file, ch=ch)

        new_motility_df = pd.read_csv(train_iminfo.pipeline_paths['branch_motility_features'])
        train_motility_df = pd.concat([train_motility_df, new_motility_df], axis=0)
        train_motility_df.reset_index(drop=True, inplace=True)

        new_label_morphology_df = pd.read_csv(train_iminfo.pipeline_paths['branch_label_features'])
        train_label_morphology_df = pd.concat([train_label_morphology_df, new_label_morphology_df], axis=0)
        train_label_morphology_df.reset_index(drop=True, inplace=True)

        new_branch_morphology_df = pd.read_csv(train_iminfo.pipeline_paths['branch_skeleton_features'])
        train_branch_morphology_df = pd.concat([train_branch_morphology_df, new_branch_morphology_df], axis=0)
        train_branch_morphology_df.reset_index(drop=True, inplace=True)

        # add the channel number to the ch_array, with length equal to the number of rows in the new dataframe
        ch_array.extend(np.ones(new_motility_df.shape[0], dtype=int) * ch)
train_combo_morphology_df = pd.concat([train_label_morphology_df, train_branch_morphology_df], axis=1)
train_combo_df = pd.concat([train_combo_morphology_df, train_motility_df], axis=1)


if df_type_to_use == 'combo':
    train_df_to_use = train_combo_df.copy()
elif df_type_to_use == 'motility':
    train_df_to_use = train_motility_df.copy()
elif df_type_to_use == 'morphology':
    train_df_to_use = train_combo_morphology_df.copy()
else:
    raise ValueError(f'Invalid df_type_to_use: {df_type_to_use}')

train_df_to_use['gt_ch'] = ch_array

# convert infinities to nan
train_df_to_use = train_df_to_use.replace([np.inf, -np.inf], np.nan)
train_df_to_use = train_df_to_use.dropna()
train_df_to_use.reset_index(drop=True, inplace=True)

og_train_df = train_df_to_use.copy()

remove_cols = [
    'main_label', 'file', 'label', 't', 'gt_ch',
    'intensity_mean', 'intensity_std', 'intensity_range',
    'frangi_mean', 'frangi_std', 'frangi_range',
]

for col in remove_cols:
    if col in test_df_to_use.columns:
        test_df_to_use.drop(columns=col, inplace=True)
    if col in train_df_to_use.columns:
        train_df_to_use.drop(columns=col, inplace=True)

# assert all columns are the same
assert all(test_df_to_use.columns == train_df_to_use.columns)

# train a model on all but test file with combined branch data from all files
# train a morphology only model, a motility only model, and a combined model

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, KFold

#shuffle the training dataset and og dataset in the same way
train_df_to_use = train_df_to_use.sample(frac=1, random_state=42)
og_train_df = og_train_df.iloc[train_df_to_use.index]

clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(train_df_to_use, og_train_df['gt_ch'])
y_pred = clf.predict(test_df_to_use)

# Define the k-fold cross-validation procedure
# cv = KFold(n_splits=5, random_state=42, shuffle=True)

# Evaluate the model with cross-validation
# scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
report = classification_report(og_test_df['gt_ch'], y_pred)

