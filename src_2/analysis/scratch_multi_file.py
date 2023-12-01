from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

import seaborn as sns
from src import logger
from src_2.im_info.im_info import ImInfo
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
# import umap
import matplotlib.pyplot as plt
# from umap import UMAP
import plotly.express as px
import plotly.io as pio
import os

from src_2.utils.general import get_reshaped_image

pio.renderers.default = 'browser'
import numpy as np
from sklearn.cluster import HDBSCAN

top_dir = r"D:\test_files\nelly_smorgasbord\output"

file_names = [
    'deskewed-iono_post',
    'deskewed-iono_pre',
    'deskewed-mt_ends',
    'deskewed-peroxisome',
]

append = ".ome-ch0-organelle_motility_features.csv"
label_motility_paths = [os.path.join(top_dir, f"{file_name}{append}") for file_name in file_names]

append = '.ome-ch0-organelle_label_features.csv'
label_morphology_paths = [os.path.join(top_dir, f"{file_name}{append}") for file_name in file_names]

append = '.ome-ch0-organelle_skeleton_features.csv'
label_skeleton_morphology_path = [os.path.join(top_dir, f"{file_name}{append}") for file_name in file_names]

append = '.ome-ch0-branch_motility_features.csv'
branch_motility_paths = [os.path.join(top_dir, f"{file_name}{append}") for file_name in file_names]

append = '.ome-ch0-branch_label_features.csv'
branch_morphology_paths = [os.path.join(top_dir, f"{file_name}{append}") for file_name in file_names]

append = '.ome-ch0-branch_skeleton_features.csv'
branch_skeleton_paths = [os.path.join(top_dir, f"{file_name}{append}") for file_name in file_names]

label_motility_df = pd.concat([pd.read_csv(path).assign(file=os.path.basename(path)) for path in label_motility_paths])
label_motility_df.reset_index(drop=True, inplace=True)
label_morphology_df = pd.concat([pd.read_csv(path).assign(file=os.path.basename(path)) for path in label_morphology_paths])
label_morphology_df.reset_index(drop=True, inplace=True)
label_skeleton_morphology_df = pd.concat([pd.read_csv(path).assign(file=os.path.basename(path)) for path in label_skeleton_morphology_path])
label_skeleton_morphology_df.reset_index(drop=True, inplace=True)

branch_motility_df = pd.concat([pd.read_csv(path).assign(file=os.path.basename(path)) for path in branch_motility_paths])
branch_motility_df.reset_index(drop=True, inplace=True)
branch_morphology_df = pd.concat([pd.read_csv(path).assign(file=os.path.basename(path)) for path in branch_morphology_paths])
branch_morphology_df.reset_index(drop=True, inplace=True)
branch_skeleton_morphology_df = pd.concat([pd.read_csv(path).assign(file=os.path.basename(path)) for path in branch_skeleton_paths])
branch_skeleton_morphology_df.reset_index(drop=True, inplace=True)

# concatenate dataframes by columns
combo_label_morphology_df = pd.concat([label_morphology_df.drop(columns='file'), label_skeleton_morphology_df], axis=1)
combo_label_all_df = pd.concat([combo_label_morphology_df.drop(columns='file'), label_motility_df], axis=1)
combo_branch_morphology_df = pd.concat([branch_morphology_df.drop(columns='file'), branch_skeleton_morphology_df], axis=1)
combo_branch_all_df = pd.concat([combo_branch_morphology_df.drop(columns='file'), branch_motility_df], axis=1)

df_to_use = combo_label_all_df.copy()

df_to_use = df_to_use.dropna()
og_df = df_to_use.copy()

metric_to_use = 'median'
keep_motility_cols = [
    f'rel_ang_vel_mag_12_{metric_to_use}','rel_ang_vel_mag_12_max',
    f'rel_ang_acc_mag_{metric_to_use}','rel_ang_acc_mag_max',
    f'rel_lin_vel_mag_12_{metric_to_use}','rel_lin_vel_mag_12_max',
    f'rel_lin_acc_mag_{metric_to_use}','rel_lin_acc_mag_max',
    f'ref_lin_vel_mag_12_mean','ref_lin_vel_mag_12_max',
    f'ref_lin_acc_mag_mean','ref_lin_acc_mag_max',
    f'com_directionality_12_{metric_to_use}','com_directionality_12_max',
    f'com_directionality_acceleration_{metric_to_use}','com_directionality_acceleration_max',
]
remove_motility_cols = [col for col in branch_motility_df.columns if col not in keep_motility_cols]
remove_cols_morphology = [
    'main_label', 'file', 'label', 't',
    # 'extent', 'solidity',
    # 'intensity_mean', 'intensity_std', 'intensity_range',
    # 'frangi_mean', 'frangi_std', 'frangi_range',
]

for col in (remove_cols_morphology + remove_motility_cols):
    if col in df_to_use.columns:
        df_to_use.drop(columns=col, inplace=True)


scaler = StandardScaler()
standardized_features = scaler.fit_transform(df_to_use)
standardized_features = pd.DataFrame(standardized_features, columns=df_to_use.columns)


from umap import UMAP
model = UMAP(n_components=2, n_neighbors=10, min_dist=0.1)
embedding = model.fit_transform(standardized_features)

colorby = f'area'
# colorby = f'ref_lin_vel_mag_12_{metric_to_use}'
embedding_fig = px.scatter(og_df, x=embedding[:, 0], y=embedding[:, 1], color=np.log10(og_df[colorby]),
                           title="2D Projection Features", hover_data=colorby)
embedding_fig.show()


#
# keep_cols_temporal = [
#     label_name,
#     'rel_ang_vel_mag_12_mean', 'rel_ang_vel_mag_12_max',
#     'rel_ang_acc_mag_mean', 'rel_ang_acc_mag_max',
#     'rel_lin_vel_mag_12_mean', 'rel_lin_vel_mag_12_max',
#     'rel_lin_acc_mag_mean', 'rel_lin_acc_mag_max',
#     'ref_lin_vel_mag_12_mean', 'ref_lin_vel_mag_12_max',
#     'ref_lin_acc_mag_mean', 'ref_lin_acc_mag_max',
#     'lin_vel_mag_12_mean', 'lin_vel_mag_12_max',
# ]
