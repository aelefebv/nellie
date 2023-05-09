from datetime import datetime
from umap import UMAP
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.linear_model import ElasticNetCV
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
import plotly.express as px
import plotly.io as pio
import os

from src.io.pickle_jar import unpickle_object
from src.io.im_info import ImInfo

pio.renderers.default = 'browser'

# check if on mac or windows
import platform
if platform.system() == 'Windows':
    top_dir = r'D:\test_files\nelly\20230406-AELxKL-dmr_lipid_droplets_mtDR'
    csv_path = os.path.join(top_dir, '20230428-020933-ch_1-track_stats.csv')
    save_path = top_dir
else:
    csv_path = '/Users/austin/Documents/Transferred/20230418-094644-track_stats.csv'
    save_path = '/Users/austin/test_files/nelly_outputs/20230418_controls'
# remove_cols = ['track_num', 'fission_mean', 'fusion_mean', 'file_num', 'file_name', 'concentration',
#                ]
remove_cols = ['track_num', 'file_num', 'file_name', 'concentration',
               ]
# remove_cols = ['track_num', 'fission_mean', 'fusion_mean', 'file_num', 'file_name',
#                'branch_length_IQR', 'branch_length_max', 'branch_length_med', 'branch_length_min',
#                'branch_length_q25', 'branch_length_q75', 'branch_length_range',
#                'branch_tortuosity_IQR', 'branch_tortuosity_max', 'branch_tortuosity_med', 'branch_tortuosity_min',
#                'branch_tortuosity_q25', 'branch_tortuosity_q75', 'branch_tortuosity_range', 'concentration']
# keep_cols = ['branch_length_med', 'branch_tortuosity_med', 'branch_width_med', 'node_branch_width_ratio_med', 'node_width_med', 'displacement_max', 'persistance_med', 'speed_max', 'speed_med']
# keep_cols = ['branch_length_med', 'branch_tortuosity_med', 'branch_width_med', 'node_branch_width_ratio_med', 'node_width_med']
# keep_cols = ['displacement_max', 'persistance_med', 'speed_max', 'speed_med']

now = datetime.now()
now = now.strftime("%Y%m%d_%H%M%S")

# open csv as a dataframe
df = pd.read_csv(csv_path)
df = df.dropna()
# keep rows where concentration is 1 or 5000
df = df[(df['concentration'] == 1) | (df['concentration'] == 5000)]
# df = df[df['concentration'] < 10]
df = df.sample(frac=1).reset_index(drop=True)
df = df[~df['file_name'].str.contains('4h')]
df_original = df.copy()
# remove the columns we don't want
df = df.filter(regex='^(?!Unnamed:)')
# df = df.drop(remove_cols, axis=1)
#remove any columns with words in remove_cols in the name
df = df.drop(columns=[col for col in df.columns if any(word in col for word in remove_cols)])
# df = df[keep_cols]
# remove any rows with filename containing 4h
# # remove correlated features
# corr_matrix = np.corrcoef(df.T)
# corr_matrix = pd.DataFrame(corr_matrix, columns=df.columns, index=df.columns)
# corr_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
# corr_matrix = corr_matrix.stack().reset_index()
# corr_matrix.columns = ['feature_1', 'feature_2', 'correlation']
# corr_matrix = corr_matrix[corr_matrix['correlation'] > 0.75]
# corr_matrix = corr_matrix.sort_values(by='correlation', ascending=False)
# corr_matrix = corr_matrix.reset_index(drop=True)
# # remove the features that are correlated
# df = df.drop(corr_matrix['feature_2'], axis=1)


# drop rows with nan
df = df.dropna()
scaled_data = StandardScaler().fit_transform(df)


#
# # make a new column for low and high concentrations
# # df_original['concentration'] = df_original['concentration'].apply(lambda x: 'low' if x < 10 else 'high')
#
# # make the concentration column a string
df_original['concentration'] = df_original['concentration'].astype(str)
#


#shuffle the data

x = scaled_data
y = df_original['concentration']
# y = df_original['concentration']

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

model = RandomForestClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
cv_scores = cross_val_score(model, x, y, cv=3)
results = {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred),
            'cross_val_scores': cv_scores,
            'mean_cross_val_score': np.mean(cv_scores)
}
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
# create a dataframe of the feature rankings and importances
feature_rankings = pd.DataFrame({'feature': df.columns[indices], 'importance': importances[indices]})
print(feature_rankings)

# only keep the top N features and rerun the model
# N = 1
# top_features = feature_rankings['feature'].head(N).values
# x = df[top_features]
# x = StandardScaler().fit_transform(x)
# y = df_original['concentration']
# # Train-test split
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
# model = RandomForestClassifier()
# model.fit(x_train, y_train)
# y_pred = model.predict(x_test)
# accuracy = accuracy_score(y_test, y_pred)
# cv_scores = cross_val_score(model, x, y, cv=3)
# results = {
#             'accuracy': accuracy,
#             'classification_report': classification_report(y_test, y_pred),
#             'cross_val_scores': cv_scores,
#             'mean_cross_val_score': np.mean(cv_scores)
# }
# importances = model.feature_importances_
# indices = np.argsort(importances)[::-1]
# # create a dataframe of the feature rankings and importances
# feature_rankings = pd.DataFrame({'feature': df.columns[indices], 'importance': importances[indices]})
# print(feature_rankings)


# save dataframe to csv
save_dir = os.path.join(top_dir, 'output', 'feature_rankings')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
feature_rankings.to_csv(os.path.join(save_dir, 'feature_rankings_' + now + '.csv'))
# save final dataframe to csv
df_original.to_csv(os.path.join(save_dir, 'df_original_' + now + '.csv'))
# save results to a txt file
with open(os.path.join(save_dir, 'random_forest_results_' + now + '.txt'), 'w') as f:
    for key, value in results.items():
        f.write(f'{key}: {value} \n')


feature_cols = ['branch_intensity_max', 'branch_intensity_IQR', 'branch_length_max', 'branch_tortuosity_max', 'branch_length_IQR']
df_to_umap = df_original[feature_cols]
# scaled_data_umap = StandardScaler().fit_transform(df_to_umap)

umap_params = {'n_neighbors': 10, 'min_dist': 0.1,
               'n_components': 2, 'metric': 'euclidean', 'random_state': 42}
umap_model = UMAP(**umap_params)
umap_results = umap_model.fit_transform(df_to_umap)
df_to_umap['UMAP 1'] = umap_results[:, 0]
df_to_umap['UMAP 2'] = umap_results[:, 1]

dbscan_eps = 0.3
dbscan_min_samples = 10
dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
cluster_labels_dbscan = dbscan.fit_predict(umap_results)
df_to_umap['cluster'] = cluster_labels_dbscan
df_to_umap['concentration'] = df_original['concentration']

df_to_umap['index'] = df.index
fig = px.scatter(df_to_umap, x='UMAP 1', y='UMAP 2', color='concentration', hover_data=['index', 'cluster'])
                 # range_color=[0, 2])
unique_clusters = df_to_umap['cluster'].unique()
fig.show()

df_volcano = df.copy()
# drop nans
df_volcano = df_volcano.dropna()
df_volcano['concentration'] = df_original['concentration']
# convert to int
df_volcano['concentration'] = df_volcano['concentration'].astype(int)
#standard scale the data, except for concentration
df_volcano.iloc[:, :-1] = StandardScaler().fit_transform(df_volcano.iloc[:, :-1])
grouped_df = df_volcano.groupby('concentration').mean()
# grouped_df = df_volcano.groupby('concentration').nanmedian()
# calculate log fold change
test = grouped_df.iloc[1] / grouped_df.iloc[0]
LFC = np.log2(grouped_df.iloc[1] / grouped_df.iloc[0])
# get first row of the grouped_df dataframe
SMDs = []
MDs = []
for col in df_volcano.columns[:-1]:
    mean1 = df_volcano[df_volcano['concentration']==1][col].mean()
    mean2 = df_volcano[df_volcano['concentration']==5000][col].mean()
    sd1 = df_volcano[df_volcano['concentration']==1][col].std(ddof=1)
    sd2 = df_volcano[df_volcano['concentration']==5000][col].std(ddof=1)
    n1, n2 = len(df_volcano[df_volcano['concentration']==0]), len(df_volcano[df_volcano['concentration']==1])
    pooled_sd = np.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2))
    MD = mean2 - mean1
    SMD = MD / pooled_sd
    SMDs.append(SMD)
    MDs.append(MD)

# convert the SMDs to a numpy array and calculate the -log10(p-values) for each column
SMDs = np.array(SMDs)
MDs = np.array(MDs)

pval = 0.05 # set a constant p-value for demonstration purposes
neg_log10_pval = -np.log10(pval)
pvals = []
#df columns

from scipy.stats import ttest_ind
for col in df_volcano.columns[:-1]:
    pvals.append(ttest_ind(df_volcano[df_volcano['concentration']==1][col], df_volcano[df_volcano['concentration']==5000][col])[1])

neg_log10_pvals = -np.log10(pvals)
# cap the values at 100
neg_log10_pvals[neg_log10_pvals > 100] = 100
df_to_plot = pd.DataFrame({'LFC': LFC.values, 'neg_log10_pvals': neg_log10_pvals,
                           'SMD': SMDs, 'MD': MDs})
# give a label of 1 to the rows that are significant
df_to_plot['significant'] = np.where(df_to_plot['neg_log10_pvals'] > neg_log10_pval, 1, 0)
# make the plotly hover be the feature name
df_to_plot['feature'] = df_volcano.columns[:-1]
fig = px.scatter(df_to_plot, x='MD', y='neg_log10_pvals', color='significant', color_continuous_scale='RdBu',
                 text='feature', hover_data=['feature', 'SMD', 'neg_log10_pvals'])
# adjust the text position to avoid overlap with points
fig.update_traces(textposition='top center')

# add labels and title to the plot
fig.update_layout(xaxis_title='Standardized Mean Difference',
                  yaxis_title='-log10(p-value)',
                  title='Volcano Plot')

fig.show()
# # get the mean values for each cluster
# df_mean = df_original.groupby('cluster').mean()
# # df_sub = df_mean[keep_cols]
#
# # get unique filenames:
# unique_files = df_original['file_name'].unique()
# cluster_to_check = 4
# df_cluster = df_original[df_original['cluster'] == cluster_to_check]
#
# for unique_file in unique_files:
#     # get the indices of the rows in the original dataframe that belong to the cluster and im_name
#     df_cluster_file = df_cluster[df_cluster['file_name'] == unique_file]
#     tracks_to_check = df_cluster_file['track_num'].values
#     if len(tracks_to_check) != 0:
#         im_name = unique_file
#         break
#
#
# pkl_dir = os.path.join(top_dir, 'output', 'pickles')
# pkl_file = f'ch1-seg-{im_name[:-4]}.pkl'
# full_pkl_file = os.path.join(pkl_dir, pkl_file)
# tracks = unpickle_object(full_pkl_file)
# im_info = ImInfo(os.path.join(top_dir, im_name), ch=1)
#
# import tifffile
# from src.utils.general import get_reshaped_image
#
# mask_im = tifffile.memmap(im_info.path_im_mask, mode='r')
# mask_im = get_reshaped_image(mask_im, im_info=im_info)
# new_im = np.zeros(mask_im.shape, dtype=np.uint8)
# label_im = np.zeros(mask_im.shape, dtype=np.uint16)
# for track_num, track in enumerate(tracks):
#     if track_num not in tracks_to_check:
#         label_num = 1
#     else:
#        continue
#     print(track_num, len(tracks))
#     for node in track:
#         frame = node.frame_num
#         coords = node.node.coords
#         for coord in coords:
#             new_im[frame, coord[0], coord[1], coord[2]] = label_num
# for track_num, track in enumerate(tracks):
#     if track_num in tracks_to_check:
#         label_num = 2
#     else:
#         continue
#     print(track_num, len(tracks))
#     for node in track:
#         frame = node.frame_num
#         coords = node.node.coords
#         for coord in coords:
#             new_im[frame, coord[0], coord[1], coord[2]] = label_num
#
# import napari
# viewer = napari.Viewer()
# viewer.dims.ndisplay = 3
# viewer.add_image(mask_im, name='mask', colormap='gray', interpolation3d='nearest', contrast_limits=[0, 1],
#                  rendering='iso', iso_threshold=0, opacity=0.2)
# # viewer.add_labels(label_im, name='track_labels')
# viewer.add_image(new_im, name='tracks of interest', colormap='turbo', interpolation3d='nearest',
#                  contrast_limits=[0, 2], opacity=1)

# labels_of_interest = [270, 655, 243]
# # get df of labels of interest for that specific im_name
# df_labels = df_original[df_original['file_name'] == im_name]
# df_labels = df_labels[df_labels['index'].isin(labels_of_interest)]
#
# # add a column to the df_original that is True if the index is in the df_labels
# df_original['highlight'] = df_original['index'].isin(df_labels['index'])
#
# # highlight the df_labels on the plot
# fig = px.scatter(df_original, x='UMAP 1', y='UMAP 2', color='highlight', hover_data=['index', 'cluster'])
# fig.show()
