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
pio.renderers.default = 'browser'

csv_path = '/Users/austin/Documents/Transferred/20230418-094644-track_stats.csv'
save_path = '/Users/austin/test_files/nelly_outputs/20230418_controls'
remove_cols = ['track_num', 'fission_mean', 'fusion_mean']
# keep_cols = ['branch_length_med', 'branch_tortuosity_med', 'branch_width_med', 'node_branch_width_ratio_med', 'node_width_med', 'displacement_max', 'persistance_med', 'speed_max', 'speed_med']
# keep_cols = ['branch_length_med', 'branch_tortuosity_med', 'branch_width_med', 'node_branch_width_ratio_med', 'node_width_med']
# keep_cols = ['displacement_max', 'persistance_med', 'speed_max', 'speed_med']

now = datetime.now()
now = now.strftime("%Y%m%d_%H%M%S")

# open csv as a dataframe
df = pd.read_csv(csv_path)
df_original = df.copy()
# remove the columns we don't want
df = df.filter(regex='^(?!Unnamed:)')
df = df.drop(remove_cols, axis=1)
# df = df[keep_cols]

# remove correlated features
corr_matrix = np.corrcoef(df.T)
corr_matrix = pd.DataFrame(corr_matrix, columns=df.columns, index=df.columns)
corr_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
corr_matrix = corr_matrix.stack().reset_index()
corr_matrix.columns = ['feature_1', 'feature_2', 'correlation']
corr_matrix = corr_matrix[corr_matrix['correlation'] > 0.9]
corr_matrix = corr_matrix.sort_values(by='correlation', ascending=False)
corr_matrix = corr_matrix.reset_index(drop=True)

# remove the features that are correlated
df = df.drop(corr_matrix['feature_2'], axis=1)
scaled_data = StandardScaler().fit_transform(df)


umap_params = {'n_neighbors': 20, 'min_dist': 0.1,
               'n_components': 2, 'metric': 'euclidean', 'random_state': 42}
umap_model = UMAP(**umap_params)
umap_results = umap_model.fit_transform(scaled_data)
df_original['UMAP 1'] = umap_results[:, 0]
df_original['UMAP 2'] = umap_results[:, 1]

dbscan_eps = 0.5
dbscan_min_samples = 20
dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
cluster_labels_dbscan = dbscan.fit_predict(umap_results)
df_original['cluster'] = cluster_labels_dbscan

df_original['index'] = df.index
fig = px.scatter(df_original, x='UMAP 1', y='UMAP 2', color='displacement_max', hover_data=['index', 'cluster'])
                 # range_color=[0, 2])
unique_clusters = df_original['cluster'].unique()
fig.show()

# get the mean values for each cluster
df_mean = df_original.groupby('cluster').mean()
# df_sub = df_mean[keep_cols]
