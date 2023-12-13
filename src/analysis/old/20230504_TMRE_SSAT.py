from datetime import datetime
from umap import UMAP
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
# from sklearn.linear_model import ElasticNetCV
import pandas as pd
import numpy as np
# from scipy.spatial import ConvexHull
import plotly.express as px
import plotly.io as pio
import os

# from src.io.pickle_jar import unpickle_object
# from src.io.im_info import ImInfo

pio.renderers.default = 'browser'

# check if on mac or windows
import platform
if platform.system() == 'Windows':
    top_dir = r'D:\test_files\nelly\20230330-AELxZL-A549-TMRE_mtG'
    csv_path = os.path.join(top_dir, '20230504-114658-ch_0-track_stats.csv')
    save_path = top_dir
else:
    csv_path = '/Users/austin/Documents/Transferred/20230418-094644-track_stats.csv'
    save_path = '/Users/austin/test_files/nelly_outputs/20230418_controls'
remove_cols = ['track_num', 'file_num', 'file_name', 'sample_type', 'condition',
               ]
now = datetime.now()
now = now.strftime("%Y%m%d_%H%M%S")

# open csv as a dataframe, drop columns with NA values
df = pd.read_csv(csv_path)
df = df.dropna(axis=1)

# add a column for the sample_type, where it finds either PQLC2_KO or Ctrl in the filename
df['sample_type'] = df['file_name'].apply(lambda x: 'ctrl' if 'ctrl' in x else 'KO' if 'KO' in x else 'OE')
# add a column for condition where it finds either FCCP or not in the filename
df['condition'] = df['file_name'].apply(lambda x: 'FCCP' if 'FCCP' in x else 'noFCCP')
# concatenate the sample_type and condition columns into a new column
df['sample_type_condition'] = df['sample_type'] + '_' + df['condition']

# create new columns of ratios between columns with "intensity" and "ch1" in the name and columns with "intensity" and "ch0"
# in the name then remove the original intensity cols
for col in df.columns:
    if 'intensity' in col and 'ch1' in col:
        df[col + '_ratio'] = df[col] / df[col.replace('ch1', 'ch0')]
        df[col + '_ratio'] = df[col + '_ratio'].replace([np.inf, -np.inf], np.nan)

for col in df.columns:
    if 'intensity' in col and ('ch1' in col or 'ch0' in col) and 'ratio' not in col:
        df = df.drop(columns=[col])

# then remove original columns with "intensity" and in the name


# keep only columns that don't have FCCP
df = df[df['condition'] != 'FCCP']

# drop the 'branch_intensity_ch1_ratio' column
df = df.drop(columns=['branch_intensity_ch1_ratio'])

# shuffle the dataframe, save the original
df = df.sample(frac=1).reset_index(drop=True)
df = df.dropna(axis=0)
df_original = df.copy()

# remove the columns we don't want, specifically anything with _branch_ anywhere in the name
#  and anything with Unnamed in the name
# df = df.filter(regex='^(?!.*_branch_).*')
# df = df.filter(regex='^(?!branch)')
# todo although, branches could be indicative of clustering! so maybe don't remove them
df = df.filter(regex='^(?!Unnamed:)')

#remove any columns with words in remove_cols in the name
df = df.drop(columns=[col for col in df.columns if any(word in col for word in remove_cols)])

# drop rows with nan
scaled_data = StandardScaler().fit_transform(df)

x = scaled_data
y = df_original['sample_type']
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


feature_cols = ['speed_med', 'node_width_med', 'displacement_max']
df_to_umap = df[feature_cols]
# df_to_umap = df
# make it a dataframe
# df_to_umap = pd.DataFrame(df_to_umap, columns=df.columns)
scaled_data_umap = StandardScaler().fit_transform(df_to_umap)

umap_params = {'n_neighbors': 30, 'min_dist': 0.5,
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
df_to_umap['sample_type'] = df_original['sample_type']

df_to_umap['index'] = df.index
fig = px.scatter(df_to_umap, x='UMAP 1', y='UMAP 2', color=df_original['file_name'], hover_data=['index', 'cluster'])
                 # range_color=[0, 2])
unique_clusters = df_to_umap['cluster'].unique()
fig.show()

variable_name = 'sample_type'
df_volcano = df.copy()
# drop nans
df_volcano = df_volcano.dropna()
df_volcano[variable_name] = df_original[variable_name]
# convert to int
#standard scale the data, except for concentration
df_volcano.iloc[:, :-1] = StandardScaler().fit_transform(df_volcano.iloc[:, :-1])
grouped_df = df_volcano.groupby(variable_name).mean()
# grouped_df = df_volcano.groupby('concentration').nanmedian()
# calculate log fold change
test = grouped_df.iloc[1] / grouped_df.iloc[0]
LFC = np.log2(grouped_df.iloc[1] / grouped_df.iloc[0])
# get first row of the grouped_df dataframe
SMDs = []
MDs = []
condition_1 = 'KO'
condition_2 = 'OE'
for col in df_volcano.columns[:-1]:
    mean1 = df_volcano[df_volcano[variable_name]==condition_1][col].mean()
    mean2 = df_volcano[df_volcano[variable_name]==condition_2][col].mean()
    sd1 = df_volcano[df_volcano[variable_name]==condition_1][col].std(ddof=1)
    sd2 = df_volcano[df_volcano[variable_name]==condition_2][col].std(ddof=1)
    n1, n2 = len(df_volcano[df_volcano[variable_name]==condition_1]), len(df_volcano[df_volcano[variable_name]==condition_2])
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
    pvals.append(ttest_ind(df_volcano[df_volcano[variable_name]==condition_1][col], df_volcano[df_volcano[variable_name]==condition_2][col])[1])

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