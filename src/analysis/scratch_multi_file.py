import pandas as pd
from sklearn.preprocessing import StandardScaler
# import umap
# from umap import UMAP
import plotly.io as pio
import os

pio.renderers.default = 'browser'

top_dir = r"D:\test_files\nelly_smorgasbord\output"

file_names = [
    'deskewed-iono_post',
    'deskewed-iono_pre',
    # 'deskewed-mt_ends',
    # 'deskewed-peroxisome',
    # 'deskewed-2023-07-13_14-58-28_000_wt_0_acquire',
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

# df_to_use = df_to_use.replace([np.inf, -np.inf], np.nan)
df_to_use = df_to_use.dropna()
# also drop infinities
og_df = df_to_use.copy()

# stats_to_use = ['median', 'max', 'std', 'q_25', 'q_75', 'min']
# features_to_use = ['rel_ang_vel_mag_12', 'rel_ang_acc_mag',
#                    'rel_lin_vel_mag_12', 'rel_lin_acc_mag',
#                    'ref_lin_vel_mag_12', 'ref_lin_acc_mag']
# keep_motility_cols = []
# for feature in features_to_use:
#     for stat in stats_to_use:
#         keep_motility_cols.append(f'{feature}_{stat}')
# keep_motility_cols.extend([
#     # f'com_directionality_12_{metric_to_use}','com_directionality_12_max',
#     # f'com_directionality_acceleration_{metric_to_use}','com_directionality_acceleration_max',
# ])
# actually, just use all columns from motility df
keep_motility_cols = branch_motility_df.columns
remove_motility_cols = [col for col in branch_motility_df.columns if col not in keep_motility_cols]
remove_cols_morphology = [
    'main_label', 'file', 'label', 't',
    # 'extent', 'solidity',
    'intensity_mean', 'intensity_std', 'intensity_range',
    'frangi_mean', 'frangi_std', 'frangi_range',
]

for col in (remove_cols_morphology + remove_motility_cols):
    if col in df_to_use.columns:
        df_to_use.drop(columns=col, inplace=True)

# df_to_save = df_to_use.copy()
# df_to_save['file'] = og_df['file']
# # save to csv
# df_to_save.to_csv(os.path.join(top_dir, 'iono_features.csv'), index=False)

scaler = StandardScaler()
standardized_features = scaler.fit_transform(df_to_use)
standardized_features = pd.DataFrame(standardized_features, columns=df_to_use.columns)


# from umap import UMAP
# model = UMAP(n_components=2, n_neighbors=5, min_dist=0.1)
# embedding = model.fit_transform(standardized_features)
#
# colorby = f'file'
# log10 = False
# if log10:
#     colorset = np.log10(og_df[colorby])
# else:
#     colorset = og_df[colorby]
# # colorby = f'ref_lin_vel_mag_12_{metric_to_use}'
# embedding_fig = px.scatter(og_df, x=embedding[:, 0], y=embedding[:, 1], color=colorset,
#                            title="2D Projection Features", hover_data=['area', 'rel_ang_vel_mag_12_max', 'ref_lin_vel_mag_12_mean'])
# embedding_fig.show()


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

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

X = df_to_use
y = og_df['file']
# Initialize the classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Define the k-fold cross-validation procedure
# cv = KFold(n_splits=5, random_state=42, shuffle=True)

# Evaluate the model with cross-validation
# scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
report = classification_report(y_test, y_pred)


# get roc curve
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# get the probabilities for each class
y_prob = clf.predict_proba(X_test)
# get the probabilities for the positive class
y_prob = y_prob[:, 1]
# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label='deskewed-iono_pre.ome-ch0-organelle_motility_features.csv')
# calculate roc auc
roc_auc = roc_auc_score(y_test, y_prob)

# generate a random guess line
r_x, r_y = [0, 1], [0, 1]
# plot the random guess line

# plot the roc curve
plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, marker='.')
plt.plot(r_x, r_y, linestyle='--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
print(f'ROC AUC: {roc_auc}')
# plt.savefig(os.path.join(top_dir, 'roc_curve.png'), dpi=500)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# Example: Assuming you have two groups "group1" and "group2"
# Replace these with your actual group columns or logic
group1 = df_to_use[og_df['file'] == 'deskewed-iono_post.ome-ch0-organelle_motility_features.csv']
group2 = df_to_use[og_df['file'] == 'deskewed-iono_pre.ome-ch0-organelle_motility_features.csv']

# Compute fold change and p-values
features = df_to_use.columns.difference(['file'])  # Adjust as needed
fold_changes = group1[features].mean() / group2[features].mean()
p_values = ttest_ind(group1[features], group2[features], axis=0).pvalue

# Prepare DataFrame for volcano plot
volcano_df = pd.DataFrame({
    'Feature': features,
    'Log2FoldChange': np.log2(fold_changes),
    '-Log10PValue': -np.log10(p_values)
})

# Plotting the Volcano Plot with Annotations
plt.figure(figsize=(12, 8))
plt.title('Volcano Plot')
plt.xlabel('Log2 Fold Change')
plt.ylabel('-Log10 P-Value')
sns.scatterplot(data=volcano_df, x='Log2FoldChange', y='-Log10PValue', edgecolor=None)

# Annotating significant features
# Define your criteria for significance
significant = volcano_df[(abs(volcano_df['Log2FoldChange']) > np.log2(0.2)) &
                         (volcano_df['-Log10PValue'] > -np.log10(0.05))]
for index, row in significant.iterrows():
    plt.text(row['Log2FoldChange'], row['-Log10PValue'], row['Feature'], horizontalalignment='left', size='small', color='black', weight='semibold')

sns.scatterplot(data=significant, x='Log2FoldChange', y='-Log10PValue', color='red', edgecolor=None)
plt.savefig(os.path.join(top_dir, 'volcano_plot.png'), dpi=500)
# # plt.show()


