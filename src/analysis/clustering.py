from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

import seaborn as sns
from src import logger
from src.im_info.im_info import ImInfo
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio

from src.utils.general import get_reshaped_image

pio.renderers.default = 'browser'
import numpy as np
from sklearn.cluster import HDBSCAN


class Clustering:
    def __init__(self, im_info: ImInfo, t=1):
        self.im_info = im_info
        self.t = t
        self.spatial_features = None
        self.temporal_features = None
        self.all_features = None
        self.embedding = None
        self.standardized_features = None

        self.embedding_fig = None
        self.feature_fig = None

        self.feature_rf_importances = None
        self.feature_perm_importances = None

        self.df_of_interest = None

        self.cluster_labels = None

        self.label_memmap = None
        self.branch_label_memmap = None
        self.use_memmap = None

        self.label_type = None
        self.label_name = None

    def get_features_dfs(self, label_type='organelle'):
        logger.debug('Importing CSV.')
        if label_type == 'organelle':
            label_path = self.im_info.pipeline_paths['organelle_label_features']
            skeleton_path = self.im_info.pipeline_paths['organelle_skeleton_features']
            motility_path = self.im_info.pipeline_paths['organelle_motility_features']
            self.use_memmap = self.label_memmap
            self.label_name = 'main_label'
        elif label_type == 'branch':
            label_path = self.im_info.pipeline_paths['branch_label_features']
            skeleton_path = self.im_info.pipeline_paths['branch_skeleton_features']
            motility_path = self.im_info.pipeline_paths['branch_motility_features']
            self.use_memmap = self.branch_label_memmap
            self.label_name = 'label'
        else:
            raise ValueError(f'Unknown label type: {label_type}')

        self.spatial_features = pd.DataFrame()
        self.morphology_label_features = pd.read_csv(label_path)
        self.morphology_skeleton_features = pd.read_csv(skeleton_path)
        # merge morphology label and skeleton dataframes based on "label" column
        self.morphology_features = pd.merge(self.morphology_label_features, self.morphology_skeleton_features, on=self.label_name)
        self.motility_label_features = pd.read_csv(motility_path)

        self.all_features = pd.merge(self.morphology_features, self.motility_label_features, on=self.label_name)

        self.spatial_features = pd.concat([self.spatial_features, self.morphology_features], axis=1)
        self.labels = self.spatial_features[self.label_name]
        # self.spatial_features = self.spatial_features.drop(columns=[self.label_name])
        remove_cols_spatial = [
            'extent', 'solidity',
            'intensity_mean', 'intensity_std', 'intensity_range',
            # 'frangi_mean', 'frangi_std', 'frangi_range',
        ]
        self.spatial_features = self.spatial_features.drop(columns=remove_cols_spatial)
        keep_cols_temporal = [
            self.label_name,
            'rel_ang_vel_mag_12_mean', 'rel_ang_vel_mag_12_max',
            'rel_ang_acc_mag_mean', 'rel_ang_acc_mag_max',
            'rel_lin_vel_mag_12_mean', 'rel_lin_vel_mag_12_max',
            'rel_lin_acc_mag_mean', 'rel_lin_acc_mag_max',
            'ref_lin_vel_mag_12_mean', 'ref_lin_vel_mag_12_max',
            'ref_lin_acc_mag_mean', 'ref_lin_acc_mag_max',
            'lin_vel_mag_12_mean', 'lin_vel_mag_12_max',
        ]
        remove_cols_temporal = [col for col in self.motility_label_features.columns if col not in keep_cols_temporal]
        # self.temporal_features = self.motility_label_features[keep_cols_temporal]
        self.temporal_features = self.motility_label_features.drop(columns=remove_cols_temporal)

        remove_cols_both = remove_cols_spatial + remove_cols_temporal
        self.all_features = self.all_features.drop(columns=remove_cols_both)

        self.spatial_features = self.spatial_features.dropna()
        self.temporal_features = self.temporal_features.dropna()
        self.all_features = self.all_features.dropna()


    def _standardize_features(self):
        logger.debug('Standardizing features.')
        scaler = StandardScaler()
        # log transform first
        # self.spatial_features = np.log(self.spatial_features)
        temp_df = self.df_of_interest.copy()
        temp_df = temp_df.drop(columns=[self.label_name])
        self.standardized_features = scaler.fit_transform(temp_df)
        self.standardized_features = pd.DataFrame(self.standardized_features, columns=temp_df.columns)

    def reduce_dimensions(self, method="tsne", **kwargs):
        if self.standardized_features is None:
            self._standardize_features()

        logger.debug(f"Reducing dimensions using {method}.")
        if method == "tsne":
            model = TSNE(**kwargs)
        elif method == "umap":
            model = umap.UMAP(**kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

        self.embedding = model.fit_transform(self.standardized_features)

    def cluster(self, min_cluster_size=5, min_samples=None, cluster_selection_epsilon=0.0, **kwargs):
        logger.debug('Clustering using HDBSCAN.')
        if self.embedding is None:
            self.reduce_dimensions()  # You might want to specify default method and kwargs

        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            **kwargs
        )

        # Fitting the HDBSCAN on the embedding.
        self.cluster_labels = clusterer.fit_predict(self.embedding)
        self.cluster_labels += 1

        return self.cluster_labels  # Returning the labels might be useful

    def feature_importance(self):
        logger.debug('Calculating feature importance.')
        temp_df = self.df_of_interest.copy()
        temp_df = temp_df.drop(columns=[self.label_name])
        rf = RandomForestClassifier(n_estimators=100)
        X = temp_df[self.cluster_labels > 0]
        y = self.cluster_labels[self.cluster_labels > 0]
        rf.fit(X, y)
        self.feature_rf_importances = rf.feature_importances_

        # Permutation Importance
        perm_importance = permutation_importance(rf, X, y)
        self.feature_perm_importances = perm_importance.importances_mean

        # Return the feature importance
        return self.feature_rf_importances, self.feature_perm_importances

    def recolor_im_labels(self):
        # label idxs and values
        valid_labels = self.df_of_interest[self.label_name].values
        label_idxs = np.argwhere(self.use_memmap > 0)
        label_values = self.use_memmap[self.use_memmap > 0]
        unique_labels = np.unique(label_values)
        # if a unique label is not included in self.cluster_labels, add it as 0
        temp_cluster_labels = self.cluster_labels.copy()
        cluster_label_mapping = self.labels.values
        for unique_label in unique_labels:
            if unique_label not in cluster_label_mapping:
                temp_cluster_labels = np.append(temp_cluster_labels, 0)
                cluster_label_mapping = np.append(cluster_label_mapping, unique_label)
            elif unique_label not in valid_labels:
                temp_cluster_labels = np.append(temp_cluster_labels, 0)
        # get cluster labels for each label, -1 for those not in the list
        cluster_labels = np.full(len(label_values), 0)
        for i, label_value in enumerate(label_values):
            # if the label value is not in valid labels, skip
            # if label_value not in valid_labels:
            #     continue
            cluster_labels[i] = temp_cluster_labels[cluster_label_mapping == label_value]

        # recolor labels
        new_label_im = np.zeros(self.use_memmap.shape, dtype=np.int32)
        for i, label_idx in enumerate(label_idxs):
            new_label_im[label_idx[0], label_idx[1], label_idx[2]] = cluster_labels[i]
        return new_label_im


    def create_embedding_plot(self, color=None):
        self.embedding_fig = px.scatter(x=self.embedding[:, 0], y=self.embedding[:, 1], color=color, title="2D Projection of Spatial Features")
        self.embedding_fig.show()

    def plot_feature_distribution(self, feature_column):
        if feature_column not in self.df_of_interest.columns:
            raise ValueError(f"{feature_column} not found in DataFrame.")

        features_with_cluster = self.df_of_interest.copy()
        features_with_cluster['cluster'] = self.cluster_labels
        features_with_cluster = features_with_cluster[features_with_cluster['cluster']>0]

        # Calculate the mean and standard deviation of the feature for each cluster
        stats = features_with_cluster.groupby('cluster')[feature_column].describe()

        # Extracting the 25th and 75th percentiles
        q1 = stats['25%']
        q3 = stats['75%']
        median = stats['50%']

        # Plotting using seaborn for a nicer appearance
        sns.set_style("whitegrid")
        plt.figure(figsize=(10, 6))

        # Using barplot to plot the median values
        sns.barplot(x=stats.index, y=median, palette="viridis")

        # Adding error bars for the IQR
        plt.errorbar(x=stats.index-1, y=median, yerr=[median - q1, q3 - median], fmt='none', c='black', capsize=5)

        plt.title(f'Median {feature_column} Value by Cluster with IQR')
        plt.xlabel('Cluster Label')
        plt.ylabel(f'Median {feature_column}')
        plt.xticks(rotation=45)
        plt.tight_layout()  # Adjusts plot to ensure everything fits without overlapping

        self.feature_fig = plt.gcf()
        self.feature_fig.show()

    def _get_memmaps(self):
        num_t = self.im_info.shape[self.im_info.axes.index('T')]
        if num_t == 1:
            self.t = 0

        label_memmap = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_instance_label'])
        self.label_memmap = get_reshaped_image(label_memmap, num_t, self.im_info)

        branch_label_memmap = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_skel_relabelled'])
        self.branch_label_memmap = get_reshaped_image(branch_label_memmap, num_t, self.im_info)

        if not self.im_info.no_t:
            self.label_memmap = self.label_memmap[self.t]
            self.branch_label_memmap = self.branch_label_memmap[self.t]

    def set_df_of_interest(self, df):
        self.df_of_interest = df

    def run(self):
        self._get_memmaps()


if __name__ == "__main__":
    im_path = r"D:\test_files\nelly_tests\deskewed-2023-07-13_14-58-28_000_wt_0_acquire.ome.tif"
    im_info = ImInfo(im_path)

    dim_red = Clustering(im_info)
    dim_red.run()
    dim_red.get_features_dfs('branch')

    dim_red.set_df_of_interest(dim_red.all_features)

    # dim_red.reduce_dimensions(method="tsne", n_components=2, perplexity=10, n_iter=1000)
    dim_red.reduce_dimensions(method="umap", n_components=2, n_neighbors=10, min_dist=0)
    dim_red.cluster(cluster_selection_epsilon=0.5)
    dim_red.create_embedding_plot(color=dim_red.cluster_labels)
    new_labels = dim_red.recolor_im_labels()

    dim_red.plot_feature_distribution('rel_ang_vel_mag_12_mean')
    dim_red.feature_importance()

    # plot area vs rel_ang_vel_mag_12_mean
    feature_1 = 'area'
    feature_2 = 'rel_lin_vel_mag_12_mean'
    plt.figure()
    plt.scatter(np.log10(dim_red.df_of_interest[feature_1]), dim_red.df_of_interest[feature_2], s=1)
    plt.xlabel(feature_1)
    plt.ylabel(feature_2)
    plt.show()

    import napari
    viewer = napari.Viewer()
    viewer.add_labels(new_labels, name='clustered_labels')
