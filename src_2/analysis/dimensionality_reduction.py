from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

import seaborn as sns
from src import logger
from src_2.im_info.im_info import ImInfo
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap
import matplotlib.pyplot as plt
from umap import UMAP
import plotly.express as px
import plotly.io as pio

from src_2.utils.general import get_reshaped_image

pio.renderers.default = 'browser'
import numpy as np
from sklearn.cluster import HDBSCAN


class DimensionalityReduction:
    def __init__(self, im_info: ImInfo):
        self.im_info = im_info
        self.spatial_features = None
        self.embedding = None
        self.standardized_features = None
        self.fig = None
        self.cluster_labels = None

        self.label_memmap = None

    def _get_spatial_features(self):
        logger.debug('Importing CSV.')
        self.spatial_features = pd.DataFrame()
        self.morphology_label_features = pd.read_csv(self.im_info.pipeline_paths['morphology_label_features'])
        self.morphology_skeleton_features = pd.read_csv(self.im_info.pipeline_paths['morphology_skeleton_features'])
        # merge morphology label and skeleton dataframes based on "label" column
        self.morphology_features = pd.merge(self.morphology_label_features, self.morphology_skeleton_features, on='label')
        self.morphology_features = self.morphology_features.dropna()
        self.spatial_features = pd.concat([self.spatial_features, self.morphology_features], axis=1)
        self.labels = self.spatial_features['label']
        self.spatial_features = self.spatial_features.drop(columns=['label'])
        remove_cols = [
            'extent', 'solidity',
            'intensity_mean', 'intensity_std', 'intensity_range',
            'frangi_mean', 'frangi_std', 'frangi_range',
        ]
        self.spatial_features = self.spatial_features.drop(columns=remove_cols)
        # self.spatial_features = self.spatial_features.drop(columns=['solidity'])
        # drop na

    def _standardize_features(self):
        logger.debug('Standardizing features.')
        scaler = StandardScaler()
        # log transform first
        # self.spatial_features = np.log(self.spatial_features)
        self.standardized_features = scaler.fit_transform(self.spatial_features)
        self.standardized_features = pd.DataFrame(self.standardized_features, columns=self.spatial_features.columns)

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
        rf = RandomForestClassifier(n_estimators=100)
        X = self.spatial_features.drop('cluster', axis=1)
        y = self.spatial_features['cluster']
        rf.fit(X, y)
        self.feature_importances = rf.feature_importances_

        # Permutation Importance
        perm_importance = permutation_importance(rf, X, y)
        self.perm_feature_importances = perm_importance.importances_mean

        # Return the feature importance
        return self.feature_importances, self.perm_feature_importances

    def recolor_im_labels(self):
        # label idxs and values
        label_idxs = np.argwhere(self.label_memmap[0] > 0)
        label_values = self.label_memmap[0][self.label_memmap[0] > 0]
        unique_labels = np.unique(label_values)
        # if a unique label is not included in self.cluster_labels, add it as 0
        temp_cluster_labels = self.cluster_labels.copy()
        for unique_label in unique_labels:
            if unique_label not in temp_cluster_labels:
                temp_cluster_labels = np.append(temp_cluster_labels, 0)
        # get cluster labels for each label, -1 for those not in the list
        cluster_labels = np.full(len(label_values), -1)
        for i, label_value in enumerate(label_values):
            cluster_labels[i] = int(temp_cluster_labels[label_value-1])
        # recolor labels
        new_label_im = np.zeros(self.label_memmap[0].shape, dtype=np.int32)
        for i, label_idx in enumerate(label_idxs):
            new_label_im[label_idx[0], label_idx[1], label_idx[2]] = cluster_labels[i]
        return new_label_im


    def create_plot(self, color=None):
        self.fig = px.scatter(x=self.embedding[:, 0], y=self.embedding[:, 1], color=color, title="2D Projection of Spatial Features")
        self.fig.show()

    def plot_feature_distribution(self, feature_column):
        if feature_column not in self.spatial_features.columns:
            raise ValueError(f"{feature_column} not found in DataFrame.")

        features_with_cluster = self.spatial_features.copy()
        features_with_cluster['cluster'] = self.cluster_labels

        # Calculate the mean and standard deviation of the feature for each cluster
        stats = features_with_cluster.groupby('cluster')[feature_column].describe()

        # Extracting the 25th and 75th percentiles
        q1 = stats['25%']
        q3 = stats['75%']
        median = stats['50%']

        # Sort the clusters by the median feature value
        # sorted_indices = median.sort_values().index
        # q1 = q1.loc[sorted_indices]
        # q3 = q3.loc[sorted_indices]
        # median = median.loc[sorted_indices]

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
        plt.show()

    def _get_memmaps(self):
        label_memmap = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_instance_label'])
        self.label_memmap = get_reshaped_image(label_memmap, 1, self.im_info)

    def run(self):
        self._get_memmaps()
        self._get_spatial_features()


if __name__ == "__main__":
    im_path = r"D:\test_files\nelly_tests\deskewed-2023-07-13_14-58-28_000_wt_0_acquire.ome.tif"
    im_info = ImInfo(im_path)
    im_info.create_output_path('morphology_label_features', ext='.csv')
    im_info.create_output_path('morphology_skeleton_features', ext='.csv')
    im_info.create_output_path('im_instance_label')


    dim_red = DimensionalityReduction(im_info)
    dim_red.run()

    # dim_red.reduce_dimensions(method="tsne", n_components=2, perplexity=20, n_iter=1000)
    dim_red.reduce_dimensions(method="umap", n_components=2, n_neighbors=50, min_dist=0.5)
    dim_red.cluster(cluster_selection_epsilon=0.5)
    dim_red.create_plot(color=dim_red.cluster_labels)
    # dim_red.create_plot(color=dim_red.standardized_features['length'])
    new_labels = dim_red.recolor_im_labels()

    import napari
    viewer = napari.Viewer()
    viewer.add_labels(new_labels)

    logger.debug('Calculating feature importance.')
    rf = RandomForestClassifier(n_estimators=100)
    # drop any clusters = 0
    X = dim_red.spatial_features[dim_red.cluster_labels != 0]
    y = dim_red.cluster_labels[dim_red.cluster_labels != 0]
    rf.fit(X, y)
    feature_importances = rf.feature_importances_

    # Permutation Importance
    perm_importance = permutation_importance(rf, X, y)
    perm_feature_importances = perm_importance.importances_mean

    dim_red.plot_feature_distribution('length')

    feature_column = 'radius_weighted'

    if feature_column not in dim_red.spatial_features.columns:
        raise ValueError(f"{feature_column} not found in DataFrame.")

    features_with_cluster = dim_red.spatial_features.copy()
    features_with_cluster['cluster'] = dim_red.cluster_labels

    # Calculate the mean and standard deviation of the feature for each cluster
    stats = features_with_cluster.groupby('cluster')[feature_column].describe()

    # Extracting the 25th and 75th percentiles
    q1 = stats['25%']
    q3 = stats['75%']
    median = stats['50%']

    # Sort the clusters by the median feature value
    # sorted_indices = median.sort_values().index
    # q1 = q1.loc[sorted_indices]
    # q3 = q3.loc[sorted_indices]
    # median = median.loc[sorted_indices]

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
    plt.show()
