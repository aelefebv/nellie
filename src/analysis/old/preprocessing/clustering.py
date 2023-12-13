import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from datetime import datetime
from umap import UMAP

from src.analysis.old.preprocessing.csv_preprocessing import fill_na_in_csv


class ClusteringAnalysis:
    def __init__(self, df_in, output_dir, file_path,
                 remove_cols=None,
                 ignore_cols=None,
                 keep_groups=None,
                 corr_threshold=None,
                 n_pca_clusters=None,
                 umap_params=None,
                 dbscan_eps=None,
                 dbscan_min_samples=None,
                 sample_size=None,
                 frames_to_keep=None,
                 save_df_out=True):
        self.date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_path = file_path
        self.df_in = df_in
        self.output_dir = output_dir
        self.remove_cols = remove_cols or ['len_weighted', 'angles', 'orientation', 'cell_center', 'direction', 'fission', 'fusion']
        self.ignore_cols = ignore_cols or ['group', 'filename', 'frame_number',
                                           'branch_ids', 'node_id', 'region_id', 'branch_id', 'node_ids']
        self.keep_groups = keep_groups
        self.corr_threshold = corr_threshold or 0.67
        self.n_pca_clusters = n_pca_clusters or 3
        self.umap_params = umap_params
        self.dbscan_eps = dbscan_eps or 0.75
        self.dbscan_min_samples = dbscan_min_samples or 5
        self.sample_size = sample_size
        self.frames_to_keep = frames_to_keep

        self.df_out = None
        self.save_df_out = save_df_out

    def preprocess_data(self):
        df_in = self.df_in
        if self.frames_to_keep is not None:
            # if frames to keep is not a list, make it a list
            if not isinstance(self.frames_to_keep, list):
                self.frames_to_keep = [self.frames_to_keep]
            df_in = df_in[df_in['frame_number'].isin(self.frames_to_keep)]
        if self.keep_groups is not None:
            df_in = df_in[df_in['group'].isin(self.keep_groups)]
        if self.sample_size is not None:
            df_in = df_in.sample(n=self.sample_size, random_state=42)
        df_all = df_in.loc[:, ~df_in.columns.str.contains('|'.join(self.remove_cols))]
        df_all = df_all.dropna(axis=1, how='all')
        # df_all = df_all.fillna(1)
        for col in self.ignore_cols:
            if col in df_all.columns:
                df_all = df_all.drop(col, axis=1)
        df = df_all.copy()
        df = df.filter(regex='^(?!.*_n$)')
        # df = df.filter(regex='^(?!.*_median$)')
        # df = df.filter(regex='^(?!.*_quartiles*)')
        # df = df.filter(regex='^(?!.*_std$)')  # maybe keep?

        corr_matrix = df.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype('bool'))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > self.corr_threshold)]
        df = df.drop(to_drop, axis=1)

        self.df_in = df_in
        self.df = df
        self.scaled_data = StandardScaler().fit_transform(df)

    def pca_clustering(self):
        pca = PCA(n_components=0.95)
        reduced_data = pca.fit_transform(self.scaled_data)

        kmeans = KMeans(n_clusters=self.n_pca_clusters, random_state=42)
        kmeans_cluster_labels = kmeans.fit_predict(reduced_data)
        df_with_labels = self.df_in.copy()
        df_with_labels['cluster'] = kmeans_cluster_labels
        df_with_labels['pca_1'] = reduced_data[:, 0]
        df_with_labels['pca_2'] = reduced_data[:, 1]

        return reduced_data, kmeans_cluster_labels, df_with_labels

    def umap_clustering(self, umap_params=None):
        if self.umap_params is None:
            umap_params = {'n_components': 2, 'n_neighbors': 15, 'min_dist': 0.1, 'metric': 'euclidean', 'random_state': 42}
            self.umap_params = umap_params
        else:
            umap_params = self.umap_params
        umap_model = UMAP(**umap_params)
        umap_results = umap_model.fit_transform(self.scaled_data)

        dbscan = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
        cluster_labels_dbscan = dbscan.fit_predict(umap_results)

        return umap_results, cluster_labels_dbscan

    def plot_pca(self, reduced_data, cluster_labels):
        integer_labels = LabelEncoder().fit_transform(cluster_labels)
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.set_xlabel('First Principal Component')
        ax.set_ylabel('Second Principal Component')
        plt.title('Clusters in PCA space')
        scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1],
                              cmap='tab20', c=integer_labels, s=1)
        legend1 = ax.legend(*scatter.legend_elements(),
                            title="Clusters", bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.,
                            handlelength=2.5)
        ax.add_artist(legend1)
        plt.subplots_adjust(right=0.80)
        full_save_path = os.path.join(self.output_dir, f'{self.date_time}-{self.file_path}-pca_kmeans_plot.png')
        plt.savefig(full_save_path, dpi=500, bbox_inches='tight')
        plt.show()
        plt.close()
        print(f"Saved plot to {full_save_path}")

    def plot_umap(self, umap_results, cluster_labels_dbscan):
        if -1 in cluster_labels_dbscan:
            integer_labels = LabelEncoder().fit_transform(cluster_labels_dbscan) - 1
        else:
            integer_labels = LabelEncoder().fit_transform(cluster_labels_dbscan)

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.set_xlabel('First UMAP Component')
        ax.set_ylabel('Second UMAP Component')
        plt.title('DBSCAN Clusters in UMAP space')
        scatter = plt.scatter(umap_results[:, 0], umap_results[:, 1],
                              cmap='tab20', c=integer_labels, s=1)

        # Manually create the legend by iterating over the unique cluster labels
        unique_labels = np.unique(integer_labels)
        handles = []
        labels = []
        for label in unique_labels:
            handles.append(plt.scatter([], [], color=scatter.cmap(scatter.norm(label)), s=10))
            labels.append(str(label))

        legend1 = ax.legend(handles, labels, title="Clusters", bbox_to_anchor=(1.01, 1),
                            loc=2, borderaxespad=0., handlelength=2.5)
        ax.add_artist(legend1)
        plt.subplots_adjust(right=0.80)
        full_save_path = os.path.join(self.output_dir, f'{self.date_time}-{self.file_path}-umap_dbscan_plot.png')
        plt.savefig(full_save_path, dpi=500, bbox_inches='tight')
        plt.show()
        plt.close()
        print(f"Saved plot to {full_save_path}")

    def save_parameters(self):
        params = {
            'remove_cols': self.remove_cols,
            'ignore_cols': self.ignore_cols,
            'keep_groups': self.keep_groups,
            'corr_threshold': self.corr_threshold,
            'n_clusters': self.n_pca_clusters,
            'umap_n_components': self.umap_params['n_components'],
            'umap_n_neighbors': self.umap_params['n_neighbors'],
            'umap_min_dist': self.umap_params['min_dist'],
            'umap_metric': self.umap_params['metric'],
            'umap_random_state': self.umap_params['random_state'],
            'dbscan_eps': self.dbscan_eps,
            'dbscan_min_samples': self.dbscan_min_samples,
        }

        # Convert non-list values to single-element lists
        params_series = pd.Series(params, name='value')
        params_df = params_series.to_frame()
        params_df.to_csv(os.path.join(self.output_dir, f'{self.date_time}-{self.file_path}-parameters.csv'))

    def run(self):
        self.preprocess_data()
        reduced_data, kmeans_cluster_labels, df_with_labels = self.pca_clustering()
        self.plot_pca(reduced_data, kmeans_cluster_labels)

        umap_results, cluster_labels_dbscan = self.umap_clustering()
        self.plot_umap(umap_results, cluster_labels_dbscan)

        # Save PCA and UMAP results with cluster labels
        df_with_labels['cluster_pca'] = kmeans_cluster_labels
        df_with_labels['cluster_umap'] = cluster_labels_dbscan
        df_with_labels['umap_1'] = umap_results[:, 0]
        df_with_labels['umap_2'] = umap_results[:, 1]
        # if the 'filename' column doesn't exist, add it
        if 'filename' not in df_with_labels.columns:
            df_with_labels['filename'] = self.file_path
        self.df_out = df_with_labels

        if self.save_df_out:
            df_with_labels.to_csv(os.path.join(self.output_dir, f'{self.date_time}-{self.file_path}-umap_pca_data.csv'), index=False)
        self.save_parameters()

        return self.df_out


def run_20230410_tips_umap_test(df_in, output_dir):
    clustering_analysis = ClusteringAnalysis(df_in, output_dir, keep_groups=['ctrl'])
    clustering_analysis.run()


if __name__ == '__main__':
    output_dir = r'D:\test_files\nelly\20230406-AELxKL-dmr_lipid_droplets_mtDR\output\csv'
    file_path = 'summary_stats_regions-deskewed-2023-04-06_17-01-43_000_AELxKL-dmr_PERK-lipid_droplets_mtDR-5000-4h.ome-ch1.csv'
    # df_in = pd.read_csv(os.path.join(output_dir, file_path))
    df_preprocessed = fill_na_in_csv(output_dir, file_path)
    # run_20230410_tips_umap_test(df_in, output_dir)
    ignore_cols = ['group', 'filename', 'frame_number',
                   'branch_ids', 'node_id', 'region_id', 'branch_id', 'node_ids']
    remove_cols = None
    # remove_cols = ['qqq']
    clustering_analysis = ClusteringAnalysis(df_preprocessed, output_dir, file_path,
                                             corr_threshold=0.8, frames_to_keep=1,
                                             ignore_cols=ignore_cols,
                                             remove_cols=remove_cols,)
    df_out = clustering_analysis.run()
