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
pio.renderers.default = 'browser'


class DimensionalityReduction:
    def __init__(self, im_info: ImInfo):
        self.im_info = im_info
        self.spatial_features = None
        self.embedding = None
        self.standardized_features = None
        self.fig = None

    def _get_spatial_features(self):
        logger.debug('Importing CSV.')
        self.spatial_features = pd.DataFrame()
        self.morphology_label_features = pd.read_csv(self.im_info.pipeline_paths['morphology_label_features'])
        self.spatial_features = pd.concat([self.spatial_features, self.morphology_label_features], axis=1)
        self.labels = self.spatial_features['label']
        self.spatial_features = self.spatial_features.drop(columns=['label'])
        self.spatial_features = self.spatial_features.drop(columns=['extent'])
        # self.spatial_features = self.spatial_features.drop(columns=['solidity'])

    def _standardize_features(self):
        logger.debug('Standardizing features.')
        scaler = StandardScaler()
        self.standardized_features = scaler.fit_transform(self.spatial_features)

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

    def create_plot(self, color=None):
        self.fig = px.scatter(x=self.embedding[:, 0], y=self.embedding[:, 1], color=color, title="2D Projection of Spatial Features")
        self.fig.show()

    def run(self):
        self._get_spatial_features()


if __name__ == "__main__":
    im_path = r"D:\test_files\nelly_tests\deskewed-2023-07-13_14-58-28_000_wt_0_acquire.ome.tif"
    im_info = ImInfo(im_path)
    im_info.create_output_path('morphology_label_features', ext='.csv')

    dim_red = DimensionalityReduction(im_info)
    dim_red.run()

    # dim_red.reduce_dimensions(method="tsne", n_components=2, perplexity=30, n_iter=1000)
    dim_red.reduce_dimensions(method="umap", n_components=2, n_neighbors=10, min_dist=0.1)
    dim_red.create_plot(color=dim_red.spatial_features['solidity'])

