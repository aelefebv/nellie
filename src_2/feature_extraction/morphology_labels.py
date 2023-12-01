from src import logger
from src_2.im_info.im_info import ImInfo
from src_2.utils.general import get_reshaped_image
import skimage.measure
import numpy as np
import pandas as pd


class MorphologyLabelFeatures:
    def __init__(self, im_info: ImInfo, t=1):
        self.im_info = im_info
        self.t = t
        if self.im_info.no_z:
            self.spacing = (self.im_info.dim_sizes['Y'], self.im_info.dim_sizes['X'])
        else:
            self.spacing = (self.im_info.dim_sizes['Z'], self.im_info.dim_sizes['Y'], self.im_info.dim_sizes['X'])
        self.im_memmap = None
        self.im_frangi_memmap = None
        self.label_memmap = None

        self.organelle_label_features_path = None
        self.branch_label_features_path = None

        self.organelle_features = {}

    def _label_morphology(self, label_im):
        label_objects_intensity = skimage.measure.regionprops(label_im, self.im_memmap, spacing=self.spacing)
        log10_frangi = np.log10(self.im_frangi_memmap)
        log10_frangi[np.isinf(log10_frangi)] = 0
        label_objects_frangi = skimage.measure.regionprops(label_im, log10_frangi, spacing=self.spacing)

        features = {'label': [label_object.label for label_object in label_objects_intensity],
                    'area': [label_object.area for label_object in label_objects_intensity],
                    'extent': [label_object.extent for label_object in label_objects_intensity],
                    'solidity': [label_object.solidity for label_object in label_objects_intensity]}

        # intensity image
        sorted_inertia_tensor_eigvals_intensity = [sorted(label_object.inertia_tensor_eigvals) for label_object in label_objects_intensity]
        features['inertia_tensor_eig_sorted_min'] = [sorted_inertia_tensor_eigvals[0] for sorted_inertia_tensor_eigvals in sorted_inertia_tensor_eigvals_intensity]
        features['inertia_tensor_eig_sorted_max'] = [sorted_inertia_tensor_eigvals[-1] for sorted_inertia_tensor_eigvals in sorted_inertia_tensor_eigvals_intensity]
        if not self.im_info.no_z:
            features['inertia_tensor_eig_sorted_mid'] = [sorted_inertia_tensor_eigvals[1] for sorted_inertia_tensor_eigvals in sorted_inertia_tensor_eigvals_intensity]

        features['intensity_mean'] = [label_object.mean_intensity for label_object in label_objects_intensity]
        features['intensity_range'] = [label_object.max_intensity - label_object.min_intensity for label_object in label_objects_intensity]
        features['intensity_std'] = [np.std(label_object.image_intensity[label_object.image_intensity>0]) for label_object in label_objects_intensity]

        # frangi image
        features['frangi_mean'] = [label_object.mean_intensity for label_object in label_objects_frangi]
        features['frangi_range'] = [label_object.max_intensity - label_object.min_intensity for label_object in label_objects_frangi]
        features['frangi_std'] = [np.nanstd(label_object.image_intensity[label_object.image_intensity!=0]) for label_object in label_objects_frangi]

        return features

    def _get_memmaps(self):
        logger.debug('Allocating memory for spatial feature extraction.')

        num_t = self.im_info.shape[self.im_info.axes.index('T')]
        if num_t == 1:
            self.t = 0

        im_memmap = self.im_info.get_im_memmap(self.im_info.im_path)
        self.im_memmap = get_reshaped_image(im_memmap, num_t, self.im_info)

        im_frangi_memmap = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_frangi'])
        self.im_frangi_memmap = get_reshaped_image(im_frangi_memmap, num_t, self.im_info)

        label_memmap = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_instance_label'])
        self.label_memmap = get_reshaped_image(label_memmap, num_t, self.im_info)

        branch_label_memmap = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_skel_relabelled'])
        self.branch_label_memmap = get_reshaped_image(branch_label_memmap, num_t, self.im_info)

        if not self.im_info.no_t:
            self.im_memmap = self.im_memmap[self.t]
            self.im_frangi_memmap = self.im_frangi_memmap[self.t]
            self.label_memmap = self.label_memmap[self.t]
            self.branch_label_memmap = self.branch_label_memmap[self.t]

        # self.im_info.create_output_path('morphology_label_features', ext='.csv')
        self.organelle_label_features_path = self.im_info.pipeline_paths['organelle_label_features']
        self.branch_label_features_path = self.im_info.pipeline_paths['branch_label_features']

        self.shape = self.label_memmap.shape

    def _save_features(self):
        logger.debug('Saving spatial features.')
        organelle_features_df = pd.DataFrame.from_dict(self.organelle_features)
        organelle_features_df.to_csv(self.organelle_label_features_path, index=False)

        branch_features_df = pd.DataFrame.from_dict(self.branch_features)
        branch_features_df.to_csv(self.branch_label_features_path, index=False)

    def run(self):
        self._get_memmaps()
        self.organelle_features = self._label_morphology(self.label_memmap)
        # rename label to main_label
        self.organelle_features['main_label'] = self.organelle_features.pop('label')
        self.branch_features = self._label_morphology(self.branch_label_memmap)
        self._save_features()


if __name__ == "__main__":
    im_path = r"D:\test_files\nelly_tests\deskewed-2023-07-13_14-58-28_000_wt_0_acquire.ome.tif"
    im_info = ImInfo(im_path)

    morphology_label_features = MorphologyLabelFeatures(im_info)
    morphology_label_features.run()