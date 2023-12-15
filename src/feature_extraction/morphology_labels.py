from src import logger
from src.im_info.im_info import ImInfo
from src.utils.general import get_reshaped_image
import skimage.measure
import numpy as np
import pandas as pd


class MorphologyLabelFeatures:
    def __init__(self, im_info: ImInfo,
                 num_t=None):
        self.im_info = im_info
        self.num_t = num_t
        if self.im_info.no_z:
            self.spacing = (self.im_info.dim_sizes['Y'], self.im_info.dim_sizes['X'])
        else:
            self.spacing = (self.im_info.dim_sizes['Z'], self.im_info.dim_sizes['Y'], self.im_info.dim_sizes['X'])
        self.im_memmap = None
        self.im_frangi_memmap = None
        self.label_memmap = None

        self.organelle_label_features_path = None
        self.branch_label_features_path = None

        self.organelle_features = {'t': [], 'label': [], 'area': [], 'extent': [], 'solidity': [],
                                   'inertia_tensor_eig_sorted_min': [], 'inertia_tensor_eig_sorted_max': [],
                                   'intensity_mean': [], 'intensity_range': [], 'intensity_std': [],
                                   'frangi_mean': [], 'frangi_range': [], 'frangi_std': []}
        self.branch_features = {'t': [], 'label': [], 'area': [], 'extent': [], 'solidity': [],
                                'inertia_tensor_eig_sorted_min': [], 'inertia_tensor_eig_sorted_max': [],
                                'intensity_mean': [], 'intensity_range': [], 'intensity_std': [],
                                'frangi_mean': [], 'frangi_range': [], 'frangi_std': []}
        if not self.im_info.no_z:
            self.organelle_features['inertia_tensor_eig_sorted_mid'] = []
            self.branch_features['inertia_tensor_eig_sorted_mid'] = []


    def _label_morphology_frame(self, label_im, t, feature_dict):
        label_objects_intensity = skimage.measure.regionprops(label_im, self.im_memmap[t], spacing=self.spacing)
        log10_frangi = np.log10(self.im_frangi_memmap[t])
        log10_frangi[np.isinf(log10_frangi)] = 0
        label_objects_frangi = skimage.measure.regionprops(label_im, log10_frangi, spacing=self.spacing)

        feature_dict['t'].extend([t] * len(label_objects_intensity))
        feature_dict['label'].extend([label_object.label for label_object in label_objects_intensity])
        feature_dict['area'].extend([label_object.area for label_object in label_objects_intensity])
        feature_dict['extent'].extend([label_object.extent for label_object in label_objects_intensity])
        feature_dict['solidity'].extend([label_object.solidity for label_object in label_objects_intensity])

        # intensity image
        sorted_inertia_tensor_eigvals_intensity = [sorted(label_object.inertia_tensor_eigvals) for label_object in label_objects_intensity]
        feature_dict['inertia_tensor_eig_sorted_min'].extend([sorted_inertia_tensor_eigvals[0] for sorted_inertia_tensor_eigvals in sorted_inertia_tensor_eigvals_intensity])
        feature_dict['inertia_tensor_eig_sorted_max'].extend([sorted_inertia_tensor_eigvals[-1] for sorted_inertia_tensor_eigvals in sorted_inertia_tensor_eigvals_intensity])
        if not self.im_info.no_z:
            feature_dict['inertia_tensor_eig_sorted_mid'].extend([sorted_inertia_tensor_eigvals[1] for sorted_inertia_tensor_eigvals in sorted_inertia_tensor_eigvals_intensity])

        feature_dict['intensity_mean'].extend([label_object.mean_intensity for label_object in label_objects_intensity])
        feature_dict['intensity_range'].extend([label_object.max_intensity - label_object.min_intensity for label_object in label_objects_intensity])
        feature_dict['intensity_std'].extend([np.std(label_object.image_intensity[label_object.image_intensity>0]) for label_object in label_objects_intensity])

        # frangi image
        feature_dict['frangi_mean'].extend([label_object.mean_intensity for label_object in label_objects_frangi])
        feature_dict['frangi_range'].extend([label_object.max_intensity - label_object.min_intensity for label_object in label_objects_frangi])
        feature_dict['frangi_std'].extend([np.nanstd(label_object.image_intensity[label_object.image_intensity!=0]) for label_object in label_objects_frangi])

    def _label_morphology(self):
        for t in range(1, self.num_t-1):
            logger.debug(f'Processing frame {t + 1} of {self.num_t}')
            self._label_morphology_frame(self.label_memmap[t], t, self.organelle_features)
            self._label_morphology_frame(self.branch_label_memmap[t], t, self.branch_features)

    def _get_t(self):
        if self.num_t is None:
            if self.im_info.no_t:
                self.num_t = 1
            else:
                self.num_t = self.im_info.shape[self.im_info.axes.index('T')]
        else:
            return

    def _get_memmaps(self):
        logger.debug('Allocating memory for spatial feature extraction.')

        im_memmap = self.im_info.get_im_memmap(self.im_info.im_path)
        self.im_memmap = get_reshaped_image(im_memmap, self.num_t, self.im_info)

        im_frangi_memmap = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_frangi'])
        self.im_frangi_memmap = get_reshaped_image(im_frangi_memmap, self.num_t, self.im_info)

        label_memmap = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_instance_label'])
        self.label_memmap = get_reshaped_image(label_memmap, self.num_t, self.im_info)

        branch_label_memmap = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_skel_relabelled'])
        self.branch_label_memmap = get_reshaped_image(branch_label_memmap, self.num_t, self.im_info)

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
        self._get_t()
        self._get_memmaps()
        self._label_morphology()
        # rename label to main_label
        self.organelle_features['main_label'] = self.organelle_features.pop('label')
        self._save_features()


if __name__ == "__main__":
    im_path = r"D:\test_files\nelly_tests\deskewed-2023-07-13_14-58-28_000_wt_0_acquire.ome.tif"
    im_info = ImInfo(im_path)

    morphology_label_features = MorphologyLabelFeatures(im_info)
    morphology_label_features.run()
