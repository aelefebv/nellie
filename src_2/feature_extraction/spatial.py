from src import logger
from src_2.im_info.im_info import ImInfo
from src_2.utils.general import get_reshaped_image
import skimage.measure
import numpy as np


class SpatialFeatures:
    def __init__(self, im_info: ImInfo):
        self.im_info = im_info
        if self.im_info.no_z:
            self.spacing = (self.im_info.dim_sizes['Y'], self.im_info.dim_sizes['X'])
        else:
            self.spacing = (self.im_info.dim_sizes['Z'], self.im_info.dim_sizes['Y'], self.im_info.dim_sizes['X'])

        self.im_memmap = None
        self.im_frangi_memmap = None
        self.label_memmap = None
        self.network_memmap = None
        self.pixel_class_memmap = None
        self.distance_memmap = None

        self.label_objects_intensity = None

        self.features = {}

    def _label_morphology(self):
        self.label_objects_intensity = skimage.measure.regionprops(self.label_memmap[0], self.im_memmap[0], spacing=self.spacing)
        log10_frangi = np.log10(self.im_frangi_memmap[0])
        log10_frangi[np.isinf(log10_frangi)] = 0
        self.label_objects_frangi = skimage.measure.regionprops(self.label_memmap[0], log10_frangi, spacing=self.spacing)

        self.features['area'] = [label_object.area for label_object in self.label_objects_intensity]

        self.features['extent'] = [label_object.extent for label_object in self.label_objects_intensity]
        self.features['solidity'] = [label_object.solidity for label_object in self.label_objects_intensity]

        # intensity image
        sorted_inertia_tensor_eigvals_intensity = [sorted(label_object.inertia_tensor_eigvals) for label_object in self.label_objects_intensity]
        self.features['inertia_tensor_eig_sorted_min'] = [sorted_inertia_tensor_eigvals[0] for sorted_inertia_tensor_eigvals in sorted_inertia_tensor_eigvals_intensity]
        self.features['inertia_tensor_eig_sorted_max'] = [sorted_inertia_tensor_eigvals[-1] for sorted_inertia_tensor_eigvals in sorted_inertia_tensor_eigvals_intensity]
        if not self.im_info.no_z:
            self.features['inertia_tensor_eig_sorted_mid'] = [sorted_inertia_tensor_eigvals[1] for sorted_inertia_tensor_eigvals in sorted_inertia_tensor_eigvals_intensity]

        self.features['intensity_mean'] = [label_object.mean_intensity for label_object in self.label_objects_intensity]
        self.features['intensity_range'] = [label_object.max_intensity - label_object.min_intensity for label_object in self.label_objects_intensity]

        # frangi image
        self.features['frangi_mean'] = [label_object.mean_intensity for label_object in self.label_objects_frangi]
        self.features['frangi_range'] = [label_object.max_intensity - label_object.min_intensity for label_object in self.label_objects_frangi]



    def _get_memmaps(self):
        logger.debug('Allocating memory for spatial feature extraction.')
        im_memmap = self.im_info.get_im_memmap(self.im_info.im_path)
        self.im_memmap = get_reshaped_image(im_memmap, 1, self.im_info)

        im_frangi_memmap = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_frangi'])
        self.im_frangi_memmap = get_reshaped_image(im_frangi_memmap, 1, self.im_info)

        label_memmap = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_instance_label'])
        self.label_memmap = get_reshaped_image(label_memmap, 1, self.im_info)

        network_memmap = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_skel'])
        self.network_memmap = get_reshaped_image(network_memmap, 1, self.im_info)

        pixel_class_memmap = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_pixel_class'])
        self.pixel_class_memmap = get_reshaped_image(pixel_class_memmap, 1, self.im_info)

        distance_memmap = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_distance'])
        self.distance_memmap = get_reshaped_image(distance_memmap, 1, self.im_info)

        self.shape = self.label_memmap.shape

    def run(self):
        self._get_memmaps()
        self._label_morphology()



if __name__ == "__main__":
    im_path = r"D:\test_files\nelly_tests\deskewed-2023-07-13_14-58-28_000_wt_0_acquire.ome.tif"
    im_info = ImInfo(im_path)
    im_info.create_output_path('im_instance_label')
    im_info.create_output_path('im_frangi')
    im_info.create_output_path('im_skel')
    im_info.create_output_path('im_pixel_class')
    im_info.create_output_path('im_distance')

    spatial_features = SpatialFeatures(im_info)
    spatial_features.run()