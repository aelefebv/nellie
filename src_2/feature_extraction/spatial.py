from src import logger
from src_2.im_info.im_info import ImInfo
from src_2.utils.general import get_reshaped_image


class SpatialFeatures:
    def __init__(self, im_info: ImInfo):
        self.im_info = im_info

        self.im_memmap = None
        self.im_frangi = None
        self.label_memmap = None
        self.network_memmap = None
        self.pixel_class_memmap = None
        self.distance_memmap = None

    def _label_morphology(self):
        label_objects =

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