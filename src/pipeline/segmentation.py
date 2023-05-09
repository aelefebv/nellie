import tifffile

from src.io.im_info import ImInfo
from src import xp, morphology, ndi, is_gpu, logger
from src.utils.general import get_reshaped_image


class Segment:
    """
    Performs semantic and instance segmentation on a Frangi filtered image.

    Attributes:
        im_info (ImInfo): An ImInfo object containing information about the image.
        threshold (float): The threshold value for the binary mask.
        min_radius_um (float): The minimum radius (in micrometers) of objects to keep in the binary mask.
        min_size_threshold_px (float): The minimum size (in pixels) of objects to keep in the binary mask.
        semantic_mask_memmap (numpy.memmap or None): A memory-mapped boolean tif file for the semantic segmentation mask, or None if not yet created.
        instance_mask_memmap (numpy.memmap or None): A memory-mapped uint32 tif file for the instance segmentation mask, or None if not yet created.
        shape (tuple): The shape of the image in (t, z, y, x) format.
    """
    # todo, min_radius should probably default to something based off of a specific organelle. LUT for size?
    # todo tests
    def __init__(self, im_info: ImInfo,
                 threshold: float = 0,
                 min_radius_um: float = 0.25,
                 min_mean_intensity = None,
                 max_size_threshold = None,):
        """
        Initializes a Segment object with the given ImInfo object, threshold, and minimum radius.

        Converts the minimum radius to a minimum size threshold in pixels based on the image dimensions.

        Args:
            im_info (ImInfo): An ImInfo object containing information about the image.
            threshold (float, optional): The threshold value for the binary mask. Defaults to 1E-04.
            min_radius_um (float, optional): The minimum radius (in micrometers) of objects to keep in the binary mask. Defaults to 0.25.
        """
        self.im_info = im_info
        self.threshold = threshold
        self.min_radius_um = min_radius_um
        self.im_memmap = None

        self.min_mean_intensity = min_mean_intensity
        if self.min_mean_intensity is not None:
            self.im_memmap = self.im_info.get_im_memmap(self.im_info.im_path)

        self.remove_in_2d = False
        if any(xp.array(
                [self.im_info.dim_sizes['Z'], self.im_info.dim_sizes['Y'], self.im_info.dim_sizes['X']]
        ) > (self.min_radius_um*2)):
            logger.warning(f"One of the dimensions' voxel sizes is greater than the minimum radius of the structure in "
                           f"question so object removal will be conducted based on 2D parameters instead of 3D. "
                           f"This may result in objects being kept that should not be.")
            self.remove_in_2d = True

        # convert min radius um to a min area / volume
        if self.im_info.is_3d and not self.remove_in_2d:
            # volume of sphere of radius min_width/2 in pixels cubed
            self.min_size_threshold_px = (4 / 3 * xp.pi * (min_radius_um / 2) ** 2) / (
                    self.im_info.dim_sizes['X'] ** 2 * self.im_info.dim_sizes['Z']
            )
            # min area of 3*3*3 pixels
            self.min_size_threshold_px = max(self.min_size_threshold_px, 27)
        else:
            self.min_size_threshold_px = (xp.pi * (min_radius_um / 2) ** 2) / (self.im_info.dim_sizes['X'] ** 2)
            # min area of 3*3 pixels
            self.min_size_threshold_px = max(self.min_size_threshold_px, 9)

        self.max_size_threshold_px = max_size_threshold

        self.semantic_mask_memmap = None
        self.instance_mask_memmap = None
        self.shape = ()

    def semantic(self, num_t: int = None):
        """
        Run semantic segmentation on the frangi filtered image.

        Args:
            num_t (int, optional): Number of timepoints to process. Defaults to None, which processes all timepoints.
        """
        frangi_memmap = tifffile.memmap(self.im_info.path_im_frangi, mode='r')
        frangi_memmap = get_reshaped_image(frangi_memmap, num_t, self.im_info)
        shape = frangi_memmap.shape

        self.im_info.allocate_memory(
            self.im_info.path_im_mask, shape=shape, dtype='uint8', description='Semantic mask image.',
        )

        self.semantic_mask_memmap = tifffile.memmap(self.im_info.path_im_mask, mode='r+')
        if len(self.semantic_mask_memmap.shape) == len(shape)-1:
            self.semantic_mask_memmap = self.semantic_mask_memmap[None, ...]

        for frame_num, frame in enumerate(frangi_memmap):
            logger.info(f'Running semantic segmentation, volume {frame_num}/{len(frangi_memmap) - 1}')
            frame_in_mem = xp.asarray(frame)
            frame_in_mem = frame_in_mem > self.threshold
            if self.remove_in_2d:
                struct = ndi.generate_binary_structure(2, 1)
                for z in range(frame_in_mem.shape[0]):
                    frame_in_mem[z] = ndi.binary_opening(frame_in_mem[z], structure=struct)
            else:
                frame_in_mem = ndi.binary_opening(frame_in_mem)
            frame_in_mem = morphology.remove_small_objects(frame_in_mem, self.min_size_threshold_px)
            if is_gpu:
                self.semantic_mask_memmap[frame_num] = frame_in_mem.get()
            else:
                self.semantic_mask_memmap[frame_num] = frame_in_mem

    def instance(self, num_t: int = None, dtype: str = 'uint32'):
        """
        Run instance segmentation on the semantic segmentation.

        Args:
            num_t (int, optional): Number of timepoints to process. Defaults to None, which processes all timepoints.
            dtype (str, optional): Data type of the output instance mask. Defaults to 'uint32'.
        """
        self.semantic_mask_memmap = tifffile.memmap(self.im_info.path_im_mask, mode='r')
        self.semantic_mask_memmap = get_reshaped_image(self.semantic_mask_memmap, num_t, self.im_info)
        self.shape = self.semantic_mask_memmap.shape

        self.im_info.allocate_memory(
            self.im_info.path_im_label_obj, shape=self.shape, dtype=dtype, description='Instance mask image.',
        )
        self.instance_mask_memmap = tifffile.memmap(self.im_info.path_im_label_obj, mode='r+')

        if len(self.instance_mask_memmap.shape) == len(self.shape)-1:
            self.instance_mask_memmap = self.instance_mask_memmap[None, ...]

        if self.im_info.is_3d:
            structure = xp.ones((3, 3, 3))
        else:
            structure = xp.ones((3, 3))
        for frame_num, frame in enumerate(self.semantic_mask_memmap):
            logger.info(f'Running instance segmentation, volume {frame_num}/{len(self.semantic_mask_memmap) - 1}')
            label_im = xp.asarray(frame).astype(bool)
            label_im, _ = ndi.label(label_im, structure=structure)
            # remove objects that have a mean intensity less that a threshold
            if self.min_mean_intensity is not None:
                unique_labels = xp.unique(label_im)
                # print('unique_labels', unique_labels)
                # print('memmap_shape', self.im_memmap[frame_num, ...].shape)

                mean_intensities = ndi.labeled_comprehension(self.im_memmap[frame_num, ...], label_im, unique_labels,
                                                             xp.mean, float, xp.nan)
                # print(mean_intensities)
                label_mean_intensity = dict(zip(unique_labels.tolist(), mean_intensities.tolist()))
                filtered_label_mean_intensity = {label: mean_intensity
                                                 for label, mean_intensity in label_mean_intensity.items()
                                                 if mean_intensity >= self.min_mean_intensity}
                label_voxel_count = {label: xp.sum(label_im == label) for label in filtered_label_mean_intensity.keys()}
                if self.max_size_threshold_px is not None:
                    filtered_label_voxel_count = {label: voxel_count for label, voxel_count in label_voxel_count.items() if
                                                  voxel_count < self.max_size_threshold_px}
                    # Create a new label image with only the remaining labels
                    filtered_label_img = xp.zeros_like(label_im)
                    for label in filtered_label_voxel_count.keys():
                        filtered_label_img[label_im == label] = label
                else:
                    # Create a new label image with only the remaining labels
                    filtered_label_img = xp.zeros_like(label_im)
                    for label in filtered_label_mean_intensity.keys():
                        filtered_label_img[label_im == label] = label
                label_im = filtered_label_img

            if is_gpu:
                self.instance_mask_memmap[frame_num] = label_im.get()
            else:
                self.instance_mask_memmap[frame_num] = label_im


if __name__ == '__main__':
    windows_filepath = (r"D:\test_files\nelly\deskewed-single.ome.tif", '')
    mac_filepath = ("/Users/austin/Documents/Transferred/deskewed-single.ome.tif", '')

    custom_filepath = (r"/Users/austin/test_files/nelly_Alireza/1.tif", 'ZYX')

    filepath = custom_filepath
    try:
        test = ImInfo(filepath[0], ch=0, dimension_order=filepath[1])
    except FileNotFoundError:
        logger.error("File not found.")
        exit(1)
    segmentation = Segment(test)
    segmentation.semantic()
    segmentation.instance()
    print('hi')
