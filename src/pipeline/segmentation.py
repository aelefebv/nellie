import tifffile

from src.io.im_info import ImInfo
from src import xp, morphology, ndi, is_gpu, logger


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
                 threshold: float = 5E-04,
                 min_radius_um: float = 0.25):
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

        # convert min radius um to a min area / volume
        if self.im_info.is_3d:
            # volume of sphere of radius min_width/2 in pixels cubed
            self.min_size_threshold_px = (4 / 3 * xp.pi * (min_radius_um / 2) ** 2) / (
                    self.im_info.dim_sizes['X'] ** 2 * self.im_info.dim_sizes['Z']
            )
        else:
            self.min_size_threshold_px = xp.pi * (min_radius_um / 2) ** 2

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
        if num_t is not None:
            num_t = min(num_t, frangi_memmap.shape[0])
            frangi_memmap = frangi_memmap[:num_t, ...]
        self.shape = frangi_memmap.shape
        self.im_info.allocate_memory(
            self.im_info.path_im_mask, shape=self.shape, dtype='uint8', description='Semantic mask image.',
        )
        self.semantic_mask_memmap = tifffile.memmap(self.im_info.path_im_mask, mode='r+')
        for frame_num, frame in enumerate(frangi_memmap):
            logger.info(f'Running semantic segmentation, volume {frame_num}/{len(frangi_memmap) - 1}')
            frame_in_mem = xp.asarray(frame) > self.threshold
            frame_in_mem = morphology.remove_small_objects(frame_in_mem, self.min_size_threshold_px)
            frame_in_mem = ndi.binary_opening(frame_in_mem)
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
        if num_t is not None:
            num_t = min(num_t, self.semantic_mask_memmap.shape[0])
            self.semantic_mask_memmap = self.semantic_mask_memmap[:num_t, ...]
        self.shape = self.semantic_mask_memmap.shape
        self.im_info.allocate_memory(
            self.im_info.path_im_label_obj, shape=self.shape, dtype=dtype, description='Instance mask image.',
        )
        self.instance_mask_memmap = tifffile.memmap(self.im_info.path_im_label_obj, mode='r+')
        if self.im_info.is_3d:
            structure = xp.ones((3, 3, 3))
        else:
            structure = xp.ones((3, 3))
        for frame_num, frame in enumerate(self.semantic_mask_memmap):
            logger.info(f'Running instance segmentation, volume {frame_num}/{len(self.semantic_mask_memmap) - 1}')
            label_im = xp.asarray(frame).astype(bool)
            label_im, _ = ndi.label(label_im, structure=structure)
            if is_gpu:
                self.instance_mask_memmap[frame_num] = label_im.get()
            else:
                self.instance_mask_memmap[frame_num] = label_im


if __name__ == '__main__':
    import os
    filepath = r"D:\test_files\nelly\deskewed-single.ome.tif"
    if not os.path.isfile(filepath):
        filepath = "/Users/austin/Documents/Transferred/deskewed-single.ome.tif"
    try:
        test = ImInfo(filepath, ch=0)
    except FileNotFoundError:
        logger.error("File not found.")
        exit(1)
    segmentation = Segment(test)
    segmentation.semantic(2)
    segmentation.instance(2)
    print('hi')
