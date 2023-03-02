import tifffile

from src.io.im_info import ImInfo
from src import xp, morphology, ndi, is_gpu, logger


class Segment:
    # takes in a path to a frangi filtered image, and saves a semantic segmentation to a boolean tif file.
    # todo, min_radius should probably default to something based off of a specific organelle. LUT for size?
    # todo docs
    # todo tests
    def __init__(self, im_info: ImInfo,
                 threshold: float = 1E-04,
                 min_radius_um: float = 0.25):
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
        self.semantic_mask_memmap = tifffile.memmap(self.im_info.path_im_mask, mode='r')
        if num_t is not None:
            num_t = min(num_t, self.semantic_mask_memmap.shape[0])
            self.semantic_mask_memmap = self.semantic_mask_memmap[:num_t, ...]
        self.shape = self.semantic_mask_memmap.shape
        self.im_info.allocate_memory(
            self.im_info.path_im_label_seg, shape=self.shape, dtype=dtype, description='Instance mask image.',
        )
        self.instance_mask_memmap = tifffile.memmap(self.im_info.path_im_label_seg, mode='r+')
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
    filepath = r"D:\test_files\nelly\deskewed-single.ome.tif"
    test = ImInfo(filepath, ch=0)
    segmentation = Segment(test)
    segmentation.semantic(2)
    segmentation.instance(2)
    print('hi')
