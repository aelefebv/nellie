from src import xp, ndi, logger, is_gpu
from src.io.im_info import ImInfo
import tifffile


class BranchSegments:
    def __init__(self, im_info: ImInfo):
        self.im_info = im_info
        self.segment_memmap = None
        self.shape = ()

    def segment_branches(self, num_t, dtype='uint32'):
        # Load the neighbor image file as memory-mapped files
        neighbor_im = tifffile.memmap(self.im_info.path_im_neighbors, mode='r')

        # Load only a subset of frames if num_t is not None
        if num_t is not None:
            num_t = min(num_t, neighbor_im.shape[0])
            neighbor_im = neighbor_im[:num_t, ...]
        self.shape = neighbor_im.shape

        # Allocate memory for the branch segment volume and load it as a memory-mapped file
        self.im_info.allocate_memory(
            self.im_info.path_im_label_seg, shape=self.shape, dtype=dtype, description='Branch segments image'
        )
        self.segment_memmap = tifffile.memmap(self.im_info.path_im_label_seg, mode='r+')

        # Label any individual branch and save it to the memmap.
        for frame_num, frame in enumerate(neighbor_im):
            logger.info(f'Running neighborhood analysis, volume {frame_num}/{len(neighbor_im)}')
