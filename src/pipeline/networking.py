from src import xp, ndi, logger, is_gpu
from src.io.im_info import ImInfo
import tifffile


class Neighbors:
    def __init__(self, im_info: ImInfo):
        self.im_info = im_info
        self.neighborhood_memmap = None
        self.shape = ()

    def find_neighbors(self, num_t):
        # Load the skeleton image file as memory-mapped files
        skeleton_im = tifffile.memmap(self.im_info.path_im_skeleton, mode='r')

        # Load only a subset of frames if num_t is not None
        if num_t is not None:
            num_t = min(num_t, skeleton_im.shape[0])
            skeleton_im = skeleton_im[:num_t, ...]
        self.shape = skeleton_im.shape

        # Allocate memory for the neighbor volume and load it as a memory-mapped file
        self.im_info.allocate_memory(
            self.im_info.path_im_neighbors, shape=self.shape, dtype='uint8', description='Neighbor image'
        )
        self.neighborhood_memmap = tifffile.memmap(self.im_info.path_im_neighbors, mode='r+')

        # Get the neighborhood for each frame in the skeleton image and save it to its memory mapped location
        for frame_num, frame in enumerate(skeleton_im):
            logger.info(f'Running neighborhood analysis, volume {frame_num}/{len(skeleton_im)}')

            # Create a 3x3x3 neighborhood template
            neighborhood = xp.ones((3, 3, 3), dtype=xp.uint8)
            neighborhood[1, 1, 1] = 0

            # Convolve the skeleton image with the neighborhood template to count neighboring skeleton pixels
            neighbors = ndi.convolve(xp.asarray(frame) > 0, neighborhood, mode='constant')
            neighbors[neighbors > 3] = 3  # set max neighbors (i.e. connection type) to 3.

            # Get a list of pixels that are branch points
            branch_point_idx = xp.argwhere(neighborhood == 3)
            for branch_point in branch_point_idx:
                z, y, x = branch_point

                # get the coords of the neighboring pixels
                coords = [(z+i, y+j, x+k)
                          for i in [-1, 0, 1] for j in [-1, 0, 1] for k in [-1, 0, 1]
                          if i != 0 or j != 0 or k != 0]

                # Check if any of the neighboring pixels have a value of 3
                for coord in coords:
                    if neighbors[coord] == 3:
                        neighbors[branch_point] = 2
                        break

            # Save the neighbor image to its corresponding memory
            if is_gpu:
                self.neighborhood_memmap[frame_num] = neighbors.get()
            else:
                self.neighborhood_memmap[frame_num] = neighbors
