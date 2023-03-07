from src import xp, ndi, logger, is_gpu
from src.io.im_info import ImInfo
import tifffile


class Neighbors:
    """
    A class that computes the neighborhood analysis of a skeleton image volume and saves the results in a memory-mapped
    image.

    Args:
        im_info (ImInfo): An instance of the ImInfo class containing information about the input and output images.

    Attributes:
        im_info (ImInfo): An instance of the ImInfo class containing information about the input and output images.
        neighborhood_memmap (numpy.memmap): A memory-mapped numpy array to store the neighborhood analysis of the
            skeleton image volume.
        shape (tuple): A tuple containing the shape of the skeleton image volume.
    """
    def __init__(self, im_info: ImInfo):
        self.im_info = im_info
        self.neighborhood_memmap = None
        self.shape = ()

    def find_neighbors(self, num_t):  # todo this is faster on cpu?
        """
        Computes the neighborhood analysis of the skeleton image volume and saves the results in a memory-mapped image.

        Args:
            num_t (int or None): The number of frames to be processed in the skeleton image volume. If None, all frames
                will be processed.

        Returns:
            None
        """
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
            logger.info(f'Running neighborhood analysis, volume {frame_num}/{len(skeleton_im)-1}')

            # Create a 3x3x3 neighborhood template
            neighborhood = xp.ones((3, 3, 3), dtype=xp.uint8)
            neighborhood[1, 1, 1] = 0
            frame_mem = xp.asarray(frame)
            frame_mask = (frame_mem > 0).astype('uint8')

            # Convolve the skeleton image with the neighborhood template to count neighboring skeleton pixels
            neighbors = ndi.convolve(frame_mask, neighborhood, mode='constant').astype('uint8')
            neighbors *= frame_mask
            neighbors = xp.max(xp.stack([neighbors, frame_mask], axis=0), axis=0)
            neighbors[neighbors > 3] = 3  # set max neighbors (i.e. connection type) to 3.

            expanded_neighbors = ndi.binary_dilation(neighbors == 3, structure=xp.ones((3, 3, 3))) * 3
            neighbors = xp.max(xp.stack([neighbors, expanded_neighbors], axis=0), axis=0) * frame_mask


        # todo label all bp neighbor pixels as tips. Then relabel tips based on connected pixel
            # # Get a list of pixels that are branch points
            # branch_point_idx = xp.argwhere(neighbors == 3)
            #
            # # Investigate each branch point to determine the connection type
            # for branch_idx, branch_point in enumerate(branch_point_idx):
            #     logger.debug(f'Investigating branch point {branch_idx}/{len(branch_point_idx)}')
            #     z, y, x = branch_point
            #
            #     # Get the coords of the neighboring pixels
            #     coords = [(z+i, y+j, x+k)
            #               for i in [-1, 0, 1] for j in [-1, 0, 1] for k in [-1, 0, 1]
            #               if i != 0 or j != 0 or k != 0]
            #
            #     # Check if any of the neighboring pixels have a value of 3
            #     for coord in coords:
            #         if neighbors[coord] == 3:
            #             neighbors[z, y, x] = 2
            #             break

            # Save the neighbor image to its corresponding memory
            if is_gpu:
                self.neighborhood_memmap[frame_num] = neighbors.get()
            else:
                self.neighborhood_memmap[frame_num] = neighbors


if __name__ == "__main__":
    import os
    filepath = r"D:\test_files\nelly\deskewed-single.ome.tif"
    if not os.path.isfile(filepath):
        filepath = "/Users/austin/Documents/Transferred/deskewed-single.ome.tif"
    try:
        test = ImInfo(filepath, ch=0)
    except:
        logger.error("File not found.")
        exit(1)
    neighbors_test = Neighbors(test)
    neighbors_test.find_neighbors(2)
    print('hi')