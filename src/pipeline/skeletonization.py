import tifffile

from src import logger
from src.io.im_info import ImInfo
from skimage import morphology  # currently no skeletonization cupy implementation
from src.utils.general import get_reshaped_image



class Skeleton:
    def __init__(self, im_info: ImInfo):
        """
        Constructor of the Skeleton class.

        Args:
            im_info: an instance of ImInfo that holds the information of the input and output images.

        Returns:
            An instance of the Skeleton class.
        """
        self.im_info = im_info
        self.skel_memmap = None
        self.shape = ()

    def skeletonize(self, num_t: int = None, dtype: str = 'uint32'):
        """
        Method that skeletonizes a 3D binary image volume and assigns instance labels to the skeleton.

        Args:
            num_t: the number of frames to process. If None, all the frames are processed.
            dtype: the data type of the saved skeletonized label image.

        Returns:
            None.
        """
        # Load the binary image file and instance label file as memory-mapped files
        semantic_mask = tifffile.memmap(self.im_info.path_im_mask, mode='r')
        label_im = tifffile.memmap(self.im_info.path_im_label_obj, mode='r')

        semantic_mask = get_reshaped_image(semantic_mask, num_t, self.im_info)
        label_im = get_reshaped_image(label_im, num_t, self.im_info)
        self.shape = semantic_mask.shape

        # Allocate memory for the skeleton volume and load it as a memory-mapped file
        self.im_info.allocate_memory(
            self.im_info.path_im_skeleton, shape=self.shape, dtype=dtype, description='Skeleton image'
        )

        self.skel_memmap = tifffile.memmap(self.im_info.path_im_skeleton, mode='r+')
        if len(self.skel_memmap.shape) == len(self.shape)-1:
            self.skel_memmap = self.skel_memmap[None, ...]

        # Skeletonize each frame in the binary image volume and multiply it by its instance label
        for frame_num, frame in enumerate(semantic_mask):
            logger.info(f'Running skeletonization, volume {frame_num}/{len(semantic_mask)}')
            self.skel_memmap[frame_num] = morphology.skeletonize(frame) * label_im[frame_num]


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
    skel_im_out = Skeleton(test)
    skel_im_out.skeletonize()
    print('hi')
