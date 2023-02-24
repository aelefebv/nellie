import os

import tifffile
from src.utils.base_logger import logger


class ImInfo:
    """
    A class that extracts metadata and image size information from a TIFF file.

    Attributes:
        im_path (str): Path to the input TIFF file.
        ch (int, optional): Channel index for multi-channel TIFF files.
        dim_sizes (dict, optional): Dictionary mapping dimension names to physical voxel sizes.

    Examples:
        >>> im_path = "../../data/test_5d.tif"
        >>> im_info = ImInfo(im_path)

    """
    def __init__(self, im_path: str, ch: int = None, dim_sizes: dict = None):
        """
        Initialize an ImInfo object for a TIFF file.

        Args:
            im_path (str): Path to the input TIFF file.
            ch (int, optional): Channel index for multi-channel TIFF files.
            dim_sizes (dict, optional): Dictionary mapping dimension names to physical voxel sizes.

        Returns:
            None.
        """
        self.im_path = im_path
        self.ch = ch
        self.dim_sizes = dim_sizes
        self.extension = self.im_path.split('.')[-1]
        self.filename = self.im_path.split(os.sep)[-1].split('.'+self.extension)[0]
        self.dirname = self.im_path.split(os.sep)[-2]
        try:
            with tifffile.TiffFile(self.im_path) as tif:
                self.metadata = tif.imagej_metadata
                self.axes = tif.series[0].axes
                self.shape = tif.series[0].shape
        except Exception as e:
            logger.error(f"Error loading file {self.im_path}: {str(e)}")
            exit(1)

        if self.dim_sizes is not None:
            self.dim_sizes = dim_sizes
        else:
            try:
                self.dim_sizes = {}
                if 'physicalsizex' in self.metadata:
                    self.dim_sizes['x'] = self.metadata['physicalsizex']
                if 'physicalsizey' in self.metadata:
                    self.dim_sizes['y'] = self.metadata['physicalsizey']
                if 'spacing' in self.metadata:
                    self.dim_sizes['z'] = self.metadata['spacing']
                if 'finterval' in self.metadata:
                    self.dim_sizes['t'] = self.metadata['finterval']
            except Exception as e:
                logger.error(f"Error loading metadata for image {self.im_path}: {str(e)}")
                self.metadata = {}
                self.axes = None
                self.shape = None
                self.dim_sizes = {}


if __name__ == "__main__":
    filepath = r"D:\test_files\nelly\deskewed-single.ome.tif"
    test = ImInfo(filepath)
