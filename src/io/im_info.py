import os

import tifffile
from src.utils.base_logger import logger


class ImInfo:
    """
    A class that extracts metadata and image size information from a TIFF file.

    Attributes:
        im_path (str): Path to the input TIFF file.
        output_dirpath (str, optional): Path to the output top directory. im_path if none given.
        ch (int, optional): Channel index for multi-channel TIFF files.
        dim_sizes (dict, optional): Dictionary mapping dimension names to physical voxel sizes.

    Examples:
        >>> im_path = "../../data/test_5d.tif"
        >>> im_info = ImInfo(im_path)

    """
    def __init__(self, im_path: str, output_dirpath: str = None, ch: int = None, dim_sizes: dict = None):
        """
        Initialize an ImInfo object for a TIFF file.

        Args:
            im_path (str): Path to the input TIFF file.
            output_dirpath (str, optional): Path to the output top directory. im_path if none given.
            ch (int, optional): Channel index for multi-channel TIFF files.
            dim_sizes (dict, optional): Dictionary mapping dimension names to physical voxel sizes.

        Returns:
            ImInfo object.
        """
        self.im_path = im_path
        self.ch = ch
        self.dim_sizes = dim_sizes
        self.extension = self.im_path.split('.')[-1]
        self.filename = self.im_path.split(os.sep)[-1].split('.'+self.extension)[0]
        try:
            self.dirname = self.im_path.split(os.sep)[-2]
        except IndexError:
            self.dirname = ''
        self.input_dirpath = self.im_path.split(os.sep+self.filename)[0]
        self._get_metadata()
        if self.dim_sizes is not None:
            self.dim_sizes = dim_sizes
        else:
            self._get_dim_sizes()

        self.output_dirpath = None
        self.output_images_dirpath = None
        self.output_pickles_dirpath = None
        self.output_csv_dirpath = None
        self._create_output_dirs(output_dirpath)

        self.path_im_frangi = None
        self.path_im_mask = None
        self.path_im_skeleton = None
        self.path_im_label_obj = None
        self.path_im_label_seg = None
        self.path_im_network = None
        self.path_im_event = None
        self.path_pickle_obj = None
        self.path_pickle_seg = None
        self.path_pickle_track = None
        self._set_output_filepaths()

    def _get_metadata(self):
        """
        Load metadata, axes and shape information from the image file using tifffile.

        Raises:
            Exception: If there was an error loading the image file, an error message is logged and the program exits.
        """
        try:
            with tifffile.TiffFile(self.im_path) as tif:
                self.metadata = tif.imagej_metadata
                self.axes = tif.series[0].axes
                self.shape = tif.series[0].shape
        except Exception as e:
            logger.error(f"Error loading file {self.im_path}: {str(e)}")
            exit(1)

    def _get_dim_sizes(self):
        """Extract physical dimensions of image from its metadata and populate the dim_sizes attribute."""
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

    def _create_output_dirs(self, output_dirpath=None):
        """
        Create output directories for a given file path if they don't exist.
        Specifically, creates output subdirectories for output images, pickle files, and csv files.

        Args:
            output_dirpath (str): The path to the directory where "output_dirpath/output" directory will be added to.
            The "output_dirpath/output" directory will be created if it doesn't exist.

        Returns:
            None
        """
        if output_dirpath is None:
            output_dirpath = self.input_dirpath
        self.output_dirpath = os.path.join(output_dirpath, 'output')
        self.output_images_dirpath = os.path.join(self.output_dirpath, 'images')
        self.output_pickles_dirpath = os.path.join(self.output_dirpath, 'pickles')
        self.output_csv_dirpath = os.path.join(self.output_dirpath, 'csv')
        dirs_to_make = [self.output_images_dirpath, self.output_pickles_dirpath, self.output_csv_dirpath]
        for dir_to_make in dirs_to_make:
            os.makedirs(dir_to_make, exist_ok=True)

    def _set_output_filepaths(self):
        """
        Set the output file paths for various file types. These file paths are based on the input file path and output
        directory.
        """
        self.path_im_frangi = os.path.join(self.output_images_dirpath, f'frangi-{self.filename}.tif')
        self.path_im_mask = os.path.join(self.output_images_dirpath, f'mask-{self.filename}.tif')
        self.path_im_skeleton = os.path.join(self.output_images_dirpath, f'skeleton-{self.filename}.tif')
        self.path_im_label_obj = os.path.join(self.output_images_dirpath, f'label_obj-{self.filename}.tif')
        self.path_im_label_seg = os.path.join(self.output_images_dirpath, f'label_seg-{self.filename}.tif')
        self.path_im_network = os.path.join(self.output_images_dirpath, f'network-{self.filename}.tif')
        self.path_im_event = os.path.join(self.output_images_dirpath, f'event-{self.filename}.tif')
        self.path_pickle_obj = os.path.join(self.output_pickles_dirpath, f'obj-{self.filename}.pkl')
        self.path_pickle_seg = os.path.join(self.output_pickles_dirpath, f'seg-{self.filename}.pkl')
        self.path_pickle_track = os.path.join(self.output_pickles_dirpath, f'track-{self.filename}.pkl')


if __name__ == "__main__":
    filepath = r"D:\test_files\nelly\deskewed-single.ome.tif"
    test = ImInfo(filepath)
