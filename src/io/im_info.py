import os

import tifffile
import ome_types
from src import logger
import numpy as np
from typing import Union, Type


# todo make this work with no "t" dimension. Just have it segment, no tracking.
# todo also make this work in 2d
class ImInfo:
    """
    A class that extracts metadata and image size information from a TIFF file.
    This will accept a path to an image, store useful info, and produce output directories for downstream functions.

    Attributes:
        im_path (str): Path to the input TIFF file.
        output_dirpath (str, optional): Path to the output top directory. im_path if none given.
        ch (int, optional): Channel index for multichannel TIFF files.
        dim_sizes (dict, optional): Dictionary mapping dimension names to physical voxel sizes.
    """
    def __init__(self, im_path: str,
                 output_dirpath: str = None,
                 ch: int = None,
                 dim_sizes: dict = None,
                 dimension_order: str = '',
                 ):
        """
        Initialize an ImInfo object for a TIFF file.

        Args:
            im_path (str): Path to the input TIFF file.
            output_dirpath (str, optional): Path to the output top directory. im_path if none given.
            ch (int, optional): Channel index for multichannel TIFF files.
            dim_sizes (dict, optional): Dictionary mapping dimension names to physical voxel sizes.

        Returns:
            ImInfo object.
        """
        self.im_path = im_path
        if ch is None:
            logger.warning(f'Multichannel file found, but no channel specified, using channel 0.')
            self.ch = 0
        else:
            self.ch = ch
        self.dim_sizes = dim_sizes
        self.extension = self.im_path.split('.')[-1]
        self.sep = os.sep if os.sep in self.im_path else '/'
        self.filename = self.im_path.split(self.sep)[-1].split('.'+self.extension)[0]
        try:
            self.dirname = self.im_path.split(self.sep)[-2]
        except IndexError:
            self.dirname = ''
        self.input_dirpath = self.im_path.split(self.sep+self.filename)[0]
        self.axes = dimension_order
        self.shape = ()
        self.metadata = None
        self._get_metadata()
        if self.dim_sizes is None:
            self._get_dim_sizes()
        if self.dim_sizes['X'] != self.dim_sizes['Y']:
            logger.warning('X and Y dimensions do not match. Rectangular pixels not yet supported, '
                           'so unexpected results and wrong measurements will occur.')
        if 'Z' not in self.axes:
            self.is_3d = False
        elif self.shape[self.axes.index('Z')] > 1:
            self.is_3d = True
        else:
            self.is_3d = False

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
        self.path_im_label_tips = None
        self.path_im_label_junctions = None
        self.path_im_node_types = None
        self.path_im_network = None
        self.path_im_event = None

        self.path_pickle_obj = None
        self.path_pickle_seg = None
        self.path_pickle_node = None
        self.path_pickle_track = None
        self._set_output_filepaths()

    def _get_metadata(self):
        """
        Load metadata, axes and shape information from the image file using tifffile.

        Raises:
            Exception: If there was an error loading the image file, an error message is logged and the program exits.
        """
        logger.debug('Getting metadata.')
        try:
            with tifffile.TiffFile(self.im_path) as tif:
                if tif.is_imagej:
                    self.metadata = tif.imagej_metadata
                    self.metadata_type = 'imagej'
                elif tif.is_ome:
                    ome_xml = tifffile.tiffcomment(self.im_path)
                    ome = ome_types.from_xml(ome_xml, parser="lxml")
                    self.metadata = ome
                    self.metadata_type = 'ome'
                else:
                    self.metadata = tif.pages[0].tags._dict
                    self.metadata_type = None
                    if 'C' in self.axes:
                        logger.error('Multichannel TIFF files must have ImageJ or OME metadata. Resubmit the'
                                     'file either with correct metadata or the single channel of interest.')
                if self.axes == '':
                    self.axes = tif.series[0].axes
                self.shape = tif.series[0].shape
                if len(self.axes) != len(self.shape):
                    logger.error(f"Dimension order {self.axes} does not match the number of dimensions in the image "
                                   f"({len(self.shape)}).")
                    exit(1)
        except Exception as e:
            logger.error(f"Error loading file {self.im_path}: {str(e)}")
            exit(1)
        accepted_axes = ['TZYX', 'TYX', 'TZCYX', 'TCYX', 'TCZYX', 'ZYX', 'YX', 'CYX', 'CZYX', 'ZCYX']
        if self.axes not in accepted_axes:
            # todo, have user optionally specify axes
            logger.warning(f"File dimension order is in unknown order {self.axes} with {len(self.shape)} dimensions. \n"
                           f"Please specify the order of the dimensions in the run. \n"
                           f"Accepted dimensions are: {accepted_axes}.")
            exit(1)

    def _get_dim_sizes(self):
        """Extract physical dimensions of image from its metadata and populate the dim_sizes attribute."""
        logger.debug('Getting dimension sizes.')
        try:
            self.dim_sizes = {'X': None, 'Y': None, 'Z': None, 'T': None}
            if self.metadata_type == 'imagej':
                if 'physicalsizex' in self.metadata:
                    self.dim_sizes['X'] = self.metadata['physicalsizex']
                if 'physicalsizey' in self.metadata:
                    self.dim_sizes['Y'] = self.metadata['physicalsizey']
                if 'spacing' in self.metadata:
                    self.dim_sizes['Z'] = self.metadata['spacing']
                if 'finterval' in self.metadata:
                    self.dim_sizes['T'] = self.metadata['finterval']
            elif self.metadata_type == 'ome':
                self.dim_sizes['X'] = self.metadata.images[0].pixels.physical_size_x
                self.dim_sizes['Y'] = self.metadata.images[0].pixels.physical_size_y
                self.dim_sizes['Z'] = self.metadata.images[0].pixels.physical_size_z
                self.dim_sizes['T'] = self.metadata.images[0].pixels.time_increment
            elif self.metadata_type is None:
                tag_names = {tag_value.name: tag_code for tag_code, tag_value in self.metadata.items()}
                if 'XResolution' in tag_names:
                    self.dim_sizes['X'] = self.metadata[tag_names['XResolution']].value[1]\
                                          / self.metadata[tag_names['XResolution']].value[0]
                else:
                    self.dim_sizes['X'] = 1
                if 'YResolution' in tag_names:
                    self.dim_sizes['Y'] = self.metadata[tag_names['YResolution']].value[1]\
                                          / self.metadata[tag_names['YResolution']].value[0]
                else:
                    self.dim_sizes['Y'] = 1
                if 'ResolutionUnit' in tag_names:
                    if self.metadata[tag_names['ResolutionUnit']].value == tifffile.TIFF.RESUNIT.CENTIMETER:
                        self.dim_sizes['X'] *= 1E-2 * 1E6
                        self.dim_sizes['Y'] *= 1E-2 * 1E6
                if 'Z' in self.axes:
                    if 'ZResolution' in tag_names:
                        self.dim_sizes['Z'] = 1 / self.metadata[tag_names['ZResolution']].value[0]
                    else:
                        self.dim_sizes['Z'] = 1
                if 'T' in self.axes:
                    if 'FrameRate' in tag_names:
                        self.dim_sizes['T'] = 1 / self.metadata[tag_names['FrameRate']].value[0]
                    else:
                        self.dim_sizes['T'] = 1
                logger.warning(f'File is not an ImageJ or OME type, estimated dimension sizes: {self.dim_sizes}')
        except Exception as e:
            logger.error(f"Error loading metadata for image {self.im_path}: {str(e)}")
            self.metadata = {}
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
        logger.debug('Creating output directories')
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
        logger.debug('Setting output filepaths.')
        if '.ome' not in self.filename:
            self.filename = self.filename + '.ome'
        self.path_im_frangi = os.path.join(self.output_images_dirpath, f'ch{self.ch}-frangi-{self.filename}.tif')
        self.path_im_mask = os.path.join(self.output_images_dirpath, f'ch{self.ch}-mask-{self.filename}.tif')
        self.path_im_skeleton = os.path.join(self.output_images_dirpath, f'ch{self.ch}-skeleton-{self.filename}.tif')
        self.path_im_label_obj = os.path.join(self.output_images_dirpath, f'ch{self.ch}-label_obj-{self.filename}.tif')
        self.path_im_label_seg = os.path.join(self.output_images_dirpath, f'ch{self.ch}-label_seg-{self.filename}.tif')
        self.path_im_label_tips = os.path.join(self.output_images_dirpath, f'ch{self.ch}-label_tips-{self.filename}.tif')
        self.path_im_label_junctions = os.path.join(self.output_images_dirpath, f'ch{self.ch}-label_junctions-{self.filename}.tif')
        self.path_im_node_types = os.path.join(self.output_images_dirpath, f'ch{self.ch}-node_types-{self.filename}.tif')
        self.path_im_network = os.path.join(self.output_images_dirpath, f'ch{self.ch}-network-{self.filename}.tif')
        self.path_im_event = os.path.join(self.output_images_dirpath, f'ch{self.ch}-event-{self.filename}.tif')

        self.path_pickle_obj = os.path.join(self.output_pickles_dirpath, f'ch{self.ch}-obj-{self.filename}.pkl')
        self.path_pickle_node = os.path.join(self.output_pickles_dirpath, f'ch{self.ch}-node-{self.filename}.pkl')
        self.path_pickle_seg = os.path.join(self.output_pickles_dirpath, f'ch{self.ch}-seg-{self.filename}.pkl')
        self.path_pickle_track = os.path.join(self.output_pickles_dirpath, f'ch{self.ch}-track-{self.filename}.pkl')

    def allocate_memory(
            self,
            path_im: str, dtype: Union[Type, str] = 'float', data=None,
            shape: tuple = None,
            description: str = 'No description.'):
        """
        Initializes a numpy array to store image data, allocates the memory for it with corresponding
        ome-types OME-TIFF metadata.

        Args:
            path_im (str): The path to the file where the memory will be allocated.
            dtype (Union[Type, str]): The datatype of the resulting file, default is 'float'
            data (optional): The data to store in the allocated memory. If none, allocated memory will be empty.
            shape (tuple, optional): The shape for which to allocate the memory block.
            description (str, optional): The OME-TIFF description tag's value.

        Returns:
            None
        """
        axes = self.axes
        axes = axes.replace('C', '') if 'C' in axes else axes
        logger.debug(f'Saving axes as {axes}')
        if data is None:
            assert shape is not None
            tifffile.imwrite(
                path_im, shape=shape, dtype=dtype, bigtiff=True, metadata={"axes": axes}
            )
        else:
            tifffile.imwrite(
                path_im, data, bigtiff=True, metadata={"axes": axes}
            )
        print(shape, axes, dtype)
        ome_xml = tifffile.tiffcomment(path_im)
        ome = ome_types.from_xml(ome_xml, parser="lxml")
        ome.images[0].pixels.physical_size_x = self.dim_sizes['X']
        ome.images[0].pixels.physical_size_y = self.dim_sizes['Y']
        ome.images[0].pixels.physical_size_z = self.dim_sizes['Z']
        ome.images[0].pixels.time_increment = self.dim_sizes['T']
        ome.images[0].description = description
        ome.images[0].pixels.type = dtype  # note: numpy uses 8 bits as smallest, so 'bit' type does nothing for bool.
        ome_xml = ome.to_xml()
        tifffile.tiffcomment(path_im, ome_xml)

    def get_im_memmap(self, path_im: str):
        """
        Loads an image from a TIFF file located at `path_im` using the `tifffile.memmap` function,
        and returns a memory-mapped array of the image data.

        If the `C` axis is present in the image and the image shape matches the number of dimensions specified in
        `self.axes`, only the channel specified in `self.ch` will be returned, otherwise the entire image will be
        returned.

        Args:
            path_im (str): The path to the TIFF file containing the image to load.

        Returns:
            np.ndarray: A memory-mapped array of the image data, with shape and data type determined by the file.
        """
        logger.debug('Getting and returning read-only memmap.')
        try:
            im_memmap = tifffile.memmap(path_im, mode='r')
        except ValueError:
            logger.warning('Could not get memmap, loading file into memory instead.')
            im_memmap = tifffile.imread(path_im)

        # Only get wanted channel
        if ('C' in self.axes) and (len(im_memmap.shape) == len(self.axes)):
            im_memmap = np.take(im_memmap, self.ch, axis=self.axes.index('C'))
        return im_memmap


if __name__ == "__main__":
    import os
    windows_filepath = (r"D:\test_files\nelly\deskewed-single.ome.tif", '')
    mac_filepath = ("/Users/austin/Documents/Transferred/deskewed-single.ome.tif", '')

    custom_filepath = (r"/Users/austin/test_files/nelly_Alireza/1.tif", 'ZYX')

    filepath = mac_filepath
    try:
        test = ImInfo(filepath[0], ch=0, dimension_order=filepath[1])
    except FileNotFoundError:
        logger.error("File not found.")
        exit(1)
    loaded_file = test.get_im_memmap(test.im_path)

    visualize = False
    if visualize:
        import napari
        viewer = napari.Viewer()
        viewer.add_image(loaded_file, name='memmap', scale=(test.dim_sizes['Z'], test.dim_sizes['Y'], test.dim_sizes['X']))
