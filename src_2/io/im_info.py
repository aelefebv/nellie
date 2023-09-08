import os

import tifffile
import ome_types
from src import logger
import numpy as np
from typing import Union, Type

class ImInfo:
    def __init__(self, im_path: str,
                 output_dirpath: str = None,
                 ch: int = None,
                 dim_sizes: dict = None,
                 dimension_order: str = '',
                 ):
        self.im_path = im_path
        self.ch = ch or 0
        self.dim_sizes = dim_sizes
        self.axes = dimension_order

        self.extension = os.path.splitext(self.im_path)[1]
        self.basename = os.path.basename(self.im_path)
        self.dirname = os.path.basename(os.path.dirname(self.im_path))
        self.input_dir = os.path.dirname(self.im_path)
        self.output_dir = output_dirpath or self.input_dir

        self.shape = ()
        self.metadata = None
        self.metadata_type = None
        self._load_metadata()

        if self.dim_sizes is None:
            self._set_dim_sizes()

        self.no_z = True
        self.no_t = True
        self.no_c = True
        self._check_axes()

    def _check_axes(self):
        if 'Z' in self.axes and self.shape[self.axes.index('Z')] > 1:
            self.no_z = False
        if 'T' in self.axes and self.shape[self.axes.index('T')] > 1:
            self.no_t = False
        if 'C' in self.axes and self.shape[self.axes.index('C')] > 1:
            self.no_c = False

    def _set_dim_sizes(self):
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
                    self.dim_sizes['X'] = self.metadata[tag_names['XResolution']].value[1] \
                                          / self.metadata[tag_names['XResolution']].value[0]
                else:
                    logger.warning('No XResolution tag found, assuming X dimension is 1 micron.')
                    self.dim_sizes['X'] = 1
                if 'YResolution' in tag_names:
                    self.dim_sizes['Y'] = self.metadata[tag_names['YResolution']].value[1] \
                                          / self.metadata[tag_names['YResolution']].value[0]
                else:
                    logger.warning('No YResolution tag found, assuming Y dimension is 1 micron.')
                    self.dim_sizes['Y'] = 1
                if 'ResolutionUnit' in tag_names:
                    if self.metadata[tag_names['ResolutionUnit']].value == tifffile.TIFF.RESUNIT.CENTIMETER:
                        self.dim_sizes['X'] *= 1E-2 * 1E6
                        self.dim_sizes['Y'] *= 1E-2 * 1E6
                if 'Z' in self.axes:
                    if 'ZResolution' in tag_names:
                        self.dim_sizes['Z'] = 1 / self.metadata[tag_names['ZResolution']].value[0]
                    else:
                        logger.warning('No ZResolution tag found, assuming Z dimension is 1 micron.')
                        self.dim_sizes['Z'] = 1
                else:
                    logger.warning('No ZResolution tag found, assuming Z dimension is 1 micron.')
                    self.dim_sizes['Z'] = 1
                if 'T' in self.axes:
                    if 'FrameRate' in tag_names:
                        self.dim_sizes['T'] = 1 / self.metadata[tag_names['FrameRate']].value[0]
                    else:
                        logger.warning('No FrameRate tag found, assuming T dimension is 1 second.')
                        self.dim_sizes['T'] = 1
                else:
                    logger.warning('No FrameRate tag found, assuming T dimension is 1 second.')
                    self.dim_sizes['T'] = 1
                logger.warning(f'File is not an ImageJ or OME type, estimated dimension sizes: {self.dim_sizes}')
            self.metadata = None

        except Exception as e:
            logger.error(f"Error loading metadata for image {self.im_path}: {str(e)}")
            self.metadata = {}
            self.dim_sizes = {}

    def _load_metadata(self):
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
        if self.dim_sizes['X'] != self.dim_sizes['Y']:
            logger.warning('X and Y dimensions do not match. Rectangular pixels not supported, '
                           'so unexpected results and wrong measurements will occur.')


if __name__ == "__main__":
    test_folder = r"D:\test_files\nelly_tests"
    all_files = os.listdir(test_folder)
    im_infos = []
    for file in all_files:
        im_path = os.path.join(test_folder, file)
        im_info = ImInfo(im_path)
        im_infos.append(im_info)
