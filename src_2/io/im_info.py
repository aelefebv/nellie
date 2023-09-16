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
                 output_suffix: str = '',
                 ):
        self.im_path = im_path
        self.ch = ch or 0
        self.dim_sizes = dim_sizes
        self.axes = dimension_order
        self.output_suffix = output_suffix

        self.extension = os.path.splitext(self.im_path)[1]
        self.basename = os.path.basename(self.im_path)
        self.basename_no_ext = os.path.splitext(self.basename)[0]
        self.dirname = os.path.basename(os.path.dirname(self.im_path))
        self.input_dir = os.path.dirname(self.im_path)
        self.output_dir = output_dirpath or os.path.join(self.input_dir, 'output')
        self.output_dir = self.output_dir + self.output_suffix

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

        self._create_output_dir()

        self.pipeline_paths = {}

    def _create_output_dir(self):
        logger.debug('Creating output directory.')
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

    def _check_axes(self):
        if 'Z' in self.axes and self.shape[self.axes.index('Z')] > 1:
            self.no_z = False
        if 'T' in self.axes and self.shape[self.axes.index('T')] > 1:
            self.no_t = False
        if 'C' in self.axes and self.shape[self.axes.index('C')] > 1:
            self.no_c = False

    def _set_dim_sizes(self):
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
            if self.dim_sizes['X'] != self.dim_sizes['Y']:
                logger.warning('X and Y dimensions do not match. Rectangular pixels not supported, '
                               'so unexpected results and wrong measurements will occur.')


        except Exception as e:
            logger.error(f"Error loading metadata for image {self.im_path}: {str(e)}")
            self.metadata = {}
            self.dim_sizes = {}

    def _load_metadata(self):
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

    def allocate_memory(
            self,
            path_im: str, dtype: Union[Type, str] = 'float', data=None,
            shape: tuple = None,
            description: str = 'No description.',
            return_memmap: bool = False, read_mode = 'r+'):
        axes = self.axes
        axes = axes.replace('C', '') if 'C' in axes else axes
        logger.debug(f'Saving axes as {axes}')
        if 'T' not in axes:
            axes = 'T' + axes
        if data is None:
            assert shape is not None
            tifffile.imwrite(
                path_im, shape=shape, dtype=dtype, bigtiff=True, metadata={"axes": axes}
            )
        else:
            tifffile.imwrite(
                path_im, data, bigtiff=True, metadata={"axes": axes}
            )
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
        if return_memmap:
            return tifffile.memmap(path_im, mode=read_mode)

    def get_im_memmap(self, path_im: str):
        logger.debug('Getting and returning read-only memmap.')
        try:
            im_memmap = tifffile.memmap(path_im, mode='r')
        except ValueError:
            logger.warning('Could not get memmap, loading file into memory instead.')
            im_memmap = tifffile.imread(path_im)

        if ('C' in self.axes) and (len(im_memmap.shape) == len(self.axes)):
            im_memmap = np.take(im_memmap, self.ch, axis=self.axes.index('C'))
        return im_memmap

    def create_output_path(self, pipeline_path: str, ext: str = '.ome.tif'):
        logger.debug('Creating output path.')
        if pipeline_path not in self.pipeline_paths:
            self.pipeline_paths[pipeline_path] = os.path.join(self.output_dir,
                                                              f'{self.basename_no_ext}-'
                                                              f'ch{self.ch}-'
                                                              f'{pipeline_path}{ext}')
        return self.pipeline_paths[pipeline_path]


if __name__ == "__main__":
    test_folder = r"D:\test_files\nelly_tests"
    all_files = os.listdir(test_folder)
    all_files = [file for file in all_files if not os.path.isdir(os.path.join(test_folder, file))]
    im_infos = []
    for file in all_files:
        im_path = os.path.join(test_folder, file)
        im_info = ImInfo(im_path)
        im_infos.append(im_info)