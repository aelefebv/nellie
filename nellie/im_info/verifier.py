import os

import nd2
import numpy as np
import ome_types
from tifffile import tifffile

from nellie import logger


class FileInfo:
    def __init__(self, filepath, output_dir=None):
        self.filepath = filepath
        self.metadata = None
        self.metadata_type = None
        self.axes = None
        self.shape = None
        self.dim_res = None

        self.input_dir = os.path.dirname(filepath)
        self.basename = os.path.basename(filepath)
        self.filename_no_ext = os.path.splitext(self.basename)[0]
        self.extension = os.path.splitext(filepath)[1]
        self.output_dir = output_dir or os.path.join(self.input_dir, 'nellie_output')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.output_path = None
        self.good_dims = False
        self.good_axes = False

        self.ch = 0
        self.t_start = 0
        self.t_end = None
        self.dtype = None

    def _find_tif_metadata(self):
        with tifffile.TiffFile(self.filepath) as tif:
            if tif.is_ome or tif.ome_metadata is not None:
                ome_xml = tifffile.tiffcomment(self.filepath)
                metadata = ome_types.from_xml(ome_xml)
                metadata_type = 'ome'
            elif tif.is_imagej:
                metadata = tif.imagej_metadata
                metadata_type = 'imagej'
                if 'physicalsizex' not in metadata:
                    metadata_type = 'imagej_tif_tags'
                    metadata = [metadata, tif.pages[0].tags._dict]
            else:
                metadata = tif.pages[0].tags._dict
                metadata_type = None

            self.metadata = metadata
            self.metadata_type = metadata_type
            self.axes = tif.series[0].axes
            self.shape = tif.series[0].shape

        return metadata, metadata_type

    def _find_nd2_metadata(self):
        with nd2.ND2File(self.filepath) as nd2_file:
            metadata = nd2_file.metadata.channels[0]
            metadata.recorded_data = nd2_file.events(orient='list')
            self.metadata = metadata
            self.metadata_type = 'nd2'
            self.axes = ''.join(nd2_file.sizes.keys())
            self.shape = tuple(nd2_file.sizes.values())

    def find_metadata(self):
        if self.filepath.endswith('.tiff') or self.filepath.endswith('.tif'):
            self._find_tif_metadata()
        elif self.filepath.endswith('.nd2'):
            self._find_nd2_metadata()
        else:
            raise ValueError('File type not supported')

    def _get_imagej_metadata(self, metadata):
        self.dim_res['X'] = metadata['physicalsizex'] if 'physicalsizex' in metadata else None
        self.dim_res['Y'] = metadata['physicalsizey'] if 'physicalsizey' in metadata else None
        self.dim_res['Z'] = metadata['spacing'] if 'spacing' in metadata else None
        self.dim_res['T'] = metadata['finterval'] if 'finterval' in metadata else None

    def _get_ome_metadata(self, metadata):
        self.dim_res['X'] = metadata.images[0].pixels.physical_size_x
        self.dim_res['Y'] = metadata.images[0].pixels.physical_size_y
        self.dim_res['Z'] = metadata.images[0].pixels.physical_size_z
        self.dim_res['T'] = metadata.images[0].pixels.time_increment

    def _get_tif_tags_metadata(self, metadata):
        tag_names = {tag_value.name: tag_code for tag_code, tag_value in metadata.items()}

        if 'XResolution' in tag_names:
            self.dim_res['X'] = metadata[tag_names['XResolution']].value[1] \
                                  / metadata[tag_names['XResolution']].value[0]
        if 'YResolution' in tag_names:
            self.dim_res['Y'] = metadata[tag_names['YResolution']].value[1] \
                                  / metadata[tag_names['YResolution']].value[0]
        if 'ResolutionUnit' in tag_names:
            if metadata[tag_names['ResolutionUnit']].value == tifffile.RESUNIT.CENTIMETER:
                self.dim_res['X'] *= 1E-4 * 1E6
                self.dim_res['Y'] *= 1E-4 * 1E6
        if 'Z' in self.axes:
            if 'ZResolution' in tag_names:
                self.dim_res['Z'] = 1 / metadata[tag_names['ZResolution']].value[0]
        if 'T' in self.axes:
            if 'FrameRate' in tag_names:
                self.dim_res['T'] = 1 / metadata[tag_names['FrameRate']].value[0]

    def _get_nd2_metadata(self, metadata):
        if 'Time [s]' in metadata.recorded_data:
            timestamps = metadata.recorded_data['Time [s]']
            self.dim_res['T'] = timestamps[-1] / len(timestamps)
        self.dim_res['X'] = metadata.volume.axesCalibration[0]
        self.dim_res['Y'] = metadata.volume.axesCalibration[1]
        self.dim_res['Z'] = metadata.volume.axesCalibration[2]

    def load_metadata(self):
        self.dim_res = {'X': None, 'Y': None, 'Z': None, 'T': None}
        if self.metadata_type == 'ome':
            self._get_ome_metadata(self.metadata)
        elif self.metadata_type == 'imagej':
            self._get_imagej_metadata(self.metadata)
        elif self.metadata_type == 'imagej_tif_tags':
            self._get_imagej_metadata(self.metadata[0])
            self._get_tif_tags_metadata(self.metadata[1])
        elif self.metadata_type == 'nd2':
            self._get_nd2_metadata(self.metadata)
        elif self.metadata_type is None:
            self._get_tif_tags_metadata(self.metadata)
        self._validate()

    def _check_axes(self):
        if len(self.shape) != len(self.axes):
            self.change_axes('Q' * len(self.shape))
        if 'Q' in self.axes:
            self.good_axes = False
            return
        # if any duplicates, not good
        if len(set(self.axes)) != len(self.axes):
            self.good_axes = False
            return
        # if X or Y are not there, not good
        if 'X' not in self.axes or 'Y' not in self.axes:
            self.good_axes = False
            return
        self.good_axes = True

    def _check_dim_res(self):
        check_dims = ['X', 'Y', 'Z', 'T']
        for dim in check_dims:
            if dim in self.axes and self.dim_res[dim] is None:
                self.good_dims = False
                return
        self.good_dims = True

    def change_axes(self, new_axes):
        # if len(new_axes) != len(self.shape):
        self.good_axes = False
            # return
            # raise ValueError('New axes must have the same length as the shape of the data')
        self.axes = new_axes
        self._validate()

    def change_dim_res(self, dim, new_size):
        if dim not in self.dim_res:
            return
            # raise ValueError('Invalid dimension')
        self.dim_res[dim] = new_size
        self._validate()

    def change_selected_channel(self, ch):
        if not self.good_dims or not self.good_axes:
            raise ValueError('Must have both valid axes and dimensions to change channel')
        if 'C' not in self.axes:
            raise KeyError('No channel dimension to change')
        if ch < 0 or ch >= self.shape[self.axes.index('C')]:
            raise IndexError('Invalid channel index')
        self.ch = ch
        self._get_output_path()

    def select_temporal_range(self, start=0, end=None):
        if not self.good_dims or not self.good_axes:
            return
            # raise ValueError('Must have both valid axes and dimensions to select temporal range')
        if 'T' not in self.axes:
            return
            # raise KeyError('No time dimension to select')
        self.t_start = start
        self.t_end = end
        if self.t_end is None:
            self.t_end = self.shape[self.axes.index('T')] - 1
        self._get_output_path()

    def _validate(self):
        self._check_axes()
        self._check_dim_res()
        self.select_temporal_range()
        self._get_output_path()

    def read_file(self):
        if self.extension == '.nd2':
            data = nd2.imread(self.filepath)
        elif self.extension == '.tif' or self.extension == '.tiff':
            try:
                data = tifffile.memmap(self.filepath)
            except:
                data = tifffile.imread(self.filepath)
        else:
            logger.error(f'Filetype {self.extension} not supported. Please convert to .nd2 or .tif.')
            raise ValueError
        self.dtype = data.dtype
        return data

    def _get_output_path(self):
        t_text = f'-t{self.t_start}_to_{self.t_end}' if 'T' in self.axes else ''
        self.output_path = os.path.join(
            self.output_dir,
            f'{self.filename_no_ext}'
            f'-{self.axes}'
            f'-ch{self.ch}'
            f'{t_text}'
            f'.ome.tif'
        )

    def save_ome_tiff(self):
        if not self.good_axes or not self.good_dims:
            raise ValueError('Cannot save file with invalid axes or dimensions')

        axes = self.axes
        data = self.read_file()
        if 'T' not in self.axes:
            data = data[np.newaxis, ...]
            axes = 'T' + self.axes
        else:
            data = np.take(data, range(self.t_start, self.t_end + 1), axis=self.axes.index('T'))
        if 'C' in axes:
            data = np.take(data, self.ch, axis=axes.index('C'))
            axes = axes.replace('C', '')

        tifffile.imwrite(
            self.output_path, data, bigtiff=True, metadata={"axes": axes}
        )

        ome_xml = tifffile.tiffcomment(self.output_path)
        ome = ome_types.from_xml(ome_xml)
        ome.images[0].pixels.physical_size_x = self.dim_res['X']
        ome.images[0].pixels.physical_size_y = self.dim_res['Y']
        ome.images[0].pixels.physical_size_z = self.dim_res['Z']
        ome.images[0].pixels.time_increment = self.dim_res['T']
        ome.images[0].pixels.type = data.dtype.name
        ome_xml = ome.to_xml()
        tifffile.tiffcomment(self.output_path, ome_xml)


class ImInfo:
    def __init__(self, file_info: FileInfo):
        self.file_info = file_info
        self.im_path = file_info.output_path
        if not os.path.exists(self.im_path):
            file_info.save_ome_tiff()
        self.im = tifffile.memmap(self.im_path)
        self.output_dir = file_info.output_dir

        self.dim_res = {'X': None, 'Y': None, 'Z': None, 'T': None}
        self.axes = None
        self.shape = None
        self.ome_metadata = None
        self._get_ome_metadata()

        self.no_z = True
        self.no_t = True
        self._check_axes_exist()

        self.pipeline_paths = {}
        self._create_output_paths()

    def _check_axes_exist(self):
        if 'Z' in self.axes and self.shape[self.axes.index('Z')] > 1:
            self.no_z = False
        if 'T' in self.axes and self.shape[self.axes.index('T')] > 1:
            self.no_t = False

    def create_output_path(self, pipeline_path: str, ext: str = '.ome.tif'):
        t_text = f'-t{self.file_info.t_start}_to_{self.file_info.t_end}' if 'T' in self.file_info.axes else ''
        output_path = os.path.join(
            self.output_dir,
            f'{self.file_info.filename_no_ext}'
            f'-{self.axes}'
            f'-ch{self.file_info.ch}'
            f'{t_text}'
            f'-{pipeline_path}'
            f'{ext}'
        )
        self.pipeline_paths[pipeline_path] = output_path
        return self.pipeline_paths[pipeline_path]

    def _create_output_paths(self):
        self.create_output_path('im_frangi')
        self.create_output_path('im_instance_label')
        self.create_output_path('im_skel')
        self.create_output_path('im_skel_relabelled')
        self.create_output_path('im_pixel_class')
        self.create_output_path('im_marker')
        self.create_output_path('im_distance')
        self.create_output_path('im_border')
        self.create_output_path('flow_vector_array', ext='.npy')
        self.create_output_path('voxel_matches', ext='.npy')
        self.create_output_path('im_branch_label_reassigned')
        self.create_output_path('im_obj_label_reassigned')
        self.create_output_path('features_voxels', ext='.csv')
        self.create_output_path('features_nodes', ext='.csv')
        self.create_output_path('features_branches', ext='.csv')
        self.create_output_path('features_components', ext='.csv')
        self.create_output_path('features_image', ext='.csv')
        self.create_output_path('adjacency_maps', ext='.pkl')

    def _get_ome_metadata(self, ):
        with tifffile.TiffFile(self.im_path) as tif:
            self.axes = tif.series[0].axes
            self.shape = tif.series[0].shape
        self.ome_metadata = ome_types.from_xml(tifffile.tiffcomment(self.im_path))
        self.dim_res['X'] = self.ome_metadata.images[0].pixels.physical_size_x
        self.dim_res['Y'] = self.ome_metadata.images[0].pixels.physical_size_y
        self.dim_res['Z'] = self.ome_metadata.images[0].pixels.physical_size_z
        self.dim_res['T'] = self.ome_metadata.images[0].pixels.time_increment

    def allocate_memory(self, output_path, dtype='float', data=None, description='No description.',
                        return_memmap=False, read_mode='r+'):
        if data is None:
            tifffile.imwrite(
                output_path, shape=self.shape, dtype=dtype, bigtiff=True, metadata={"axes": self.axes}
            )
        else:
            dtype = dtype or data.dtype.name
            tifffile.imwrite(
                output_path, data, bigtiff=True, metadata={"axes": self.axes}
            )
        ome = self.ome_metadata
        ome.images[0].description = description
        ome.images[0].pixels.type = dtype
        ome_xml = ome.to_xml()
        tifffile.tiffcomment(output_path, ome_xml)
        if return_memmap:
            return tifffile.memmap(output_path, mode=read_mode)


if __name__ == "__main__":
    test_dir = '/Users/austin/test_files/nellie_all_tests'
    all_paths = os.listdir(test_dir)
    all_paths = [os.path.join(test_dir, path) for path in all_paths if path.endswith('.tiff') or path.endswith('.tif') or path.endswith('.nd2')]
    # for filepath in all_paths:
    #     file_info = FileInfo(filepath)
    #     file_info.find_metadata()
    #     file_info.load_metadata()
    #     print(file_info.metadata_type)
    #     print(file_info.axes)
    #     print(file_info.shape)
    #     print(file_info.dim_res)
    #     print('\n\n')

    test_file = all_paths[1]
    file_info = FileInfo(test_file)
    file_info.find_metadata()
    file_info.load_metadata()
    print(f'{file_info.metadata_type=}')
    print(f'{file_info.axes=}')
    print(f'{file_info.shape=}')
    print(f'{file_info.dim_res=}')
    print(f'{file_info.good_axes=}')
    print(f'{file_info.good_dims=}')
    print('\n')

    file_info.change_axes('TZYX')
    print('Axes changed')
    print(f'{file_info.axes=}')
    print(f'{file_info.dim_res=}')
    print(f'{file_info.good_axes=}')
    print(f'{file_info.good_dims=}')
    print('\n')

    file_info.change_dim_res('T', 0.5)
    file_info.change_dim_res('Z', 0.2)

    print('Dimension resolutions changed')
    print(f'{file_info.axes=}')
    print(f'{file_info.dim_res=}')
    print(f'{file_info.good_axes=}')
    print(f'{file_info.good_dims=}')
    print('\n')

    # print(f'{file_info.ch=}')
    # file_info.change_selected_channel(3)
    # print('Channel changed')
    # print(f'{file_info.ch=}')

    print(f'{file_info.t_start=}')
    print(f'{file_info.t_end=}')
    file_info.select_temporal_range(1, 3)
    print('Temporal range selected')
    print(f'{file_info.t_start=}')
    print(f'{file_info.t_end=}')

    # file_info.save_ome_tiff()
    im_info = ImInfo(file_info)
