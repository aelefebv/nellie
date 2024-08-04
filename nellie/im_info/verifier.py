import os

import nd2
import ome_types
from tifffile import tifffile

from nellie import logger


# takes in a file path
# if not already an ome-tiff with correct metadata
# returns a message saying so

# user then has to correct metadata (if not correct).
# Once correct, they can tell it to save the file as an ome-tiff

class FileInfo:
    def __init__(self, filepath):
        self.filepath = filepath
        self.metadata = None
        self.metadata_type = None
        self.axes = None
        self.shape = None
        self.dim_sizes = None
        self.ch = 0
        self.good_dims = False

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
        self.dim_sizes['X'] = metadata['physicalsizex'] if 'physicalsizex' in metadata else None
        self.dim_sizes['Y'] = metadata['physicalsizey'] if 'physicalsizey' in metadata else None
        self.dim_sizes['Z'] = metadata['spacing'] if 'spacing' in metadata else None
        self.dim_sizes['T'] = metadata['finterval'] if 'finterval' in metadata else None

    def _get_ome_metadata(self, metadata):
        self.dim_sizes['X'] = metadata.images[0].pixels.physical_size_x
        self.dim_sizes['Y'] = metadata.images[0].pixels.physical_size_y
        self.dim_sizes['Z'] = metadata.images[0].pixels.physical_size_z
        self.dim_sizes['T'] = metadata.images[0].pixels.time_increment

    def _get_tif_tags_metadata(self, metadata):
        tag_names = {tag_value.name: tag_code for tag_code, tag_value in metadata.items()}

        if 'XResolution' in tag_names:
            self.dim_sizes['X'] = metadata[tag_names['XResolution']].value[1] \
                                  / metadata[tag_names['XResolution']].value[0]
        if 'YResolution' in tag_names:
            self.dim_sizes['Y'] = metadata[tag_names['YResolution']].value[1] \
                                  / metadata[tag_names['YResolution']].value[0]
        if 'ResolutionUnit' in tag_names:
            if metadata[tag_names['ResolutionUnit']].value == tifffile.RESUNIT.CENTIMETER:
                self.dim_sizes['X'] *= 1E-4 * 1E6
                self.dim_sizes['Y'] *= 1E-4 * 1E6
        if 'Z' in self.axes:
            if 'ZResolution' in tag_names:
                self.dim_sizes['Z'] = 1 / metadata[tag_names['ZResolution']].value[0]
        if 'T' in self.axes:
            if 'FrameRate' in tag_names:
                self.dim_sizes['T'] = 1 / metadata[tag_names['FrameRate']].value[0]

    def _get_nd2_metadata(self, metadata):
        if 'Time [s]' in metadata.recorded_data:
            timestamps = metadata.recorded_data['Time [s]']
            self.dim_sizes['T'] = timestamps[-1] / len(timestamps)
        self.dim_sizes['X'] = metadata.volume.axesCalibration[0]
        self.dim_sizes['Y'] = metadata.volume.axesCalibration[1]
        self.dim_sizes['Z'] = metadata.volume.axesCalibration[2]

    def load_metadata(self):
        self.dim_sizes = {'X': None, 'Y': None, 'Z': None, 'T': None}
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
        self._check_axes()

    def _check_axes(self):
        if len(self.shape) != len(self.axes):
            self.good_dims = False
            self.change_axes('Q' * len(self.shape))

    def _check_dim_sizes(self):
        if 'Z' in self.axes and self.dim_sizes['Z'] is None:
            self.good_dims = False
            return
        if 'T' in self.axes and self.dim_sizes['T'] is None:
            self.good_dims = False
            return
        if self.dim_sizes['X'] is None or self.dim_sizes['Y'] is None:
            self.good_dims = False
            return
        if 'Q' in self.axes:
            self.good_dims = False
            return
        self.good_dims = True

    def change_axes(self, new_axes):
        if len(new_axes) != len(self.shape):
            raise ValueError('New axes must have the same length as the shape of the data')
        self.axes = new_axes
        self._check_axes()
        self._check_dim_sizes()


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
    #     print(file_info.dim_sizes)
    #     print('\n\n')

    test_file = all_paths[1]
    file_info = FileInfo(test_file)
    file_info.find_metadata()
    file_info.load_metadata()
    print(file_info.metadata_type)
    print(file_info.axes)
    print(file_info.shape)
    print(file_info.dim_sizes)
    file_info.change_axes('TZYX')
    print(file_info.axes)
    print(file_info.good_dims)
