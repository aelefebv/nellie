import os

import nd2
import numpy as np
import ome_types
from tifffile import tifffile

from nellie import logger


class FileInfo:
    """
    A class to handle file information, metadata extraction, and basic file operations for microscopy image files.

    Attributes
    ----------
    filepath : str
        Path to the input file.
    metadata : dict or None
        Stores the metadata extracted from the file.
    metadata_type : str or None
        Type of metadata detected (e.g., 'ome', 'imagej', 'nd2').
    axes : str or None
        String representing the axes in the file (e.g., 'TZCYX').
    shape : tuple or None
        Shape of the image file.
    dim_res : dict or None
        Dictionary of physical dimensions (X, Y, Z, T) resolution in microns or seconds.
    input_dir : str
        Directory of the input file.
    basename : str
        Filename with extension.
    filename_no_ext : str
        Filename without the extension.
    extension : str
        File extension (e.g., '.tiff', '.nd2').
    output_dir : str
        Output directory for processed files.
    nellie_necessities_dir : str
        Directory for internal processing data.
    ome_output_path : str or None
        Path for OME TIFF output.
    good_dims : bool
        Whether the dimensional metadata is valid.
    good_axes : bool
        Whether the axes metadata is valid.
    ch : int
        Selected channel.
    t_start : int
        Start timepoint for processing.
    t_end : int or None
        End timepoint for processing.
    dtype : type or None
        Data type of the image.

    Methods
    -------
    _find_tif_metadata()
        Extract metadata from TIFF or OME-TIFF files.
    _find_nd2_metadata()
        Extract metadata from ND2 files.
    find_metadata()
        Detect file type and extract corresponding metadata.
    _get_imagej_metadata(metadata)
        Extract dimensional resolution from ImageJ metadata.
    _get_ome_metadata(metadata)
        Extract dimensional resolution from OME metadata.
    _get_tif_tags_metadata(metadata)
        Extract dimensional resolution from generic TIFF tags.
    _get_nd2_metadata(metadata)
        Extract dimensional resolution from ND2 metadata.
    load_metadata()
        Load and validate dimensional metadata based on the file type.
    _check_axes()
        Validate the axes metadata for correctness.
    _check_dim_res()
        Validate the dimensional resolution metadata for correctness.
    change_axes(new_axes)
        Change the axes string and revalidate the metadata.
    change_dim_res(dim, new_size)
        Modify the resolution of a specific dimension.
    change_selected_channel(ch)
        Select a different channel in the file for processing.
    select_temporal_range(start=0, end=None)
        Select a temporal range for processing.
    _validate()
        Validate the current state of axes and dimension metadata.
    read_file()
        Read the image file based on its type.
    _get_output_path()
        Generate the output file path based on the current axes, resolution, and channel.
    save_ome_tiff()
        Save the processed image file as an OME-TIFF file with updated metadata.
    """
    def __init__(self, filepath, output_dir=None):
        """
        Initializes the FileInfo object and creates directories for outputs if they do not exist.

        Parameters
        ----------
        filepath : str
            Path to the input file.
        output_dir : str, optional
            Directory for saving output files. Defaults to a subdirectory within the input file's directory.
        """
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

        self.nellie_necessities_dir = os.path.join(self.output_dir, 'nellie_necessities')
        if not os.path.exists(self.nellie_necessities_dir):
            os.makedirs(self.nellie_necessities_dir)

        self.ome_output_path = None
        self.good_dims = False
        self.good_axes = False

        self.ch = 0
        self.t_start = 0
        self.t_end = None
        self.dtype = None

    def _find_tif_metadata(self):
        """
        Extracts metadata from TIFF or OME-TIFF files and updates relevant class attributes.

        Returns
        -------
        tuple
            Metadata and metadata type extracted from the TIFF file.
        """
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
        """
        Extracts metadata from ND2 files and updates relevant class attributes.
        """
        with nd2.ND2File(self.filepath) as nd2_file:
            metadata = nd2_file.metadata.channels[0]
            metadata.recorded_data = nd2_file.events(orient='list')
            self.metadata = metadata
            self.metadata_type = 'nd2'
            self.axes = ''.join(nd2_file.sizes.keys())
            self.shape = tuple(nd2_file.sizes.values())

    def find_metadata(self):
        """
        Detects file type (e.g., TIFF or ND2) and calls the appropriate metadata extraction method.

        Raises
        ------
        ValueError
            If the file type is not supported.
        """
        if self.filepath.endswith('.tiff') or self.filepath.endswith('.tif'):
            self._find_tif_metadata()
        elif self.filepath.endswith('.nd2'):
            self._find_nd2_metadata()
        else:
            raise ValueError('File type not supported')

    def _get_imagej_metadata(self, metadata):
        """
        Extracts dimensional resolution from ImageJ metadata and stores it in `dim_res`.

        Parameters
        ----------
        metadata : dict
            ImageJ metadata extracted from the file.
        """
        self.dim_res['X'] = metadata['physicalsizex'] if 'physicalsizex' in metadata else None
        self.dim_res['Y'] = metadata['physicalsizey'] if 'physicalsizey' in metadata else None
        self.dim_res['Z'] = metadata['spacing'] if 'spacing' in metadata else None
        self.dim_res['T'] = metadata['finterval'] if 'finterval' in metadata else None

    def _get_ome_metadata(self, metadata):
        """
        Extracts dimensional resolution from OME metadata and stores it in `dim_res`.

        Parameters
        ----------
        metadata : ome_types.OME
            OME metadata object.
        """
        self.dim_res['X'] = metadata.images[0].pixels.physical_size_x
        self.dim_res['Y'] = metadata.images[0].pixels.physical_size_y
        self.dim_res['Z'] = metadata.images[0].pixels.physical_size_z
        self.dim_res['T'] = metadata.images[0].pixels.time_increment

    def _get_tif_tags_metadata(self, metadata):
        """
        Extracts dimensional resolution from TIFF tag metadata and stores it in `dim_res`.

        Parameters
        ----------
        metadata : dict
            Dictionary of TIFF tags.
        """
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
        """
        Extracts dimensional resolution from ND2 metadata and stores it in `dim_res`.

        Parameters
        ----------
        metadata : dict
            ND2 metadata object.
        """
        if 'Time [s]' in metadata.recorded_data:
            timestamps = metadata.recorded_data['Time [s]']
            self.dim_res['T'] = timestamps[-1] / len(timestamps)
        self.dim_res['X'] = metadata.volume.axesCalibration[0]
        self.dim_res['Y'] = metadata.volume.axesCalibration[1]
        self.dim_res['Z'] = metadata.volume.axesCalibration[2]

    def load_metadata(self):
        """
        Loads and validates dimensional metadata based on the file type (OME, ImageJ, ND2, or generic TIFF).
        """
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
        """
        Validates the axes metadata and ensures the required axes (X, Y) are present.
        """
        if len(self.shape) != len(self.axes):
            self.change_axes('Q' * len(self.shape))
        for axis in self.axes:
            if axis not in ['T', 'Z', 'Y', 'X', 'C']:
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
        """
        Validates the dimensional resolution metadata, ensuring that X, Y, Z, and T dimensions have valid resolutions.
        """
        check_dims = ['X', 'Y', 'Z', 'T']
        for dim in check_dims:
            if dim in self.axes and self.dim_res[dim] is None:
                self.good_dims = False
                return
        self.good_dims = True

    def change_axes(self, new_axes):
        """
        Changes the axes string and revalidates the metadata.

        Parameters
        ----------
        new_axes : str
            New axes string to replace the existing one.
        """
        # if len(new_axes) != len(self.shape):
        self.good_axes = False
            # return
            # raise ValueError('New axes must have the same length as the shape of the data')
        self.axes = new_axes
        self._validate()

    def change_dim_res(self, dim, new_size):
        """
        Modifies the resolution of a specific dimension.

        Parameters
        ----------
        dim : str
            Dimension to modify (e.g., 'X', 'Y', 'Z', 'T').
        new_size : float
            New resolution for the specified dimension.
        """
        if dim not in self.dim_res:
            return
            # raise ValueError('Invalid dimension')
        self.dim_res[dim] = new_size
        self._validate()

    def change_selected_channel(self, ch):
        """
        Changes the selected channel for processing.

        Parameters
        ----------
        ch : int
            Index of the new channel to select.

        Raises
        ------
        ValueError
            If the axes or dimension metadata are invalid.
        KeyError
            If no channel dimension is available.
        IndexError
            If the selected channel index is out of range.
        """
        if not self.good_dims or not self.good_axes:
            raise ValueError('Must have both valid axes and dimensions to change channel')
        if 'C' not in self.axes:
            raise KeyError('No channel dimension to change')
        if ch < 0 or ch >= self.shape[self.axes.index('C')]:
            raise IndexError('Invalid channel index')
        self.ch = ch
        self._get_output_path()

    def select_temporal_range(self, start=0, end=None):
        """
        Selects a temporal range for processing.

        Parameters
        ----------
        start : int, optional
            Start index of the temporal range. Defaults to 0.
        end : int, optional
            End index of the temporal range. Defaults to None, which includes all timepoints.
        """
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
        """
        Validates the current state of the axes and dimensional metadata, then updates output paths.

        This method performs several validation steps:
        1. Calls `_check_axes()` to ensure the axes are valid.
        2. Calls `_check_dim_res()` to ensure that the dimensional resolutions are valid.
        3. Calls `select_temporal_range()` to set or update the time range if applicable.
        4. Calls `_get_output_path()` to update the output paths based on the current metadata.

        The method ensures that all aspects of the metadata (axes, dimensional resolutions, and temporal range)
        are consistent and correctly applied before further processing.

        Raises
        ------
        ValueError
            If any aspect of the metadata is invalid or inconsistent.
        """
        self._check_axes()
        self._check_dim_res()
        self.select_temporal_range()
        self._get_output_path()

    def read_file(self):
        """
        Reads the image file into memory, supporting TIFF and ND2 formats.

        Returns
        -------
        np.ndarray
            Numpy array representing the image data.

        Raises
        ------
        ValueError
            If the file type is unsupported.
        """
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
        """
        Generates output paths for the processed image file, based on the current axes, resolutions, and selected channel.

        This method constructs a filename that incorporates the axes, the rounded dimensional resolutions (up to four
        decimal places), the selected channel, and the temporal range. It also generates both user-facing and internal
        processing paths.

        The generated paths include:
        - `user_output_path_no_ext`: Path for the user output file (excluding file extension).
        - `nellie_necessities_output_path_no_ext`: Path for internal processing output (excluding file extension).
        - `ome_output_path`: Full path for the OME-TIFF output file.

        The method replaces periods in dimensional resolutions with 'p' to avoid issues with file systems.

        Notes
        -----
        - Temporal range information is added if the 'T' (time) axis is present.
        - If any dimensional resolution is `None`, the string 'None' is used in the filename.
        """
        t_text = f'-t{self.t_start}_to_{self.t_end}' if 'T' in self.axes else ''
        dim_texts = []
        for axis in self.axes:
            if axis not in self.dim_res:
                continue
            dim_res = self.dim_res[axis]
            # round to 4 decimal places
            if dim_res is None:
                dim_res = 'None'
            else:
                dim_res = str(round(dim_res, 4))
            # convert '.' to 'p'
            dim_res = dim_res.replace('.', 'p')
            dim_texts.append(f'{axis}{dim_res}')
        dim_text = f"-{'_'.join(dim_texts)}"
        output_name = f'{self.filename_no_ext}-{self.axes}{dim_text}-ch{self.ch}{t_text}'
        self.user_output_path_no_ext = os.path.join(self.output_dir, output_name)
        self.nellie_necessities_output_path_no_ext = os.path.join(self.nellie_necessities_dir, output_name)
        self.ome_output_path = self.nellie_necessities_output_path_no_ext + '.ome.tif'

    def save_ome_tiff(self):
        """
        Saves the processed image data as an OME-TIFF file, including updated metadata.

        Raises
        ------
        ValueError
            If the axes or dimensional resolution metadata is invalid.
        """
        if not self.good_axes or not self.good_dims:
            raise ValueError('Cannot save file with invalid axes or dimensions')

        axes = self.axes
        data = self.read_file()
        if 'T' not in self.axes:
            data = data[np.newaxis, ...]
            axes = 'T' + self.axes
        else:
            t_index = self.axes.index('T')
            selected_range = range(self.t_start, self.t_end + 1)
            data = np.take(data, selected_range, axis=t_index)
            # if len(selected_range) == 1:
            #     data = np.expand_dims(data, axis=t_index)
        if 'C' in axes:
            data = np.take(data, self.ch, axis=axes.index('C'))
            axes = axes.replace('C', '')

        # ensure 'T' is the 0th dimension
        if 'T' in axes:
            t_index = axes.index('T')
            data = np.moveaxis(data, t_index, 0)
            axes = 'T' + axes.replace('T', '')

        tifffile.imwrite(
            self.ome_output_path, data, bigtiff=True, metadata={"axes": axes}
        )

        ome_xml = tifffile.tiffcomment(self.ome_output_path)
        ome = ome_types.from_xml(ome_xml)
        ome.images[0].pixels.physical_size_x = self.dim_res['X']
        ome.images[0].pixels.physical_size_y = self.dim_res['Y']
        ome.images[0].pixels.physical_size_z = self.dim_res['Z']
        ome.images[0].pixels.time_increment = self.dim_res['T']
        dtype_name = data.dtype.name
        if data.dtype.name == 'float64':
            dtype_name = 'double'
        if data.dtype.name == 'float32':
            dtype_name = 'float'
        ome.images[0].pixels.type = dtype_name
        ome_xml = ome.to_xml()
        tifffile.tiffcomment(self.ome_output_path, ome_xml)


class ImInfo:
    """
    A class to manage image data and file outputs related to microscopy image processing.

    This class handles the initialization of memory-mapped image data, creation of output paths,
    extraction of OME metadata, and memory allocation for various stages of an image processing pipeline.

    Attributes
    ----------
    file_info : FileInfo
        The FileInfo object containing metadata and file paths.
    im_path : str
        Path to the OME-TIFF image file.
    im : np.ndarray
        Memory-mapped image data loaded from the file.
    screenshot_dir : str
        Directory for saving screenshots of processed images.
    graph_dir : str
        Directory for saving graphs of processed data.
    dim_res : dict
        Dictionary storing the resolution of the image along the dimensions (X, Y, Z, T).
    axes : str
        Axes string representing the dimensions in the image (e.g., 'TZYX').
    new_axes : str
        Modified axes string if additional dimensions are added.
    shape : tuple
        Shape of the image data.
    ome_metadata : ome_types.OME
        OME metadata object extracted from the image.
    no_z : bool
        Flag indicating if the Z dimension is absent or has a single slice.
    no_t : bool
        Flag indicating if the T dimension is absent or has a single timepoint.
    pipeline_paths : dict
        Dictionary storing output paths for different stages of the image processing pipeline.

    Methods
    -------
    _check_axes_exist()
        Checks if the Z and T dimensions exist and updates the flags `no_z` and `no_t` accordingly.
    create_output_path(pipeline_path: str, ext: str = '.ome.tif', for_nellie=True)
        Creates a file path for a specific stage of the image processing pipeline.
    _create_output_paths()
        Creates all necessary output paths for various stages in the image processing pipeline.
    remove_intermediates()
        Removes intermediate files created during the image processing pipeline, except for .csv files.
    _get_ome_metadata()
        Extracts OME metadata from the image and updates resolution, axes, and shape information.
    get_memmap(file_path: str, read_mode: str = 'r+')
        Returns a memory-mapped array for the image data from the specified file.
    allocate_memory(output_path: str, dtype: str = 'float', data: np.ndarray = None, description: str = 'No description.',
                    return_memmap: bool = False, read_mode: str = 'r+')
        Allocates memory for new image data, saves it to the specified file, and writes updated OME metadata.
    """
    def __init__(self, file_info: FileInfo):
        """
        Initializes the ImInfo object, loading image data and setting up directories for screenshots and graphs.

        If the OME-TIFF file does not exist, it creates one by calling `save_ome_tiff()` from the FileInfo class.

        Parameters
        ----------
        file_info : FileInfo
            An instance of the FileInfo class, containing metadata and paths for the image file.
        """
        self.file_info = file_info
        self.im_path = file_info.ome_output_path
        if not os.path.exists(self.im_path):
            file_info.save_ome_tiff()
        self.im = tifffile.memmap(self.im_path)

        self.screenshot_dir = os.path.join(self.file_info.output_dir, 'screenshots')
        self.graph_dir = os.path.join(self.file_info.output_dir, 'graphs')

        self.dim_res = {'X': None, 'Y': None, 'Z': None, 'T': None}
        self.axes = None
        self.new_axes = None
        self.shape = None
        self.ome_metadata = None
        self._get_ome_metadata()

        self.no_z = True
        self.no_t = True
        self._check_axes_exist()

        self.pipeline_paths = {}
        self._create_output_paths()

    def _check_axes_exist(self):
        """
        Checks the existence of the Z and T dimensions in the image data.

        Updates the `no_z` and `no_t` flags based on whether the Z and T axes are present and have more than one slice or timepoint.
        """
        if 'Z' in self.axes and self.shape[self.new_axes.index('Z')] > 1:
            self.no_z = False
        if 'T' in self.axes and self.shape[self.new_axes.index('T')] > 1:
            self.no_t = False

    def create_output_path(self, pipeline_path: str, ext: str = '.ome.tif', for_nellie=True):
        """
        Creates a file path for a specific stage of the image processing pipeline.

        Parameters
        ----------
        pipeline_path : str
            A descriptive string representing the stage of the image processing pipeline (e.g., 'im_preprocessed').
        ext : str, optional
            The file extension to use (default is '.ome.tif').
        for_nellie : bool, optional
            Whether the output is for internal use by Nellie (default is True).

        Returns
        -------
        str
            The full file path for the given stage of the image processing pipeline.
        """
        if for_nellie:
            output_path = f'{self.file_info.nellie_necessities_output_path_no_ext}-{pipeline_path}{ext}'
        else:
            output_path = f'{self.file_info.user_output_path_no_ext}-{pipeline_path}{ext}'
        self.pipeline_paths[pipeline_path] = output_path
        return self.pipeline_paths[pipeline_path]

    def _create_output_paths(self):
        """
        Creates all necessary output paths for different stages in the image processing pipeline.

        This method creates paths for various pipeline stages such as preprocessed images, instance labels, skeletons,
        pixel classifications, flow vectors, adjacency maps, and various feature extraction results (voxels, nodes, branches, organelles, and images).
        """
        self.create_output_path('im_preprocessed')
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
        self.create_output_path('features_voxels', ext='.csv', for_nellie=False)
        self.create_output_path('features_nodes', ext='.csv', for_nellie=False)
        self.create_output_path('features_branches', ext='.csv', for_nellie=False)
        self.create_output_path('features_organelles', ext='.csv', for_nellie=False)
        self.create_output_path('features_image', ext='.csv', for_nellie=False)
        self.create_output_path('adjacency_maps', ext='.pkl')

    def remove_intermediates(self):
        """
        Removes intermediate files created during the image processing pipeline, except for CSV files.

        This method loops through all pipeline paths and deletes files (except .csv files) that were created during
        processing. It also deletes the main image file if it exists.
        """
        all_pipeline_paths = [self.pipeline_paths[pipeline_path] for pipeline_path in self.pipeline_paths]
        for pipeline_path in all_pipeline_paths + [self.im_path]:
            if 'csv' in pipeline_path:
                continue
            elif os.path.exists(pipeline_path):
                os.remove(pipeline_path)

    def _get_ome_metadata(self, ):
        """
        Extracts OME metadata from the image and updates the `axes`, `new_axes`, `shape`, and `dim_res` attributes.

        If the 'T' axis is not present, it adds a new temporal axis to the image data and updates the axes accordingly.
        """
        with tifffile.TiffFile(self.im_path) as tif:
            self.axes = tif.series[0].axes
            self.new_axes = self.axes
        if 'T' not in self.axes:
            self.im = self.im[np.newaxis, ...]
            self.new_axes = 'T' + self.axes
        self.shape = self.im.shape
        self.ome_metadata = ome_types.from_xml(tifffile.tiffcomment(self.im_path))
        self.dim_res['X'] = self.ome_metadata.images[0].pixels.physical_size_x
        self.dim_res['Y'] = self.ome_metadata.images[0].pixels.physical_size_y
        self.dim_res['Z'] = self.ome_metadata.images[0].pixels.physical_size_z
        self.dim_res['T'] = self.ome_metadata.images[0].pixels.time_increment

    def get_memmap(self, file_path, read_mode='r+'):
        """
        Returns a memory-mapped array for the image data from the specified file.

        Parameters
        ----------
        file_path : str
            Path to the image file to be memory-mapped.
        read_mode : str, optional
            Mode for reading the memory-mapped file (default is 'r+').

        Returns
        -------
        np.ndarray
            A memory-mapped numpy array representing the image data.
        """
        memmap = tifffile.memmap(file_path, mode=read_mode)
        if 'T' not in self.axes:
            memmap = memmap[np.newaxis, ...]
        return memmap

    def allocate_memory(self, output_path, dtype='float', data=None, description='No description.',
                        return_memmap=False, read_mode='r+'):
        """
        Allocates memory for new image data or writes new data to an output file.

        This method creates an empty OME-TIFF file with the specified `dtype` and shape, or writes the given `data` to the file.
        It also updates the OME metadata with a description and the correct pixel type.

        Parameters
        ----------
        output_path : str
            Path to the output file.
        dtype : str, optional
            Data type for the new image (default is 'float').
        data : np.ndarray, optional
            Numpy array containing image data to write (default is None, which allocates empty memory).
        description : str, optional
            Description for the OME metadata (default is 'No description.').
        return_memmap : bool, optional
            Whether to return a memory-mapped array for the newly allocated file (default is False).
        read_mode : str, optional
            Mode for reading the memory-mapped file if `return_memmap` is True (default is 'r+').

        Returns
        -------
        np.ndarray, optional
            A memory-mapped numpy array if `return_memmap` is set to True.
        """
        axes = self.axes
        if 'T' not in self.axes:
            axes = 'T' + axes
            if data is not None:
                data = data[np.newaxis, ...]
        if data is None:
            tifffile.imwrite(
                output_path, shape=self.shape, dtype=dtype, bigtiff=True, metadata={"axes": axes}
            )
        else:
            dtype = dtype or data.dtype.name
            tifffile.imwrite(
                output_path, data, bigtiff=True, metadata={"axes": axes}
            )
        ome = self.ome_metadata
        ome.images[0].description = description
        ome.images[0].pixels.type = dtype
        ome_xml = ome.to_xml()
        tifffile.tiffcomment(output_path, ome_xml)
        if return_memmap:
            return self.get_memmap(output_path, read_mode=read_mode)


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
