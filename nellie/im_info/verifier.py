"""
File verification and image metadata handling for microscopy images.

This module provides FileInfo and ImInfo classes for loading, validating, and managing
metadata from various microscopy file formats (TIFF, OME-TIFF, ND2).
"""
import json
import os

import nd2
import numpy as np
import ome_types
from tifffile import tifffile

from nellie.im_info.extractors import MetadataExtractor, detect_extractor
from nellie.im_info.types import DimRes, infer_t_axis
from nellie.utils.base_logger import logger

# Backwards-compat re-exports: ``DimRes`` and ``infer_t_axis`` moved to
# ``nellie.im_info.types`` in Slice 6 to break the verifier ↔ extractors
# circular import. Existing ``from nellie.im_info.verifier import DimRes``
# (or ``infer_t_axis``) imports keep working via the re-exports above.
__all__ = ['FileInfo', 'ImInfo', 'DimRes', 'infer_t_axis', 'transform_to_axes']


def transform_to_axes(
    data: np.ndarray,
    source_axes: str | None,
    target_axes: str | None = None,
) -> tuple[np.ndarray, str]:
    """
    Transform a data array so its axes match a target layout.

    Two modes:

    - **Canonical mode** (``target_axes=None``): derive a canonical
      ``T[Z]YX`` target — T is prepended if absent (or moved to index 0
      if present elsewhere); singleton Z is always squeezed; the result
      must contain Y and X and use only ``{T, Z, Y, X}``.
    - **Match mode** (``target_axes`` is a string): rearrange to match
      the supplied target. T is prepended if source lacks T; Z is
      squeezed only when the target lacks Z (raises if Z>1 in source
      but missing in target). Source axes set must equal target axes
      set after T-handling and the conditional Z-squeeze.

    Parameters
    ----------
    data : np.ndarray
        Data array to transform.
    source_axes : str
        Axes string describing ``data``'s current layout.
    target_axes : str or None, optional
        Desired axes layout. ``None`` triggers canonical mode.

    Returns
    -------
    (np.ndarray, str)
        Transformed array and its final axes string.

    Raises
    ------
    ValueError
        If ``source_axes`` is ``None``, validation fails, or
        dimensions become inconsistent with the resulting axes.
    """
    if source_axes is None:
        raise ValueError("Axes metadata is not initialized")

    axes_list = list(source_axes)

    # Step 1: ensure T is at index 0 (prepend or moveaxis).
    if 'T' not in axes_list:
        data = data[np.newaxis, ...]
        axes_list = ['T'] + axes_list
    else:
        t_index = axes_list.index('T')
        if t_index != 0:
            data = np.moveaxis(data, t_index, 0)
            axes_list = ['T'] + [ax for i, ax in enumerate(axes_list) if i != t_index]

    target_list = list(target_axes) if target_axes is not None else None

    # Step 2: Z handling. Canonical mode always squeezes singleton Z;
    # match mode squeezes only when target lacks Z (and raises on Z>1).
    if 'Z' in axes_list:
        z_index = axes_list.index('Z')
        if target_list is None:
            if data.shape[z_index] == 1:
                data = np.squeeze(data, axis=z_index)
                axes_list.pop(z_index)
        elif 'Z' not in target_list:
            if data.shape[z_index] == 1:
                data = np.squeeze(data, axis=z_index)
                axes_list.pop(z_index)
            else:
                raise ValueError(
                    "Z axis present with size > 1, but target axes lacks Z"
                )

    # Step 3: validate and pick the final ordering.
    if target_list is None:
        allowed_axes = {'T', 'Z', 'Y', 'X'}
        extra_axes = [ax for ax in axes_list if ax not in allowed_axes]
        if extra_axes:
            raise ValueError(f"Unsupported axes found: {extra_axes}")
        if 'Y' not in axes_list or 'X' not in axes_list:
            raise ValueError("Axes must include both Y and X")
        final_axes = ['T']
        if 'Z' in axes_list:
            final_axes.append('Z')
        final_axes.extend(['Y', 'X'])
    else:
        if set(axes_list) != set(target_list):
            extra = sorted(set(axes_list) - set(target_list))
            missing = sorted(set(target_list) - set(axes_list))
            raise ValueError(f"Axes mismatch. Extra: {extra}, missing: {missing}")
        final_axes = target_list

    if axes_list != final_axes:
        order = [axes_list.index(ax) for ax in final_axes]
        data = np.transpose(data, order)

    if data.ndim != len(final_axes):
        raise ValueError("Data dimensions do not match normalized axes")

    return data, ''.join(final_axes)


class FileInfo:
    """
    A class to handle file information, metadata extraction, and basic file operations for microscopy image files.

    Attributes
    ----------
    filepath : str
        Path to the input file.
    metadata_type : str or None
        Type of metadata detected (e.g., 'ome', 'imagej', 'nd2').
        Slice 6: this is the only metadata-discriminator field on
        ``FileInfo``; the polymorphic ``self.metadata`` field was
        retired in favor of a stashed extractor instance
        (``self._extractor``) consumed by ``load_metadata``.
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
    output_naming : str
        Output naming strategy ("detailed" or "stable").
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
    find_metadata()
        Detect file type via the extractor factory and stash the
        extractor instance, plus axes / shape / metadata_type. Calls
        ``prepare_output_dirs`` first.
    load_metadata()
        Project the stashed extractor into ``self.dim_res`` via
        ``parse_dim_res()``, then runs ``_validate``.
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
    def __init__(self, filepath, output_dir=None, output_naming="detailed"):
        """
        Initializes the FileInfo object — pure-data, no I/O.

        Output directories are created lazily by ``prepare_output_dirs``,
        which is called automatically from ``find_metadata``. Construction
        itself does not touch the filesystem.

        Parameters
        ----------
        filepath : str
            Path to the input file.
        output_dir : str, optional
            Directory for saving output files. Defaults to a subdirectory within the input file's directory.
        output_naming : str, optional
            Output naming strategy ("detailed" or "stable"). Defaults to "detailed".
        """
        self.filepath = filepath
        self.metadata_type: str | None = None
        self.axes = None
        self.shape = None
        self.dim_res: DimRes | None = None
        # Slice 6: extractor instance stashed by ``find_metadata`` and
        # consumed by ``load_metadata`` (whose only job is now
        # ``self._extractor.parse_dim_res()``). Not part of the public
        # surface — callers should read ``metadata_type``/``axes``/
        # ``shape``/``dim_res`` instead.
        self._extractor: MetadataExtractor | None = None

        self.input_dir = os.path.dirname(filepath)
        self.basename = os.path.basename(filepath)
        self.filename_no_ext = os.path.splitext(self.basename)[0]
        self.extension = os.path.splitext(filepath)[1].lower()
        self.output_naming = output_naming
        self.output_dir = output_dir or os.path.join(self.input_dir, 'nellie_output')
        self.nellie_necessities_dir = os.path.join(self.output_dir, 'nellie_necessities')

        self.ome_output_path = None
        self.good_dims = False
        self.good_axes = False
        self.validation_errors = []

        self.ch = 0
        self.t_start = 0
        self.t_end = None
        self.dtype = None

    def prepare_output_dirs(self):
        """
        Create ``output_dir`` and ``nellie_necessities_dir`` if missing.

        Idempotent (uses ``exist_ok=True``). Called automatically from
        ``find_metadata``; can be invoked explicitly by callers that
        want directories created without metadata extraction.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.nellie_necessities_dir, exist_ok=True)

    def find_metadata(self):
        """
        Detect file type and read its metadata via the appropriate extractor.

        Slice 6 thinned this method to a 1-line factory call. The
        legacy 6-method per-format dispatch
        (``_find_tif_metadata``/``_find_nd2_metadata`` →
        ``_get_*_metadata``) was extracted into the
        ``nellie.im_info.extractors`` subpackage; ``detect_extractor``
        opens the file ONCE, classifies it, and returns the appropriate
        ``MetadataExtractor`` instance carrying ``axes``, ``shape``,
        ``metadata_type``, and the format-specific raw fields.

        Also calls ``prepare_output_dirs`` so output directories exist
        before any subsequent ``save_ome_tiff`` call. Construction
        itself does not touch the filesystem; this method does.

        Raises
        ------
        ValueError
            If the file type is not supported.
        """
        self.prepare_output_dirs()
        self._extractor = detect_extractor(self.filepath)
        self.metadata_type = self._extractor.metadata_type
        self.axes = self._extractor.axes
        self.shape = self._extractor.shape

    def load_metadata(self):
        """
        Parse ``dim_res`` from the previously-detected extractor.

        Must be called after ``find_metadata`` (which stashes the
        extractor instance on ``self._extractor``). Populates
        ``self.dim_res`` with the per-format physical pixel sizes and
        time interval, then runs ``_validate``.

        Raises
        ------
        ValueError
            If ``find_metadata`` was not called first (no extractor
            stashed on ``self._extractor``).
        """
        if self._extractor is None:
            raise ValueError("find_metadata must be called before load_metadata")
        self.dim_res = self._extractor.parse_dim_res()
        self._validate()

    def _axis_errors(self):
        errors = []
        if self.axes is None or self.shape is None:
            errors.append('Axes or shape metadata not loaded')
            return errors
        if len(self.shape) != len(self.axes):
            errors.append('Axes length does not match data shape')
        allowed_axes = {'T', 'Z', 'Y', 'X', 'C'}
        invalid_axes = [axis for axis in self.axes if axis not in allowed_axes]
        if invalid_axes:
            errors.append('Axes must only use T, Z, C, Y, X')
        if len(set(self.axes)) != len(self.axes):
            errors.append('Axes must not contain duplicates')
        if 'X' not in self.axes or 'Y' not in self.axes:
            errors.append('Axes must include both X and Y')
        return errors

    def _dim_errors(self):
        errors = []
        if self.axes is None or self.dim_res is None:
            return errors
        check_dims = ['X', 'Y', 'Z', 'T']
        for dim in check_dims:
            if dim in self.axes and self.dim_res.get(dim) is None:
                errors.append(f"Missing {dim} resolution")
        return errors

    def _time_range_errors(self):
        errors = []
        if self.axes is None or self.shape is None:
            return errors
        if 'T' not in self.axes:
            return errors
        if self.t_start is None or self.t_end is None:
            return errors
        if self.t_start < 0 or self.t_end < 0:
            errors.append('Temporal range must be >= 0')
        t_index = self.axes.index('T')
        max_t = self.shape[t_index] - 1
        if self.t_start > self.t_end:
            errors.append('Start frame must be <= end frame')
        if self.t_start > max_t or self.t_end > max_t:
            errors.append('Temporal range out of bounds')
        return errors

    def compute_errors(self):
        """
        Aggregate axis, dim, and time-range errors. Pure read.

        Returns
        -------
        list[str]
            Empty if all three error sources are clean. Otherwise the
            concatenation of ``_axis_errors`` + ``_dim_errors`` +
            ``_time_range_errors`` results.
        """
        return self._axis_errors() + self._dim_errors() + self._time_range_errors()

    def apply_defaults(self):
        """
        Fill ``t_start``/``t_end`` defaults when axes are valid and T present.

        No-op when ``good_axes`` is False, when ``T`` is absent from
        ``self.axes``, or when ``self.shape`` is None. Otherwise:
        ``t_start`` defaults to 0 (when None); ``t_end`` defaults to
        ``shape[axes.index('T')] - 1`` (when None).
        """
        if not self.good_axes:
            return
        if self.axes is None or 'T' not in self.axes:
            return
        if self.shape is None:
            return
        if self.t_start is None:
            self.t_start = 0
        t_index = self.axes.index('T')
        max_t = self.shape[t_index] - 1
        if self.t_end is None:
            self.t_end = max_t

    def change_axes(self, new_axes):
        """
        Changes the axes string and revalidates the metadata.

        Parameters
        ----------
        new_axes : str
            New axes string to replace the existing one.

        Raises
        ------
        ValueError
            If ``len(new_axes)`` does not match ``len(self.shape)``.

        Preconditions
        -------------
        ``find_metadata`` and ``load_metadata`` must have run so that
        ``self.shape`` is populated. The napari fileselect widget
        pre-validates length before calling this method; programmatic
        callers must pass a length-matching axes string or handle the
        ValueError.
        """
        if self.shape is None or len(new_axes) != len(self.shape):
            raise ValueError(
                'New axes must have the same length as the shape of the data'
            )
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

        Raises
        ------
        ValueError
            If ``dim_res`` is not initialized (call ``load_metadata``
            first) or ``dim`` is not one of {'X', 'Y', 'Z', 'T'}.

        Preconditions
        -------------
        ``find_metadata`` and ``load_metadata`` must have run so that
        ``self.dim_res`` is initialized to the canonical 4-key dict.
        Note: the ``dim_res is None`` check fires BEFORE the invalid-dim
        check, so calling with both conditions raises the
        "not initialized" message.
        """
        if self.dim_res is None:
            raise ValueError('Dimension resolutions are not initialized')
        if dim not in self.dim_res:
            raise ValueError(f"Invalid dimension '{dim}'")
        self.dim_res[dim] = new_size  # type: ignore[literal-required]
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

        Preconditions
        -------------
        ``find_metadata`` and ``load_metadata`` must have run, AND
        ``good_axes`` and ``good_dims`` must both be True. The file's
        axes string must include 'C'.
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

        Raises
        ------
        ValueError
            If axes/shape are not loaded, lengths mismatch, or
            ``start > end``.
        KeyError
            If 'T' is not in the file's axes.
        IndexError
            If ``start < 0``, ``end < 0``, or either exceeds ``max_t``.

        Preconditions
        -------------
        ``find_metadata`` and ``load_metadata`` must have run. The
        file's axes string must include 'T'. Distinct error messages
        for ``start < 0`` ('Start frame must be >= 0') and ``end < 0``
        ('End frame must be >= 0').
        """
        if self.axes is None or self.shape is None:
            raise ValueError('Axes or shape metadata not loaded')
        if len(self.axes) != len(self.shape):
            raise ValueError('Axes and shape length mismatch')
        if 'T' not in self.axes:
            raise KeyError('No time dimension to select')
        if start < 0:
            raise IndexError('Start frame must be >= 0')
        t_index = self.axes.index('T')
        max_t = self.shape[t_index] - 1
        if end is None:
            end = max_t
        if end < 0:
            raise IndexError('End frame must be >= 0')
        if start > end:
            raise ValueError('Start frame must be <= end frame')
        if start > max_t or end > max_t:
            raise IndexError('Temporal range out of bounds')
        self.t_start = start
        self.t_end = end
        self._get_output_path()

    def _validate(self):
        """
        Recompute validation flags + ``validation_errors`` and update output paths.

        Thin orchestrator that:
        1. Sets ``good_axes`` and ``good_dims`` from the pure
           ``_axis_errors`` / ``_dim_errors`` results.
        2. Calls ``apply_defaults`` to fill ``t_start``/``t_end`` when
           the axes are valid and T is present.
        3. Sets ``validation_errors`` to ``compute_errors()``.
        4. Calls ``_get_output_path`` to refresh output filename strings.

        Never raises. Slice 5 made validation symmetric — failures are
        always surfaced via ``validation_errors`` and the boolean flags.
        Programmer-error mutators (``select_temporal_range``,
        ``change_axes``, ``change_dim_res``) still raise directly on
        invalid input at the entry point.
        """
        self.good_axes = not self._axis_errors()
        self.good_dims = not self._dim_errors()
        self.apply_defaults()
        self.validation_errors = self.compute_errors()
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
        elif self.extension in ('.tif', '.tiff'):
            try:
                data = tifffile.memmap(self.filepath)
            except Exception:
                try:
                    data = tifffile.imread(self.filepath)
                except Exception as read_exc:
                    message = f'Failed to read TIFF file {self.filepath}: {read_exc}'
                    logger.error(message)
                    raise ValueError(message) from read_exc
        else:
            message = f'Filetype {self.extension} not supported. Please convert to .nd2 or .tif.'
            logger.error(message)
            raise ValueError(message)
        self.dtype = data.dtype
        return data

    def _get_output_path(self):
        """
        Generates output paths for the processed image file using the configured output naming strategy.

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
        - Temporal range information is added for the "detailed" naming strategy if the 'T' (time) axis is present.
        - If any dimensional resolution is `None`, the string 'None' is used in the filename for the "detailed" strategy.
        """
        if self.output_naming not in ('detailed', 'stable'):
            raise ValueError(f"Unsupported output naming strategy '{self.output_naming}'")

        if self.output_naming == 'stable':
            output_name = f'{self.filename_no_ext}'
        else:
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
        if data.ndim != len(axes):
            if 'T' in axes and data.ndim == len(axes) - 1:
                data = np.expand_dims(data, axis=axes.index('T'))
            else:
                message = 'Data dimensions do not match axes'
                logger.error(message)
                raise ValueError(message)
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
            self.ome_output_path,
            data,
            bigtiff=True,
            metadata={"axes": axes},
            photometric="minisblack",
        )

        ome_xml = tifffile.tiffcomment(self.ome_output_path)
        ome = ome_types.from_xml(ome_xml)
        ome.images[0].pixels.physical_size_x = self.dim_res['X']
        ome.images[0].pixels.physical_size_y = self.dim_res['Y']
        ome.images[0].pixels.physical_size_z = self.dim_res['Z']
        ome.images[0].pixels.time_increment = self.dim_res['T']
        def _normalize_value(value):
            if isinstance(value, np.generic):
                return value.item()
            return value

        provenance = {
            "source_axes": self.axes,
            "output_axes": axes,
            "dim_res": {key: _normalize_value(val) for key, val in self.dim_res.items()},
            "channel": self.ch,
            "t_start": self.t_start,
            "t_end": self.t_end,
        }
        ome.images[0].description = json.dumps(provenance, sort_keys=True)
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
        Thin constructor — pure-data, no I/O.

        Stores ``file_info`` and computes the path-derived attributes
        (``im_path``, ``screenshot_dir``, ``graph_dir``). Initializes
        all loaded-state attributes to None / empty defaults.

        Use ``ImInfo.from_file_info(file_info)`` for the standard
        construct-and-load entry point. Calling ``__init__`` alone
        leaves ``self.im`` as None and ``pipeline_paths`` empty;
        downstream stages will fail until ``load()`` runs.

        Parameters
        ----------
        file_info : FileInfo
            An instance of the FileInfo class, containing metadata and paths for the image file.
        """
        self.file_info = file_info
        self.im_path = file_info.ome_output_path
        self.screenshot_dir = os.path.join(self.file_info.output_dir, 'screenshots')
        self.graph_dir = os.path.join(self.file_info.output_dir, 'graphs')

        self.im = None
        self.dim_res: DimRes = {'X': None, 'Y': None, 'Z': None, 'T': None}
        self.axes = None
        self.new_axes = None
        self.shape = None
        self.ome_metadata = None
        self.file_axes = None
        self.file_shape = None
        self.no_z = True
        self.no_t = True
        self.pipeline_paths = {}

    @classmethod
    def from_file_info(cls, file_info: FileInfo) -> "ImInfo":
        """
        Construct an ``ImInfo`` and load it in one step.

        Equivalent to ``info = ImInfo(file_info); info.load()``.
        This is the canonical entry point — every production caller
        should use it. The thin ``__init__`` exists for tests that
        want to inspect path computations without the I/O cost of a
        full load.
        """
        instance = cls(file_info)
        instance.load()
        return instance

    def load(self) -> None:
        """
        Perform the I/O sequence: regen-on-stale, memmap, metadata, paths.

        Idempotent — calling ``load()`` twice re-loads the memmap and
        re-derives axes/shape/dim_res. Used by ``from_file_info`` and
        by tests that want to load explicitly after a thin construction.
        """
        file_info = self.file_info
        needs_regen = not os.path.exists(self.im_path)
        if not needs_regen:
            with tifffile.TiffFile(self.im_path) as tif:
                existing_axes = tif.series[0].axes
            if 'T' not in existing_axes and file_info.axes is not None and 'T' in file_info.axes:
                needs_regen = True
        if needs_regen:
            file_info.save_ome_tiff()
        self.im = tifffile.memmap(self.im_path)

        self._get_ome_metadata()
        self._check_axes_exist()

        self.pipeline_paths = {}
        self._create_output_paths()

    def _check_axes_exist(self):
        """
        Checks the existence of the Z and T dimensions in the image data.

        Updates the `no_z` and `no_t` flags based on whether the Z and T axes are present and have more than one slice or timepoint.

        Resets both flags to True at the top so the result reflects the
        current ``axes`` and ``shape`` (not stale state from a prior
        call). This makes the method idempotent and safe to re-invoke
        after axes mutation.
        """
        self.no_z = True
        self.no_t = True
        if 'Z' in self.axes and self.shape[self.axes.index('Z')] > 1:
            self.no_z = False
        if 'T' in self.axes and self.shape[self.axes.index('T')] > 1:
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

        If the OME-TIFF lacks a 'T' axis, one is added to the in-memory representation.
        Axes are normalized to canonical order (T, Z, Y, X) and singleton Z is squeezed.
        """
        with tifffile.TiffFile(self.im_path) as tif:
            self.file_axes = tif.series[0].axes
            self.file_shape = tif.series[0].shape
        self.im, self.axes = transform_to_axes(self.im, self.file_axes)
        self.new_axes = self.axes
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
        file_axes = None
        try:
            with tifffile.TiffFile(file_path) as tif:
                file_axes = tif.series[0].axes
        except Exception:
            file_axes = None
        if file_axes is None:
            return memmap
        return transform_to_axes(memmap, file_axes, target_axes=self.axes)[0]

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
        axes = self.new_axes or self.axes
        if axes is None:
            raise ValueError('Axes metadata is not initialized')
        if data is not None and len(axes) != data.ndim:
            if axes.startswith('T') and data.ndim == len(axes) - 1:
                data = data[np.newaxis, ...]
            elif 'T' not in axes and data.ndim == len(axes) + 1:
                axes = 'T' + axes
            else:
                raise ValueError('Data dimensions do not match axes')
        if data is None:
            if len(axes) != len(self.shape):
                raise ValueError('Shape does not match axes')
            tifffile.imwrite(
                output_path,
                shape=self.shape,
                dtype=dtype,
                bigtiff=True,
                metadata={"axes": axes},
                photometric="minisblack",
            )
            dtype_name = np.dtype(dtype).name if dtype is not None else 'float'
        else:
            tifffile.imwrite(
                output_path,
                data,
                bigtiff=True,
                metadata={"axes": axes},
                photometric="minisblack",
            )
            dtype_name = data.dtype.name
        ome = ome_types.from_xml(tifffile.tiffcomment(output_path))
        ome.images[0].description = description
        if self.dim_res.get('X') is not None:
            ome.images[0].pixels.physical_size_x = self.dim_res['X']
        if self.dim_res.get('Y') is not None:
            ome.images[0].pixels.physical_size_y = self.dim_res['Y']
        if self.dim_res.get('Z') is not None:
            ome.images[0].pixels.physical_size_z = self.dim_res['Z']
        if self.dim_res.get('T') is not None:
            ome.images[0].pixels.time_increment = self.dim_res['T']

        if dtype_name == 'float64':
            dtype_name = 'double'
        if dtype_name == 'float32':
            dtype_name = 'float'
        ome.images[0].pixels.type = dtype_name
        ome_xml = ome.to_xml()
        tifffile.tiffcomment(output_path, ome_xml)
        if return_memmap:
            return self.get_memmap(output_path, read_mode=read_mode)
