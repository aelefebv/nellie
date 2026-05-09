"""
Factory for choosing the right ``MetadataExtractor`` for a given
file path.

Replaces the legacy two-stage dispatch in ``FileInfo`` (extension-based
``find_metadata`` + string-tag ``load_metadata`` switch) with a single
``detect_extractor(path)`` call. The file is opened ONCE for both
classification and raw-metadata reads — subsequent
``extractor.parse_dim_res()`` calls do not reopen the file.
"""

from __future__ import annotations

import os

import nd2
import ome_types
from tifffile import tifffile

from nellie.im_info.extractors.imagej import ImageJExtractor
from nellie.im_info.extractors.imagej_tif_tags import ImageJTifTagExtractor
from nellie.im_info.extractors.nd2 import Nd2Extractor
from nellie.im_info.extractors.ome import OmeExtractor
from nellie.im_info.extractors.protocol import MetadataExtractor
from nellie.im_info.extractors.raw_tiff import RawTiffTagExtractor
from nellie.im_info.types import infer_t_axis


def detect_extractor(filepath: str) -> MetadataExtractor:
    """
    Detect file format and return the appropriate ``MetadataExtractor``.

    Opens the file once to classify and to extract format-specific raw
    metadata + axes/shape. Subsequent calls to the returned extractor's
    ``parse_dim_res()`` do not reopen the file.

    Parameters
    ----------
    filepath : str
        Path to the input file. Extension is matched case-insensitively
        against ``.nd2``, ``.tif``, and ``.tiff``.

    Returns
    -------
    MetadataExtractor
        One of ``Nd2Extractor``, ``OmeExtractor``, ``ImageJExtractor``,
        ``ImageJTifTagExtractor``, or ``RawTiffTagExtractor``.

    Raises
    ------
    ValueError
        If the file extension is not supported.
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.nd2':
        with nd2.ND2File(filepath) as nd2_file:
            source_axes = ''.join(nd2_file.sizes.keys())
            shape = tuple(nd2_file.sizes.values())
            # ``nd2_file.events(orient='list')`` returns a ``DictOfLists``
            # that pyright treats as Mapping rather than dict; cast for
            # the dataclass annotation.
            return Nd2Extractor(
                root_meta=nd2_file.metadata,
                recorded_data=dict(nd2_file.events(orient='list')),
                axes=infer_t_axis(source_axes, shape),
                shape=shape,
            )
    elif ext in ('.tif', '.tiff'):
        with tifffile.TiffFile(filepath) as tif:
            source_axes = tif.series[0].axes
            shape = tif.series[0].shape
            axes = infer_t_axis(source_axes, shape)
            if tif.is_ome or tif.ome_metadata is not None:
                # ``tifffile.tiffcomment`` is typed as
                # ``str | bytes | None``; ``from_xml`` requires non-None.
                # In the ``is_ome``/``ome_metadata is not None`` branch
                # the comment is always present — narrow with an assert.
                ome_xml = tifffile.tiffcomment(filepath)
                assert ome_xml is not None
                return OmeExtractor(
                    ome=ome_types.from_xml(ome_xml),
                    axes=axes,
                    shape=shape,
                )
            elif tif.is_imagej:
                # ``tif.imagej_metadata`` is typed as ``dict | None``;
                # in the ``is_imagej`` branch it is always a dict.
                imagej_meta = tif.imagej_metadata
                assert imagej_meta is not None
                if 'physicalsizex' in imagej_meta:
                    return ImageJExtractor(
                        imagej_meta=imagej_meta,
                        axes=axes,
                        shape=shape,
                    )
                else:
                    return ImageJTifTagExtractor(
                        imagej_meta=imagej_meta,
                        tif_tags=tif.pages[0].tags._dict,  # type: ignore[union-attr]
                        axes=axes,
                        shape=shape,
                    )
            else:
                return RawTiffTagExtractor(
                    tags=tif.pages[0].tags._dict,  # type: ignore[union-attr]
                    axes=axes,
                    shape=shape,
                )
    else:
        raise ValueError('File type not supported')
