"""
Per-format metadata extractors for ``FileInfo``.

Layout:

- ``protocol.py`` defines the structural ``MetadataExtractor`` Protocol
  that every concrete extractor satisfies.
- One file per format: ``ome.py``, ``imagej.py``, ``imagej_tif_tags.py``,
  ``nd2.py``, ``raw_tiff.py``.
- ``factory.py`` exposes ``detect_extractor(path)``, the single
  classifier+constructor entry point used by ``FileInfo.find_metadata``.
- ``_tif_tags.py`` carries the shared TIFF-tag projection helper used
  by ``RawTiffTagExtractor`` and ``ImageJTifTagExtractor`` (the
  fallback case where the imagej dict lacks ``physicalsizex``).
"""

from nellie.im_info.extractors.factory import detect_extractor
from nellie.im_info.extractors.imagej import ImageJExtractor
from nellie.im_info.extractors.imagej_tif_tags import ImageJTifTagExtractor
from nellie.im_info.extractors.nd2 import Nd2Extractor
from nellie.im_info.extractors.ome import OmeExtractor
from nellie.im_info.extractors.protocol import MetadataExtractor
from nellie.im_info.extractors.raw_tiff import RawTiffTagExtractor

__all__ = [
    'MetadataExtractor',
    'OmeExtractor',
    'ImageJExtractor',
    'ImageJTifTagExtractor',
    'Nd2Extractor',
    'RawTiffTagExtractor',
    'detect_extractor',
]
