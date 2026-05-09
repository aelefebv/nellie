"""
``RawTiffTagExtractor`` — ``MetadataExtractor`` for raw TIFFs that
carry neither OME-XML nor ImageJ metadata.

Mirrors the legacy ``FileInfo._get_tif_tags_metadata`` exactly when
called from the ``metadata_type is None`` branch — starts with an
all-None ``DimRes`` and overwrites entries from the TIFF tags.

The ``metadata_type`` discriminator is ``None`` per the existing
``FileInfo`` convention (the dispatcher's "raw TIFF" branch was keyed
off ``metadata_type is None`` rather than a string tag).
"""

from __future__ import annotations

from dataclasses import dataclass

from nellie.im_info.extractors._tif_tags import apply_tif_tags
from nellie.im_info.types import DimRes


@dataclass
class RawTiffTagExtractor:
    """Per-format extractor for plain TIFFs (no OME, no ImageJ)."""
    tags: dict
    axes: str | None
    shape: tuple[int, ...] | None
    # Typed as ``str | None`` (not the literal ``None``) for invariance
    # compatibility with the Protocol's ``metadata_type: str | None``
    # field. The default value ``None`` preserves the legacy
    # ``FileInfo`` convention (raw TIFF dispatch was keyed off
    # ``metadata_type is None``).
    metadata_type: str | None = None

    def parse_dim_res(self) -> DimRes:
        """Build an all-None ``DimRes`` then layer in TIFF-tag values via ``apply_tif_tags``."""
        result: DimRes = {'X': None, 'Y': None, 'Z': None, 'T': None}
        apply_tif_tags(result, self.tags, self.axes)
        return result
