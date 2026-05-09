"""
``ImageJTifTagExtractor`` — ``MetadataExtractor`` for ImageJ TIFFs
that LACK the ``physicalsizex`` key.

Combines the imagej-meta extraction (same logic as
``ImageJExtractor``) with the raw-TIFF tag fallback (same logic as
``RawTiffTagExtractor``). The legacy code shoehorned both into a
list-of-dicts (``self.metadata = [imagej_meta, tif_tags]``) and routed
two extractor calls in ``load_metadata``; this dataclass carries both
as proper named fields.

The ordering is preserved from the legacy: imagej-meta values are
written first (X/Y from ``physicalsizex/y``, Z from ``spacing``, T from
``finterval``), then the raw TIFF tag fallback layers in non-None
values for any axis the imagej dict didn't carry.
"""

from __future__ import annotations

from dataclasses import dataclass

from nellie.im_info.extractors._tif_tags import apply_tif_tags
from nellie.im_info.types import DimRes


@dataclass
class ImageJTifTagExtractor:
    """Per-format extractor for ImageJ TIFFs without ``physicalsizex``."""
    imagej_meta: dict
    tif_tags: dict
    axes: str | None
    shape: tuple[int, ...] | None
    # Widened to ``str | None`` for Protocol invariance compatibility;
    # default value is the discriminator ``'imagej_tif_tags'``.
    metadata_type: str | None = 'imagej_tif_tags'

    def parse_dim_res(self) -> DimRes:
        """Pull imagej-meta first, then layer in any TIFF-tag values via ``apply_tif_tags``."""
        m = self.imagej_meta
        result: DimRes = {
            'X': m.get('physicalsizex'),
            'Y': m.get('physicalsizey'),
            'Z': m.get('spacing'),
            'T': m.get('finterval'),
        }
        apply_tif_tags(result, self.tif_tags, self.axes)
        return result
