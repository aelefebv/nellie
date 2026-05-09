"""
``ImageJExtractor`` — ``MetadataExtractor`` for ImageJ-flavored TIFFs
that carry the ``physicalsizex`` key.

Mirrors the legacy ``FileInfo._get_imagej_metadata`` exactly: pulls
``physicalsizex/y`` for X/Y, ``spacing`` for Z, ``finterval`` for T.
Each key is independently optional; missing keys produce ``None``.

For ImageJ TIFFs that LACK ``physicalsizex``, see
``ImageJTifTagExtractor`` (which combines this extractor's logic with
the raw-TIFF tag fallback).
"""

from __future__ import annotations

from dataclasses import dataclass

from nellie.im_info.types import DimRes


@dataclass
class ImageJExtractor:
    """Per-format metadata extractor for ImageJ TIFFs with ``physicalsizex``."""
    imagej_meta: dict
    axes: str | None
    shape: tuple[int, ...] | None
    # Widened to ``str | None`` for Protocol invariance compatibility;
    # default value is the discriminator ``'imagej'``.
    metadata_type: str | None = 'imagej'

    def parse_dim_res(self) -> DimRes:
        """Pull X/Y from ``physicalsize{x,y}``, Z from ``spacing``, T from ``finterval``."""
        m = self.imagej_meta
        return {
            'X': m.get('physicalsizex'),
            'Y': m.get('physicalsizey'),
            'Z': m.get('spacing'),
            'T': m.get('finterval'),
        }
