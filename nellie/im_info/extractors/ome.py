"""
``OmeExtractor`` — ``MetadataExtractor`` for OME-TIFF files.

Reads X/Y/Z physical sizes and the time increment off the first image's
pixels block in the OME-XML metadata tree (``ome.images[0].pixels``).
Mirrors the legacy ``FileInfo._get_ome_metadata`` exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nellie.im_info.types import DimRes


@dataclass
class OmeExtractor:
    """Per-format metadata extractor for OME-TIFF.

    Construct with the parsed ``ome_types.OME`` object plus the file's
    axes/shape (already normalized via ``infer_t_axis`` by the factory).
    """
    ome: Any
    axes: str | None
    shape: tuple[int, ...] | None
    # Typed as ``str | None`` (not just ``str``) for invariance
    # compatibility with the Protocol's ``metadata_type: str | None``
    # field. Default value is the discriminator ``'ome'``.
    metadata_type: str | None = 'ome'

    def parse_dim_res(self) -> DimRes:
        """Return X/Y/Z in microns + T in seconds from ``ome.images[0].pixels``."""
        pixels = self.ome.images[0].pixels
        return {
            'X': pixels.physical_size_x,
            'Y': pixels.physical_size_y,
            'Z': pixels.physical_size_z,
            'T': pixels.time_increment,
        }
