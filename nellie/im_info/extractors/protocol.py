"""
``MetadataExtractor`` Protocol — the structural seam shared by every
per-format extractor in this subpackage.

Per PEP 544 (structural typing): any class whose attribute and method
signatures match this Protocol is a valid ``MetadataExtractor`` without
needing to inherit from it. The 5 concrete extractors
(``OmeExtractor``, ``ImageJExtractor``, ``ImageJTifTagExtractor``,
``Nd2Extractor``, ``RawTiffTagExtractor``) all satisfy this Protocol
without an ``isinstance`` check or a base class.

The Protocol fields ``axes`` / ``shape`` carry the source-reader's
view of the file's axis layout; ``parse_dim_res`` projects the raw
format-specific metadata into the canonical ``DimRes`` 4-key dict that
``FileInfo`` consumes.
"""

from __future__ import annotations

from typing import Protocol

from nellie.im_info.types import DimRes


class MetadataExtractor(Protocol):
    """Structural type for per-format metadata extractors.

    Attributes
    ----------
    axes : str | None
        Axes string for the file (e.g. 'TZYX'), normalized via
        ``infer_t_axis`` by the factory before construction.
    shape : tuple[int, ...] | None
        Shape of the file's primary series.
    metadata_type : str | None
        Extractor's discriminator string. One of
        ``'ome'``, ``'imagej''``, ``'imagej_tif_tags'``, ``'nd2'``, or
        ``None`` (raw TIFF, by existing FileInfo convention).

    Methods
    -------
    parse_dim_res() -> DimRes
        Project the per-format raw metadata into the canonical
        ``DimRes`` 4-key dict (X/Y in microns, Z in microns, T in
        seconds; any missing axis is ``None``).
    """

    axes: str | None
    shape: tuple[int, ...] | None
    metadata_type: str | None

    def parse_dim_res(self) -> DimRes: ...
