"""
Private helper shared by ``RawTiffTagExtractor`` and
``ImageJTifTagExtractor``: compute X/Y/Z/T entries from a raw TIFF
``tags`` dict (the output of ``tif.pages[0].tags._dict``).

Mirrors the legacy ``FileInfo._get_tif_tags_metadata`` logic exactly:

- X/Y from XResolution/YResolution, scaled by ResolutionUnit
  (CENTIMETER → ×1e4, INCH → ×25400, NONE/absent → no scale).
- Z from ZResolution, but only when ``axes`` contains 'Z'.
- T from FrameRate, but only when ``axes`` contains 'T'.

The function MUTATES ``result`` in place rather than returning a new
dict so it can be used both by ``RawTiffTagExtractor`` (which starts
with an all-None ``DimRes``) and by ``ImageJTifTagExtractor`` (which
starts with the imagej-derived dict and layers raw tags on top).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tifffile import tifffile

if TYPE_CHECKING:
    from nellie.im_info.types import DimRes


def apply_tif_tags(
    result: "DimRes",
    tags: dict,
    axes: str | None,
) -> None:
    """Mutate ``result`` with X/Y/Z/T values pulled from the TIFF tags dict.

    Parameters
    ----------
    result : DimRes
        Dict to mutate. Existing entries (e.g. populated by an earlier
        imagej-meta pass) are overwritten when the corresponding tag is
        present.
    tags : dict
        Raw TIFF tag dict — ``tif.pages[0].tags._dict``. Values are
        ``TiffTag`` objects with ``.name`` and ``.value`` attributes.
    axes : str | None
        Axes string for the file (used to gate Z/T extraction). The
        Z/T extraction only fires when the corresponding letter is in
        this string.
    """
    tag_names = {tag_value.name: tag_code for tag_code, tag_value in tags.items()}

    if 'XResolution' in tag_names:
        result['X'] = tags[tag_names['XResolution']].value[1] \
                       / tags[tag_names['XResolution']].value[0]
    if 'YResolution' in tag_names:
        result['Y'] = tags[tag_names['YResolution']].value[1] \
                       / tags[tag_names['YResolution']].value[0]
    if 'ResolutionUnit' in tag_names:
        unit = tags[tag_names['ResolutionUnit']].value
        if unit == tifffile.RESUNIT.CENTIMETER:
            result['X'] *= 1E4  # type: ignore[operator]
            result['Y'] *= 1E4  # type: ignore[operator]
        elif unit == tifffile.RESUNIT.INCH:
            result['X'] *= 25400  # type: ignore[operator]
            result['Y'] *= 25400  # type: ignore[operator]
    if axes is not None and 'Z' in axes:
        if 'ZResolution' in tag_names:
            result['Z'] = 1 / tags[tag_names['ZResolution']].value[0]
    if axes is not None and 'T' in axes:
        if 'FrameRate' in tag_names:
            result['T'] = 1 / tags[tag_names['FrameRate']].value[0]
