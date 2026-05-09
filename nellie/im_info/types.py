"""
Shared types and small helpers for the im_info subpackage.

Lives at this seam to break the circular import between
``nellie.im_info.verifier`` and ``nellie.im_info.extractors`` —
verifier imports the extractor factory and Protocol, while extractors
need ``DimRes`` (and the ``infer_t_axis`` helper that the factory uses
to prepend a missing leading T axis).

``verifier.py`` re-exports ``DimRes`` for backwards compat so existing
``from nellie.im_info.verifier import DimRes`` imports keep working.
"""

from __future__ import annotations

from typing import TypedDict


class DimRes(TypedDict):
    """Per-axis physical resolution: X/Y/Z in micrometers, T in seconds.

    All four keys are always present; values are ``None`` until populated
    by ``FileInfo.load_metadata`` (per-format extractor) or until the
    corresponding axis is determined to be absent from the file.
    """
    X: float | None
    Y: float | None
    Z: float | None
    T: float | None


def infer_t_axis(
    source_axes: str | None,
    source_shape: tuple[int, ...] | None,
) -> str | None:
    """
    Infer a leading T axis when the source library stripped it.

    Some readers (notably tifffile on certain inputs) return an axes
    string that omits a leading singleton T dim while keeping it in the
    shape — e.g. axes='ZYX' but shape=(1, 16, 512, 512). This helper
    detects that pattern and returns the prepended axes string.

    Parameters
    ----------
    source_axes : str or None
        Axes string from the source reader (e.g. 'ZYX', 'TZYX').
    source_shape : tuple of int or None
        Shape from the source reader.

    Returns
    -------
    str or None
        ``source_axes`` with 'T' prepended when the leading-singleton
        heuristic fires; otherwise ``source_axes`` unchanged. Returns
        ``None`` when either input is ``None``.
    """
    if source_axes is None or source_shape is None:
        return source_axes
    if 'T' in source_axes:
        return source_axes
    if len(source_shape) == len(source_axes) + 1 and source_shape[0] == 1:
        return 'T' + source_axes
    return source_axes
