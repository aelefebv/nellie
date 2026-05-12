from pathlib import Path

from .types import DimRes
from .verifier import (
    CSVS_ONLY_PRESET,
    DROPPABLE_KEYS,
    FileInfo,
    ImInfo,
    KEEP_EVERYTHING_PRESET,
    MASKS_AND_CSVS_PRESET,
)


def load_image(
    path: str | Path,
    *,
    output_dir: str | None = None,
    output_naming: str = "detailed",
) -> ImInfo:
    """
    Load an image from ``path`` and return a fully-loaded ``ImInfo``.

    The canonical one-call entry point for programmatic image loading.
    Equivalent to::

        file_info = FileInfo(path, output_dir=output_dir, output_naming=output_naming)
        file_info.find_metadata()
        file_info.load_metadata()
        return ImInfo.from_file_info(file_info)

    The napari fileselect widget intentionally splits these steps so
    the UI can surface auto-detected axes for user override before
    triggering ``ImInfo`` construction; programmatic callers should use
    this function instead.

    Parameters
    ----------
    path : str or Path
        Path to the image file (.tif, .tiff, or .nd2).
    output_dir : str, optional
        Override the output directory. Defaults to
        ``<input_dir>/nellie_output``.
    output_naming : str, optional
        Output naming strategy: ``"detailed"`` (default) bakes axes +
        ``dim_res`` into the filename; ``"stable"`` preserves the
        input filename.

    Returns
    -------
    ImInfo
        A fully-loaded ``ImInfo`` with ``axes``, ``dim_res``,
        ``pipeline_paths`` populated and the canonical OME-TIFF
        memmapped.
    """
    file_info = FileInfo(
        str(path), output_dir=output_dir, output_naming=output_naming
    )
    file_info.find_metadata()
    file_info.load_metadata()
    return ImInfo.from_file_info(file_info)


__all__ = [
    "FileInfo",
    "ImInfo",
    "DimRes",
    "load_image",
    "DROPPABLE_KEYS",
    "KEEP_EVERYTHING_PRESET",
    "MASKS_AND_CSVS_PRESET",
    "CSVS_ONLY_PRESET",
]
