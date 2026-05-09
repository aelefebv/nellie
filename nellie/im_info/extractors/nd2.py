"""
``Nd2Extractor`` — ``MetadataExtractor`` for Nikon ND2 files.

Mirrors the legacy ``FileInfo._get_nd2_metadata`` exactly. ND2's
metadata is split across two surfaces:

- ``recorded_data`` — a dict-of-lists from ``nd2.events(orient='list')``
  carrying per-frame timestamps. ``T`` is the **median** of timestamp
  diffs (not first-diff or mean) to be robust against a single
  long-tail outlier.
- ``root_meta`` — the structured metadata object from
  ``nd2.metadata``. X/Y/Z come from ``volume.axesCalibration`` (a
  3-tuple of microns), with a fallback chain through
  ``channels[0].volume.axesCalibration`` because some ND2s organize
  the calibration under the channel rather than the file root.

Both ``root_meta`` and (each entry it traverses) can be either a
``dict`` or an attribute-bearing object — the legacy code used
``isinstance``-aware getters that this extractor preserves verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from nellie.im_info.types import DimRes


@dataclass
class Nd2Extractor:
    """Per-format extractor for ND2.

    Parameters
    ----------
    root_meta : Any
        ``nd2.ND2File.metadata`` — either an attribute-bearing object
        (typical) or a dict (legacy fallback). The legacy
        ``_get_nd2_metadata`` accepted both shapes, so this extractor
        does too.
    recorded_data : dict
        ``nd2.ND2File.events(orient='list')`` — dict of per-frame
        recorded values. We read ``"Time [s]"``.
    """
    root_meta: Any
    recorded_data: dict
    axes: str | None
    shape: tuple[int, ...] | None
    # Widened to ``str | None`` for Protocol invariance compatibility;
    # default value is the discriminator ``'nd2'``.
    metadata_type: str | None = 'nd2'

    def parse_dim_res(self) -> DimRes:
        """Compute T from median frame-diff and X/Y/Z from axesCalibration with fallback."""
        result: DimRes = {'X': None, 'Y': None, 'Z': None, 'T': None}

        recorded_data = self.recorded_data or {}
        timestamps = recorded_data.get("Time [s]")
        if timestamps is not None:
            if len(timestamps) >= 2:
                diffs = np.diff(timestamps)
                result['T'] = float(np.median(diffs))
            else:
                result['T'] = None

        root_metadata = self.root_meta
        axes_calibration = None
        if root_metadata is not None:
            if isinstance(root_metadata, dict):
                volume = root_metadata.get("volume")
            else:
                volume = getattr(root_metadata, "volume", None)
            axes_calibration = getattr(volume, "axesCalibration", None)

        if axes_calibration is None and root_metadata is not None:
            if isinstance(root_metadata, dict):
                channels = root_metadata.get("channels")
            else:
                channels = getattr(root_metadata, "channels", None)
            if channels:
                channel = channels[0]
                if isinstance(channel, dict):
                    channel_volume = channel.get("volume")
                else:
                    channel_volume = getattr(channel, "volume", None)
                axes_calibration = getattr(channel_volume, "axesCalibration", None)

        if axes_calibration is not None:
            if len(axes_calibration) > 0:
                result['X'] = axes_calibration[0]
            if len(axes_calibration) > 1:
                result['Y'] = axes_calibration[1]
            if len(axes_calibration) > 2:
                result['Z'] = axes_calibration[2]

        return result
