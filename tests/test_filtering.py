"""Characterization tests for ``nellie.segmentation.filtering.Filter``.

The full test suite is built out in slice 2 (issue #53). Slice 1 only
ships the end-to-end smoke test that proves the framework, the fixtures,
and the Filter wiring all hang together.
"""

from __future__ import annotations

from nellie.segmentation.filtering import Filter


def test_filter_runs_end_to_end(imageinfo_3d) -> None:
    """Filter completes on the 3D fixture and writes an output of matching shape."""
    filt = Filter(imageinfo_3d, num_t=2, device="cpu")
    filt.run()

    assert filt.frangi_memmap is not None
    assert filt.frangi_memmap.shape == imageinfo_3d.shape
