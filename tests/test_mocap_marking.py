import numpy as np
from types import SimpleNamespace

from nellie.segmentation.mocap_marking import Markers


def _make_im_info(no_z=True):
    if no_z:
        shape = (1, 9, 9)
        axes = "TYX"
        z_res = None
    else:
        shape = (1, 3, 9, 9)
        axes = "TZYX"
        z_res = 0.4
    return SimpleNamespace(
        no_t=True,
        no_z=no_z,
        shape=shape,
        axes=axes,
        dim_res={"X": 0.2, "Y": 0.2, "Z": z_res, "T": 1.0},
    )


def _setup_markers(im_info, intensity, labels, **kwargs):
    markers = Markers(im_info, num_t=1, device="cpu", **kwargs)
    markers.im_memmap = intensity
    markers.label_memmap = labels
    markers.shape = labels.shape
    markers._set_default_sigmas()
    return markers


def test_mocap_marking_low_memory_matches_full_2d():
    im_info = _make_im_info(no_z=True)
    intensity = np.zeros((1, 9, 9), dtype=np.float32)
    intensity[0, 4, 4] = 10.0
    labels = np.zeros((1, 9, 9), dtype=np.uint8)
    labels[0, 2:7, 2:7] = 1

    markers_full = _setup_markers(im_info, intensity, labels, num_sigma=3, low_memory=False)
    full_marker, full_dist, full_border = markers_full._run_frame_impl(0, low_memory=False)

    markers_low = _setup_markers(
        im_info,
        intensity,
        labels,
        num_sigma=3,
        low_memory=True,
        max_chunk_voxels=20,
    )
    low_marker, low_dist, low_border = markers_low._run_frame_impl(
        0, low_memory=True, chunk_voxels=20
    )

    assert np.array_equal(full_marker, low_marker)
    assert np.array_equal(full_dist, low_dist)
    assert np.array_equal(full_border, low_border)


def test_border_is_outside_mask():
    im_info = _make_im_info(no_z=True)
    markers = Markers(im_info, num_t=1, device="cpu")

    mask = np.zeros((7, 7), dtype=bool)
    mask[2:5, 2:5] = True
    _, border = markers._distance_im(mask)

    assert border.shape == mask.shape
    assert not np.any(border & mask)
