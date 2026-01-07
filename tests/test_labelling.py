import numpy as np
from types import SimpleNamespace

from nellie.segmentation.labelling import Label


def _make_im_info(no_z=True):
    if no_z:
        shape = (1, 5, 5)
        axes = "TYX"
        z_res = None
    else:
        shape = (1, 3, 5, 5)
        axes = "TZYX"
        z_res = 1.0
    return SimpleNamespace(
        no_t=True,
        no_z=no_z,
        shape=shape,
        axes=axes,
        dim_res={"X": 1.0, "Y": 1.0, "Z": z_res, "T": 1.0},
    )


def test_label_ids_reset_per_frame():
    im_info = _make_im_info(no_z=True)
    labeler = Label(im_info, num_t=2, device="cpu")

    original = np.zeros((5, 5), dtype=np.float32)
    original[1:4, 1:4] = 1.0
    frangi = original.copy()

    labels_0 = labeler._run_frame_full_volume(
        0,
        original,
        frangi,
        intensity_thresh=None,
        frangi_thresh=0.5,
    )
    labels_1 = labeler._run_frame_full_volume(
        1,
        original,
        frangi,
        intensity_thresh=None,
        frangi_thresh=0.5,
    )

    assert labels_0 is not None
    assert labels_1 is not None
    assert labels_0.max() == 1
    assert labels_1.max() == 1
    assert set(np.unique(labels_0)) <= {0, 1}
    assert set(np.unique(labels_1)) <= {0, 1}


def test_masking_does_not_mutate_inputs():
    im_info = _make_im_info(no_z=True)
    labeler = Label(im_info, num_t=1, device="cpu")

    original = np.zeros((5, 5), dtype=np.float32)
    original[1:4, 1:4] = 1.0
    frangi = original.copy()

    original_copy = original.copy()
    frangi_copy = frangi.copy()

    labels = labeler._run_frame_full_volume(
        0,
        original,
        frangi,
        intensity_thresh=0.5,
        frangi_thresh=0.5,
    )

    assert labels is not None
    assert np.array_equal(original, original_copy)
    assert np.array_equal(frangi, frangi_copy)
