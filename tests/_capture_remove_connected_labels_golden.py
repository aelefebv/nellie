"""One-shot capture of the dense ``_remove_connected_label_pixels_impl`` output.

Rerun if the input fixture (``tests/fixtures/yeast_3d_t0_to_1.ome.tif``) or
the dense impl in ``nellie/segmentation/networking.py`` changes; the output
``.npy`` is committed for regression-test consumption by
``test_remove_connected_label_pixels_matches_golden_3d`` (in
``tests/test_networking.py``).

Usage::

    python tests/_capture_remove_connected_labels_golden.py

Mirrors the pipeline's per-frame setup: runs Filter then Label on yeast 3D
frame 0 to produce ``im_instance_label``, builds a CPU ``Network``
(via the same ``_build_cpu_network`` helper used in
``tests/test_networking.py``), skeletonizes frame 0 the same way
``_run_frame_backend`` does, then calls the dense ``_impl`` directly with
``np`` and ``scipy.ndimage`` as the ``xp`` / ``ndi`` args.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi_cpu

from nellie.im_info import load_image
from nellie.segmentation.filtering import Filter, FrangiConfig
from nellie.segmentation.labelling import Label, LabelConfig
from nellie.segmentation.networking import Network, NetworkConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_3D_PATH = REPO_ROOT / "tests" / "fixtures" / "yeast_3d_t0_to_1.ome.tif"
GOLDEN_PATH = REPO_ROOT / "tests" / "fixtures" / "remove_connected_labels_3d_golden.npy"


def _build_cpu_network(info, **kwargs):
    """Mirror ``tests/test_networking.py::_build_cpu_network``.

    Construct a Network without running ``run()``; allocate memory and pin
    the backend to CPU so we can call internal methods directly.
    """
    kwargs.setdefault("device", "cpu")
    net = Network(info, NetworkConfig(**kwargs), num_t=2)
    net._set_backend("cpu")
    net._set_low_memory(kwargs.get("low_memory", False))
    net._allocate_memory()
    return net


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="capture_golden_") as tmp_dir:
        workdir = Path(tmp_dir)
        # Copy the fixture so ImInfo's writes land in the temp dir, not the
        # committed ``tests/fixtures/`` directory.
        local_source = workdir / FIXTURE_3D_PATH.name
        shutil.copy(FIXTURE_3D_PATH, local_source)
        info = load_image(local_source)

        # Pipeline cascade: Filter -> Label, both pinned to CPU for
        # deterministic output. Network's per-frame backend reads the
        # ``im_instance_label`` memmap that Label produces.
        flt = Filter(info, FrangiConfig(device="cpu"), num_t=2)
        flt.run()
        lbl = Label(info, LabelConfig(device="cpu"), num_t=2)
        lbl.run()

        net = _build_cpu_network(info)
        assert net.label_memmap is not None  # populated by _allocate_memory
        label_frame = np.asarray(net.label_memmap[0]).copy()
        skel_frame = net._skeletonize(label_frame)

        cleaned = net._remove_connected_label_pixels_impl(skel_frame, np, ndi_cpu)

        # Drop memmap handles before tmp_dir cleanup (Windows-safe; harmless
        # elsewhere).
        net.skel_memmap = None
        net.pixel_class_memmap = None
        net.skel_relabelled_memmap = None
        net.label_memmap = None
        net.im_memmap = None
        net.im_frangi_memmap = None

    np.save(GOLDEN_PATH, cleaned)
    print(
        f"Wrote {GOLDEN_PATH} "
        f"(shape={cleaned.shape}, dtype={cleaned.dtype}, "
        f"size={GOLDEN_PATH.stat().st_size} bytes)"
    )


if __name__ == "__main__":
    main()
