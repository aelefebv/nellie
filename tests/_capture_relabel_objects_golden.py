"""One-shot capture of the ``_relabel_objects`` golden triple.

Captures the ``label_frame`` input, the ``branch_skel_labels`` input, AND
the ``_relabel_objects`` golden output, so tests can run
``_relabel_objects`` against fixed inputs regardless of upstream platform
drift.

Rerun if the input fixture (``tests/fixtures/yeast_3d_t0_to_1.ome.tif``)
or the serial impl in ``nellie/segmentation/networking.py`` changes; the
``.npy`` files are committed for regression-test consumption by
``test_relabel_objects_matches_golden_3d`` (in ``tests/test_networking.py``).

Usage::

    python tests/_capture_relabel_objects_golden.py

Mirrors the per-frame setup of ``Network._run_frame_backend`` to produce
the exact ``(branch_skel_labels, label_frame)`` pair that
``_relabel_objects`` consumes in production: runs Filter then Label on
yeast 3D frame 0 to produce the label memmap, then walks the cleanup
chain ``_skeletonize`` → ``_remove_connected_label_pixels`` →
``_add_missing_skeleton_labels`` → ``_get_pixel_class`` →
``_get_branch_skel_labels`` (all on CPU, force_cpu where applicable).
The label frame, the branch skel labels, and the ``_relabel_objects``
output are all saved.

All three are cached because the upstream Filter+Label+skeleton+pixel-
class chain includes Frangi vesselness (floating-point SIMD-sensitive)
and connected-component labeling, which produce slightly different
intermediate outputs across platforms (macOS / Linux / Windows) for
byte-identical TIFF input. PRD #168 Slice 1 originally cached only the
output and CI failed on Linux + Windows for exactly this reason — this
capture caches the inputs alongside the golden so the snapshot test
exercises only the platform-deterministic ``_relabel_objects`` algorithm
on a fixed integer/scaling input.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np

from nellie.im_info import load_image
from nellie.segmentation.filtering import Filter, FrangiConfig
from nellie.segmentation.labelling import Label, LabelConfig
from nellie.segmentation.networking import Network, NetworkConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_3D_PATH = REPO_ROOT / "tests" / "fixtures" / "yeast_3d_t0_to_1.ome.tif"
INPUT_PATH = REPO_ROOT / "tests" / "fixtures" / "relabel_objects_3d_input.npy"
BRANCH_PATH = REPO_ROOT / "tests" / "fixtures" / "relabel_objects_3d_branch.npy"
GOLDEN_PATH = REPO_ROOT / "tests" / "fixtures" / "relabel_objects_3d_golden.npy"


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
        # ``im_instance_label`` and ``im_frangi`` memmaps these produce.
        flt = Filter(info, FrangiConfig(device="cpu"), num_t=2)
        flt.run()
        lbl = Label(info, LabelConfig(device="cpu"), num_t=2)
        lbl.run()

        net = _build_cpu_network(info)
        assert net.label_memmap is not None  # populated by _allocate_memory
        # Mirror ``_run_frame_backend`` for t=0:
        label_frame = np.asarray(net.label_memmap[0]).copy()
        frangi_frame = np.asarray(net.im_frangi_memmap[0])
        skel = net._skeletonize(label_frame)
        skel = net._remove_connected_label_pixels(skel)
        skel = net._add_missing_skeleton_labels(skel, label_frame, frangi_frame)
        skel_pre_cpu = (skel > 0) * label_frame
        pixel_class = net._get_pixel_class(skel_pre_cpu, force_cpu=True)
        branch_skel_labels = net._get_branch_skel_labels(pixel_class, force_cpu=True)

        # Save inputs first so the golden is exactly what the snapshot
        # test will reproduce from these saved arrays.
        np.save(INPUT_PATH, label_frame.astype(np.int32, copy=False))
        np.save(BRANCH_PATH, branch_skel_labels.astype(np.int32, copy=False))

        golden = net._relabel_objects(branch_skel_labels, label_frame)

        # Drop memmap handles before tmp_dir cleanup (Windows-safe; harmless
        # elsewhere).
        net.skel_memmap = None
        net.pixel_class_memmap = None
        net.skel_relabelled_memmap = None
        net.label_memmap = None
        net.im_memmap = None
        net.im_frangi_memmap = None

    np.save(GOLDEN_PATH, golden)
    print(
        f"Wrote {INPUT_PATH} "
        f"(shape={label_frame.shape}, dtype={np.int32}, "
        f"size={INPUT_PATH.stat().st_size} bytes)"
    )
    print(
        f"Wrote {BRANCH_PATH} "
        f"(shape={branch_skel_labels.shape}, dtype={np.int32}, "
        f"size={BRANCH_PATH.stat().st_size} bytes)"
    )
    print(
        f"Wrote {GOLDEN_PATH} "
        f"(shape={golden.shape}, dtype={golden.dtype}, "
        f"size={GOLDEN_PATH.stat().st_size} bytes)"
    )


if __name__ == "__main__":
    main()
