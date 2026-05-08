"""Characterization tests for ``nellie.segmentation.networking.Network``.

Pins the wiki-documented invariants on both the 3D and 2D paths:
- Output dtypes (``im_skel`` int32, ``im_pixel_class`` uint8,
  ``im_skel_relabelled`` uint32)
- ``im_skel`` carries branch IDs at non-junction skel voxels (0 at
  junction voxels and off-skeleton)
- ``im_pixel_class`` values lie in {0,1,2,3,4} and clip at 4 even with
  ≥5 skel neighbors
- ``im_skel_relabelled`` covers every voxel inside an object with a
  branch ID and is 0 outside any object
- ``_add_missing_skeleton_labels`` guarantees every label in
  ``im_instance_label`` ends up with at least one skel voxel
- ``_remove_connected_label_pixels_impl`` preserves boundary voxels
  even when ambiguous (synthetic 2D)
- ``_get_branch_skel_labels`` excludes pixel-class 4 from CC labeling
- ``_relabel_objects`` honors anisotropic ``sampling=self.scaling``
  (synthetic 3D where the nearest-seed answer flips between isotropic
  and anisotropic)
- Input memmaps (raw + Frangi + Label) are not mutated
- Full-volume vs low-memory chunked equivalence for ``_get_pixel_class``
  and ``_remove_connected_label_pixels``

The Frangi and Label memmaps that ``Network`` consumes are precomputed
once per session by ``conftest.frangi_*_path`` / ``conftest.label_*_path``;
per-test ImInfos copy both memmaps into a fresh working directory so
each test gets isolated ``im_skel`` / ``im_pixel_class`` /
``im_skel_relabelled`` targets.
"""

from __future__ import annotations

import gc
import hashlib
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage as ndi_cpu

from nellie.im_info.verifier import ImInfo
from nellie.segmentation.networking import Network


def _release_network(net: Network) -> None:
    """Drop a Network's memmap references and force gc.

    Mirrors ``test_labelling._release_label``. Required on Windows: the
    output memmaps can stay file-locked until the handles are dropped,
    blocking any later overwrite of the same paths.
    """
    net.skel_memmap = None
    net.pixel_class_memmap = None
    net.skel_relabelled_memmap = None
    net.label_memmap = None
    net.im_memmap = None
    net.im_frangi_memmap = None
    gc.collect()


def _run_network(info: ImInfo, **kwargs) -> dict[str, np.ndarray]:
    """Run ``Network`` on ``info`` and return copies of the on-disk outputs."""
    kwargs.setdefault("device", "cpu")
    net = Network(info, num_t=2, **kwargs)
    net.run()
    out = {
        "skel": np.asarray(net.skel_memmap).copy(),
        "pixel_class": np.asarray(net.pixel_class_memmap).copy(),
        "skel_relabelled": np.asarray(net.skel_relabelled_memmap).copy(),
    }
    _release_network(net)
    return out


def _build_cpu_network(info: ImInfo, **kwargs) -> Network:
    """Construct a Network without running ``run()``; allocate memory and set backend.

    Useful for tests that need to call internal methods (e.g.
    ``_remove_connected_label_pixels``, ``_relabel_objects``,
    ``_get_pixel_class_impl``) directly against real or synthetic inputs.
    Always pinned to CPU for deterministic behavior.
    """
    kwargs.setdefault("device", "cpu")
    net = Network(info, num_t=2, **kwargs)
    net._set_backend("cpu")
    net._set_low_memory(kwargs.get("low_memory", False))
    net._get_t()
    net._allocate_memory()
    return net


# Module-scoped: run Network once on the 3D fixture and share outputs
# across the read-only invariant tests (dtype, set membership, contracts).

@pytest.fixture(scope="module")
def network_outputs_3d(make_network_imageinfo_3d_module) -> dict[str, np.ndarray]:
    info = make_network_imageinfo_3d_module()
    return _run_network(info)


@pytest.fixture(scope="module")
def network_outputs_2d(make_network_imageinfo_2d_module) -> dict[str, np.ndarray]:
    info = make_network_imageinfo_2d_module()
    return _run_network(info)


@pytest.fixture(scope="module")
def label_volume_3d(make_network_imageinfo_3d_module) -> np.ndarray:
    """Module-scoped: copy of the 3D ``im_instance_label`` memmap.

    Used by tests that compare Network outputs against the upstream
    label volume (``_add_missing_skeleton_labels`` guarantee, per-object
    coverage of ``im_skel_relabelled``, etc.). Reading from a fresh
    ImInfo so the file handle is independent of any Network instance's
    memmap.
    """
    info = make_network_imageinfo_3d_module()
    return np.asarray(info.get_memmap(info.pipeline_paths["im_instance_label"])).copy()


# -------------------------------------------------------------------------
# 3D path — output dtype / contract assertions
# -------------------------------------------------------------------------


def test_network_runs_end_to_end_3d(network_outputs_3d, imageinfo_3d) -> None:
    assert network_outputs_3d["skel"].shape == imageinfo_3d.shape
    assert network_outputs_3d["pixel_class"].shape == imageinfo_3d.shape
    assert network_outputs_3d["skel_relabelled"].shape == imageinfo_3d.shape


def test_im_skel_dtype_int32_3d(network_outputs_3d) -> None:
    assert network_outputs_3d["skel"].dtype == np.int32


def test_im_pixel_class_dtype_and_value_set_3d(network_outputs_3d) -> None:
    """``im_pixel_class`` is uint8 with values strictly in {0,1,2,3,4}."""
    pc = network_outputs_3d["pixel_class"]
    assert pc.dtype == np.uint8
    unique = np.unique(pc)
    assert set(unique.tolist()).issubset({0, 1, 2, 3, 4}), (
        f"Unexpected pixel-class values: {unique.tolist()}"
    )


def test_im_skel_relabelled_dtype_uint32_3d(network_outputs_3d) -> None:
    assert network_outputs_3d["skel_relabelled"].dtype == np.uint32


def test_im_skel_carries_branch_ids_not_parent_labels_3d(network_outputs_3d) -> None:
    """``im_skel`` stores branch IDs (CC labels) at non-junction skel voxels.

    Pins the recently-corrected wiki contract: wherever pixel-class > 0
    and != 4, im_skel must be > 0; wherever pixel-class == 4 (junction)
    or == 0 (background), im_skel must be 0. This is the actual code
    behavior in ``_run_frame_backend`` (returns ``branch_skel_labels``,
    which is the CC of non-junction pixels, as the first tuple element
    written to ``skel_memmap``).
    """
    skel = network_outputs_3d["skel"]
    pc = network_outputs_3d["pixel_class"]

    non_junction_skel = (pc > 0) & (pc != 4)
    junction = pc == 4
    background = pc == 0

    # Sanity: the fixture exercises the categories we can pin reliably.
    # Junction presence varies across scipy/skimage versions (junctions
    # appear on macOS but not consistently on Linux/Windows for this
    # fixture), so we don't gate on `junction.any()`. The junction-branch
    # claim below is vacuously satisfied when junctions are absent — the
    # contract still holds, the platform just didn't exercise that arm.
    assert non_junction_skel.any(), "Fixture has no non-junction skeleton voxels"
    assert background.any(), "Fixture has no background voxels"

    assert (skel[non_junction_skel] > 0).all(), (
        "Non-junction skeleton voxels must carry a branch ID"
    )
    assert (skel[junction] == 0).all(), "Junction voxels must be 0 in im_skel"
    assert (skel[background] == 0).all(), "Background voxels must be 0 in im_skel"


def test_im_skel_relabelled_covers_every_object_voxel_3d(
    network_outputs_3d, label_volume_3d
) -> None:
    """Every voxel inside any object label gets a branch ID in im_skel_relabelled."""
    relabelled = network_outputs_3d["skel_relabelled"]
    object_mask = label_volume_3d > 0
    assert object_mask.any(), "Fixture has no labelled objects"
    assert (relabelled[object_mask] > 0).all(), (
        "Some object voxels were left unlabeled by _relabel_objects"
    )


def test_im_skel_relabelled_zero_outside_objects_3d(
    network_outputs_3d, label_volume_3d
) -> None:
    """Voxels outside any object label are 0 in im_skel_relabelled."""
    relabelled = network_outputs_3d["skel_relabelled"]
    outside = label_volume_3d == 0
    assert outside.any()
    assert (relabelled[outside] == 0).all(), (
        "im_skel_relabelled has nonzero values outside any object"
    )


def test_every_label_has_at_least_one_skel_voxel_3d(
    network_outputs_3d, label_volume_3d
) -> None:
    """``_add_missing_skeleton_labels`` guarantees every object label survives.

    For each timepoint, every nonzero label ID present in the upstream
    ``im_instance_label`` volume must have at least one voxel inside it
    where ``im_skel_relabelled > 0``. The patch step plants a seed at the
    Frangi-maximum within the label if skeletonization missed it.
    """
    relabelled = network_outputs_3d["skel_relabelled"]
    for t in range(label_volume_3d.shape[0]):
        labels_t = label_volume_3d[t]
        relabelled_t = relabelled[t]
        unique_labels = np.unique(labels_t)
        unique_labels = unique_labels[unique_labels != 0]
        for lab in unique_labels:
            obj_mask = labels_t == lab
            assert (relabelled_t[obj_mask] > 0).any(), (
                f"t={t} label={int(lab)} has no skel voxel after "
                f"_add_missing_skeleton_labels"
            )


def test_get_branch_skel_labels_excludes_junctions_3d(network_outputs_3d) -> None:
    """``_get_branch_skel_labels`` runs CC on ``(pc > 0) & (pc != 4)``.

    The downstream ``im_skel`` therefore must be 0 at every pixel-class-4
    voxel — pinned independently from
    ``test_im_skel_carries_branch_ids_not_parent_labels_3d`` to single
    out the junction-exclusion contract from the broader im_skel shape.
    """
    skel = network_outputs_3d["skel"]
    pc = network_outputs_3d["pixel_class"]
    junction_mask = pc == 4
    if not junction_mask.any():
        pytest.skip("Fixture has no junction voxels; nothing to assert")
    assert (skel[junction_mask] == 0).all(), (
        "im_skel has nonzero values at pixel-class-4 (junction) voxels"
    )


def test_input_memmaps_not_mutated_3d(make_network_imageinfo_3d) -> None:
    """Hash raw, Frangi, and Label inputs before/after Network.run()."""
    info = make_network_imageinfo_3d()
    raw_path = Path(info.im_path)
    frangi_path = Path(info.pipeline_paths["im_preprocessed"])
    label_path = Path(info.pipeline_paths["im_instance_label"])

    raw_before = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    frangi_before = hashlib.sha256(frangi_path.read_bytes()).hexdigest()
    label_before = hashlib.sha256(label_path.read_bytes()).hexdigest()

    _run_network(info)

    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == raw_before, (
        "Network mutated the raw input memmap"
    )
    assert hashlib.sha256(frangi_path.read_bytes()).hexdigest() == frangi_before, (
        "Network mutated the Frangi input memmap"
    )
    assert hashlib.sha256(label_path.read_bytes()).hexdigest() == label_before, (
        "Network mutated the Label input memmap"
    )


# -------------------------------------------------------------------------
# 3D path — equivalence under low_memory chunking
# -------------------------------------------------------------------------


def test_full_vs_chunked_pixel_class_equivalence_3d(make_network_imageinfo_3d) -> None:
    """Full-volume and chunked-low-memory runs produce identical ``im_pixel_class``.

    The chunked variant of ``_get_pixel_class`` uses a 1-voxel halo; the
    convolution kernel is 3×3×3, so the chunk-internal results must be
    byte-identical to the non-chunked version.
    """
    full = _run_network(make_network_imageinfo_3d(), low_memory=False)["pixel_class"]
    # Force multi-chunk processing on the small yeast-3d fixture (~910k voxels).
    chunked = _run_network(
        make_network_imageinfo_3d(), low_memory=True, max_chunk_voxels=50_000
    )["pixel_class"]
    assert np.array_equal(full, chunked), (
        "im_pixel_class differs between full-volume and chunked runs"
    )


def test_remove_connected_label_pixels_full_vs_chunked_equivalence_3d(
    make_network_imageinfo_3d,
) -> None:
    """Direct call: ``_remove_connected_label_pixels_impl`` (full) equals chunked output.

    Builds the same input the pipeline feeds into the cleanup step
    (skeletonized labels) and compares the full-volume implementation
    with the chunked dispatcher. The chunked path uses a 1-voxel halo,
    which exactly covers the 3×3×3 min/max neighborhood — output should
    be byte-identical.
    """
    info = make_network_imageinfo_3d()
    net = _build_cpu_network(info, low_memory=False, max_chunk_voxels=50_000)
    assert net.label_memmap is not None  # populated by _allocate_memory in helper
    label_frame = np.asarray(net.label_memmap[0]).copy()
    skel_frame = net._skeletonize(label_frame)

    full = net._remove_connected_label_pixels_impl(skel_frame, np, ndi_cpu)
    chunked = net._remove_connected_label_pixels_chunked(skel_frame)
    _release_network(net)

    assert np.array_equal(full, chunked), (
        "_remove_connected_label_pixels chunked output differs from full-volume"
    )


# -------------------------------------------------------------------------
# Targeted tests on synthetic inputs
# -------------------------------------------------------------------------


def test_pixel_class_clips_at_four_with_dense_neighborhood_3d(
    make_network_imageinfo_3d,
) -> None:
    """A 3×3×3 fully-skel cube has 26 neighbors at the center; clip→4.

    ``_get_pixel_class_impl`` convolves a 3×3×3 all-ones kernel over the
    binary skeleton mask; the center voxel of a fully-skel cube sums to
    27 (kernel includes the center weight), then clips to 4. Any value
    above 4 is wiki-documented to be clipped.
    """
    info = make_network_imageinfo_3d()
    net = _build_cpu_network(info, low_memory=False)
    skel = np.ones((3, 3, 3), dtype=np.int32)

    pc = net._get_pixel_class_impl(skel, np, ndi_cpu)
    _release_network(net)

    assert pc[1, 1, 1] == 4, (
        f"Pixel class at the center of a dense neighborhood was {pc[1, 1, 1]}, "
        "expected 4 (clipped)"
    )


def test_remove_connected_label_pixels_preserves_boundary_voxels_2d(
    make_network_imageinfo_2d,
) -> None:
    """Boundary skel voxels touching multiple labels are preserved; interior ones are removed.

    Synthetic 2D 5×5 input with three label-1/label-2 pairs:
    - row 0 (top edge): pair preserved (boundary)
    - row 2 (interior): pair removed (interior + ambiguous)
    - row 4 (bottom edge): pair preserved (boundary)

    Direct call to ``_remove_connected_label_pixels_impl`` via a
    Network configured against a 2D ImInfo so ``self.im_info.no_z``
    gives the 2D footprint.
    """
    info = make_network_imageinfo_2d()
    net = _build_cpu_network(info, low_memory=False)

    skel = np.array(
        [
            [1, 2, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 2, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 2],
        ],
        dtype=np.int32,
    )
    expected = np.array(
        [
            [1, 2, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 2],
        ],
        dtype=np.int32,
    )

    cleaned = net._remove_connected_label_pixels_impl(skel, np, ndi_cpu)
    _release_network(net)

    assert np.array_equal(cleaned, expected), (
        f"Boundary preservation broken; got\n{cleaned}\nexpected\n{expected}"
    )


def test_relabel_objects_uses_anisotropic_sampling_3d(
    make_network_imageinfo_3d,
) -> None:
    """``_relabel_objects`` honors ``sampling=self.scaling`` (anisotropic EDT).

    Synthetic 3D setup: shape (3, 6, 3), one object filling everything
    (label=1). Two branch seeds inside:
      - seed A (branch label 5) at (z=2, y=1, x=1)
      - seed B (branch label 7) at (z=1, y=4, x=1)
    Query voxel at (z=1, y=1, x=1):
      - Isotropic: dist(A) = 1, dist(B) = 3 → A wins (label 5)
      - Anisotropic with z=4.0, y=x=1.0: dist(A) = 4, dist(B) = 3 → B wins (label 7)
    The function must select the anisotropic answer (7) because it
    passes ``sampling=self.scaling`` to ``distance_transform_edt``.
    """
    info = make_network_imageinfo_3d()
    net = _build_cpu_network(info, low_memory=False)

    # Override scaling to the test-specific anisotropic spacing. The
    # constructor pulls ``self.scaling`` from ``im_info.dim_res``;
    # overriding here drives the EDT sampling argument.
    net.scaling = (4.0, 1.0, 1.0)  # type: ignore[assignment]

    label_frame = np.ones((3, 6, 3), dtype=np.int32)
    branch_skel_labels = np.zeros((3, 6, 3), dtype=np.int32)
    branch_skel_labels[2, 1, 1] = 5  # seed A (closer in voxel-distance, far in physical Z)
    branch_skel_labels[1, 4, 1] = 7  # seed B (farther in voxel-distance, closer in physical Y)

    relabelled = net._relabel_objects(branch_skel_labels, label_frame)
    _release_network(net)

    assert relabelled[1, 1, 1] == 7, (
        f"Anisotropic EDT picked the wrong nearest seed at (1,1,1): "
        f"got {int(relabelled[1, 1, 1])}, expected 7. "
        "If this is 5, _relabel_objects is using isotropic sampling."
    )


# -------------------------------------------------------------------------
# 2D path
# -------------------------------------------------------------------------


def test_network_runs_end_to_end_2d(network_outputs_2d, imageinfo_2d) -> None:
    assert network_outputs_2d["skel"].shape == imageinfo_2d.shape


def test_2d_output_invariants(
    network_outputs_2d, make_network_imageinfo_2d
) -> None:
    """2D mirror of the dtype / value-set / contract / unmutated suite."""
    skel = network_outputs_2d["skel"]
    pc = network_outputs_2d["pixel_class"]
    relabelled = network_outputs_2d["skel_relabelled"]

    # dtypes
    assert skel.dtype == np.int32
    assert pc.dtype == np.uint8
    assert relabelled.dtype == np.uint32

    # pixel-class value set
    unique = np.unique(pc)
    assert set(unique.tolist()).issubset({0, 1, 2, 3, 4}), (
        f"Unexpected 2D pixel-class values: {unique.tolist()}"
    )

    # im_skel branch-ID contract: nonzero on non-junction skel voxels;
    # zero on junctions and background.
    non_junction_skel = (pc > 0) & (pc != 4)
    background = pc == 0
    assert non_junction_skel.any(), "2D fixture has no non-junction skeleton voxels"
    assert (skel[non_junction_skel] > 0).all()
    assert (skel[background] == 0).all()

    # Input memmaps not mutated (raw + Frangi + Label).
    info = make_network_imageinfo_2d()
    raw_path = Path(info.im_path)
    frangi_path = Path(info.pipeline_paths["im_preprocessed"])
    label_path = Path(info.pipeline_paths["im_instance_label"])
    raw_before = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    frangi_before = hashlib.sha256(frangi_path.read_bytes()).hexdigest()
    label_before = hashlib.sha256(label_path.read_bytes()).hexdigest()

    _run_network(info)

    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == raw_before
    assert hashlib.sha256(frangi_path.read_bytes()).hexdigest() == frangi_before
    assert hashlib.sha256(label_path.read_bytes()).hexdigest() == label_before
