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
- ``_remove_connected_label_pixels`` preserves boundary voxels
  even when ambiguous (synthetic 2D)
- ``_get_branch_skel_labels`` excludes pixel-class 4 from CC labeling
- ``_relabel_objects`` honors anisotropic ``sampling=self.scaling``
  (synthetic 3D where the nearest-seed answer flips between isotropic
  and anisotropic)
- Input memmaps (raw + Frangi + Label) are not mutated
- Full-volume vs low-memory chunked equivalence for ``_get_pixel_class``

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
from nellie.segmentation.networking import Network, NetworkConfig


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
    net = Network(info, NetworkConfig(**kwargs), num_t=2)
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
    net = Network(info, NetworkConfig(**kwargs), num_t=2)
    net._set_backend("cpu")
    net._set_low_memory(kwargs.get("low_memory", False))
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

    Direct call to ``_remove_connected_label_pixels`` via a
    Network configured against a 2D ImInfo so ``labels.ndim`` gives the
    2D footprint.
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

    cleaned = net._remove_connected_label_pixels(skel)
    _release_network(net)

    assert np.array_equal(cleaned, expected), (
        f"Boundary preservation broken; got\n{cleaned}\nexpected\n{expected}"
    )


# -------------------------------------------------------------------------
# _remove_connected_label_pixels: snapshot + 3D synthetic edge cases.
#
# Pins the contract that Slice 2 of PRD #168 will preserve byte-for-byte
# when it replaces the dense ``_impl`` (and the chunked variant) with a
# sparse skeleton-coordinate scan. See
# ``wiki/decisions/0004-skel-boundary-preservation.md`` for the
# boundary-exemption ADR exercised by the 2D test above and Test 4 below.
# -------------------------------------------------------------------------

INPUT_REMOVE_CONNECTED_LABELS_3D = (
    Path(__file__).parent / "fixtures" / "remove_connected_labels_3d_input.npy"
)
GOLDEN_REMOVE_CONNECTED_LABELS_3D = (
    Path(__file__).parent / "fixtures" / "remove_connected_labels_3d_golden.npy"
)


def test_remove_connected_label_pixels_matches_golden_3d(
    make_network_imageinfo_3d,
) -> None:
    """Snapshot regression: sparse ``_remove_connected_label_pixels`` output on the cached input matches the committed golden.

    Loads a pre-captured skeletonized input fixture (saved by
    ``tests/_capture_remove_connected_labels_golden.py`` from yeast 3D
    frame 0) and feeds it directly into
    ``_remove_connected_label_pixels``. The Network instance is only
    needed for the unbound method call, not to regenerate the input.
    Slice 1 (#169) captured the golden against the dense ``_impl``;
    Slice 2 (#170) retargets this assertion to the new sparse top-level
    method — bit-for-bit equality is the regression bar.

    The input is cached because the upstream Filter+Label pipeline
    (Frangi vesselness + connected components) produces slightly
    different intermediate outputs across platforms (macOS / Linux /
    Windows) for byte-identical TIFF input. The cleanup algorithm
    itself is platform-deterministic on a fixed integer input, so
    caching the input lets this test exercise only the function under
    test.
    """
    info = make_network_imageinfo_3d()
    net = _build_cpu_network(info, low_memory=False)

    input_skel = np.load(INPUT_REMOVE_CONNECTED_LABELS_3D)
    cleaned = net._remove_connected_label_pixels(input_skel)
    _release_network(net)

    golden = np.load(GOLDEN_REMOVE_CONNECTED_LABELS_3D)
    assert np.array_equal(cleaned, golden), (
        "Sparse _remove_connected_label_pixels output drifted from the committed "
        "golden snapshot. If this drift is intentional, rerun "
        "tests/_capture_remove_connected_labels_golden.py and review the diff."
    )


def test_remove_connected_label_pixels_empty_3d(
    make_network_imageinfo_3d,
) -> None:
    """All-zeros input: no skeleton voxels, no ambiguity, output stays all-zeros."""
    info = make_network_imageinfo_3d()
    net = _build_cpu_network(info, low_memory=False)

    skel = np.zeros((3, 3, 3), dtype=np.int32)
    expected = np.zeros((3, 3, 3), dtype=np.int32)

    cleaned = net._remove_connected_label_pixels(skel)
    _release_network(net)

    assert np.array_equal(cleaned, expected), (
        f"Empty input should round-trip; got\n{cleaned}"
    )


def test_remove_connected_label_pixels_single_object_3d(
    make_network_imageinfo_3d,
) -> None:
    """Single connected label: only one positive value anywhere in the neighborhood, so no ambiguity."""
    info = make_network_imageinfo_3d()
    net = _build_cpu_network(info, low_memory=False)

    skel = np.zeros((3, 3, 3), dtype=np.int32)
    skel[1, 1, :] = 1  # single label-1 line through the center
    expected = skel.copy()

    cleaned = net._remove_connected_label_pixels(skel)
    _release_network(net)

    assert np.array_equal(cleaned, expected), (
        f"Single-label input should round-trip; got\n{cleaned}"
    )


def test_remove_connected_label_pixels_multi_object_touch_interior_3d(
    make_network_imageinfo_3d,
) -> None:
    """Two labels overlapping at interior voxels: every interior label-1 voxel and every label-2 voxel except the lone island gets zeroed.

    5×5×5: ``arr[1:3, 1:3, 1:3] = 1`` then ``arr[2:4, 2:4, 2:4] = 2`` (label 2
    last, so the (2,2,2) overlap voxel ends up label 2 in the input). All
    label-1 voxels are interior and see at least one label-2 neighbor in
    their 3×3×3 window, so they all zero out. Label-2 voxels at the
    seven positions adjacent to the label-1 cluster also zero out for the
    same reason. The lone exception is (3,3,3): its 3×3×3 neighborhood
    {2..4}×{2..4}×{2..4} contains only label-2 voxels (the (2,2,2) corner
    is label 2 by the overwrite ordering), so it is unambiguous and
    survives. Expected literal is hand-derived from the predicate.
    """
    info = make_network_imageinfo_3d()
    net = _build_cpu_network(info, low_memory=False)

    skel = np.zeros((5, 5, 5), dtype=np.int32)
    skel[1:3, 1:3, 1:3] = 1
    skel[2:4, 2:4, 2:4] = 2  # set last so (2,2,2) overlap resolves to label 2

    expected = np.zeros((5, 5, 5), dtype=np.int32)
    expected[3, 3, 3] = 2  # only voxel whose 3×3×3 sees only one positive label

    cleaned = net._remove_connected_label_pixels(skel)
    _release_network(net)

    assert np.array_equal(cleaned, expected), (
        f"Multi-object touch (interior) wrong; got\n{cleaned}\nexpected\n{expected}"
    )


def test_remove_connected_label_pixels_junction_on_boundary_3d(
    make_network_imageinfo_3d,
) -> None:
    """Two labels meeting at a voxel on the volume boundary: boundary exemption preserves both.

    4×4×4 with ``(0,1,1)=1`` and ``(0,1,2)=2``. Both voxels are on the z=0
    face, so the boundary mask exempts them even though the predicate
    would otherwise flag them ambiguous (each sees the other in its 3×3×3
    window). Expected: input unchanged.
    """
    info = make_network_imageinfo_3d()
    net = _build_cpu_network(info, low_memory=False)

    skel = np.zeros((4, 4, 4), dtype=np.int32)
    skel[0, 1, 1] = 1
    skel[0, 1, 2] = 2
    expected = skel.copy()

    cleaned = net._remove_connected_label_pixels(skel)
    _release_network(net)

    assert np.array_equal(cleaned, expected), (
        f"Boundary-junction voxel should be preserved; got\n{cleaned}\nexpected\n{expected}"
    )


def test_remove_connected_label_pixels_all_boundary_3d(
    make_network_imageinfo_3d,
) -> None:
    """Skeleton entirely on the 6 face-center voxels of a 3×3×3 (all boundary): every potentially-ambiguous voxel is exempt.

    Six face-center voxels with mixed labels (1 vs 2). Every voxel sees
    the others in its 3×3×3 window and would be flagged ambiguous, but
    every voxel has at least one coord at 0 or 2 (shape-1), so the
    boundary mask exempts all of them. Expected: input unchanged.
    """
    info = make_network_imageinfo_3d()
    net = _build_cpu_network(info, low_memory=False)

    skel = np.zeros((3, 3, 3), dtype=np.int32)
    # Mix the labels so every voxel sees a different positive label nearby;
    # the exemption is the only reason none of them get zeroed.
    skel[0, 1, 1] = 1
    skel[2, 1, 1] = 2
    skel[1, 0, 1] = 1
    skel[1, 2, 1] = 2
    skel[1, 1, 0] = 1
    skel[1, 1, 2] = 2
    expected = skel.copy()

    cleaned = net._remove_connected_label_pixels(skel)
    _release_network(net)

    assert np.array_equal(cleaned, expected), (
        f"All-boundary voxels should be preserved; got\n{cleaned}\nexpected\n{expected}"
    )


def test_remove_connected_label_pixels_high_density_3d(
    make_network_imageinfo_3d,
) -> None:
    """Every voxel of a 3×3×3 cube has the same label: no ambiguity anywhere.

    Single-label dense volume; ``min_labels == max_labels`` everywhere, so
    the predicate never fires regardless of boundary status. Expected:
    input unchanged.
    """
    info = make_network_imageinfo_3d()
    net = _build_cpu_network(info, low_memory=False)

    skel = np.ones((3, 3, 3), dtype=np.int32)
    expected = skel.copy()

    cleaned = net._remove_connected_label_pixels(skel)
    _release_network(net)

    assert np.array_equal(cleaned, expected), (
        f"High-density single-label input should round-trip; got\n{cleaned}"
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
# _relabel_objects: 3D synthetic edge cases.
#
# Pins the contract that Slice 2 of PRD #173 will preserve byte-for-byte
# when it threads the per-object EDT loop. See
# ``wiki/decisions/0005-relabel-objects-serialized-writeback.md`` for the
# serialized-writeback ADR exercised by the overlapping-bboxes test below
# (the race-condition pin). No realistic-data snapshot test: scipy's EDT
# tie-breaking with ``return_indices=True`` is platform-implementation-
# dependent (cross-platform-different "nearest" seed for equidistant
# ties), so a yeast-3D snapshot is not cross-platform-deterministic. The
# byte-for-byte regression bar for Slice 2 is the synthetic suite plus
# the explicit serial-vs-threaded equivalence test added by Slice 2
# itself (intra-platform deterministic).
# -------------------------------------------------------------------------


def test_relabel_objects_empty_3d(
    make_network_imageinfo_3d,
) -> None:
    """All-zeros input: no labels, no seeds, output is all-zero uint32."""
    info = make_network_imageinfo_3d()
    net = _build_cpu_network(info, low_memory=False)

    labels = np.zeros((3, 3, 3), dtype=np.int32)
    branch = np.zeros((3, 3, 3), dtype=np.int32)
    expected = np.zeros((3, 3, 3), dtype=np.uint32)

    relabelled = net._relabel_objects(branch, labels)
    _release_network(net)

    assert np.array_equal(relabelled, expected), (
        f"Empty input should produce all-zero output; got\n{relabelled}"
    )
    assert relabelled.dtype == np.uint32, (
        f"Output dtype should be uint32; got {relabelled.dtype}"
    )


def test_relabel_objects_single_object_one_seed_3d(
    make_network_imageinfo_3d,
) -> None:
    """Single object, single seed: every voxel of the object gets the seed's branch ID."""
    info = make_network_imageinfo_3d()
    net = _build_cpu_network(info, low_memory=False)

    labels = np.zeros((5, 5, 5), dtype=np.int32)
    labels[1:4, 1:4, 1:4] = 1  # 3x3x3 cube of label 1

    branch = np.zeros((5, 5, 5), dtype=np.int32)
    branch[2, 2, 2] = 7  # one seed at the center of the object

    expected = np.zeros((5, 5, 5), dtype=np.uint32)
    expected[1:4, 1:4, 1:4] = 7  # every label-1 voxel gets branch 7

    relabelled = net._relabel_objects(branch, labels)
    _release_network(net)

    assert np.array_equal(relabelled, expected), (
        f"Single-seed propagation wrong; got\n{relabelled}\nexpected\n{expected}"
    )


def test_relabel_objects_two_non_overlapping_bboxes_3d(
    make_network_imageinfo_3d,
) -> None:
    """Two objects with non-overlapping bboxes: each object's voxels get its seed's branch ID."""
    info = make_network_imageinfo_3d()
    net = _build_cpu_network(info, low_memory=False)

    labels = np.zeros((7, 7, 7), dtype=np.int32)
    labels[0:3, 0:3, 0:3] = 1
    labels[4:7, 4:7, 4:7] = 2

    branch = np.zeros((7, 7, 7), dtype=np.int32)
    branch[1, 1, 1] = 4  # seed for object 1
    branch[5, 5, 5] = 9  # seed for object 2

    expected = np.zeros((7, 7, 7), dtype=np.uint32)
    expected[0:3, 0:3, 0:3] = 4
    expected[4:7, 4:7, 4:7] = 9

    relabelled = net._relabel_objects(branch, labels)
    _release_network(net)

    assert np.array_equal(relabelled, expected), (
        f"Two-object propagation wrong; got\n{relabelled}\nexpected\n{expected}"
    )


def test_relabel_objects_two_overlapping_bboxes_3d(
    make_network_imageinfo_3d,
) -> None:
    """Pins that overlapping bboxes don't cause cross-contamination — label 1's bbox contains label 2's voxels but label 2's mask excludes them. Critical for the eventual threading rewrite.

    6×6×6. Label 1 fills the **outer shell** (every voxel where any
    coordinate is 0 or 5), so ``find_objects`` returns its bbox as the
    whole volume ``[0:6, 0:6, 0:6]``. Label 2 is a small interior cluster
    at ``[2:4, 2:4, 2:4]``; its bbox is ``[2:4, 2:4, 2:4]``. Label 1's
    bbox **contains** label 2's voxels, but the masks don't overlap (each
    voxel has exactly one label). Branch seeds at (0,0,0)=4 (inside label
    1's mask) and (3,3,3)=9 (inside label 2's mask). Expected: every
    shell voxel gets 4; every interior-cluster voxel gets 9; background
    stays 0. A future maintainer who parallelizes the writeback would
    break this test silently if they assumed bbox non-overlap.
    """
    info = make_network_imageinfo_3d()
    net = _build_cpu_network(info, low_memory=False)

    labels = np.zeros((6, 6, 6), dtype=np.int32)
    # Shell: label 1 wherever any coordinate is on the volume boundary.
    shell_mask = np.zeros((6, 6, 6), dtype=bool)
    shell_mask[0, :, :] = True
    shell_mask[5, :, :] = True
    shell_mask[:, 0, :] = True
    shell_mask[:, 5, :] = True
    shell_mask[:, :, 0] = True
    shell_mask[:, :, 5] = True
    labels[shell_mask] = 1
    # Interior cluster: label 2 at [2:4, 2:4, 2:4] (all 8 voxels are
    # interior, so they're disjoint from the shell).
    labels[2:4, 2:4, 2:4] = 2

    branch = np.zeros((6, 6, 6), dtype=np.int32)
    branch[0, 0, 0] = 4  # seed for label 1 (a shell voxel)
    branch[3, 3, 3] = 9  # seed for label 2 (an interior-cluster voxel)

    expected = np.zeros((6, 6, 6), dtype=np.uint32)
    expected[shell_mask] = 4
    expected[2:4, 2:4, 2:4] = 9

    relabelled = net._relabel_objects(branch, labels)
    _release_network(net)

    assert np.array_equal(relabelled, expected), (
        f"Overlapping-bbox propagation wrong; got\n{relabelled}\nexpected\n{expected}"
    )


def test_relabel_objects_sparse_label_ids_3d(
    make_network_imageinfo_3d,
) -> None:
    """Pins find_objects sparse-IDs handling (None entries for missing labels).

    6×6×6 with labels 1, 5, 17 only — gaps at 2-4 and 6-16.
    ``find_objects`` returns ``len == 17`` with ``None`` at indices for
    missing labels. The function must skip those without erroring.
    """
    info = make_network_imageinfo_3d()
    net = _build_cpu_network(info, low_memory=False)

    labels = np.zeros((6, 6, 6), dtype=np.int32)
    labels[0:2, 0:2, 0:2] = 1
    labels[2:4, 2:4, 2:4] = 5
    labels[4:6, 4:6, 4:6] = 17

    branch = np.zeros((6, 6, 6), dtype=np.int32)
    branch[0, 0, 0] = 11  # seed for label 1
    branch[3, 3, 3] = 22  # seed for label 5
    branch[5, 5, 5] = 33  # seed for label 17

    expected = np.zeros((6, 6, 6), dtype=np.uint32)
    expected[0:2, 0:2, 0:2] = 11
    expected[2:4, 2:4, 2:4] = 22
    expected[4:6, 4:6, 4:6] = 33

    relabelled = net._relabel_objects(branch, labels)
    _release_network(net)

    assert np.array_equal(relabelled, expected), (
        f"Sparse-IDs propagation wrong; got\n{relabelled}\nexpected\n{expected}"
    )


def test_relabel_objects_object_with_no_seeds_3d(
    make_network_imageinfo_3d,
) -> None:
    """Pins the no-seeds-in-object branch — output stays 0 for that label.

    Label 1 has voxels but ``branch_skel_labels`` is all zeros, so
    ``seed_mask.any()`` is False and the function ``continue``s without
    writing anything for that label.
    """
    info = make_network_imageinfo_3d()
    net = _build_cpu_network(info, low_memory=False)

    labels = np.zeros((5, 5, 5), dtype=np.int32)
    labels[1:4, 1:4, 1:4] = 1

    branch = np.zeros((5, 5, 5), dtype=np.int32)  # no seeds anywhere
    expected = np.zeros((5, 5, 5), dtype=np.uint32)  # nothing gets written

    relabelled = net._relabel_objects(branch, labels)
    _release_network(net)

    assert np.array_equal(relabelled, expected), (
        f"No-seeds object should leave output zero; got\n{relabelled}"
    )


def test_relabel_objects_serial_vs_threaded_equivalence_3d(
    make_network_imageinfo_3d,
) -> None:
    """Threading must not perturb output: serial and threaded paths produce byte-identical results.

    Runs the upstream pipeline on the test machine to get realistic
    ``(label_frame, branch_skel_labels)`` inputs, then calls
    ``_relabel_objects`` twice — once with ``low_memory=True`` (forces
    serial), once with ``low_memory=False`` (uses ``ThreadPoolExecutor``
    when ``len(work) > 1``). Asserts ``np.array_equal`` between the two
    outputs.

    Intra-platform deterministic by construction: both calls run on the
    same machine with the same scipy build, so any EDT tie-break
    non-determinism affects both equally. This is the regression bar
    that no cross-platform snapshot can be (see
    ``wiki/decisions/0005-relabel-objects-serialized-writeback.md``
    Consequences).
    """
    info = make_network_imageinfo_3d()
    net = _build_cpu_network(info, low_memory=False)
    assert net.label_memmap is not None
    assert net.im_frangi_memmap is not None

    # Run the upstream pipeline mirror of _run_frame_backend to get realistic input.
    label_frame = np.asarray(net.label_memmap[0]).copy()
    frangi_frame = np.asarray(net.im_frangi_memmap[0])
    skel = net._skeletonize(label_frame)
    skel = net._remove_connected_label_pixels(skel)
    skel = net._add_missing_skeleton_labels(skel, label_frame, frangi_frame)
    skel_pre_cpu = (skel > 0) * label_frame
    pixel_class = net._get_pixel_class(skel_pre_cpu, force_cpu=True)
    branch_skel_labels = net._get_branch_skel_labels(pixel_class, force_cpu=True)

    # Run both paths on the same machine.
    net.low_memory = True
    serial_output = net._relabel_objects(branch_skel_labels, label_frame)
    net.low_memory = False
    threaded_output = net._relabel_objects(branch_skel_labels, label_frame)
    _release_network(net)

    assert np.array_equal(serial_output, threaded_output), (
        "Threaded _relabel_objects output diverged from serial output. "
        "This indicates the threading rewrite perturbed observable behavior — "
        "verify that the writeback is still serialized through as_completed "
        "(see wiki/decisions/0005-relabel-objects-serialized-writeback.md)."
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


# -------------------------------------------------------------------------
# NetworkConfig validation (__post_init__)
# -------------------------------------------------------------------------

def test_network_config_default_constructs() -> None:
    NetworkConfig()


def test_network_config_rejects_bad_device() -> None:
    with pytest.raises(ValueError, match="device"):
        NetworkConfig(device="bogus")


@pytest.mark.parametrize("field", [
    "min_radius_um", "max_radius_um", "max_chunk_voxels",
])
def test_network_config_rejects_nonpositive_numeric(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        NetworkConfig(**{field: 0})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=field):
        NetworkConfig(**{field: -1})  # type: ignore[arg-type]


def test_network_config_rejects_inverted_radius_range() -> None:
    with pytest.raises(ValueError, match="min_radius_um"):
        NetworkConfig(min_radius_um=2.0, max_radius_um=1.0)


def test_network_config_equal_radii_ok() -> None:
    NetworkConfig(min_radius_um=0.5, max_radius_um=0.5)
