"""Shared pytest fixtures for the nellie test suite.

Each ImInfo fixture copies its source file into a session-scoped tmp dir
before constructing ImInfo. ImInfo's constructor writes a sibling
``nellie_output/`` tree next to the input file; the copy keeps those
writes out of ``tests/fixtures/`` so the committed fixtures stay clean
across runs.

For pipeline stages downstream of Filter (Label, Network, ...), the
session-scoped ``frangi_*_path`` fixtures run Filter once per session and
expose the resulting ``im_preprocessed`` memmap path. Per-test factories
(``make_label_imageinfo_*``) copy that precomputed Frangi memmap into a
fresh ImInfo's pipeline tree, so each test gets an isolated working
directory (clean ``im_instance_label`` target, no Windows file-lock
issues) without paying the Filter cost per test.

The cascade extends one more layer for Network: ``label_*_path``
fixtures run Filter+Label once per session and expose the resulting
``im_instance_label`` memmap path; ``make_network_imageinfo_*``
per-test factories copy in BOTH the Frangi and Label memmaps.

Markers piggybacks on the same Filter+Label session caches:
``make_markers_imageinfo_*`` per-test factories copy in BOTH the Frangi
and Label memmaps unconditionally. Markers' inputs are a strict subset
of Network's (raw + Frangi only when ``use_im='frangi'`` + Label always),
but copying both memmaps lets a single factory shape serve both
``use_im='distance'`` and ``use_im='frangi'`` tests without
re-parametrizing.

The cascade extends one more layer for HuMomentTracking:
``markers_*_paths`` session-scoped fixtures run Markers once per session
on top of the Filter+Label caches and expose paths to BOTH the
``im_marker`` and ``im_distance`` memmaps; ``make_hu_imageinfo_*``
per-test factories copy in FOUR memmaps (Frangi + Label + Marker +
Distance) so each test gets an isolated working directory ready for
``HuMomentTracking(info)`` without re-running any upstream stage.

The cascade extends two more layers for VoxelReassigner: a Network
session cache (``network_*_path``) runs Filter+Label+Network once per
session and exposes the resulting ``im_skel_relabelled`` memmap, and a
Hu-output session cache (``hu_outputs_*_path``) runs
Filter+Label+Markers+Hu once per session and exposes the resulting
``flow_vector_array.npy``. ``make_voxel_reassign_imageinfo_*`` per-test
factories copy in THREE files (``im_instance_label`` from the Label
cache, ``im_skel_relabelled`` from the Network cache, and
``flow_vector_array.npy`` from the Hu-output cache) so each test gets an
isolated working directory ready for ``VoxelReassigner(info)`` without
re-running any upstream stage. The Hu output must be materialized
BEFORE constructing the VoxelReassigner because
``VoxelReassigner.__init__`` constructs two ``FlowInterpolator``
instances unconditionally (when not ``no_t``), and FlowInterpolator
loads ``flow_vector_array.npy`` immediately at construction.
"""

from __future__ import annotations

import gc
import shutil
from pathlib import Path

import pytest

from nellie.im_info.verifier import FileInfo, ImInfo
from nellie.segmentation.filtering import Filter, FrangiConfig
from nellie.segmentation.labelling import Label
from nellie.segmentation.mocap_marking import Markers
from nellie.segmentation.networking import Network
from nellie.tracking.hu_tracking import HuMomentTracking

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_3D_PATH = REPO_ROOT / "tests" / "fixtures" / "yeast_3d_t0_to_1.ome.tif"
FIXTURE_2D_PATH = REPO_ROOT / "tests" / "fixtures" / "yeast_2d_t0_to_1.ome.tif"


def _build_iminfo(source: Path, workdir: Path) -> ImInfo:
    dst = workdir / source.name
    shutil.copy(source, dst)
    file_info = FileInfo(str(dst))
    file_info.find_metadata()
    file_info.load_metadata()
    return ImInfo(file_info)


@pytest.fixture(scope="session")
def imageinfo_3d(tmp_path_factory: pytest.TempPathFactory) -> ImInfo:
    """Session-scoped ImInfo for the 3D yeast fixture (T=2, Z=17, Y=192, X=279)."""
    workdir = tmp_path_factory.mktemp("yeast_3d")
    return _build_iminfo(FIXTURE_3D_PATH, workdir)


@pytest.fixture(scope="session")
def imageinfo_2d(tmp_path_factory: pytest.TempPathFactory) -> ImInfo:
    """Session-scoped ImInfo for the 2D yeast fixture (T=2, Y=192, X=279)."""
    workdir = tmp_path_factory.mktemp("yeast_2d")
    return _build_iminfo(FIXTURE_2D_PATH, workdir)


@pytest.fixture
def make_imageinfo_3d(tmp_path: Path):
    """Factory: build a fresh 3D ImInfo for each call.

    Use this in tests that construct multiple Filter instances. Each call
    returns an ImInfo with an isolated `im_preprocessed` path, which
    sidesteps the Windows file-lock issue when overwriting an mmap'd
    output file.
    """
    counter = {"n": 0}

    def _factory() -> ImInfo:
        counter["n"] += 1
        sub = tmp_path / f"info_{counter['n']}"
        sub.mkdir()
        return _build_iminfo(FIXTURE_3D_PATH, sub)

    return _factory


@pytest.fixture
def make_imageinfo_2d(tmp_path: Path):
    """Factory: build a fresh 2D ImInfo for each call (see `make_imageinfo_3d`)."""
    counter = {"n": 0}

    def _factory() -> ImInfo:
        counter["n"] += 1
        sub = tmp_path / f"info_{counter['n']}"
        sub.mkdir()
        return _build_iminfo(FIXTURE_2D_PATH, sub)

    return _factory


# ------------------------------------------------------------------
# Frangi-output fixtures (session-scoped) for downstream pipeline tests
# ------------------------------------------------------------------


def _run_filter_to_disk(im_info: ImInfo) -> Path:
    """Run Filter to populate ``im_preprocessed`` on disk and release its memmap handles."""
    filt = Filter(im_info, FrangiConfig(device="cpu"), num_t=2)
    filt.run()
    # Drop memmap handles so Windows lets us read/copy the file later.
    filt.frangi_memmap = None
    filt.im_memmap = None
    gc.collect()
    return Path(im_info.pipeline_paths["im_preprocessed"])


@pytest.fixture(scope="session")
def frangi_3d_path(imageinfo_3d) -> Path:
    """Session-scoped: path to a precomputed 3D Frangi memmap.

    Runs Filter once per session against the 3D yeast fixture so that
    downstream Label tests can reuse the result via
    ``make_label_imageinfo_3d``.
    """
    return _run_filter_to_disk(imageinfo_3d)


@pytest.fixture(scope="session")
def frangi_2d_path(imageinfo_2d) -> Path:
    """Session-scoped: path to a precomputed 2D Frangi memmap (see :func:`frangi_3d_path`)."""
    return _run_filter_to_disk(imageinfo_2d)


def _make_label_imageinfo_factory(
    tmp_path: Path, source_image: Path, frangi_source: Path
):
    """Build a factory that produces fresh ImInfos with the Frangi memmap pre-populated.

    Each call:
      1. Copies the raw source image into a fresh subdirectory.
      2. Constructs an ImInfo (which sets up `pipeline_paths`).
      3. Copies the precomputed Frangi memmap into the
         `im_preprocessed` path that ImInfo just computed.

    The returned ImInfo is ready for `Label(info)` without re-running
    Filter.
    """
    counter = {"n": 0}

    def _factory() -> ImInfo:
        counter["n"] += 1
        sub = tmp_path / f"label_info_{counter['n']}"
        sub.mkdir()
        info = _build_iminfo(source_image, sub)
        dst = Path(info.pipeline_paths["im_preprocessed"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(frangi_source, dst)
        return info

    return _factory


@pytest.fixture
def make_label_imageinfo_3d(tmp_path: Path, frangi_3d_path: Path):
    """Factory: per-test 3D ImInfo with the precomputed Frangi memmap copied in.

    Each call returns an ImInfo with an isolated ``im_instance_label``
    path, so multiple Label runs in one test do not collide.
    """
    return _make_label_imageinfo_factory(tmp_path, FIXTURE_3D_PATH, frangi_3d_path)


@pytest.fixture
def make_label_imageinfo_2d(tmp_path: Path, frangi_2d_path: Path):
    """Factory: per-test 2D ImInfo with the precomputed Frangi memmap copied in."""
    return _make_label_imageinfo_factory(tmp_path, FIXTURE_2D_PATH, frangi_2d_path)


@pytest.fixture(scope="module")
def make_label_imageinfo_3d_module(
    tmp_path_factory: pytest.TempPathFactory, frangi_3d_path: Path
):
    """Module-scoped variant of :func:`make_label_imageinfo_3d`.

    Provides a factory whose ImInfos persist for the lifetime of the
    test module. Use this when a module-scoped Label-output fixture
    needs a stable workdir.
    """
    workdir = tmp_path_factory.mktemp("label_3d_module")
    return _make_label_imageinfo_factory(workdir, FIXTURE_3D_PATH, frangi_3d_path)


@pytest.fixture(scope="module")
def make_label_imageinfo_2d_module(
    tmp_path_factory: pytest.TempPathFactory, frangi_2d_path: Path
):
    """Module-scoped variant of :func:`make_label_imageinfo_2d`."""
    workdir = tmp_path_factory.mktemp("label_2d_module")
    return _make_label_imageinfo_factory(workdir, FIXTURE_2D_PATH, frangi_2d_path)


# ------------------------------------------------------------------
# Label-output fixtures (session-scoped) for downstream pipeline tests
# ------------------------------------------------------------------


def _run_label_to_disk(im_info: ImInfo) -> Path:
    """Run Label to populate ``im_instance_label`` on disk and release its memmap handles.

    Assumes the Frangi memmap (``im_preprocessed``) is already on disk in
    ``im_info``'s pipeline tree (pre-populated by ``_run_filter_to_disk``
    or by copying a session-cached Frangi).
    """
    lbl = Label(im_info, num_t=2, device="cpu")
    lbl.run()
    # Drop memmap handles so Windows lets us read/copy the file later.
    lbl.instance_label_memmap = None
    lbl.frangi_memmap = None
    lbl.im_memmap = None
    gc.collect()
    return Path(im_info.pipeline_paths["im_instance_label"])


def _run_filter_then_label_for_session(
    tmp_path_factory: pytest.TempPathFactory, source_image: Path, sub_name: str
) -> Path:
    """Build a session ImInfo, run Filter then Label, return the label path.

    Used by ``label_3d_path`` / ``label_2d_path`` to materialize a Label
    output once per session in its own workdir (independent of the
    session-scoped ``imageinfo_*`` workdir, which the Frangi fixture
    already populated). The returned path is read-only from the
    perspective of downstream tests.
    """
    workdir = tmp_path_factory.mktemp(sub_name)
    info = _build_iminfo(source_image, workdir)
    _run_filter_to_disk(info)
    return _run_label_to_disk(info)


@pytest.fixture(scope="session")
def label_3d_path(
    tmp_path_factory: pytest.TempPathFactory, frangi_3d_path: Path
) -> Path:
    """Session-scoped: path to a precomputed 3D ``im_instance_label`` memmap.

    Runs Filter + Label once per session against the 3D yeast fixture so
    that downstream Network tests can reuse the result via
    ``make_network_imageinfo_3d``. Depends on ``frangi_3d_path`` only to
    serialize relative to the existing Frangi cascade — the actual Label
    run uses its own workdir to keep output paths isolated from any
    other test that mutates the Frangi-stage workdir.
    """
    del frangi_3d_path  # only used to serialize the cascade order
    return _run_filter_then_label_for_session(
        tmp_path_factory, FIXTURE_3D_PATH, "label_3d_session"
    )


@pytest.fixture(scope="session")
def label_2d_path(
    tmp_path_factory: pytest.TempPathFactory, frangi_2d_path: Path
) -> Path:
    """Session-scoped: path to a precomputed 2D ``im_instance_label`` memmap (see :func:`label_3d_path`)."""
    del frangi_2d_path
    return _run_filter_then_label_for_session(
        tmp_path_factory, FIXTURE_2D_PATH, "label_2d_session"
    )


def _make_network_imageinfo_factory(
    tmp_path: Path,
    source_image: Path,
    frangi_source: Path,
    label_source: Path,
):
    """Build a factory that produces fresh ImInfos with both Frangi and Label memmaps pre-populated.

    Each call:
      1. Copies the raw source image into a fresh subdirectory.
      2. Constructs an ImInfo (which sets up `pipeline_paths`).
      3. Copies the precomputed Frangi memmap into ``im_preprocessed``.
      4. Copies the precomputed Label memmap into ``im_instance_label``.

    The returned ImInfo is ready for ``Network(info)`` without re-running
    Filter or Label.
    """
    counter = {"n": 0}

    def _factory() -> ImInfo:
        counter["n"] += 1
        sub = tmp_path / f"network_info_{counter['n']}"
        sub.mkdir()
        info = _build_iminfo(source_image, sub)
        for src, key in (
            (frangi_source, "im_preprocessed"),
            (label_source, "im_instance_label"),
        ):
            dst = Path(info.pipeline_paths[key])
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)
        return info

    return _factory


@pytest.fixture
def make_network_imageinfo_3d(
    tmp_path: Path, frangi_3d_path: Path, label_3d_path: Path
):
    """Factory: per-test 3D ImInfo with both Frangi and Label memmaps copied in.

    Each call returns an ImInfo with isolated ``im_skel`` /
    ``im_pixel_class`` / ``im_skel_relabelled`` paths, so multiple
    Network runs in one test do not collide.
    """
    return _make_network_imageinfo_factory(
        tmp_path, FIXTURE_3D_PATH, frangi_3d_path, label_3d_path
    )


@pytest.fixture
def make_network_imageinfo_2d(
    tmp_path: Path, frangi_2d_path: Path, label_2d_path: Path
):
    """Factory: per-test 2D ImInfo with both Frangi and Label memmaps copied in."""
    return _make_network_imageinfo_factory(
        tmp_path, FIXTURE_2D_PATH, frangi_2d_path, label_2d_path
    )


@pytest.fixture(scope="module")
def make_network_imageinfo_3d_module(
    tmp_path_factory: pytest.TempPathFactory,
    frangi_3d_path: Path,
    label_3d_path: Path,
):
    """Module-scoped variant of :func:`make_network_imageinfo_3d`.

    Provides a factory whose ImInfos persist for the lifetime of the
    test module. Use this when a module-scoped Network-output fixture
    needs a stable workdir.
    """
    workdir = tmp_path_factory.mktemp("network_3d_module")
    return _make_network_imageinfo_factory(
        workdir, FIXTURE_3D_PATH, frangi_3d_path, label_3d_path
    )


@pytest.fixture(scope="module")
def make_network_imageinfo_2d_module(
    tmp_path_factory: pytest.TempPathFactory,
    frangi_2d_path: Path,
    label_2d_path: Path,
):
    """Module-scoped variant of :func:`make_network_imageinfo_2d`."""
    workdir = tmp_path_factory.mktemp("network_2d_module")
    return _make_network_imageinfo_factory(
        workdir, FIXTURE_2D_PATH, frangi_2d_path, label_2d_path
    )


# ------------------------------------------------------------------
# Markers-input fixtures (per-test + module-scoped) for
# ``Markers`` characterization tests.
# ------------------------------------------------------------------


def _make_markers_imageinfo_factory(
    tmp_path: Path,
    source_image: Path,
    frangi_source: Path,
    label_source: Path,
):
    """Build a factory that produces fresh ImInfos with both Frangi and Label memmaps pre-populated.

    Mirrors :func:`_make_network_imageinfo_factory` exactly. Markers'
    inputs are a strict subset of Network's: raw image (already
    populated by ``_build_iminfo``), ``im_instance_label`` (always),
    and ``im_preprocessed`` (only when ``use_im='frangi'``). Copying
    both memmaps unconditionally lets a single factory shape serve
    both ``use_im`` modes without re-parametrizing.

    Each call:
      1. Copies the raw source image into a fresh subdirectory.
      2. Constructs an ImInfo (which sets up `pipeline_paths`).
      3. Copies the precomputed Frangi memmap into ``im_preprocessed``.
      4. Copies the precomputed Label memmap into ``im_instance_label``.

    The returned ImInfo is ready for ``Markers(info)`` without re-running
    Filter or Label.
    """
    counter = {"n": 0}

    def _factory() -> ImInfo:
        counter["n"] += 1
        sub = tmp_path / f"markers_info_{counter['n']}"
        sub.mkdir()
        info = _build_iminfo(source_image, sub)
        for src, key in (
            (frangi_source, "im_preprocessed"),
            (label_source, "im_instance_label"),
        ):
            dst = Path(info.pipeline_paths[key])
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)
        return info

    return _factory


@pytest.fixture
def make_markers_imageinfo_3d(
    tmp_path: Path, frangi_3d_path: Path, label_3d_path: Path
):
    """Factory: per-test 3D ImInfo with both Frangi and Label memmaps copied in.

    Each call returns an ImInfo with isolated ``im_marker`` /
    ``im_distance`` / ``im_border`` paths, so multiple Markers runs in
    one test do not collide.
    """
    return _make_markers_imageinfo_factory(
        tmp_path, FIXTURE_3D_PATH, frangi_3d_path, label_3d_path
    )


@pytest.fixture
def make_markers_imageinfo_2d(
    tmp_path: Path, frangi_2d_path: Path, label_2d_path: Path
):
    """Factory: per-test 2D ImInfo with both Frangi and Label memmaps copied in."""
    return _make_markers_imageinfo_factory(
        tmp_path, FIXTURE_2D_PATH, frangi_2d_path, label_2d_path
    )


@pytest.fixture(scope="module")
def make_markers_imageinfo_3d_module(
    tmp_path_factory: pytest.TempPathFactory,
    frangi_3d_path: Path,
    label_3d_path: Path,
):
    """Module-scoped variant of :func:`make_markers_imageinfo_3d`.

    Provides a factory whose ImInfos persist for the lifetime of the
    test module. Use this when a module-scoped Markers-output fixture
    needs a stable workdir.
    """
    workdir = tmp_path_factory.mktemp("markers_3d_module")
    return _make_markers_imageinfo_factory(
        workdir, FIXTURE_3D_PATH, frangi_3d_path, label_3d_path
    )


@pytest.fixture(scope="module")
def make_markers_imageinfo_2d_module(
    tmp_path_factory: pytest.TempPathFactory,
    frangi_2d_path: Path,
    label_2d_path: Path,
):
    """Module-scoped variant of :func:`make_markers_imageinfo_2d`."""
    workdir = tmp_path_factory.mktemp("markers_2d_module")
    return _make_markers_imageinfo_factory(
        workdir, FIXTURE_2D_PATH, frangi_2d_path, label_2d_path
    )


# ------------------------------------------------------------------
# Markers-output fixtures (session-scoped) for downstream pipeline tests
# ------------------------------------------------------------------


def _run_markers_to_disk(im_info: ImInfo) -> dict[str, Path]:
    """Run Markers on ``im_info``, return the ``im_marker`` / ``im_distance`` paths.

    Drops Markers' memmap handles so Windows lets us copy the files later
    (mirrors ``_run_filter_to_disk`` / ``_run_label_to_disk``). ``im_border``
    is intentionally not exposed — HuMomentTracking does not consume it.
    """
    m = Markers(im_info, num_t=2, device="cpu")
    m.run()
    m.im_marker_memmap = None
    m.im_distance_memmap = None
    m.im_border_memmap = None
    m.label_memmap = None
    m.im_memmap = None
    m.im_frangi_memmap = None
    gc.collect()
    return {
        "im_marker": Path(im_info.pipeline_paths["im_marker"]),
        "im_distance": Path(im_info.pipeline_paths["im_distance"]),
    }


def _run_filter_label_markers_for_session(
    tmp_path_factory: pytest.TempPathFactory, source_image: Path, sub_name: str
) -> dict[str, Path]:
    """Build a session ImInfo, run Filter+Label+Markers, return the marker paths.

    Used by ``markers_3d_paths`` / ``markers_2d_paths`` to materialize
    Markers' ``im_marker`` / ``im_distance`` outputs once per session in
    its own workdir (independent of the session-scoped ``imageinfo_*``
    workdir, which earlier fixtures may have populated).
    """
    workdir = tmp_path_factory.mktemp(sub_name)
    info = _build_iminfo(source_image, workdir)
    _run_filter_to_disk(info)
    _run_label_to_disk(info)
    return _run_markers_to_disk(info)


@pytest.fixture(scope="session")
def markers_3d_paths(
    tmp_path_factory: pytest.TempPathFactory,
    frangi_3d_path: Path,
    label_3d_path: Path,
) -> dict[str, Path]:
    """Session-scoped: precomputed 3D Markers ``im_marker`` / ``im_distance`` paths.

    Runs Filter + Label + Markers once per session against the 3D yeast
    fixture so that downstream HuMomentTracking tests can reuse the
    result via ``make_hu_imageinfo_3d``. Depends on ``frangi_3d_path``
    and ``label_3d_path`` only to serialize relative to the existing
    cascade — the actual Markers run uses its own workdir to keep output
    paths isolated from any other test that mutates the upstream
    workdirs.
    """
    del frangi_3d_path  # only used to serialize the cascade order
    del label_3d_path
    return _run_filter_label_markers_for_session(
        tmp_path_factory, FIXTURE_3D_PATH, "markers_3d_session"
    )


@pytest.fixture(scope="session")
def markers_2d_paths(
    tmp_path_factory: pytest.TempPathFactory,
    frangi_2d_path: Path,
    label_2d_path: Path,
) -> dict[str, Path]:
    """Session-scoped: precomputed 2D Markers ``im_marker`` / ``im_distance`` paths (see :func:`markers_3d_paths`)."""
    del frangi_2d_path
    del label_2d_path
    return _run_filter_label_markers_for_session(
        tmp_path_factory, FIXTURE_2D_PATH, "markers_2d_session"
    )


# ------------------------------------------------------------------
# Hu-input fixtures (per-test + module-scoped) for
# ``HuMomentTracking`` characterization tests.
# ------------------------------------------------------------------


def _make_hu_imageinfo_factory(
    tmp_path: Path,
    source_image: Path,
    frangi_source: Path,
    label_source: Path,
    markers_sources: dict[str, Path],
):
    """Build a factory that produces fresh ImInfos with Frangi + Label + Marker + Distance memmaps pre-populated.

    Mirrors :func:`_make_markers_imageinfo_factory` and
    :func:`_make_network_imageinfo_factory` in shape, just with one
    extra layer of inputs. HuMomentTracking reads four memmaps in
    ``_allocate_memory`` (`hu_tracking.py:508-512`):
    ``im_instance_label`` (Label), the raw image (already populated by
    ``_build_iminfo``), ``im_preprocessed`` (Filter), ``im_marker``
    (Markers), ``im_distance`` (Markers). Copying all four
    unconditionally keeps the factory shape uniform with the rest of
    the cascade.

    Each call:
      1. Copies the raw source image into a fresh subdirectory.
      2. Constructs an ImInfo (which sets up ``pipeline_paths``).
      3. Copies the precomputed Frangi memmap into ``im_preprocessed``.
      4. Copies the precomputed Label memmap into ``im_instance_label``.
      5. Copies the precomputed Marker memmap into ``im_marker``.
      6. Copies the precomputed Distance memmap into ``im_distance``.

    The returned ImInfo is ready for ``HuMomentTracking(info)`` without
    re-running Filter, Label, or Markers.
    """
    counter = {"n": 0}

    def _factory() -> ImInfo:
        counter["n"] += 1
        sub = tmp_path / f"hu_info_{counter['n']}"
        sub.mkdir()
        info = _build_iminfo(source_image, sub)
        for src, key in (
            (frangi_source, "im_preprocessed"),
            (label_source, "im_instance_label"),
            (markers_sources["im_marker"], "im_marker"),
            (markers_sources["im_distance"], "im_distance"),
        ):
            dst = Path(info.pipeline_paths[key])
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)
        return info

    return _factory


@pytest.fixture
def make_hu_imageinfo_3d(
    tmp_path: Path,
    frangi_3d_path: Path,
    label_3d_path: Path,
    markers_3d_paths: dict[str, Path],
):
    """Factory: per-test 3D ImInfo with Frangi + Label + Marker + Distance memmaps copied in.

    Each call returns an ImInfo with an isolated
    ``flow_vector_array`` path, so multiple HuMomentTracking runs in
    one test do not collide.
    """
    return _make_hu_imageinfo_factory(
        tmp_path, FIXTURE_3D_PATH, frangi_3d_path, label_3d_path, markers_3d_paths
    )


@pytest.fixture
def make_hu_imageinfo_2d(
    tmp_path: Path,
    frangi_2d_path: Path,
    label_2d_path: Path,
    markers_2d_paths: dict[str, Path],
):
    """Factory: per-test 2D ImInfo with Frangi + Label + Marker + Distance memmaps copied in."""
    return _make_hu_imageinfo_factory(
        tmp_path, FIXTURE_2D_PATH, frangi_2d_path, label_2d_path, markers_2d_paths
    )


@pytest.fixture(scope="module")
def make_hu_imageinfo_3d_module(
    tmp_path_factory: pytest.TempPathFactory,
    frangi_3d_path: Path,
    label_3d_path: Path,
    markers_3d_paths: dict[str, Path],
):
    """Module-scoped variant of :func:`make_hu_imageinfo_3d`.

    Provides a factory whose ImInfos persist for the lifetime of the
    test module. Use this when a module-scoped Hu-output fixture needs
    a stable workdir.
    """
    workdir = tmp_path_factory.mktemp("hu_3d_module")
    return _make_hu_imageinfo_factory(
        workdir, FIXTURE_3D_PATH, frangi_3d_path, label_3d_path, markers_3d_paths
    )


@pytest.fixture(scope="module")
def make_hu_imageinfo_2d_module(
    tmp_path_factory: pytest.TempPathFactory,
    frangi_2d_path: Path,
    label_2d_path: Path,
    markers_2d_paths: dict[str, Path],
):
    """Module-scoped variant of :func:`make_hu_imageinfo_2d`."""
    workdir = tmp_path_factory.mktemp("hu_2d_module")
    return _make_hu_imageinfo_factory(
        workdir, FIXTURE_2D_PATH, frangi_2d_path, label_2d_path, markers_2d_paths
    )


# ------------------------------------------------------------------
# Network-output fixtures (session-scoped) for downstream pipeline tests
# ------------------------------------------------------------------


def _run_network_to_disk(im_info: ImInfo) -> Path:
    """Run Network to populate ``im_skel_relabelled`` on disk and release its memmap handles.

    Assumes the Frangi memmap (``im_preprocessed``) and Label memmap
    (``im_instance_label``) are already on disk in ``im_info``'s pipeline
    tree (pre-populated by ``_run_filter_to_disk`` + ``_run_label_to_disk``
    or by copying session-cached outputs).
    """
    net = Network(im_info, num_t=2, device="cpu")
    net.run()
    # Drop memmap handles so Windows lets us read/copy the file later.
    net.skel_relabelled_memmap = None
    net.skel_memmap = None
    net.pixel_class_memmap = None
    net.label_memmap = None
    net.im_memmap = None
    net.im_frangi_memmap = None
    gc.collect()
    return Path(im_info.pipeline_paths["im_skel_relabelled"])


def _run_filter_label_network_for_session(
    tmp_path_factory: pytest.TempPathFactory, source_image: Path, sub_name: str
) -> Path:
    """Build a session ImInfo, run Filter+Label+Network, return the relabelled-skel path.

    Used by ``network_3d_path`` / ``network_2d_path`` to materialize
    Network's ``im_skel_relabelled`` output once per session in its own
    workdir (independent of the session-scoped ``imageinfo_*`` workdir,
    which earlier fixtures may have populated).
    """
    workdir = tmp_path_factory.mktemp(sub_name)
    info = _build_iminfo(source_image, workdir)
    _run_filter_to_disk(info)
    _run_label_to_disk(info)
    return _run_network_to_disk(info)


@pytest.fixture(scope="session")
def network_3d_path(
    tmp_path_factory: pytest.TempPathFactory,
    frangi_3d_path: Path,
    label_3d_path: Path,
) -> Path:
    """Session-scoped: path to a precomputed 3D ``im_skel_relabelled`` memmap.

    Runs Filter + Label + Network once per session against the 3D yeast
    fixture so that downstream VoxelReassigner tests can reuse the
    result via ``make_voxel_reassign_imageinfo_3d``. Depends on
    ``frangi_3d_path`` and ``label_3d_path`` only to serialize relative
    to the existing cascade — the actual Network run uses its own
    workdir to keep output paths isolated from any other test that
    mutates the upstream workdirs.
    """
    del frangi_3d_path  # only used to serialize the cascade order
    del label_3d_path
    return _run_filter_label_network_for_session(
        tmp_path_factory, FIXTURE_3D_PATH, "network_3d_session"
    )


@pytest.fixture(scope="session")
def network_2d_path(
    tmp_path_factory: pytest.TempPathFactory,
    frangi_2d_path: Path,
    label_2d_path: Path,
) -> Path:
    """Session-scoped: path to a precomputed 2D ``im_skel_relabelled`` memmap (see :func:`network_3d_path`)."""
    del frangi_2d_path
    del label_2d_path
    return _run_filter_label_network_for_session(
        tmp_path_factory, FIXTURE_2D_PATH, "network_2d_session"
    )


# ------------------------------------------------------------------
# Hu-output fixtures (session-scoped) for downstream pipeline tests
# ------------------------------------------------------------------


def _run_hu_to_disk(im_info: ImInfo) -> Path:
    """Run HuMomentTracking to write ``flow_vector_array.npy`` and return its path.

    Hu writes ``flow_vector_array.npy`` via ``np.save`` (not a memmap),
    so no handle-drop dance is needed for the ``.npy`` file itself.
    The upstream Markers/Label/Frangi memmap handles must still be
    released so Windows lets us read/copy them later.
    """
    h = HuMomentTracking(im_info, num_t=2, device="cpu")
    h.run()
    h.label_memmap = None
    h.im_memmap = None
    h.im_frangi_memmap = None
    h.im_marker_memmap = None
    h.im_distance_memmap = None
    gc.collect()
    return Path(im_info.pipeline_paths["flow_vector_array"])


def _run_filter_label_markers_hu_for_session(
    tmp_path_factory: pytest.TempPathFactory, source_image: Path, sub_name: str
) -> Path:
    """Build a session ImInfo, run Filter+Label+Markers+Hu, return the flow-vector-array path.

    Used by ``hu_outputs_3d_path`` / ``hu_outputs_2d_path`` to
    materialize Hu's ``flow_vector_array.npy`` once per session in its
    own workdir. This is the heaviest new session fixture — full
    upstream pipeline (Filter + Label + Markers + Hu).
    """
    workdir = tmp_path_factory.mktemp(sub_name)
    info = _build_iminfo(source_image, workdir)
    _run_filter_to_disk(info)
    _run_label_to_disk(info)
    _run_markers_to_disk(info)
    return _run_hu_to_disk(info)


@pytest.fixture(scope="session")
def hu_outputs_3d_path(
    tmp_path_factory: pytest.TempPathFactory,
    frangi_3d_path: Path,
    label_3d_path: Path,
    markers_3d_paths: dict[str, Path],
) -> Path:
    """Session-scoped: path to a precomputed 3D ``flow_vector_array.npy``.

    Runs Filter + Label + Markers + Hu once per session against the 3D
    yeast fixture so that downstream VoxelReassigner tests can reuse
    the result via ``make_voxel_reassign_imageinfo_3d``. Depends on the
    upstream session caches only to serialize relative to the existing
    cascade — the actual Hu run uses its own workdir.
    """
    del frangi_3d_path  # only used to serialize the cascade order
    del label_3d_path
    del markers_3d_paths
    return _run_filter_label_markers_hu_for_session(
        tmp_path_factory, FIXTURE_3D_PATH, "hu_outputs_3d_session"
    )


@pytest.fixture(scope="session")
def hu_outputs_2d_path(
    tmp_path_factory: pytest.TempPathFactory,
    frangi_2d_path: Path,
    label_2d_path: Path,
    markers_2d_paths: dict[str, Path],
) -> Path:
    """Session-scoped: path to a precomputed 2D ``flow_vector_array.npy`` (see :func:`hu_outputs_3d_path`)."""
    del frangi_2d_path
    del label_2d_path
    del markers_2d_paths
    return _run_filter_label_markers_hu_for_session(
        tmp_path_factory, FIXTURE_2D_PATH, "hu_outputs_2d_session"
    )


# ------------------------------------------------------------------
# VoxelReassigner-input fixtures (per-test + module-scoped) for
# ``VoxelReassigner`` characterization tests.
# ------------------------------------------------------------------


def _make_voxel_reassign_imageinfo_factory(
    tmp_path: Path,
    source_image: Path,
    label_source: Path,
    network_source: Path,
    hu_source: Path,
):
    """Build a factory that produces fresh ImInfos with Label + Network + Hu outputs pre-populated.

    VoxelReassigner reads three on-disk inputs through ImInfo's pipeline
    tree (``voxel_reassignment.py:866-867`` for the two memmaps;
    ``flow_interpolation.py:124-125`` for the .npy):

    - ``im_instance_label`` (Label memmap) — used as ``obj_label_memmap``.
    - ``im_skel_relabelled`` (Network memmap) — used as ``branch_label_memmap``.
    - ``flow_vector_array.npy`` (Hu np.save'd array) — loaded by
      ``FlowInterpolator._allocate_memory`` at construction time of
      ``VoxelReassigner.__init__``.

    The Frangi memmap is NOT copied in: VoxelReassigner does not
    consume ``im_preprocessed`` directly, and the upstream stages have
    already produced everything VoxelReassigner needs.

    **Important**: ``flow_vector_array.npy`` MUST be copied in BEFORE
    the caller constructs ``VoxelReassigner(info)``. Two
    ``FlowInterpolator`` instances are constructed unconditionally in
    ``VoxelReassigner.__init__`` (lines 108-109), and FlowInterpolator
    loads the file immediately at construction (not lazily on
    ``.run()``). The factory copies all three files in before
    returning the ImInfo, so test code can construct the
    VoxelReassigner immediately.

    Each call:
      1. Copies the raw source image into a fresh subdirectory.
      2. Constructs an ImInfo (which sets up ``pipeline_paths``).
      3. Copies the precomputed Label memmap into ``im_instance_label``.
      4. Copies the precomputed Network memmap into ``im_skel_relabelled``.
      5. Copies the precomputed Hu .npy into ``flow_vector_array``.

    The returned ImInfo is ready for ``VoxelReassigner(info)`` without
    re-running Filter, Label, Network, Markers, or Hu.
    """
    counter = {"n": 0}

    def _factory() -> ImInfo:
        counter["n"] += 1
        sub = tmp_path / f"voxel_reassign_info_{counter['n']}"
        sub.mkdir()
        info = _build_iminfo(source_image, sub)
        for src, key in (
            (label_source, "im_instance_label"),
            (network_source, "im_skel_relabelled"),
            (hu_source, "flow_vector_array"),
        ):
            dst = Path(info.pipeline_paths[key])
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)
        return info

    return _factory


@pytest.fixture
def make_voxel_reassign_imageinfo_3d(
    tmp_path: Path,
    label_3d_path: Path,
    network_3d_path: Path,
    hu_outputs_3d_path: Path,
):
    """Factory: per-test 3D ImInfo with Label + Network + Hu outputs copied in.

    Each call returns an ImInfo with isolated
    ``im_branch_label_reassigned`` / ``im_obj_label_reassigned`` /
    ``voxel_matches`` paths, so multiple VoxelReassigner runs in one
    test do not collide.
    """
    return _make_voxel_reassign_imageinfo_factory(
        tmp_path,
        FIXTURE_3D_PATH,
        label_3d_path,
        network_3d_path,
        hu_outputs_3d_path,
    )


@pytest.fixture
def make_voxel_reassign_imageinfo_2d(
    tmp_path: Path,
    label_2d_path: Path,
    network_2d_path: Path,
    hu_outputs_2d_path: Path,
):
    """Factory: per-test 2D ImInfo with Label + Network + Hu outputs copied in."""
    return _make_voxel_reassign_imageinfo_factory(
        tmp_path,
        FIXTURE_2D_PATH,
        label_2d_path,
        network_2d_path,
        hu_outputs_2d_path,
    )


@pytest.fixture(scope="module")
def make_voxel_reassign_imageinfo_3d_module(
    tmp_path_factory: pytest.TempPathFactory,
    label_3d_path: Path,
    network_3d_path: Path,
    hu_outputs_3d_path: Path,
):
    """Module-scoped variant of :func:`make_voxel_reassign_imageinfo_3d`.

    Provides a factory whose ImInfos persist for the lifetime of the
    test module. Use this when a module-scoped VoxelReassigner-output
    fixture needs a stable workdir.
    """
    workdir = tmp_path_factory.mktemp("voxel_reassign_3d_module")
    return _make_voxel_reassign_imageinfo_factory(
        workdir,
        FIXTURE_3D_PATH,
        label_3d_path,
        network_3d_path,
        hu_outputs_3d_path,
    )


@pytest.fixture(scope="module")
def make_voxel_reassign_imageinfo_2d_module(
    tmp_path_factory: pytest.TempPathFactory,
    label_2d_path: Path,
    network_2d_path: Path,
    hu_outputs_2d_path: Path,
):
    """Module-scoped variant of :func:`make_voxel_reassign_imageinfo_2d`."""
    workdir = tmp_path_factory.mktemp("voxel_reassign_2d_module")
    return _make_voxel_reassign_imageinfo_factory(
        workdir,
        FIXTURE_2D_PATH,
        label_2d_path,
        network_2d_path,
        hu_outputs_2d_path,
    )
