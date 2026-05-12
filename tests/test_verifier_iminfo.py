"""Characterization tests for ``nellie.im_info.verifier.ImInfo``.

Slice 1 of the verifier dechaos refactor pins existing ``ImInfo``
behavior end-to-end so the verifier can be safely refactored in
subsequent slices (constructor I/O extraction, unified normalizer,
validation split, metadata extractor strategy, OME-TIFF writer
extraction, ``load_image`` orchestrator). The companion file
``tests/test_verifier_fileinfo.py`` covers the ``FileInfo`` half.

These tests pin **current** behavior, including the design quirks the
dechaos report calls out — the ``_normalize_axes`` allowed-set divergence
from ``FileInfo._axis_errors`` (C-axis), the auto-regen on T-axis stale,
and the partial-flag semantics of ``_check_axes_exist``.

The tests intentionally exercise the module-level normalizer
``transform_to_axes`` directly with synthetic numpy arrays. End-to-end
testing through the constructor would re-write the synthetic file via
``save_ome_tiff`` (which has its own normalization), so direct calls
are the cheapest way to pin each branch. Slice 4 of the dechaos refactor
collapsed the three normalizer copies (``_normalize_time_axis``,
``_normalize_axes``, ``_normalize_memmap``) into two pure functions
(``infer_t_axis``, ``transform_to_axes``); the canonical-mode tests
exercise ``transform_to_axes(data, axes)`` and the match-mode tests
exercise ``transform_to_axes(data, file_axes, target_axes=info.axes)``.

See ``wiki/outputs/dechaos-verifier.md`` for the full structural review
and refactor plan.
"""

from __future__ import annotations

import gc
import os
import time
from pathlib import Path

import numpy as np
import ome_types
import pytest
import tifffile

from nellie.im_info.verifier import (
    CSVS_ONLY_PRESET,
    DROPPABLE_KEYS,
    FileInfo,
    ImInfo,
    KEEP_EVERYTHING_PRESET,
    MASKS_AND_CSVS_PRESET,
    transform_to_axes,
)


def _release_iminfo_memmap(info: ImInfo) -> None:
    """Drop the ``im_info.im`` memmap so Windows lets us delete the file.

    Required for tests that call ``remove_intermediates`` (or any test
    that needs to delete or overwrite ``im_info.im_path``). Mirrors
    ``_release_filter`` in test_filtering.py.
    """
    info.im = None  # type: ignore[assignment]
    gc.collect()

# ``conftest`` is on ``sys.path`` because pytest's rootdir-based collection
# adds the test directory before importing test modules. A relative
# ``from .conftest import …`` does not work because ``tests/`` is not a
# package (no ``__init__.py``).
from conftest import (  # type: ignore[import-not-found]
    FIXTURE_3D_PATH,
    copy_fixture_to_tmp,
)


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


_PIPELINE_PATH_KEYS = {
    'im_preprocessed', 'im_instance_label', 'im_skel', 'im_skel_relabelled',
    'im_pixel_class', 'im_marker', 'im_distance', 'im_border',
    'flow_vector_array', 'voxel_matches',
    'im_branch_label_reassigned', 'im_obj_label_reassigned',
    'features_voxels', 'features_nodes', 'features_branches',
    'features_organelles', 'features_image', 'adjacency_maps',
}

# Per dechaos Pass 5: the user-output keys (5 features_*) are routed to
# ``user_output_path_no_ext`` (top-level ``output_dir``); everything else
# routes to ``nellie_necessities_output_path_no_ext`` (the
# ``nellie_necessities`` subdir). Pinning the per-key routing is one of
# the key contracts that downstream stages rely on.
_USER_OUTPUT_KEYS = {
    'features_voxels',
    'features_nodes',
    'features_branches',
    'features_organelles',
    'features_image',
}
_NELLIE_NECESSITIES_KEYS = _PIPELINE_PATH_KEYS - _USER_OUTPUT_KEYS


def _make_minimal_file_info(workdir: Path, *, source_axes: str = 'YX') -> FileInfo:
    """Build a FileInfo from a synthetic minimal TIFF and run find/load_metadata.

    The resulting FileInfo has ``ome_output_path`` populated and is ready
    to drive ``ImInfo(file_info)``. Used for tests that need a
    single-timepoint or otherwise non-fixture shape.
    """
    src = workdir / f'tiny_{source_axes}.tif'
    if source_axes == 'YX':
        data = np.zeros((16, 16), dtype=np.uint16)
    elif source_axes == 'TYX':
        data = np.zeros((1, 16, 16), dtype=np.uint16)
    else:
        raise AssertionError(f"unsupported helper source axes: {source_axes!r}")
    tifffile.imwrite(str(src), data, photometric='minisblack')
    fi = FileInfo(str(src))
    fi.find_metadata()
    # Synthetic file lacks calibration; backfill so _validate doesn't flag
    # missing dim_res. Keeps the test focused on ImInfo behavior.
    fi.dim_res = {'X': 0.1, 'Y': 0.1, 'Z': None, 'T': None}
    fi._validate()
    return fi


# ============================================================================
# A. Construction
# ============================================================================


def test_construction_populates_im_path(imageinfo_3d: ImInfo) -> None:
    assert imageinfo_3d.im_path is not None


def test_construction_populates_memmap_im(imageinfo_3d: ImInfo) -> None:
    """``ImInfo.im`` is a (memory-mapped) numpy-compatible array."""
    assert isinstance(imageinfo_3d.im, np.ndarray)


def test_construction_populates_axes(imageinfo_3d: ImInfo) -> None:
    """Canonical 3D fixture lands at ``axes='TZYX'`` after construction."""
    assert imageinfo_3d.axes == 'TZYX'


def test_construction_populates_shape(imageinfo_3d: ImInfo) -> None:
    """3D fixture has shape ``(T=2, Z=17, Y=192, X=279)``."""
    assert imageinfo_3d.shape == (2, 17, 192, 279)


def test_construction_populates_dim_res(imageinfo_3d: ImInfo) -> None:
    """``dim_res`` carries the OME-TIFF physical-size + time-increment values."""
    assert imageinfo_3d.dim_res == {
        'X': 0.0655,
        'Y': 0.0655,
        'Z': 0.25,
        'T': 4.535566806793213,
    }


def test_construction_dim_res_matches_file_info(imageinfo_3d: ImInfo) -> None:
    """``ImInfo.dim_res`` round-trips through the OME XML and matches FileInfo's."""
    assert imageinfo_3d.dim_res == imageinfo_3d.file_info.dim_res


def test_construction_populates_ome_metadata(imageinfo_3d: ImInfo) -> None:
    """``ome_metadata`` is an ``ome_types.OME`` instance."""
    assert isinstance(imageinfo_3d.ome_metadata, ome_types.OME)


def test_construction_populates_pipeline_paths(imageinfo_3d: ImInfo) -> None:
    """``pipeline_paths`` is populated after construction (not None / empty)."""
    assert imageinfo_3d.pipeline_paths
    assert isinstance(imageinfo_3d.pipeline_paths, dict)


def test_construction_populates_screenshot_dir(imageinfo_3d: ImInfo) -> None:
    expected = os.path.join(imageinfo_3d.file_info.output_dir, 'screenshots')
    assert imageinfo_3d.screenshot_dir == expected


def test_construction_populates_graph_dir(imageinfo_3d: ImInfo) -> None:
    expected = os.path.join(imageinfo_3d.file_info.output_dir, 'graphs')
    assert imageinfo_3d.graph_dir == expected


def test_construction_populates_no_z_3d(imageinfo_3d: ImInfo) -> None:
    """3D fixture has Z=17, so ``no_z`` is False."""
    assert imageinfo_3d.no_z is False


def test_construction_populates_no_t_3d(imageinfo_3d: ImInfo) -> None:
    """3D fixture has T=2, so ``no_t`` is False."""
    assert imageinfo_3d.no_t is False


def test_canonical_ome_tiff_exists_after_construction(imageinfo_3d: ImInfo) -> None:
    """The canonical OME-TIFF that ``ImInfo`` memmaps must exist on disk."""
    assert imageinfo_3d.im_path is not None
    assert os.path.exists(imageinfo_3d.im_path)


# ------ Thin constructor (Slice 3 boundary) ------


def test_thin_constructor_does_no_io(tmp_path: Path) -> None:
    """``ImInfo(file_info)`` is a thin constructor — no filesystem I/O.

    Slice 3 split ``ImInfo``'s I/O into ``load()``. The bare
    constructor stores ``file_info`` and computes path-derived
    attributes only; it does NOT call ``save_ome_tiff``, does NOT
    open the canonical OME-TIFF, and does NOT memmap. ``info.im``
    is None until ``load()`` (or ``from_file_info``) runs.

    Pinned so that pure-logic tests in subsequent slices can construct
    an ImInfo to inspect path computations without triggering OME-TIFF
    regen or memmap allocation.
    """
    workdir = tmp_path / 'wd'
    workdir.mkdir()
    src = copy_fixture_to_tmp(FIXTURE_3D_PATH, workdir)
    fi = FileInfo(str(src))
    fi.find_metadata()
    fi.load_metadata()

    # Construct without load — no I/O should fire.
    info = ImInfo(fi)
    assert info.im is None
    assert info.axes is None
    assert info.shape is None
    assert info.ome_metadata is None
    assert info.pipeline_paths == {}
    # Path-derived attrs ARE populated (they're pure-string ops)
    assert info.im_path == fi.ome_output_path
    assert info.screenshot_dir == os.path.join(fi.output_dir, 'screenshots')


def test_load_is_idempotent(tmp_path: Path) -> None:
    """Calling ``load()`` twice is safe — re-loads memmap + paths."""
    workdir = tmp_path / 'wd'
    workdir.mkdir()
    src = copy_fixture_to_tmp(FIXTURE_3D_PATH, workdir)
    fi = FileInfo(str(src))
    fi.find_metadata()
    fi.load_metadata()
    info = ImInfo(fi)
    info.load()
    first_axes = info.axes
    first_pipeline_paths = dict(info.pipeline_paths)
    info.load()  # second call should re-load without error
    assert info.axes == first_axes
    assert info.pipeline_paths == first_pipeline_paths


def test_from_file_info_equivalent_to_explicit_load(tmp_path: Path) -> None:
    """``ImInfo.from_file_info(fi)`` == ``ImInfo(fi); info.load()``."""
    workdir = tmp_path / 'wd'
    workdir.mkdir()
    src = copy_fixture_to_tmp(FIXTURE_3D_PATH, workdir)
    fi = FileInfo(str(src))
    fi.find_metadata()
    fi.load_metadata()

    via_classmethod = ImInfo.from_file_info(fi)
    via_explicit = ImInfo(fi)
    via_explicit.load()

    assert via_classmethod.axes == via_explicit.axes
    assert via_classmethod.shape == via_explicit.shape
    assert via_classmethod.dim_res == via_explicit.dim_res
    assert via_classmethod.no_z == via_explicit.no_z
    assert via_classmethod.no_t == via_explicit.no_t
    assert set(via_classmethod.pipeline_paths.keys()) == set(via_explicit.pipeline_paths.keys())


# ------ Auto-regen path (verifier.py:765-772) ------


def test_construction_skips_regen_when_ome_exists(tmp_path: Path) -> None:
    """Re-constructing ``ImInfo`` against an existing canonical OME-TIFF does NOT regen.

    The constructor checks ``os.path.exists(self.im_path)`` and only calls
    ``file_info.save_ome_tiff()`` when the file is missing or T-axis
    stale. Pinning this prevents Slice 3's constructor refactor from
    silently re-writing the cached file on every load.
    """
    workdir = tmp_path / 'wd'
    workdir.mkdir()
    src = copy_fixture_to_tmp(FIXTURE_3D_PATH, workdir)
    fi = FileInfo(str(src))
    fi.find_metadata()
    fi.load_metadata()

    info_first = ImInfo.from_file_info(fi)
    assert info_first.im_path is not None
    mtime_after_first = os.path.getmtime(info_first.im_path)
    # Sleep across the filesystem mtime resolution boundary so a
    # spurious regen would actually shift the mtime.
    time.sleep(1.1)
    info_second = ImInfo.from_file_info(fi)
    assert info_second.im_path is not None
    mtime_after_second = os.path.getmtime(info_second.im_path)
    assert mtime_after_second == mtime_after_first


def test_construction_regens_when_t_axis_missing(tmp_path: Path) -> None:
    """Cached OME-TIFF without T axis triggers regen when file_info.axes has T.

    Verifier.py:765-772: if the on-disk OME-TIFF lacks ``T`` but the
    FileInfo's axes string includes ``T``, the constructor calls
    ``file_info.save_ome_tiff()`` to regenerate. Pinned per [[im-info]]
    gotcha (mtime shifts on first re-open of pre-T-normalization caches).
    """
    workdir = tmp_path / 'wd'
    workdir.mkdir()
    src = copy_fixture_to_tmp(FIXTURE_3D_PATH, workdir)
    fi = FileInfo(str(src))
    fi.find_metadata()
    fi.load_metadata()

    # Plant a synthetic OME-TIFF at the canonical path WITHOUT T axis.
    assert fi.ome_output_path is not None
    os.makedirs(os.path.dirname(fi.ome_output_path), exist_ok=True)
    synth = np.zeros((4, 16, 16), dtype=np.uint16)  # ZYX, no T
    tifffile.imwrite(
        fi.ome_output_path, synth,
        metadata={'axes': 'ZYX'}, photometric='minisblack',
    )
    mtime_before = os.path.getmtime(fi.ome_output_path)
    time.sleep(1.1)  # cross filesystem mtime resolution boundary

    # file_info.axes='TZYX' — mismatch should trigger regen.
    assert fi.axes is not None
    assert 'T' in fi.axes
    info = ImInfo.from_file_info(fi)
    assert info.im_path is not None
    mtime_after = os.path.getmtime(info.im_path)
    assert mtime_after > mtime_before


# ============================================================================
# B. ``pipeline_paths`` 18-key surface
# ============================================================================


def test_pipeline_paths_has_18_expected_keys(imageinfo_3d: ImInfo) -> None:
    """The 18 pipeline keys form the on-disk contract for the entire pipeline.

    See [[pipeline_paths]] glossary entry. Each of the 7 algorithmic
    stages reads/writes via ``im_info.pipeline_paths[key]``; adding,
    removing, or renaming a key is a cross-cutting change.
    """
    assert set(imageinfo_3d.pipeline_paths.keys()) == _PIPELINE_PATH_KEYS


def test_pipeline_paths_contain_key_as_filename_suffix(imageinfo_3d: ImInfo) -> None:
    """Every pipeline path's filename ends with ``-{key}{ext}`` (per ``create_output_path``)."""
    for key, path in imageinfo_3d.pipeline_paths.items():
        # Strip extension and check that the basename ends with -{key}.
        # ``create_output_path`` formats: ``{base}-{pipeline_path}{ext}``.
        basename = os.path.basename(path)
        # Drop everything after the key onwards (extension may be .ome.tif / .npy / .csv / .pkl).
        assert f'-{key}' in basename, (
            f"pipeline_path {key!r} missing from filename {basename!r}"
        )


def test_pipeline_paths_user_output_routing(imageinfo_3d: ImInfo) -> None:
    """The 5 ``features_*`` keys route to user ``output_dir``, not ``nellie_necessities_dir``.

    Per ``_create_output_paths``: ``for_nellie=False`` is passed only for
    the 5 features CSVs. Every other key uses ``for_nellie=True``
    (default) and routes through ``nellie_necessities_output_path_no_ext``.
    """
    necessities_dir = imageinfo_3d.file_info.nellie_necessities_dir
    for key in _USER_OUTPUT_KEYS:
        path = imageinfo_3d.pipeline_paths[key]
        # User output paths must NOT be inside nellie_necessities/.
        assert necessities_dir not in path, (
            f"features key {key!r} unexpectedly routed to nellie_necessities: {path!r}"
        )


def test_pipeline_paths_nellie_necessities_routing(imageinfo_3d: ImInfo) -> None:
    """All non-features keys route into the ``nellie_necessities`` subdir."""
    necessities_dir = imageinfo_3d.file_info.nellie_necessities_dir
    for key in _NELLIE_NECESSITIES_KEYS:
        path = imageinfo_3d.pipeline_paths[key]
        assert necessities_dir in path, (
            f"key {key!r} expected in nellie_necessities, got {path!r}"
        )


# ============================================================================
# C0. transform_to_axes (pure-logic)
# ============================================================================
#
# Slice 4 promoted ``_normalize_axes`` / ``_normalize_memmap`` into the
# module-level ``transform_to_axes`` function with two modes (canonical
# when ``target_axes=None``; match when a target string is supplied).
# These pure-logic tests exercise edge cases that wouldn't fire through
# the existing integration paths below.


def test_transform_to_axes_canonical_t_in_middle_position() -> None:
    """``axes='ZTYX'`` with T at position 1 reorders to canonical ``'TZYX'``.

    Distinct from ``test_transform_to_axes_moves_t_to_position_zero``
    in that no ImInfo state is needed — pure module-level call.
    """
    data = np.zeros((16, 2, 512, 512), dtype=np.uint16)
    out, axes = transform_to_axes(data, 'ZTYX')
    assert axes == 'TZYX'
    assert out.shape == (2, 16, 512, 512)


def test_transform_to_axes_match_z_in_target_but_not_source_raises() -> None:
    """Target has Z but source lacks Z → set-equality check raises ``ValueError``.

    This path was un-exercised before Slice 4 because ``_normalize_memmap``'s
    caller always passed ``self.axes`` as target. In any realistic workflow
    the target axes set is a subset of (or equal to) the source axes set;
    Slice 4's set-equality check now also detects the inverse mismatch.
    """
    data = np.zeros((2, 16, 16), dtype=np.uint16)
    with pytest.raises(ValueError, match="Axes mismatch"):
        transform_to_axes(data, 'TYX', target_axes='TZYX')


def test_transform_to_axes_canonical_raises_on_none_source_axes() -> None:
    """``source_axes=None`` raises ``ValueError`` — the new contract.

    Slice 3's ``_normalize_axes`` raised the same message; Slice 4's
    ``transform_to_axes`` re-pins it as the function's first guard.
    The ``get_memmap`` caller short-circuits on ``file_axes is None``
    BEFORE calling ``transform_to_axes`` (verifier.py: ``if file_axes
    is None: return memmap``), so this guard only fires for direct
    callers.
    """
    data = np.zeros((2, 16, 16), dtype=np.uint16)
    with pytest.raises(ValueError, match="Axes metadata is not initialized"):
        transform_to_axes(data, None)


def test_transform_to_axes_canonical_rejects_c_in_source() -> None:
    """``axes='TCYX'`` raises ``ValueError("Unsupported axes found: ['C']")``.

    Pins the canonical-mode allowed-set check (``{'T','Z','Y','X'}``).
    Distinct from ``test_transform_to_axes_rejects_c_axis`` below in
    that the data here is 4D-with-T-already-present; the older test
    drives the T-prepend path.
    """
    data = np.zeros((2, 3, 16, 16), dtype=np.uint16)
    with pytest.raises(ValueError, match=r"Unsupported axes found: \['C'\]"):
        transform_to_axes(data, 'TCYX')


# ============================================================================
# C. ``transform_to_axes`` (canonical mode, called from ``_get_ome_metadata``)
# ============================================================================
#
# These tests call the module-level ``transform_to_axes`` function directly
# with synthetic numpy arrays. End-to-end testing through the constructor
# would re-write the synthetic file via ``save_ome_tiff`` (which has its
# own normalization), so the only practical way to characterize each branch
# is the direct call. Slice 4 of the dechaos refactor unified the three
# normalizer copies into one — these tests pin the canonical-mode surface
# (no ``target_axes``) that the unified normalizer must reproduce.


def test_transform_to_axes_prepends_t_when_missing() -> None:
    """``axes='ZYX'`` with non-singleton Z gets a T prepended."""
    data = np.zeros((4, 16, 16), dtype=np.uint16)
    norm_data, norm_axes = transform_to_axes(data, 'ZYX')
    assert norm_axes == 'TZYX'
    assert norm_data.shape == (1, 4, 16, 16)


def test_transform_to_axes_moves_t_to_position_zero() -> None:
    """``axes='ZTYX'`` with T at position 1 gets reordered to ``'TZYX'``."""
    data = np.zeros((16, 2, 16, 16), dtype=np.uint16)
    norm_data, norm_axes = transform_to_axes(data, 'ZTYX')
    assert norm_axes == 'TZYX'
    assert norm_data.shape == (2, 16, 16, 16)


def test_transform_to_axes_squeezes_singleton_z() -> None:
    """``axes='ZYX'`` with ``Z=1`` collapses to ``'TYX'`` after T-prepend + Z-squeeze."""
    data = np.zeros((1, 16, 16), dtype=np.uint16)
    norm_data, norm_axes = transform_to_axes(data, 'ZYX')
    assert norm_axes == 'TYX'
    assert norm_data.shape == (1, 16, 16)


def test_transform_to_axes_preserves_z_when_not_singleton() -> None:
    """``axes='ZYX'`` with ``Z>1`` preserves Z in the canonical ``'TZYX'`` order."""
    data = np.zeros((16, 16, 16), dtype=np.uint16)
    norm_data, norm_axes = transform_to_axes(data, 'ZYX')
    assert norm_axes == 'TZYX'
    assert norm_data.shape == (1, 16, 16, 16)


def test_transform_to_axes_2d_yx_gets_t_prepended() -> None:
    """Pure ``'YX'`` data gains a single-timepoint T prefix → ``'TYX'``."""
    data = np.zeros((16, 16), dtype=np.uint16)
    norm_data, norm_axes = transform_to_axes(data, 'YX')
    assert norm_axes == 'TYX'
    assert norm_data.shape == (1, 16, 16)


def test_transform_to_axes_missing_yx_raises() -> None:
    """Axes lacking Y or X raise ``ValueError``."""
    data = np.zeros((4, 16), dtype=np.uint16)
    with pytest.raises(ValueError, match="Axes must include both Y and X"):
        transform_to_axes(data, 'ZX')


def test_transform_to_axes_rejects_c_axis() -> None:
    """Axes containing C raise ``ValueError("Unsupported axes found: ['C']")``.

    Pins current intentional behavior: ``transform_to_axes`` (canonical
    mode) excludes ``C`` from its allowed set (``{'T', 'Z', 'Y', 'X'}``).
    ``FileInfo._axis_errors`` accepts multichannel input via its allowed
    set (verifier.py:373 — ``{'T', 'Z', 'Y', 'X', 'C'}``); ``save_ome_tiff``
    collapses ``C`` via channel-take before writing the canonical
    OME-TIFF, so ``ImInfo`` should only ever read post-collapse files.
    The two allowed sets diverge intentionally — see dechaos report
    Pass 5 ("normalizer excludes C; ``_axis_errors`` includes C in
    allowed set") and ``test_verifier_fileinfo.py``.
    """
    data = np.zeros((1, 4, 16, 16), dtype=np.uint16)
    with pytest.raises(ValueError, match=r"Unsupported axes found: \['C'\]"):
        transform_to_axes(data, 'CZYX')


def test_transform_to_axes_returns_string_in_canonical_order() -> None:
    """Returned axes is a string with T first, Z (if present) second, then YX."""
    # Already-canonical input — output should equal input.
    data = np.zeros((2, 4, 16, 16), dtype=np.uint16)
    _, norm_axes = transform_to_axes(data, 'TZYX')
    assert isinstance(norm_axes, str)
    assert norm_axes == 'TZYX'

    # Reordered input — output must still be canonical.
    data = np.zeros((4, 2, 16, 16), dtype=np.uint16)
    _, norm_axes = transform_to_axes(data, 'ZTYX')
    assert norm_axes == 'TZYX'


# ============================================================================
# D. ``transform_to_axes`` (match mode, called from ``get_memmap``)
# ============================================================================
#
# The 3D ``imageinfo_3d`` has self.axes='TZYX'; the 2D variant has
# self.axes='TYX'. ``transform_to_axes`` (with ``target_axes=info.axes``)
# uses the target string to decide whether to squeeze Z. Use whichever
# fixture matches the test's intent.


def test_transform_to_axes_match_no_op_when_axes_match(imageinfo_3d: ImInfo) -> None:
    """When ``file_axes == self.axes``, the memmap shape passes through."""
    data = np.zeros((2, 17, 192, 279), dtype=np.uint16)
    out, _ = transform_to_axes(data, 'TZYX', target_axes=imageinfo_3d.axes)
    assert out.shape == (2, 17, 192, 279)


def test_transform_to_axes_match_prepends_t_when_missing(imageinfo_3d: ImInfo) -> None:
    """``file_axes='ZYX'`` against ``self.axes='TZYX'`` gains a T prefix."""
    data = np.zeros((17, 192, 279), dtype=np.uint16)
    out, _ = transform_to_axes(data, 'ZYX', target_axes=imageinfo_3d.axes)
    assert out.shape == (1, 17, 192, 279)


def test_transform_to_axes_match_squeezes_singleton_z(imageinfo_2d: ImInfo) -> None:
    """When ``self.axes`` lacks Z but the memmap has ``Z=1``, Z is squeezed."""
    # 2D ImInfo has self.axes='TYX' — so a TZYX memmap with Z=1 must squeeze.
    data = np.zeros((2, 1, 192, 279), dtype=np.uint16)
    out, _ = transform_to_axes(data, 'TZYX', target_axes=imageinfo_2d.axes)
    assert out.shape == (2, 192, 279)


def test_transform_to_axes_match_z_gt_1_when_target_lacks_z_raises(
    imageinfo_2d: ImInfo,
) -> None:
    """``Z>1`` against a target that lacks Z is unrecoverable — raises ``ValueError``."""
    data = np.zeros((2, 5, 192, 279), dtype=np.uint16)
    with pytest.raises(
        ValueError,
        match="Z axis present with size > 1, but target axes lacks Z",
    ):
        transform_to_axes(data, 'TZYX', target_axes=imageinfo_2d.axes)


def test_get_memmap_returns_memmap_unchanged_when_axes_lookup_fails(
    imageinfo_3d: ImInfo, tmp_path: Path, monkeypatch,
) -> None:
    """``get_memmap`` returns the raw memmap when the axes-lookup branch raises.

    Slice 4 moved the ``file_axes is None`` early-return out of the
    normalizer (which now raises ``ValueError`` on ``None``) and into
    the ``get_memmap`` caller. We let ``tifffile.memmap`` succeed
    normally, then swap ``TiffFile`` to a raising stub for the
    axes-lookup that follows — the ``except`` branch fires, ``file_axes``
    stays ``None``, and ``get_memmap`` returns the raw memmap
    untransformed. (The old ``_normalize_memmap`` had this early-return
    baked into the normalizer itself; the new contract cleanly separates
    the data-shape transform from the missing-axes fallback.)
    """
    # Build a synthetic TIFF the memmap call can succeed against.
    p = tmp_path / "synthetic.tif"
    arr = np.zeros((2, 17, 192, 279), dtype=np.uint16)
    tifffile.imwrite(str(p), arr, metadata={'axes': 'TZYX'},
                     photometric='minisblack')

    from nellie.im_info import verifier as verifier_module

    # Pre-build the memmap so tifffile.memmap inside get_memmap can
    # succeed without invoking the raising TiffFile we're about to
    # install. We monkeypatch ``tifffile.memmap`` in the verifier module
    # to return our pre-built memmap, then replace ``TiffFile`` with a
    # stub that always raises so the axes-lookup ``except`` branch fires.
    real_memmap = verifier_module.tifffile.memmap(str(p), mode='r+')

    def _stub_memmap(*_args, **_kwargs):
        return real_memmap

    def _raise(*_args, **_kwargs):
        raise RuntimeError("synthetic axes lookup failure")

    monkeypatch.setattr(verifier_module.tifffile, 'memmap', _stub_memmap)
    monkeypatch.setattr(verifier_module.tifffile, 'TiffFile', _raise)

    out = imageinfo_3d.get_memmap(str(p))

    # Memmap returned unchanged (no transform).
    assert out is real_memmap
    assert out.shape == (2, 17, 192, 279)


def test_transform_to_axes_match_raises_on_none_source_axes() -> None:
    """Pin the new contract: ``transform_to_axes`` raises on ``None`` source axes.

    The old ``_normalize_memmap`` short-circuited on
    ``file_axes is None`` (returned the raw memmap unchanged). Slice 4
    moved that early-return into the ``get_memmap`` caller; the
    normalizer itself now raises ``ValueError`` so direct callers that
    forget to pre-validate their axes string fail loudly instead of
    silently bypassing the transform.
    """
    data = np.zeros((2, 17, 192, 279), dtype=np.uint16)
    with pytest.raises(ValueError, match="Axes metadata is not initialized"):
        transform_to_axes(data, None, target_axes='TZYX')


# ============================================================================
# E. ``_check_axes_exist``
# ============================================================================


def test_check_axes_exist_3d_fixture(imageinfo_3d: ImInfo) -> None:
    """3D fixture: Z=17, T=2 → ``no_z=False``, ``no_t=False``."""
    assert imageinfo_3d.no_z is False
    assert imageinfo_3d.no_t is False


def test_check_axes_exist_2d_fixture(imageinfo_2d: ImInfo) -> None:
    """2D fixture: no Z axis, T=2 → ``no_z=True``, ``no_t=False``."""
    assert imageinfo_2d.no_z is True
    assert imageinfo_2d.no_t is False


def test_check_axes_exist_single_timepoint_2d(tmp_path: Path) -> None:
    """Single-timepoint 2D source → ``no_z=True`` and ``no_t=True``.

    A YX TIFF is treated as ``axes='TYX'`` with ``T=1`` after the
    constructor's ``_normalize_axes`` prepends a singleton T. Because
    ``_check_axes_exist`` only flips ``no_t=False`` when ``T>1``, the
    flag stays True for single-frame inputs.
    """
    workdir = tmp_path / 'wd'
    workdir.mkdir()
    fi = _make_minimal_file_info(workdir, source_axes='YX')
    info = ImInfo.from_file_info(fi)
    assert info.axes == 'TYX'
    assert info.shape == (1, 16, 16)
    assert info.no_z is True
    assert info.no_t is True


# ============================================================================
# F. ``get_memmap``
# ============================================================================


def test_get_memmap_returns_array(imageinfo_3d: ImInfo) -> None:
    """``get_memmap`` returns a numpy-compatible array on the canonical path."""
    mm = imageinfo_3d.get_memmap(imageinfo_3d.im_path)
    assert isinstance(mm, np.ndarray)
    assert mm.shape == imageinfo_3d.shape


def test_get_memmap_default_mode_is_writable(imageinfo_3d: ImInfo) -> None:
    """``read_mode='r+'`` (default) returns a writable memmap."""
    mm = imageinfo_3d.get_memmap(imageinfo_3d.im_path)
    assert mm.flags.writeable is True


def test_get_memmap_read_only_mode(imageinfo_3d: ImInfo) -> None:
    """``read_mode='r'`` returns a read-only memmap."""
    mm = imageinfo_3d.get_memmap(imageinfo_3d.im_path, read_mode='r')
    assert mm.flags.writeable is False


# ============================================================================
# G. ``allocate_memory``
# ============================================================================


def test_allocate_memory_empty_creates_file_with_self_shape(
    make_imageinfo_3d,
) -> None:
    """``data=None`` writes an empty OME-TIFF at ``self.shape`` and ``dtype``."""
    info = make_imageinfo_3d()
    out_path = info.pipeline_paths['im_preprocessed']
    info.allocate_memory(out_path, dtype='float32')
    assert os.path.exists(out_path)
    with tifffile.TiffFile(out_path) as tif:
        assert tif.series[0].shape == info.shape
        assert tif.series[0].dtype == np.float32


def test_allocate_memory_with_data_round_trips(make_imageinfo_3d) -> None:
    """``data`` is written to disk and re-reads identically."""
    info = make_imageinfo_3d()
    out_path = info.pipeline_paths['im_instance_label']
    data = np.arange(int(np.prod(info.shape)), dtype=np.uint16).reshape(info.shape)
    info.allocate_memory(out_path, dtype='uint16', data=data)
    round_trip = tifffile.imread(out_path)
    assert np.array_equal(round_trip, data)


def test_allocate_memory_return_memmap(make_imageinfo_3d) -> None:
    """``return_memmap=True`` returns a numpy-compatible memmap-shaped array."""
    info = make_imageinfo_3d()
    out_path = info.pipeline_paths['im_skel']
    mm = info.allocate_memory(out_path, dtype='float32', return_memmap=True)
    assert isinstance(mm, np.ndarray)
    assert mm.shape == info.shape


def test_allocate_memory_writes_dim_res_to_ome_metadata(
    make_imageinfo_3d,
) -> None:
    """X / Y / Z / T physical sizes round-trip into the OME pixel metadata."""
    info = make_imageinfo_3d()
    out_path = info.pipeline_paths['im_preprocessed']
    info.allocate_memory(out_path, dtype='float32')
    comment = tifffile.tiffcomment(out_path)
    assert comment is not None
    ome = ome_types.from_xml(comment)
    assert info.dim_res is not None
    assert ome.images[0].pixels.physical_size_x == info.dim_res['X']
    assert ome.images[0].pixels.physical_size_y == info.dim_res['Y']
    assert ome.images[0].pixels.physical_size_z == info.dim_res['Z']
    assert ome.images[0].pixels.time_increment == info.dim_res['T']


def test_allocate_memory_writes_description_to_ome_metadata(
    make_imageinfo_3d,
) -> None:
    """The ``description`` argument round-trips into ``ome.images[0].description``."""
    info = make_imageinfo_3d()
    out_path = info.pipeline_paths['im_preprocessed']
    info.allocate_memory(out_path, dtype='float32', description='hello world')
    comment = tifffile.tiffcomment(out_path)
    assert comment is not None
    ome = ome_types.from_xml(comment)
    assert ome.images[0].description == 'hello world'


def test_allocate_memory_data_shape_mismatch_raises(make_imageinfo_3d) -> None:
    """Data dims that aren't the auto-fixable cases raise ``ValueError``.

    The auto-fixes are:
    - axes starts with T and ``data.ndim == len(axes) - 1`` → prepend T axis.
    - axes lacks T and ``data.ndim == len(axes) + 1`` → prepend T to axes.
    Anything else (e.g. a 2D array against a 4D ``axes='TZYX'``) raises.
    """
    info = make_imageinfo_3d()
    out_path = info.pipeline_paths['im_preprocessed']
    data = np.zeros((5, 5), dtype=np.float32)  # 2D, but info.axes='TZYX' (4D)
    with pytest.raises(ValueError, match='Data dimensions do not match axes'):
        info.allocate_memory(out_path, dtype='float32', data=data)


# ============================================================================
# H. ``remove_intermediates``
# ============================================================================


def test_remove_intermediates_preserves_csv_files(make_imageinfo_3d) -> None:
    """The 5 ``features_*`` CSVs survive ``remove_intermediates``.

    The implementation skips paths whose string contains ``'csv'`` (a
    substring match, not an extension match — see verifier.py:865-866).
    Pinned so Slice 2's clarification doesn't accidentally delete user
    output.
    """
    info = make_imageinfo_3d()
    # Plant dummy files at a few intermediate keys + a CSV key.
    intermediate_keys = ('im_preprocessed', 'im_instance_label', 'flow_vector_array')
    csv_keys = ('features_voxels', 'features_image')
    for key in intermediate_keys + csv_keys:
        path = info.pipeline_paths[key]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(b'x')

    _release_iminfo_memmap(info)  # Windows: release im_path mmap before delete
    info.remove_intermediates()

    # Intermediates deleted.
    for key in intermediate_keys:
        assert not os.path.exists(info.pipeline_paths[key]), (
            f"{key!r} should have been removed"
        )
    # CSVs survived.
    for key in csv_keys:
        assert os.path.exists(info.pipeline_paths[key]), (
            f"{key!r} (CSV) should have survived"
        )


def test_remove_intermediates_deletes_canonical_im_path(make_imageinfo_3d) -> None:
    """The canonical OME-TIFF at ``self.im_path`` is also removed.

    Per verifier.py:864, ``remove_intermediates`` iterates over
    ``pipeline_paths.values() + [self.im_path]``. The main file is part
    of the cleanup contract, not exempt from it.
    """
    info = make_imageinfo_3d()
    assert os.path.exists(info.im_path)
    _release_iminfo_memmap(info)  # Windows: release im_path mmap before delete
    info.remove_intermediates()
    assert not os.path.exists(info.im_path)


# ----------------------------------------------------------------------------
# H.1 ``DROPPABLE_KEYS`` + presets + ``remove_marked_intermediates``
# ----------------------------------------------------------------------------


_EXPECTED_DROPPABLE_KEYS = frozenset({
    'im_preprocessed',
    'im_instance_label',
    'im_skel',
    'im_skel_relabelled',
    'im_pixel_class',
    'im_marker',
    'im_distance',
    'im_border',
    'flow_vector_array',
    'voxel_matches',
    'im_branch_label_reassigned',
    'im_obj_label_reassigned',
    'adjacency_maps',
    'im_path',
})


def test_droppable_keys_membership_exact() -> None:
    """``DROPPABLE_KEYS`` is the 14-key universe, no CSVs."""
    assert DROPPABLE_KEYS == _EXPECTED_DROPPABLE_KEYS
    csv_keys = {
        'features_voxels', 'features_nodes', 'features_branches',
        'features_organelles', 'features_image',
    }
    assert DROPPABLE_KEYS.isdisjoint(csv_keys)


def test_keep_everything_preset_is_empty() -> None:
    assert KEEP_EVERYTHING_PRESET == frozenset()


def test_csvs_only_preset_equals_droppable_keys() -> None:
    assert CSVS_ONLY_PRESET == DROPPABLE_KEYS


def test_masks_and_csvs_preset_keeps_three_labels_plus_im_path() -> None:
    """Preset drops everything except the three label maps and ``im_path``."""
    kept = DROPPABLE_KEYS - MASKS_AND_CSVS_PRESET
    assert kept == frozenset({
        'im_instance_label',
        'im_branch_label_reassigned',
        'im_obj_label_reassigned',
        'im_path',
    })


def _plant_dummy_files(info: ImInfo, keys) -> None:
    """Create a small placeholder file at each pipeline_paths key."""
    for key in keys:
        if key == 'im_path':
            continue  # already exists from ImInfo construction
        path = info.pipeline_paths[key]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(b'x')


def test_remove_marked_intermediates_drops_only_marked_keys(make_imageinfo_3d) -> None:
    """Only the keys passed in are deleted; CSVs always survive."""
    info = make_imageinfo_3d()
    droppable_non_im_path = DROPPABLE_KEYS - {'im_path'}
    csv_keys = {
        'features_voxels', 'features_nodes', 'features_branches',
        'features_organelles', 'features_image',
    }
    _plant_dummy_files(info, droppable_non_im_path | csv_keys)

    drop = frozenset({'im_preprocessed', 'im_skel', 'flow_vector_array'})
    info.remove_marked_intermediates(drop_keys=drop)

    for key in drop:
        assert not os.path.exists(info.pipeline_paths[key]), (
            f"{key!r} should have been deleted"
        )
    for key in (droppable_non_im_path - drop):
        assert os.path.exists(info.pipeline_paths[key]), (
            f"{key!r} should have survived (not in drop set)"
        )
    for key in csv_keys:
        assert os.path.exists(info.pipeline_paths[key]), (
            f"{key!r} (CSV) should always survive"
        )
    # im_path was not in the drop set, should still exist.
    assert os.path.exists(info.im_path)


def test_remove_marked_intermediates_empty_set_is_noop(make_imageinfo_3d) -> None:
    """Empty drop set deletes nothing."""
    info = make_imageinfo_3d()
    droppable_non_im_path = DROPPABLE_KEYS - {'im_path'}
    _plant_dummy_files(info, droppable_non_im_path)

    info.remove_marked_intermediates(drop_keys=frozenset())

    for key in droppable_non_im_path:
        assert os.path.exists(info.pipeline_paths[key])
    assert os.path.exists(info.im_path)


@pytest.mark.parametrize('bad_key', ['not_a_real_key', 'features_voxels'])
def test_remove_marked_intermediates_rejects_unknown_keys(
    make_imageinfo_3d, bad_key: str,
) -> None:
    """Keys outside ``DROPPABLE_KEYS`` (including CSV keys) raise."""
    info = make_imageinfo_3d()
    with pytest.raises(AssertionError, match=bad_key):
        info.remove_marked_intermediates(drop_keys=frozenset({bad_key}))


def test_remove_marked_intermediates_silently_skips_missing_files(
    make_imageinfo_3d,
) -> None:
    """Drop set includes a key whose file was never created — no error."""
    info = make_imageinfo_3d()
    # Don't plant adjacency_maps; it's only created when skip_nodes=False
    # at Hierarchy time, which the fixture doesn't run.
    assert not os.path.exists(info.pipeline_paths['adjacency_maps'])
    info.remove_marked_intermediates(drop_keys=frozenset({'adjacency_maps'}))
    # No exception raised; nothing changed.
    assert not os.path.exists(info.pipeline_paths['adjacency_maps'])


def test_remove_marked_intermediates_handles_im_path(make_imageinfo_3d) -> None:
    """``im_path`` resolves to ``self.im_path`` and gets deleted."""
    info = make_imageinfo_3d()
    assert os.path.exists(info.im_path)
    _release_iminfo_memmap(info)  # Windows: release mmap before delete
    info.remove_marked_intermediates(drop_keys=frozenset({'im_path'}))
    assert not os.path.exists(info.im_path)


def test_remove_intermediates_legacy_shim_matches_new_method(make_imageinfo_3d) -> None:
    """Legacy ``remove_intermediates()`` is equivalent to dropping all of ``DROPPABLE_KEYS``.

    Re-asserts the contract pinned by the two pre-existing legacy tests
    (CSVs survive, im_path deleted, all 12 image intermediates +
    adjacency_maps deleted if present) via the new shim path.
    """
    info = make_imageinfo_3d()
    droppable_non_im_path = DROPPABLE_KEYS - {'im_path'}
    csv_keys = {
        'features_voxels', 'features_nodes', 'features_branches',
        'features_organelles', 'features_image',
    }
    _plant_dummy_files(info, droppable_non_im_path | csv_keys)

    _release_iminfo_memmap(info)
    info.remove_intermediates()

    for key in droppable_non_im_path:
        assert not os.path.exists(info.pipeline_paths[key]), (
            f"{key!r} should have been deleted by legacy shim"
        )
    assert not os.path.exists(info.im_path)
    for key in csv_keys:
        assert os.path.exists(info.pipeline_paths[key]), (
            f"{key!r} (CSV) should have survived legacy shim"
        )


# ============================================================================
# I. ``load_image`` orchestrator (Slice 8 — final slice)
# ============================================================================


def test_load_image_returns_loaded_iminfo(tmp_path: Path) -> None:
    """``load_image(path)`` returns a fully-loaded ImInfo with axes/shape/dim_res populated."""
    from nellie.im_info import load_image

    workdir = tmp_path / 'wd'
    workdir.mkdir()
    src = copy_fixture_to_tmp(FIXTURE_3D_PATH, workdir)

    info = load_image(src)
    assert info.axes == 'TZYX'
    assert info.shape == (2, 17, 192, 279)
    assert info.dim_res == pytest.approx(
        {'X': 0.0655, 'Y': 0.0655, 'Z': 0.25, 'T': 4.535566806793213}
    )
    assert info.im is not None
    assert info.pipeline_paths
    assert info.im_path is not None
    assert os.path.exists(info.im_path)


def test_load_image_equivalent_to_explicit_boot(tmp_path: Path) -> None:
    """``load_image(path)`` produces an ImInfo equivalent to the manual 4-step boot."""
    from nellie.im_info import load_image

    workdir = tmp_path / 'wd'
    workdir.mkdir()
    src_a = copy_fixture_to_tmp(FIXTURE_3D_PATH, workdir / 'a')
    src_b = copy_fixture_to_tmp(FIXTURE_3D_PATH, workdir / 'b')

    via_orchestrator = load_image(src_a)

    fi = FileInfo(str(src_b))
    fi.find_metadata()
    fi.load_metadata()
    via_explicit = ImInfo.from_file_info(fi)

    assert via_orchestrator.axes == via_explicit.axes
    assert via_orchestrator.shape == via_explicit.shape
    assert via_orchestrator.dim_res == via_explicit.dim_res
    assert via_orchestrator.no_z == via_explicit.no_z
    assert via_orchestrator.no_t == via_explicit.no_t
    assert set(via_orchestrator.pipeline_paths.keys()) == set(via_explicit.pipeline_paths.keys())


def test_load_image_forwards_output_dir_kwarg(tmp_path: Path) -> None:
    """``load_image(path, output_dir=...)`` forwards the kwarg to FileInfo."""
    from nellie.im_info import load_image

    src = copy_fixture_to_tmp(FIXTURE_3D_PATH, tmp_path / 'src')
    custom_out = tmp_path / 'custom_output'

    info = load_image(src, output_dir=str(custom_out))
    assert info.file_info.output_dir == str(custom_out)
    assert custom_out.exists()
