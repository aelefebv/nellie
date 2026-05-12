"""Tests for the ``cleanup_drop_keys`` kwarg on ``nellie.run.run()``.

Slice 2 of #245 wires per-output retention through the standalone
pipeline driver. The cleanup contract is:

- ``cleanup_drop_keys=None`` (default) leaves all on-disk state intact.
- An empty ``frozenset`` is also a no-op (truthiness check).
- A non-empty ``frozenset`` calls ``ImInfo.remove_marked_intermediates``
  after ``Hierarchy.run()`` succeeds.
- A mid-pipeline failure propagates without running cleanup.

The heavy pipeline stages are monkeypatched with cheap stubs that just
``touch`` the expected output files for their stage. This keeps each
test sub-second while still exercising the full ``run()`` orchestrator
and the post-Hierarchy cleanup branch end-to-end.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

import nellie.run as run_mod
from nellie.im_info.verifier import CSVS_ONLY_PRESET, FileInfo
from nellie.run import run

# ``conftest`` is on ``sys.path`` because pytest's rootdir-based collection
# adds the test directory before importing test modules.
from conftest import FIXTURE_2D_PATH  # type: ignore[import-not-found]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fresh_file_info(workdir: Path) -> FileInfo:
    """Copy the 2D fixture into a fresh subdir and build a FileInfo."""
    dst = workdir / FIXTURE_2D_PATH.name
    shutil.copy(FIXTURE_2D_PATH, dst)
    fi = FileInfo(str(dst))
    fi.find_metadata()
    fi.load_metadata()
    return fi


def _make_stage_stub(touch_keys: tuple[str, ...]):
    """Build a stage stub class that touches the given pipeline_paths keys when run."""
    class _StageStub:
        def __init__(self, im_info, config=None, *args, **kwargs):
            self.im_info = im_info

        def run(self):
            for key in touch_keys:
                path = self.im_info.pipeline_paths[key]
                os.makedirs(os.path.dirname(path), exist_ok=True)
                if not os.path.exists(path):
                    with open(path, 'wb') as f:
                        f.write(b'x')
    return _StageStub


_STAGE_TO_OUTPUTS: dict[str, tuple[str, ...]] = {
    'Filter': ('im_preprocessed',),
    'Label': ('im_instance_label',),
    'Network': ('im_skel', 'im_skel_relabelled', 'im_pixel_class'),
    'Markers': ('im_marker', 'im_distance', 'im_border'),
    'HuMomentTracking': ('flow_vector_array', 'voxel_matches'),
    'VoxelReassigner': ('im_branch_label_reassigned', 'im_obj_label_reassigned'),
    'Hierarchy': ('adjacency_maps',),
}

# Every droppable key in pipeline_paths that the stubs touch (excludes
# the special ``im_path`` key, which is the canonical OME-TIFF and
# always exists from FileInfo construction).
_ALL_TOUCHED_KEYS: frozenset[str] = frozenset(
    key for keys in _STAGE_TO_OUTPUTS.values() for key in keys
)


@pytest.fixture
def stub_run_stages(monkeypatch):
    """Replace heavy pipeline stages in ``nellie.run`` with cheap stubs.

    Each stub touches the on-disk paths that the real stage would have
    written. Cleanup logic in ``run()`` then has real files to delete.
    """
    for stage_name, outputs in _STAGE_TO_OUTPUTS.items():
        monkeypatch.setattr(run_mod, stage_name, _make_stage_stub(outputs))
    # Configs are passed as positional args to the stubs which ignore them.
    for cfg_name in (
        'FrangiConfig', 'LabelConfig', 'NetworkConfig', 'MarkersConfig',
        'HuMomentTrackingConfig', 'VoxelReassignerConfig', 'HierarchyConfig',
    ):
        monkeypatch.setattr(run_mod, cfg_name, lambda *a, **kw: None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_run_with_no_cleanup_drop_keys_leaves_all_intermediates(
    stub_run_stages, tmp_path: Path,
) -> None:
    """Default ``cleanup_drop_keys=None`` → no files deleted."""
    fi = _make_fresh_file_info(tmp_path)
    info = run(fi)

    for key in _ALL_TOUCHED_KEYS:
        assert os.path.exists(info.pipeline_paths[key]), (
            f"{key!r} should still exist (no cleanup configured)"
        )
    assert os.path.exists(info.im_path)


def test_run_with_keep_everything_preset_is_noop(
    stub_run_stages, tmp_path: Path,
) -> None:
    """Empty frozenset is a no-op (falsy → cleanup branch skipped)."""
    fi = _make_fresh_file_info(tmp_path)
    info = run(fi, cleanup_drop_keys=frozenset())

    for key in _ALL_TOUCHED_KEYS:
        assert os.path.exists(info.pipeline_paths[key])
    assert os.path.exists(info.im_path)


def test_run_with_csvs_only_preset_drops_all_droppable(
    stub_run_stages, tmp_path: Path,
) -> None:
    """``CSVS_ONLY_PRESET`` deletes every droppable intermediate touched by the stubs.

    CSV survival is pinned in the Slice 1 tests
    (``test_remove_marked_intermediates_drops_only_marked_keys``); here
    we only verify the wiring: passing ``CSVS_ONLY_PRESET`` to ``run()``
    deletes the droppable intermediates AND the canonical OME-TIFF.
    """
    fi = _make_fresh_file_info(tmp_path)
    info = run(fi, cleanup_drop_keys=CSVS_ONLY_PRESET)

    for key in _ALL_TOUCHED_KEYS:
        assert not os.path.exists(info.pipeline_paths[key]), (
            f"{key!r} should have been deleted by CSVS_ONLY_PRESET"
        )
    assert not os.path.exists(info.im_path), (
        "im_path should have been deleted by CSVS_ONLY_PRESET"
    )


def test_run_with_partial_drop_keys_drops_only_marked(
    stub_run_stages, tmp_path: Path,
) -> None:
    """A subset drop set deletes only the marked file; others survive."""
    fi = _make_fresh_file_info(tmp_path)
    drop = frozenset({'im_preprocessed'})
    info = run(fi, cleanup_drop_keys=drop)

    assert not os.path.exists(info.pipeline_paths['im_preprocessed']), (
        "im_preprocessed should have been deleted"
    )
    for key in (_ALL_TOUCHED_KEYS - drop):
        assert os.path.exists(info.pipeline_paths[key]), (
            f"{key!r} should still exist (not in drop set)"
        )
    assert os.path.exists(info.im_path)


def test_run_propagates_stage_failure_without_cleanup(
    monkeypatch, tmp_path: Path,
) -> None:
    """Mid-pipeline failure → exception propagates, no cleanup runs.

    Stub Filter to write its output (so we can detect whether cleanup
    fired), then stub Label to raise. The failure must propagate and the
    Filter output must survive (cleanup would have deleted it).
    """
    monkeypatch.setattr(run_mod, 'Filter', _make_stage_stub(('im_preprocessed',)))
    monkeypatch.setattr(run_mod, 'FrangiConfig', lambda *a, **kw: None)

    class _RaisingLabel:
        def __init__(self, *a, **kw):
            pass

        def run(self):
            raise RuntimeError("intentional mid-pipeline failure for test")

    monkeypatch.setattr(run_mod, 'Label', _RaisingLabel)
    monkeypatch.setattr(run_mod, 'LabelConfig', lambda *a, **kw: None)

    fi = _make_fresh_file_info(tmp_path)
    with pytest.raises(RuntimeError, match="intentional mid-pipeline failure"):
        run(fi, cleanup_drop_keys=CSVS_ONLY_PRESET)

    # Build an ImInfo to resolve pipeline_paths for the post-failure check.
    from nellie.im_info.verifier import ImInfo
    info = ImInfo.from_file_info(fi)
    assert os.path.exists(info.pipeline_paths['im_preprocessed']), (
        "im_preprocessed should survive — cleanup should NOT have run "
        "after the Label failure"
    )
    assert os.path.exists(info.im_path), (
        "im_path should survive — cleanup should NOT have run after failure"
    )
