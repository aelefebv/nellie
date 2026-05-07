"""Shared pytest fixtures for the nellie test suite.

Each ImInfo fixture copies its source file into a session-scoped tmp dir
before constructing ImInfo. ImInfo's constructor writes a sibling
``nellie_output/`` tree next to the input file; the copy keeps those
writes out of ``tests/fixtures/`` so the committed fixtures stay clean
across runs.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from nellie.im_info.verifier import FileInfo, ImInfo

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
