"""Opt-in performance tests for the full ``nellie.run.run()`` orchestrator.

Skipped by default — invoke with ``pytest -m benchmark`` to run.
Mirrors the per-stage perf-test pattern (see :mod:`tests.test_filtering_perf`)
but at the pipeline level: the seven algorithmic stages chained
together via the ``ImInfo`` on-disk contract.

This test surfaces whether stage-level perf wins compound into
pipeline-level wins. The per-stage perf files measure each stage with
its upstream outputs pre-loaded from session caches; the pipeline-level
test measures the full chain start-to-finish, including any I/O,
memmap, or allocation overhead that lives between stages but isn't
attributable to any one of them.

Two outputs per fixture:

1. **Total wall-clock** — single end-to-end ``run()`` time, after one
   warmup pass.
2. **Per-stage breakdown** — captured from ``run(timeit=True)``'s
   stdout printout. Re-emitted as ``[perf]`` lines so the per-stage
   numbers are visible alongside the total without consulting captured
   output separately.

Iters = 1 (post-warmup). The pipeline is heavy enough that median-of-3
becomes wall-clock-prohibitive for dev iteration; absolute numbers
vary per machine anyway, and the per-stage breakdown gives enough
signal to spot a regression in any single stage even from one run.

**No MPS variant.** The per-stage MPS perf tests already cover MPS
coverage. A pipeline-level MPS test would mostly measure CPU work
(only filter + labelling + networking + hu_tracking are MPS-onboarded;
markers + voxel_reassign + hierarchy fall back to CPU), so the signal
is weak and confusingly mixed. Add an MPS variant when all seven
stages are onboarded.
"""

from __future__ import annotations

import gc
import shutil
import time
from pathlib import Path

import pytest

from nellie.im_info.verifier import FileInfo
from nellie.run import run


pytestmark = pytest.mark.benchmark


def _make_fresh_file_info(source: Path, workdir: Path) -> FileInfo:
    """Copy the source fixture into a fresh subdir and build a FileInfo.

    ``run()`` calls ``ImInfo.from_file_info(file_info)`` itself, so we
    pass the FileInfo (not an ImInfo). Each invocation gets its own
    workdir so the output tree is clean.
    """
    dst = workdir / source.name
    shutil.copy(source, dst)
    file_info = FileInfo(str(dst))
    file_info.find_metadata()
    file_info.load_metadata()
    return file_info


@pytest.mark.parametrize("dim", ["2d", "3d"])
def test_pipeline_run_baseline_prints_wall_clock(
    dim, tmp_path: Path, capsys
) -> None:
    """Print a full pipeline wall-clock + per-stage breakdown.

    Runs ``run(file_info, timeit=True)`` on the 2D and 3D yeast
    fixtures. Captures the per-stage printout from ``timeit=True``
    and re-emits each line as a ``[perf]`` line. No assertion — read
    the output.
    """
    repo_root = Path(__file__).resolve().parents[1]
    fixture_path = (
        repo_root / "tests" / "fixtures" /
        ("yeast_2d_t0_to_1.ome.tif" if dim == "2d" else "yeast_3d_t0_to_1.ome.tif")
    )

    def _one_run() -> None:
        # Each run gets its own subdir so the output tree is clean
        # (and so subsequent runs don't accidentally short-circuit on
        # cached intermediates).
        sub = tmp_path / f"pipeline_{dim}_{time.perf_counter_ns()}"
        sub.mkdir()
        file_info = _make_fresh_file_info(fixture_path, sub)
        run(file_info, timeit=True)
        # Drop strong refs so memmaps + temp dirs can be cleaned up
        # promptly (Windows file-lock concern in other perf files).
        del file_info
        gc.collect()

    # Warm up
    _one_run()
    capsys.readouterr()  # discard warmup printouts

    start = time.perf_counter()
    _one_run()
    elapsed = time.perf_counter() - start

    # Capture the per-stage printout from timeit=True. The lines look
    # like "Nellie Pipeline: Filter step took X seconds" — strip the
    # "Nellie Pipeline: " prefix and re-emit as [perf] lines.
    captured = capsys.readouterr()
    timeit_lines = [
        line for line in captured.out.splitlines()
        if line.startswith("Nellie Pipeline:")
    ]

    with capsys.disabled():
        print(f"\n[perf] Pipeline run() {dim} total wall-clock: {elapsed * 1000:.1f} ms")
        for line in timeit_lines:
            # Reformat "Nellie Pipeline: Filter step took 0.1234 seconds" →
            # "[perf] Pipeline {dim} Filter: 123.4 ms"
            print(f"[perf] Pipeline {dim} {line.removeprefix('Nellie Pipeline: ')}")
