"""MPS / CPU numerical equivalence tests for the nellie pipeline stages.

Each test runs the same stage twice on the same input — once with
``device="cpu"``, once with ``device="mps"`` — and asserts the
backends agree within float32 tolerance.

All tests are gated by the ``mps`` pytest marker (deselected by default
in ``pyproject.toml``). Run with ``pytest -m mps`` on a Mac with
``pip install 'nellie[mps]'`` and a working MPS device. Tests skip
cleanly when torch is missing or MPS is unavailable.

**On the chosen tolerances.** The PRD (#140 § Testing Decisions) calls
for vesselness mean within 0.5% relative + per-voxel max abs diff
within "float32 tolerance for typical work." Empirically the real
numbers on the yeast fixtures are way below those:

  3D yeast fixture:  rel_mean_diff ~1e-6  max_abs_diff ~1e-8
  2D yeast fixture:  rel_mean_diff ~1e-7  max_abs_diff ~1e-7

The tolerances here use the PRD ceiling (0.5% / 1e-4) on purpose, so
the test still passes if MPS rounding drifts a touch across torch
versions. Tightening them further would be a regression-detection
win, but the current bar is what the slice acceptance criteria
specify and it's what reviewers should evaluate against.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest

from nellie.segmentation.filtering import Filter, FrangiConfig
from nellie.segmentation.labelling import Label, LabelConfig
from nellie.segmentation.networking import Network, NetworkConfig
from nellie.tracking.hu_tracking import HuMomentTracking, HuMomentTrackingConfig
from nellie.utils import adaptive_run


pytestmark = pytest.mark.mps


def _require_mps() -> None:
    """Skip the test cleanly if torch+MPS isn't actually available.

    The ``mps`` marker only deselects the tests by default; it doesn't
    enforce hardware. A user who runs ``pytest -m mps`` on a machine
    without torch+MPS still needs a graceful skip rather than a hard
    failure. ``mps_available`` returns False when torch is missing,
    when MPS isn't built into the wheel, or when the OS doesn't expose
    a Metal-capable device.
    """
    if not adaptive_run.mps_available():
        pytest.skip(
            "torch+MPS not available — install with `pip install 'nellie[mps]'` "
            "and run on Apple Silicon."
        )


def _release_filter(filt: Filter) -> None:
    """Drop a Filter's memmap references and force gc.

    Required on Windows: the ``im_preprocessed`` memmap holds an
    exclusive handle that blocks subsequent overwrites.
    """
    filt.frangi_memmap = None
    filt.im_memmap = None
    gc.collect()


def _release_label(lbl: Label) -> None:
    """Drop a Label's memmap references and force gc.

    Mirrors :func:`_release_filter` — same Windows file-lock concern,
    extra ``instance_label_memmap`` handle to release.
    """
    lbl.instance_label_memmap = None
    lbl.frangi_memmap = None
    lbl.im_memmap = None
    gc.collect()


def _release_network(net: Network) -> None:
    """Drop a Network's memmap references and force gc.

    Mirrors :func:`_release_label` — same Windows file-lock concern,
    six memmap handles to release (three inputs, three outputs).
    """
    net.skel_memmap = None
    net.pixel_class_memmap = None
    net.skel_relabelled_memmap = None
    net.label_memmap = None
    net.im_memmap = None
    net.im_frangi_memmap = None
    gc.collect()


def _release_hu(h: HuMomentTracking) -> None:
    """Drop a HuMomentTracking's memmap references and force gc.

    Mirrors :func:`_release_network` — five memmap handles to release
    (all five inputs that ``HuMomentTracking._allocate_memory`` opens).
    The ``flow_vector_array`` output is a regular ``.npy`` written via
    ``np.save``, not a memmap, so it doesn't need releasing.
    """
    h.label_memmap = None
    h.im_memmap = None
    h.im_frangi_memmap = None
    h.im_marker_memmap = None
    h.im_distance_memmap = None
    gc.collect()


def _run_filter(im_info, *, device: str) -> np.ndarray:
    """Run Filter end-to-end on ``device`` and return a numpy copy of the output."""
    filt = Filter(im_info, FrangiConfig(device=device), num_t=2)
    filt.run()
    out = np.array(filt.frangi_memmap)
    _release_filter(filt)
    return out


def _vesselness_stats(
    cpu: np.ndarray, mps: np.ndarray
) -> dict[str, float]:
    """Summary statistics comparing CPU and MPS vesselness outputs.

    Returned keys:
        cpu_mean / mps_mean — overall mean response magnitude
        rel_mean_diff — abs(mps_mean - cpu_mean) / max(cpu_mean, 1e-12)
        max_abs_diff — per-voxel L_inf difference
        l2_rel_diff — relative L2 distance (norm of diff / norm of cpu)
    """
    cpu_mean = float(cpu.mean())
    mps_mean = float(mps.mean())
    diff = mps - cpu
    max_abs = float(np.max(np.abs(diff)))
    cpu_norm = float(np.linalg.norm(cpu))
    diff_norm = float(np.linalg.norm(diff))
    return {
        "cpu_mean": cpu_mean,
        "mps_mean": mps_mean,
        "rel_mean_diff": abs(mps_mean - cpu_mean) / max(cpu_mean, 1e-12),
        "max_abs_diff": max_abs,
        "l2_rel_diff": diff_norm / max(cpu_norm, 1e-12),
    }


# ---------------------------------------------------------------------------
# Filter equivalence
# ---------------------------------------------------------------------------


def test_filter_equivalence_cpu_vs_mps_3d(make_imageinfo_3d, capsys) -> None:
    """``Filter.run()`` on the 3D yeast fixture: CPU vs MPS within tolerance.

    Tolerances:
      - ``rel_mean_diff <= 0.005`` — PRD ceiling of 0.5% relative mean
        difference. Real number on this fixture sits well below 1e-3.
      - ``max_abs_diff <= 1e-4`` — per-voxel L_inf difference in
        float32 territory. Empirically ~1e-5 on this fixture.

    If you tighten these, run a few times locally first; MPS reduction
    order drifts a touch run-to-run on cumulative-sum-ish ops.
    """
    _require_mps()

    cpu_out = _run_filter(make_imageinfo_3d(), device="cpu")
    mps_out = _run_filter(make_imageinfo_3d(), device="mps")

    assert mps_out.shape == cpu_out.shape
    assert mps_out.dtype == cpu_out.dtype == np.float32

    stats = _vesselness_stats(cpu_out, mps_out)

    with capsys.disabled():
        print(
            f"\n[mps eq 3D] cpu_mean={stats['cpu_mean']:.6e} "
            f"mps_mean={stats['mps_mean']:.6e} "
            f"rel_mean_diff={stats['rel_mean_diff']:.3e} "
            f"max_abs_diff={stats['max_abs_diff']:.3e} "
            f"l2_rel_diff={stats['l2_rel_diff']:.3e}"
        )

    assert stats["rel_mean_diff"] <= 0.005, (
        f"3D vesselness mean drifted {stats['rel_mean_diff']:.3%} between "
        f"CPU ({stats['cpu_mean']:.3e}) and MPS ({stats['mps_mean']:.3e}); "
        f"expected <= 0.5% per PRD #140."
    )
    assert stats["max_abs_diff"] <= 1e-4, (
        f"3D vesselness per-voxel max abs diff "
        f"({stats['max_abs_diff']:.3e}) exceeds the float32 tolerance "
        f"of 1e-4 used as the slice acceptance bar."
    )


def test_filter_equivalence_cpu_vs_mps_2d(make_imageinfo_2d, capsys) -> None:
    """``Filter.run()`` on the 2D yeast fixture: CPU vs MPS within tolerance.

    The 2D path additionally fuses a multi-scale LoG response (via the
    shim's ``gaussian_laplace``), so this exercises a different op
    surface than the 3D test. Same tolerance bar; empirically the LoG
    fusion adds a touch more drift than the 3D Hessian path but it
    stays within the ceiling.
    """
    _require_mps()

    cpu_out = _run_filter(make_imageinfo_2d(), device="cpu")
    mps_out = _run_filter(make_imageinfo_2d(), device="mps")

    assert mps_out.shape == cpu_out.shape
    assert mps_out.dtype == cpu_out.dtype == np.float32

    stats = _vesselness_stats(cpu_out, mps_out)

    with capsys.disabled():
        print(
            f"\n[mps eq 2D] cpu_mean={stats['cpu_mean']:.6e} "
            f"mps_mean={stats['mps_mean']:.6e} "
            f"rel_mean_diff={stats['rel_mean_diff']:.3e} "
            f"max_abs_diff={stats['max_abs_diff']:.3e} "
            f"l2_rel_diff={stats['l2_rel_diff']:.3e}"
        )

    assert stats["rel_mean_diff"] <= 0.005, (
        f"2D vesselness mean drifted {stats['rel_mean_diff']:.3%} between "
        f"CPU ({stats['cpu_mean']:.3e}) and MPS ({stats['mps_mean']:.3e}); "
        f"expected <= 0.5% per PRD #140."
    )
    assert stats["max_abs_diff"] <= 1e-4, (
        f"2D vesselness per-voxel max abs diff "
        f"({stats['max_abs_diff']:.3e}) exceeds the float32 tolerance "
        f"of 1e-4 used as the slice acceptance bar."
    )


# ---------------------------------------------------------------------------
# Label equivalence
# ---------------------------------------------------------------------------


def _run_label(im_info, *, device: str) -> np.ndarray:
    """Run Label end-to-end on ``device`` and return a numpy copy of the labels."""
    lbl = Label(im_info, LabelConfig(device=device), num_t=2)
    lbl.run()
    out = np.asarray(lbl.instance_label_memmap).copy()
    _release_label(lbl)
    return out


def _label_iou_max_match(cpu: np.ndarray, mps: np.ndarray) -> float:
    """Return the mean IoU after matching each MPS label to its best CPU overlap.

    For each foreground id in the MPS frame, find the CPU id with the
    largest spatial overlap, then compute IoU = |A ∩ B| / |A ∪ B|. Return
    the mean across MPS ids (background id 0 excluded). MPS ids that
    don't overlap any CPU foreground voxel score 0.0.

    The PRD #140 § Testing Decisions bar for labelling is "label IoU >
    0.95". Empirically on this fixture the labels are byte-identical
    (the structural ``ndi.label`` round-trips to scipy on CPU in both
    backends; only the upstream ``uniform_filter`` mask thresholding
    can drift), so this score sits at exactly 1.0 — the > 0.95 bar
    leaves slack for any future drift introduced by float32 rounding
    cascading through the mask threshold.
    """
    cpu_ids = np.unique(cpu)
    cpu_ids = cpu_ids[cpu_ids != 0]
    mps_ids = np.unique(mps)
    mps_ids = mps_ids[mps_ids != 0]
    if mps_ids.size == 0:
        # No foreground in MPS output: return 1.0 if CPU also has no
        # foreground (perfect agreement on empty), else 0.0.
        return 1.0 if cpu_ids.size == 0 else 0.0

    ious: list[float] = []
    for mid in mps_ids:
        m_mask = mps == int(mid)
        # Pick the CPU id with the largest overlap with this MPS id.
        overlaps_with_cpu_ids = cpu[m_mask]
        overlaps_with_cpu_ids = overlaps_with_cpu_ids[overlaps_with_cpu_ids != 0]
        if overlaps_with_cpu_ids.size == 0:
            ious.append(0.0)
            continue
        best_cpu = int(np.bincount(overlaps_with_cpu_ids).argmax())
        c_mask = cpu == best_cpu
        intersection = int((m_mask & c_mask).sum())
        union = int((m_mask | c_mask).sum())
        ious.append(intersection / union if union > 0 else 0.0)
    return float(np.mean(ious))


def _label_diagnostics(cpu: np.ndarray, mps: np.ndarray) -> dict[str, float]:
    """Summary stats for diagnosing CPU vs MPS Label drift.

    Returned keys:
        cpu_count / mps_count — total foreground component count
        cpu_voxels / mps_voxels — total foreground voxel count
        mean_iou — mean per-MPS-label IoU vs best CPU match
    """
    cpu_ids = np.unique(cpu)
    mps_ids = np.unique(mps)
    return {
        "cpu_count": float((cpu_ids != 0).sum()),
        "mps_count": float((mps_ids != 0).sum()),
        "cpu_voxels": float((cpu != 0).sum()),
        "mps_voxels": float((mps != 0).sum()),
        "mean_iou": _label_iou_max_match(cpu, mps),
    }


def test_label_equivalence_cpu_vs_mps_3d(make_label_imageinfo_3d, capsys) -> None:
    """``Label.run()`` on the 3D yeast fixture: CPU vs MPS within IoU tolerance.

    Tolerances:
      - ``mean_iou >= 0.95`` — PRD #140 § Testing Decisions bar.

    Note on tightness: the structural ``ndi.label`` and
    ``binary_fill_holes`` ops round-trip to scipy on CPU in both
    backends, so the only source of CPU/MPS drift is the convolutional
    ``uniform_filter`` (and the upstream Frangi memmap they share —
    which is precomputed once on CPU by the conftest cache, so it's
    byte-identical here). Empirically labels come out byte-identical on
    this hardware (mean_iou = 1.0); the > 0.95 bar leaves headroom for
    any future drift that float32 ``uniform_filter`` rounding might
    cascade through the mask threshold.
    """
    _require_mps()

    cpu_out = _run_label(make_label_imageinfo_3d(), device="cpu")
    mps_out = _run_label(make_label_imageinfo_3d(), device="mps")

    assert mps_out.shape == cpu_out.shape
    assert mps_out.dtype == cpu_out.dtype == np.int32

    stats = _label_diagnostics(cpu_out, mps_out)

    with capsys.disabled():
        print(
            f"\n[mps eq label 3D] cpu_count={int(stats['cpu_count'])} "
            f"mps_count={int(stats['mps_count'])} "
            f"cpu_voxels={int(stats['cpu_voxels'])} "
            f"mps_voxels={int(stats['mps_voxels'])} "
            f"mean_iou={stats['mean_iou']:.4f}"
        )

    assert stats["mean_iou"] >= 0.95, (
        f"3D Label mean IoU between CPU and MPS dropped to "
        f"{stats['mean_iou']:.3f}; expected >= 0.95 per PRD #140 "
        f"§ Testing Decisions."
    )


def test_label_equivalence_cpu_vs_mps_2d(make_label_imageinfo_2d, capsys) -> None:
    """``Label.run()`` on the 2D yeast fixture: CPU vs MPS within IoU tolerance.

    Same > 0.95 bar as the 3D test. The 2D path skips
    ``binary_fill_holes`` (only 3D fills holes — see
    ``Label._get_labels``) so the op surface exercised here is even
    smaller; expect equally tight or tighter agreement.
    """
    _require_mps()

    cpu_out = _run_label(make_label_imageinfo_2d(), device="cpu")
    mps_out = _run_label(make_label_imageinfo_2d(), device="mps")

    assert mps_out.shape == cpu_out.shape
    assert mps_out.dtype == cpu_out.dtype == np.int32

    stats = _label_diagnostics(cpu_out, mps_out)

    with capsys.disabled():
        print(
            f"\n[mps eq label 2D] cpu_count={int(stats['cpu_count'])} "
            f"mps_count={int(stats['mps_count'])} "
            f"cpu_voxels={int(stats['cpu_voxels'])} "
            f"mps_voxels={int(stats['mps_voxels'])} "
            f"mean_iou={stats['mean_iou']:.4f}"
        )

    assert stats["mean_iou"] >= 0.95, (
        f"2D Label mean IoU between CPU and MPS dropped to "
        f"{stats['mean_iou']:.3f}; expected >= 0.95 per PRD #140 "
        f"§ Testing Decisions."
    )


# ---------------------------------------------------------------------------
# Network equivalence
# ---------------------------------------------------------------------------


def _run_network(im_info, *, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run Network end-to-end on ``device``, return ``(skel, pixel_class, skel_relabelled)`` numpy copies.

    The factory fixtures (``make_network_imageinfo_*``) pre-populate the
    Frangi and Label memmaps from a session cache, so this only pays the
    Network cost — not Filter + Label on top.
    """
    net = Network(im_info, NetworkConfig(device=device), num_t=2)
    net.run()
    skel = np.asarray(net.skel_memmap).copy()
    pc = np.asarray(net.pixel_class_memmap).copy()
    relab = np.asarray(net.skel_relabelled_memmap).copy()
    _release_network(net)
    return skel, pc, relab


def _network_diagnostics(
    cpu: tuple[np.ndarray, np.ndarray, np.ndarray],
    mps: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, float]:
    """Summary stats for diagnosing CPU vs MPS Network drift.

    Inputs are ``(skel, pixel_class, skel_relabelled)`` triples.

    Returned keys:
        skel_jaccard — Jaccard of the binary skeleton mask
            (``skel > 0``). Acceptance bar; PRD #140 § Testing Decisions
            calls for "Jaccard > 0.95".
        junctions_cpu / junctions_mps — count of pixel_class == 4 (junction)
            voxels. Acceptance bar; PRD asks for exact match on junction
            count for typical inputs.
        skel_array_equal — whether the int32 skeleton labels match
            byte-for-byte. Empirically True on the yeast fixtures (the
            structural ``ndi.label`` round-trips to scipy on CPU on both
            backends).
        pc_array_equal — whether the uint8 pixel-class image matches
            byte-for-byte. Empirically True on the yeast fixtures.
        relab_jaccard — Jaccard of the relabelled skeleton mask. Same
            byte-identical empirical result on the yeast fixtures.
    """
    skel_cpu, pc_cpu, relab_cpu = cpu
    skel_mps, pc_mps, relab_mps = mps

    cpu_mask = skel_cpu > 0
    mps_mask = skel_mps > 0
    inter = int((cpu_mask & mps_mask).sum())
    union = int((cpu_mask | mps_mask).sum())
    skel_jacc = inter / union if union else 1.0

    relab_cpu_mask = relab_cpu > 0
    relab_mps_mask = relab_mps > 0
    relab_inter = int((relab_cpu_mask & relab_mps_mask).sum())
    relab_union = int((relab_cpu_mask | relab_mps_mask).sum())
    relab_jacc = relab_inter / relab_union if relab_union else 1.0

    return {
        "skel_jaccard": skel_jacc,
        "junctions_cpu": float(int((pc_cpu == 4).sum())),
        "junctions_mps": float(int((pc_mps == 4).sum())),
        "skel_array_equal": float(np.array_equal(skel_cpu, skel_mps)),
        "pc_array_equal": float(np.array_equal(pc_cpu, pc_mps)),
        "relab_jaccard": relab_jacc,
    }


def test_network_equivalence_cpu_vs_mps_3d(make_network_imageinfo_3d, capsys) -> None:
    """``Network.run()`` on the 3D yeast fixture: CPU vs MPS within tolerance.

    Tolerances:
      - ``skel_jaccard >= 0.95`` — PRD #140 § Testing Decisions bar for
        the binary skeleton mask.
      - ``junctions_cpu == junctions_mps`` — PRD asks for exact match on
        junction count for typical inputs.

    Note on tightness: the only convolutional op in Network is
    ``ndi.convolve`` inside ``_get_pixel_class_impl`` (a 3×3×3 kernel of
    ones over a uint8 binary mask). That's integer arithmetic over a
    small kernel, so the float32-rounding-cascade story that surfaces
    in Filter doesn't apply here. The skeletonization, structural
    ``ndi.label``, and per-object distance transforms all force-CPU.
    Empirically the entire ``(skel, pixel_class, skel_relabelled)``
    triple is byte-identical to CPU on this hardware (skel_jaccard =
    1.0, byte-equal labels); the > 0.95 bar leaves headroom for any
    future drift introduced by torch's MPS conv path picking a different
    accumulation order on borderline values.
    """
    _require_mps()

    cpu_outs = _run_network(make_network_imageinfo_3d(), device="cpu")
    mps_outs = _run_network(make_network_imageinfo_3d(), device="mps")

    skel_cpu, pc_cpu, relab_cpu = cpu_outs
    skel_mps, pc_mps, relab_mps = mps_outs
    assert skel_mps.shape == skel_cpu.shape
    assert pc_mps.shape == pc_cpu.shape
    assert relab_mps.shape == relab_cpu.shape
    assert skel_mps.dtype == skel_cpu.dtype == np.int32
    assert pc_mps.dtype == pc_cpu.dtype == np.uint8
    assert relab_mps.dtype == relab_cpu.dtype == np.uint32

    stats = _network_diagnostics(cpu_outs, mps_outs)

    with capsys.disabled():
        print(
            f"\n[mps eq net 3D] skel_jaccard={stats['skel_jaccard']:.4f} "
            f"junctions_cpu={int(stats['junctions_cpu'])} "
            f"junctions_mps={int(stats['junctions_mps'])} "
            f"skel_array_equal={bool(stats['skel_array_equal'])} "
            f"pc_array_equal={bool(stats['pc_array_equal'])} "
            f"relab_jaccard={stats['relab_jaccard']:.4f}"
        )

    assert stats["skel_jaccard"] >= 0.95, (
        f"3D Network skeleton mask Jaccard between CPU and MPS dropped "
        f"to {stats['skel_jaccard']:.3f}; expected >= 0.95 per PRD "
        f"#140 § Testing Decisions."
    )
    assert stats["junctions_cpu"] == stats["junctions_mps"], (
        f"3D Network junction count differs between CPU "
        f"({int(stats['junctions_cpu'])}) and MPS "
        f"({int(stats['junctions_mps'])}); PRD #140 asks for exact "
        f"match on junction count for typical inputs."
    )


def test_network_equivalence_cpu_vs_mps_2d(make_network_imageinfo_2d, capsys) -> None:
    """``Network.run()`` on the 2D yeast fixture: CPU vs MPS within tolerance.

    Same > 0.95 / exact-junction bar as the 3D test. The 2D path uses a
    (3, 3) convolution kernel instead of (3, 3, 3) — different shim
    dispatch (``conv2d`` vs ``conv3d``) but same arithmetic story.
    Empirically the 2D yeast fixture has zero junctions on both
    backends, which is itself an exact match (the > 0.95 skeleton
    Jaccard bar still applies).
    """
    _require_mps()

    cpu_outs = _run_network(make_network_imageinfo_2d(), device="cpu")
    mps_outs = _run_network(make_network_imageinfo_2d(), device="mps")

    skel_cpu, pc_cpu, relab_cpu = cpu_outs
    skel_mps, pc_mps, relab_mps = mps_outs
    assert skel_mps.shape == skel_cpu.shape
    assert pc_mps.shape == pc_cpu.shape
    assert relab_mps.shape == relab_cpu.shape
    assert skel_mps.dtype == skel_cpu.dtype == np.int32
    assert pc_mps.dtype == pc_cpu.dtype == np.uint8
    assert relab_mps.dtype == relab_cpu.dtype == np.uint32

    stats = _network_diagnostics(cpu_outs, mps_outs)

    with capsys.disabled():
        print(
            f"\n[mps eq net 2D] skel_jaccard={stats['skel_jaccard']:.4f} "
            f"junctions_cpu={int(stats['junctions_cpu'])} "
            f"junctions_mps={int(stats['junctions_mps'])} "
            f"skel_array_equal={bool(stats['skel_array_equal'])} "
            f"pc_array_equal={bool(stats['pc_array_equal'])} "
            f"relab_jaccard={stats['relab_jaccard']:.4f}"
        )

    assert stats["skel_jaccard"] >= 0.95, (
        f"2D Network skeleton mask Jaccard between CPU and MPS dropped "
        f"to {stats['skel_jaccard']:.3f}; expected >= 0.95 per PRD "
        f"#140 § Testing Decisions."
    )
    assert stats["junctions_cpu"] == stats["junctions_mps"], (
        f"2D Network junction count differs between CPU "
        f"({int(stats['junctions_cpu'])}) and MPS "
        f"({int(stats['junctions_mps'])}); PRD #140 asks for exact "
        f"match on junction count for typical inputs."
    )


# ---------------------------------------------------------------------------
# HuMomentTracking equivalence
# ---------------------------------------------------------------------------


def _run_hu(im_info, *, device: str) -> np.ndarray:
    """Run HuMomentTracking end-to-end on ``device``, return the saved flow array.

    The factory fixtures (``make_hu_imageinfo_*``) pre-populate Frangi /
    Label / Marker / Distance memmaps from a session cache, so this only
    pays the HuMomentTracking cost — not the four upstream stages.
    """
    h = HuMomentTracking(im_info, HuMomentTrackingConfig(device=device), num_t=2)
    h.run()
    flow = np.load(h.flow_vector_array_path)
    _release_hu(h)
    return flow


def _flow_assignments(flow: np.ndarray) -> set[tuple[int, ...]]:
    """Build the set of ``(t, src..., dst...)`` integer-index tuples per row.

    The flow array's row order is not deterministic across backends —
    different reduction orders in ``_find_best_matches`` can shuffle
    rows even when the underlying matching set is identical. Quotienting
    out row order via ``set`` lets a Jaccard comparison capture the true
    "did we make the same matches?" notion of equivalence.

    Schema (see :mod:`tests.test_hu_tracking`):
      - 3D ``(N, 8)``: ``[t, z, y, x, dz, dy, dx, cost]``
      - 2D ``(N, 6)``: ``[t, y, x, dy, dx, cost]``

    The cost column is dropped from the assignment key — it's only used
    to gate matches via ``cost_cutoff``, not to identify the assignment
    itself. Source and destination indices are added together in voxel
    space so a row reads as a directed edge "from (src) to (dst)" at
    time ``t``.
    """
    if flow.shape[0] == 0:
        return set()
    n_cols = flow.shape[1]
    if n_cols == 8:
        # 3D
        t = flow[:, 0].astype(np.int64)
        z = flow[:, 1].astype(np.int64)
        y = flow[:, 2].astype(np.int64)
        x = flow[:, 3].astype(np.int64)
        zd = z + flow[:, 4].astype(np.int64)
        yd = y + flow[:, 5].astype(np.int64)
        xd = x + flow[:, 6].astype(np.int64)
        return set(zip(
            t.tolist(), z.tolist(), y.tolist(), x.tolist(),
            zd.tolist(), yd.tolist(), xd.tolist(),
        ))
    # 2D
    t = flow[:, 0].astype(np.int64)
    y = flow[:, 1].astype(np.int64)
    x = flow[:, 2].astype(np.int64)
    yd = y + flow[:, 3].astype(np.int64)
    xd = x + flow[:, 4].astype(np.int64)
    return set(zip(
        t.tolist(), y.tolist(), x.tolist(),
        yd.tolist(), xd.tolist(),
    ))


def _hu_diagnostics(cpu: np.ndarray, mps: np.ndarray) -> dict[str, float]:
    """Summary stats for diagnosing CPU vs MPS HuMomentTracking drift.

    Returned keys:
        cpu_rows / mps_rows — total flow-vector row count
        asg_cpu / asg_mps — distinct assignment count after dropping
            duplicate rows (row order is not deterministic across
            backends; ``_find_best_matches`` emits both row- and
            column-minimum matches so a single match can appear twice).
        intersection / union — set ops on the assignment sets
        jaccard — assignment-Jaccard, ``intersection / union`` (or 1.0
            when both sets are empty). Acceptance bar; PRD #140
            § Testing Decisions calls for "tracking assignment Jaccard
            > 0.9".
        cost_mean_cpu / cost_mean_mps — per-row mean of the cost column.
            Costs are sums of z-scored components and can be very
            negative for excellent matches; comparing means catches
            wholesale precision drift in the cost-matrix machinery.
    """
    asg_cpu = _flow_assignments(cpu)
    asg_mps = _flow_assignments(mps)
    inter = len(asg_cpu & asg_mps)
    union = len(asg_cpu | asg_mps)
    jaccard = inter / union if union else 1.0

    cost_cpu = float(cpu[:, -1].mean()) if cpu.shape[0] else 0.0
    cost_mps = float(mps[:, -1].mean()) if mps.shape[0] else 0.0

    return {
        "cpu_rows": float(cpu.shape[0]),
        "mps_rows": float(mps.shape[0]),
        "asg_cpu": float(len(asg_cpu)),
        "asg_mps": float(len(asg_mps)),
        "intersection": float(inter),
        "union": float(union),
        "jaccard": jaccard,
        "cost_mean_cpu": cost_cpu,
        "cost_mean_mps": cost_mps,
    }


def test_hu_tracking_equivalence_cpu_vs_mps_3d(make_hu_imageinfo_3d, capsys) -> None:
    """``HuMomentTracking.run()`` on the 3D yeast fixture: CPU vs MPS within tolerance.

    Tolerances:
      - ``jaccard >= 0.9`` — PRD #140 § Testing Decisions bar for
        tracking-assignment Jaccard. The bar is intentionally looser
        than the labelling (>= 0.95) and filtering / network bars (1e-4
        max abs diff / >= 0.95 Jaccard) because of the **stacked
        precision losses** specific to hu_tracking on MPS.

    On the precision cascade:

      1. ``_get_difference_matrix`` casts the moment-distance matrix to
         ``xp.float64``. Per PRD #140 § Implementation Decisions, the
         MPS shim silently coerces ``xp.float64`` to ``torch.float32``
         (MPS does not support double precision). This is the *first*
         precision loss vs the CPU baseline.

      2. ``_get_cost_matrix`` then casts each z-scored component plus
         the final cost matrix to ``xp.float16``. This is a deliberate
         memory-management step (the cost matrix can be huge) but it
         also caps the precision of every match decision at half-float
         resolution. This is the *second* precision loss, and it
         applies on both CPU and MPS — but combined with the float64
         coercion above, the cumulative drift on MPS is materially
         more than on the other onboarded stages.

    Empirically on this fixture the assignments come out
    byte-identical (Jaccard = 1.0) — the > 0.9 bar leaves headroom for
    any future drift introduced by torch's MPS reduction order picking
    a different accumulation path on borderline z-scored values.
    """
    _require_mps()

    cpu_flow = _run_hu(make_hu_imageinfo_3d(), device="cpu")
    mps_flow = _run_hu(make_hu_imageinfo_3d(), device="mps")

    # Schema parity (per :mod:`tests.test_hu_tracking`):
    # 3D = (N, 8) — columns ``[t, z, y, x, dz, dy, dx, cost]``.
    assert mps_flow.ndim == cpu_flow.ndim == 2
    assert mps_flow.shape[1] == cpu_flow.shape[1] == 8

    stats = _hu_diagnostics(cpu_flow, mps_flow)

    with capsys.disabled():
        print(
            f"\n[mps eq hu 3D] cpu_rows={int(stats['cpu_rows'])} "
            f"mps_rows={int(stats['mps_rows'])} "
            f"asg_cpu={int(stats['asg_cpu'])} asg_mps={int(stats['asg_mps'])} "
            f"inter={int(stats['intersection'])} union={int(stats['union'])} "
            f"jaccard={stats['jaccard']:.4f} "
            f"cost_mean_cpu={stats['cost_mean_cpu']:.4f} "
            f"cost_mean_mps={stats['cost_mean_mps']:.4f}"
        )

    assert stats["jaccard"] >= 0.9, (
        f"3D HuMomentTracking assignment Jaccard between CPU and MPS "
        f"dropped to {stats['jaccard']:.3f}; expected >= 0.9 per PRD "
        f"#140 § Testing Decisions (bar is looser than other stages "
        f"because of the stacked float64 → float32 → float16 precision "
        f"loss documented in this test's docstring)."
    )


def test_hu_tracking_equivalence_cpu_vs_mps_2d(make_hu_imageinfo_2d, capsys) -> None:
    """``HuMomentTracking.run()`` on the 2D yeast fixture: CPU vs MPS within tolerance.

    Same > 0.9 bar as the 3D test. The 2D path computes 6 Hu moments
    per marker (vs 18 in 3D — no orthogonal projections) and uses a
    (3, 3) ``maximum_filter`` footprint instead of (3, 3, 3), but the
    moment-distance and cost-matrix machinery is identical so the same
    float64 → float32 → float16 precision cascade applies.
    Empirically the assignments come out byte-identical on this
    hardware (Jaccard = 1.0).
    """
    _require_mps()

    cpu_flow = _run_hu(make_hu_imageinfo_2d(), device="cpu")
    mps_flow = _run_hu(make_hu_imageinfo_2d(), device="mps")

    # 2D = (N, 6) — columns ``[t, y, x, dy, dx, cost]``.
    assert mps_flow.ndim == cpu_flow.ndim == 2
    assert mps_flow.shape[1] == cpu_flow.shape[1] == 6

    stats = _hu_diagnostics(cpu_flow, mps_flow)

    with capsys.disabled():
        print(
            f"\n[mps eq hu 2D] cpu_rows={int(stats['cpu_rows'])} "
            f"mps_rows={int(stats['mps_rows'])} "
            f"asg_cpu={int(stats['asg_cpu'])} asg_mps={int(stats['asg_mps'])} "
            f"inter={int(stats['intersection'])} union={int(stats['union'])} "
            f"jaccard={stats['jaccard']:.4f} "
            f"cost_mean_cpu={stats['cost_mean_cpu']:.4f} "
            f"cost_mean_mps={stats['cost_mean_mps']:.4f}"
        )

    assert stats["jaccard"] >= 0.9, (
        f"2D HuMomentTracking assignment Jaccard between CPU and MPS "
        f"dropped to {stats['jaccard']:.3f}; expected >= 0.9 per PRD "
        f"#140 § Testing Decisions."
    )
