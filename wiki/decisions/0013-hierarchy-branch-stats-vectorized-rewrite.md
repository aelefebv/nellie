---
created: 2026-05-11
modified: 2026-05-11
---

# `Branches._get_branch_stats` per-label loops vectorized; float32 cast point shifts to a single end-of-pipeline cast

`Branches._get_branch_stats` (in
`nellie/feature_extraction/hierarchical.py`) computes per-branch
length, thickness, aspect ratio, and tortuosity. The pre-rewrite
implementation walked `unique_labels` four times in Python:

1. **Base-length gather** — `for i, lbl in enumerate(unique_labels): if lbl < len(label_lengths): base_lengths[i] = label_lengths[int(lbl)]`.
2. **Tip-radius adjustment (×2)** — `for lbl, radius in zip(tip_labels, tip_radii): idx = np.where(unique_labels == lbl)[0]; base_lengths[idx[0]] += 2 * radius` (one loop for `lone_tip_labels`, one for `tip_labels`).
3. **Median thickness** — `for i, lbl in enumerate(unique_labels): mask = labels_branch_vox == lbl; median_thickness[i] = np.median(thicknesses[mask])`.
4. **Tortuosity** — `for i, lbl in enumerate(unique_labels): mask = tip_labels == lbl; coords_lbl = tip_coords[mask]; if coords_lbl.shape[0] >= 2: ... compute distance and ratio`.

Each loop is `O(N · L)` (full-volume comparison per label). The
rewrite replaces them with vectorized equivalents:

- Base-length gather → single `label_lengths[unique_labels_int]`
  with a bounds-mask.
- Tip-radius adjustment → `np.searchsorted(unique_labels, tip_labels)`
  + `np.add.at(base_lengths_64, idx, addend)` so duplicate indices
  accumulate correctly.
- Median thickness → `scipy.ndimage.median(thicknesses, labels, index)`
  (already imported as `ndi_cpu` for this PR).
- Tortuosity → stable `np.argsort(tip_labels)` + boundary scan
  (`np.unique(... return_index=True, return_counts=True)`) to grab
  the first two tips per qualifying label, then a single vectorized
  distance + division.

The trade-off needs an ADR because the rewrite is **not bit-identical
on the float32 noise floor**: the legacy tip-radius accumulation casts
to float32 after every `+=` (one cast per tip), while the rewrite
accumulates in float64 and casts once at the end. For labels with
multiple tips this can drift by 1 float32 ULP — observed on yeast 3D
fixture, label 2 at t=0 (the longest branch in the test data):
`branch_length_raw` 16.6453686 → 16.6453667, a 1.9e-6 absolute /
2e-7 relative diff. Tortuosity (= length / tip_dist) inherits the
same drift in the same row. **The rewrite is more numerically
accurate** in the IEEE sense (single rounding vs. accumulated
round-after-every-step), but the SHA-level bit-identity bar of ADRs
0010-0012 doesn't hold. Same approx-equivalence shape as ADRs 0008
(Hu matmul, `rtol=1e-5`) and 0009 (Hu cost-matrix streaming,
`rtol=1e-3`).

Status: **Accepted** — rewrite landed in PR #243 (hierarchy perf audit
PR B).

## Considered Options

- **Accumulate in float64 with single end-cast (chosen).** Read
  `label_lengths[unique_labels]` into a float64 buffer; do all tip
  adjustments via `np.add.at` in float64; cast once to float32 at the
  end. More precise than the legacy pattern; one cast vs N casts.
  Approx-equivalence verified on yeast 2D (bit-identical SHAs:
  `fc233f30...` voxel, `29766e55...` branches) and yeast 3D (SHAs
  shift but only one row of `branch_length_raw` differs, by 1 ULP).
- **Per-tip Python loop with intermediate `float32(...)` cast
  (rejected).** Would match the legacy bit-identity exactly, but
  defeats the perf goal of removing per-tip Python overhead. The whole
  point of switching to `np.add.at` is to eliminate that loop.
- **Pure-float32 accumulation via `np.add.at` (rejected — and tried
  first).** Triggers double-rounding (cast addend to f32, then add in
  f32) — drifted further from the legacy than the float64 path. The
  initial-attempt SHA showed `branch_length_raw` row 3 differing by
  6e-8 (a SMALL drift, but a different row than the float64 path
  drifts on). Inconsistent rounding direction across labels, no
  cleaner.
- **Detect single-tip vs multi-tip labels and use vectorized for
  single, Python loop for multi (rejected).** Would let single-tip
  labels match exactly while preserving the per-tip cast pattern for
  the rare multi-tip case. Adds branching complexity for a 1-ULP
  preservation that nobody downstream depends on; multi-tip labels
  are the long branches where the original loop already accumulated
  N ULPs of error (less accurate than the float64 single-cast path
  we ship).
- **Defer entirely (rejected).** PRD-style 3-PR audit of
  `hierarchical.py` (PRs A/B/C) — A vectorized the per-label group
  construction (5 sites, bit-identical), this is B, C is cleanups.
  `_get_branch_stats` is one of the three top hot paths cProfile
  identified per the user's audit message; the four per-label loops
  here are the actionable inside-this-file part. Skipping would leave
  measurable perf on the table.

## Consequences

- **Approx-equivalent for the `branch_length` / `branch_aspect_ratio`
  / `branch_tortuosity` columns on multi-tip labels.** Bar:
  `rtol=1e-5, atol=1e-5` (well below float32 ULP of the values
  involved). Verified on yeast 3D: max abs diff 3.8e-6 (tortuosity
  for label 2 at t=0), max rel diff 2e-7. All other rows
  bit-identical. 2D fixture has zero diffs (no multi-tip label in the
  fixture data).
- **Bit-identical for the `branch_thickness` column.**
  `scipy.ndimage.median` matches per-group `np.median` exactly on the
  test fixture (verified empirically on a 5000-element synthetic case
  with 200 labels: max abs diff 0.0; verified on yeast 2D + 3D
  fixtures via per-row diff). Both routines partition + take the
  middle (or average two middles for even-N groups).
- **`scipy.ndimage.median` returns 0.0 for absent labels**, while the
  legacy code set NaN for absent labels. Defensive `np.bincount`-based
  NaN-fill preserves the legacy semantic. By construction
  `unique_labels = np.unique(L[L>0])` and `labels_branch_vox = L[branch_idxs]`,
  so every label has at least one matching voxel — NaN-fill is a
  no-op in practice.
- **Tortuosity vectorization changes the 2D/3D split.** The legacy
  loop branched on `no_z` (an attribute of `im_info`) to compute
  `dz*dz + dy*dy + dx*dx` (3D) vs `dy*dy + dx*dx` (2D). The
  vectorized version derives dimensionality from `tip_coords.shape`
  directly: `delta = (p0 - p1) * spacing_arr` followed by
  `(delta * delta).sum(axis=1)`. Sum along axis=1 is `axis_0 + axis_1`
  for 2D and `axis_0 + axis_1 + axis_2` for 3D — same per-axis
  ordering as the legacy expression. The `no_z` local variable in
  `_get_branch_stats` is now unused and removed.
- **One new dependency import: `from scipy import ndimage as ndi_cpu`.**
  scipy is already a transitive dependency (used in `flow_interpolation.py`
  and `networking.py` per the existing per-stage code). No requirements
  changes needed.
- **Test bar pinned in `test_hierarchical.py`:**
  - `test_get_branch_stats_2d_post_rewrite_bit_identical` —
    bit-identical to dechao baseline on the 2D fixture (no multi-tip
    labels in test data; cast point shift contributes zero drift).
    Empirically cross-platform stable on Linux x86 / macOS Apple
    Silicon / Windows x86 (verified in CI).
  - `test_get_branch_stats_multi_tip_label_synthetic_post_rewrite` —
    constructs a 9-voxel "+" skeleton (4 tips on one label) directly,
    bypassing the SIMD-sensitive Frangi → Network upstream. Asserts
    the post-rewrite single-end-cast value equals the formula-derived
    expected value, AND that the legacy per-tip-cast pattern stays
    within `rtol=1e-5, atol=1e-5` of the new value. Platform-stable
    by construction (all inputs are hand-set integers + irrational-
    at-float32 distances).
  - `test_get_branch_stats_run_is_deterministic` — in-band
    determinism check across two independent runs on fresh 3D
    fixtures. Catches non-determinism (e.g. floating-point reduction-
    order dependence introduced by future threading) that wouldn't
    fail the value-pinning tests.

  Cross-platform fixture-output pinning intentionally avoided for the
  3D path: Frangi runs at session-start (`frangi_3d_path` →
  `_run_filter_to_disk`) and is SIMD-sensitive across macOS Apple
  Silicon, Linux x86, and Windows x86. Different Frangi outputs flow
  through to different segmentation, different branch counts, and
  different per-row values — so a hardcoded 3D baseline can't be
  cross-platform stable. The synthetic test covers the multi-tip code
  path on platform-stable inputs; the determinism test covers
  runtime-determinism; the 2D bit-identity test covers the no-drift
  case empirically.
