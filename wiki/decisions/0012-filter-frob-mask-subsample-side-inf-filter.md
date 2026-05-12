---
created: 2026-05-11
modified: 2026-05-11
---

# `Filter._get_frob_mask` switches inf-filtering from full volume to subsample

`Filter._get_frob_mask` (in `nellie/segmentation/filtering.py`) thresholds
a Frobenius-norm volume against a triangle/Otsu-derived threshold to
decide which voxels are eligible for Frangi vesselness response. The
pre-rewrite implementation always paid a full-volume `xp.isinf` scan
+ `xp.any` reduction up front so it could exclude infs from the
threshold input (`triangle_threshold` / `otsu_threshold` both call
`xp.histogram(range=(min, max))`, which breaks on infs). For a 100M-voxel
float32 frob_norm × 5 sigmas × 10 frames, that's ~30 GB of memory
bandwidth burned on what is, in practice, scalar control flow — the
common case (real microscopy data passing through `compute_hessian`'s
`max_abs > 0` guard) has no infs at all.

PRD #233 moves the inf-filter from the full volume to the
**subsample**. The threshold is already computed from a strided positive-
voxel subsample (`chunking.subsample_for_thresholds`, capped at
`max_threshold_samples`), so all the threshold computation needs is a
finite subsample. The full-volume `> threshold` comparison at the end
of `_get_frob_mask` already handles infs correctly on its own
(`+inf > finite` is `True`), so no full-volume inf scan is needed in
the common case. A pathological fallback (`if subsample is empty or
all-inf, check the full volume`) preserves the all-inf → all-False
contract that the prior implementation guaranteed.

The trade-off needs an ADR because the rewrite is **not bit-identical
on inf-containing data**: the subsample-side filter computes the
threshold from a smaller finite subsample (some samples were dropped
as infs) than the prior approach (which boolean-indexed the full
volume first and then subsampled). For finite-only data — which is
every production microscopy fixture — the rewrite is bit-identical
(verified locally on the yeast 3D + 2D fixtures: pre/post
`Filter.run()` SHAs match exactly). For inf-containing data the
threshold differs by sampling noise; the resulting mask is
approx-equivalent, not bit-equivalent. Same equivalence bar as ADRs
0008 (Hu matmul rewrite, `rtol=1e-5`) and 0009 (Hu cost-matrix
streaming, `rtol=1e-3`).

Status: **Accepted** — rewrite landed in PRD #233 Slice 2 (#235).

## Considered Options

- **Subsample-side `isfinite` filter (chosen).** Subsample first
  (`chunking.subsample_for_thresholds` already strides + filters
  `arr > 0`); apply `xp.isfinite(positive)` to the small array; if any
  infs survived the subsampling, drop them before passing to
  `triangle_threshold` / `otsu_threshold`. Pathological fallback for
  the empty / all-inf-subsample case checks the full volume (only path
  that pays the original full-volume `xp.isinf` scan). For finite-only
  data: bit-identical mask, no full-volume inf scan, no full-volume
  `xp.any` reduction. Common-case win: ~30 GB / frame less memory
  bandwidth at 100M-voxel scale.
- **Per-sigma cached `xp.isfinite(frobenius_norm)` (rejected).** Cache
  the full-volume isfinite mask once per sigma, reuse it across the
  threshold computation and the eventual `> thresh` comparison.
  Doesn't help: the `> thresh` comparison handles infs correctly on
  its own, so the cache has no second consumer; we'd just be paying
  the full-volume scan whether we cache or not.
- **Skip inf handling entirely (rejected).** `triangle_threshold` and
  `otsu_threshold` would crash on inf input via
  `xp.histogram(range=(min, max))`. We can't drop the inf filter — we
  can only push it later in the pipeline (subsample) or earlier
  (clamp at compute_hessian). Subsample is the right place because
  it's the smallest array that needs to be inf-clean.
- **Clamp infs at `compute_hessian` instead (rejected).** Would
  require a full-volume `nan_to_num` per sigma inside `compute_hessian`
  itself (a math-pure module). The current `compute_hessian` already
  has `max_abs > 0` guarding sqrt-divisions; adding a defensive
  full-volume clamp on top would re-introduce the same bandwidth cost
  in a less appropriate place.
- **Defer entirely (rejected).** PRD #233 audited filtering.py for
  per-sigma reductions; the inf-handling overhead was the largest
  bandwidth waste in the file. The rewrite is small (~15 lines net)
  and the test bar is well-defined (existing inf-handling tests
  already pin the semantic contract; pre/post SHA comparison pins
  bit-identity for finite data).

## Consequences

- **Bit-identical for finite-only data.** Every production microscopy
  fixture lives here. Pre/post SHAs verified locally on yeast 3D
  (`5a8626916e34538e0dd7088c1f2ffa7d4675b993e36dc30a10f0ce84a9bb9c77`)
  and 2D (`bc62abeafbd2335227d52c7a7e56765ac2af1386f725e3eb2b7c7c0441bc357d`)
  before merge. CI runs an in-band determinism check
  (`test_run_filter_3d_snapshot_post_rewrite` /
  `test_run_filter_2d_snapshot_post_rewrite`) that catches any future
  drift; cross-platform SHA comparison is intentionally avoided
  because Frangi math is SIMD-sensitive across macOS / Linux /
  Windows.
- **Approx-equivalent for inf-containing data.** The triangle/Otsu
  threshold may differ by sampling noise: the prior code subsampled
  from `frob_norm[finite]` (a 1D dense array of every finite voxel,
  then strided by ratio); the new code subsamples first
  (strided over the full volume) and filters out the infs that
  survived the stride. Both approaches yield a roughly uniform sample
  of finite values; the threshold derived from each may differ by the
  per-bin granularity of the histogram. Existing tests
  (`test_get_frob_mask_with_infs_keeps_them_and_preserves_input`,
  `test_get_frob_mask_all_infs_yields_no_signal`) keep their semantic
  assertions: inf voxels stay in the mask, all-inf input yields
  all-False mask, input is never mutated.
- **`_get_frob_mask` no longer mutates the input.** This was already
  true pre-rewrite (the `frobenius_norm[~inf_mask]` boolean-indexed
  copy was non-mutating); reaffirmed post-rewrite (`xp.isfinite` on
  the subsample doesn't touch the full volume).
- **Pathological all-inf fallback path.** The fallback fires when the
  subsample is empty (no positive voxels) or all-inf. In the all-inf
  subsample case it checks `bool(self.xp.isinf(frobenius_norm).all())`
  — the only post-rewrite path that pays a full-volume scan. The
  empty-subsample case (frob_norm has no positive values) returns a
  zero threshold; the resulting mask is `frobenius_norm > 0`, same as
  the `frob_thresh_division == 0` short-circuit at the top of the
  function.
- Future maintainers should not move the inf-filter back to the full
  volume to "simplify" without re-measuring. The bandwidth cost
  scales linearly with volume size; on 100M-voxel confocal stacks it
  is the dominant per-sigma scalar-control-flow cost.

## Bonus: dispatch reduction fusion (not a separate ADR — pure refactor)

PRD #233 Slice 2 also fused the per-sigma `xp.any(h_mask)` skip-check
(in `_compute_vesselness`, line 532) and the
`bool(h_mask.all())` dispatch (in `_compute_vesselness_chunkwise`,
line 408) into a single `xp.sum(h_mask)`. The sum gives both answers:
`is_empty = (true_count == 0)` and `is_dense = (true_count == total)`.
`_compute_vesselness_chunkwise` gained an `is_dense: bool | None = None`
keyword; when the caller threads it through, the routine skips its
own `bool(h_mask.all())` reduction. Default `None` preserves the
contract for direct unit-test callers in `tests/test_filtering.py`.

This is a strict simplification, bit-identical to the prior dispatch,
and doesn't warrant a separate ADR — but is mentioned here so the
post-rewrite reduction count makes sense (one `xp.sum(h_mask)` per
sigma, zero `xp.any(h_mask)`, zero `bool(h_mask.all())` in the dispatch
path).

## References

- PRD #233 — slim per-sigma reductions in `Filter._compute_vesselness`
  and `_get_frob_mask`
- Slice 1 #234 — pin pre-rewrite reduction pattern + isinf scope
  (test-side only)
- Slice 2 #235 — subsample-first inf in `_get_frob_mask` + fuse
  `h_mask` `any`/`all` into `sum` (this rewrite)
- ADRs 0008 / 0009 — preceding optimization ADRs that adopted the
  approx-equivalence test-bar pattern (`rtol=1e-5` / `rtol=1e-3`)
- `wiki/segmentation/filtering.md` — Performance section bullet
  documenting the prior `_get_frob_mask` no-volume-copy optimization;
  this PRD takes the next step (no full-volume scan either)
