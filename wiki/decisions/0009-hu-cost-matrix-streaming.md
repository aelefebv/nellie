---
created: 2026-05-11
modified: 2026-05-11
---

# `HuMomentTracking._get_cost_matrix` switches from broadcast `(N, N, F)` float64 tensors to per-feature streaming float32

`HuMomentTracking._get_cost_matrix` (in
`nellie/tracking/hu_tracking.py`) builds the per-frame matching cost
matrix from three feature groups: distance, stats, and hu. The
current implementation allocates `(N_post, N_pre, F)` **float64**
difference matrices via `_get_difference_matrix` and z-scores them
via `_zscore_normalize` (which itself materializes
`(m - mean)^2 * mask` temporaries). For typical N=1000 with
F_stats=4 + F_hu=18, peak transient memory is ~350 MB per matching
call. The cost matrix downstream is cast to float16 then float32 —
float64 promotion is wasted precision (the existing
`hu_tracking.py:807` comment notes that MPS silently coerces it
back to float32 anyway).

PRD #196 refactors `_get_cost_matrix` to per-feature streaming with
explicit float32 throughout. Three coupled decisions in the design
need to be pinned ahead of the rewrite in Slice 2 (#198): (1) the
reformulation streams per-feature into a running `(N, N)` cost
matrix rather than building `(N, N, F)` intermediates, (2) the test
bar is approx-equivalence (`rtol=1e-3, atol=1e-3`) — looser than
ADR 0008 due to the float64 → float32 drop, and (3)
`_get_difference_matrix` and `_zscore_normalize` are deleted (no
external callers; the dechaos wiki acknowledged `_zscore_normalize`
as a "clean reusable primitive — defer until a second consumer
materializes" but no second consumer ever materialized). Status:
**Accepted (preventive)** — documents non-obvious decisions ahead
of the rewrite in Slice 2 (#198).

## Considered Options

- **Per-feature streaming + float32 (chosen).** For each feature
  `f`, compute `diff_f[i, j] = |a[i, f] - b[j, f]|` (an `(N, N)`
  matrix), z-score it, divide by the feature group's F, and
  accumulate into a running `(N, N)` cost matrix. Mask is applied
  once at the end via `xp.where(mask, cost, xp.inf)`. Variance uses
  the sum-of-squares formula `var = E[X²] - mean²` (clamped against
  negative variance from catastrophic cancellation) to avoid the
  `(m - mean)^2 * mask` temporary per feature. Peak intermediate
  drops from ~350 MB to ~4 MB (~80× memory reduction). Speed should
  be similar (bandwidth-bound either way).
- **Per-feature streaming + keep float64 (rejected).** Half the
  memory savings; doesn't match the downstream float16/float32
  precision floor; doesn't match the existing MPS coercion behavior.
- **Keep `(N, N, F)` tensor + float32 only (rejected).** Modest
  memory win (~2×) without the bigger F× streaming benefit. Doesn't
  warrant the API surface area change.
- **Keep `(N, N, F)` tensor + float64 (status quo, rejected).** The
  problem statement.
- **Use `xp.einsum` to compute the F-axis sum without materializing
  (rejected).** `xp.einsum` reduction order is implementation-defined
  across backends (cf. ADR 0008's same concern with
  `torch.einsum`); explicit per-feature loop is reviewable and
  predictable. The per-feature loop is also a clearer mental model
  than einsum-with-sum-axis.
- **Threading the per-feature loop (rejected).** F is small (~22)
  and each feature's `(N, N)` work is bounded; threading overhead
  would dominate. The perf budget for `_get_cost_matrix` is already
  small after streaming.

## Consequences

- **Test bar: approx-equivalence (`rtol=1e-3, atol=1e-3`)**, looser
  than ADR 0008. Two precision changes layer:
    1. Float64 → float32 (drops ~8 decimal digits of precision)
    2. Sum-of-squares variance has slightly different numerical
       properties than `(m - mean)^2` — catastrophic cancellation
       when `var ≈ mean²` (very small variance with large mean).
       Clamped against negative variance to defend against the
       worst case, but the implementations are not bit-equal.
  The downstream cost matrix is cast to float16 anyway (~3-4
  decimal digits), so practically the two implementations agree at
  float16 precision. `rtol=1e-3` captures this with margin.
- **`_get_difference_matrix` and `_zscore_normalize` are deleted**
  by Slice 2 (#198). Both are private (underscore-prefixed) helpers
  with no external callers. The
  `wiki/outputs/dechaos-hu-tracking.md` Pass 6 article noted
  `_zscore_normalize` as a "clean reusable image-statistics
  primitive" with the explicit deferral "no extraction warranted
  — defer until a second consumer materializes". No second consumer
  ever materialized; the deletion is consistent with the deferred
  decision. The article needs updating to drop the claim.
- **The `test_get_difference_matrix_scaling` perf microbenchmark
  is deleted** in Slice 2 (#198) — it tests a deleted method. The
  existing `test_get_cost_matrix_scaling` perf benchmark covers
  the production-path behavior end-to-end.
- The MPS-related test/comment site at `tests/test_mps_smoke.py:412`
  and `tests/test_mps_equivalence.py:687` both reference "the
  moment-distance matrix in `_get_difference_matrix` casts to
  `xp.float64` which silently coerces to `float32` on MPS" —
  comments need updating in Slice 2 (#198) since the helper goes
  away and the float32 cast is now explicit (no silent coercion).
- The Slice 2 equivalence test pins approx-equivalence end-to-end
  on the yeast 3D fixture — both paths run in the same process so
  BLAS / cdist implementation is constant, deterministic by
  construction.
- Future maintainers should not re-introduce the `(N, N, F)` tensor
  formulation — the memory blowup is the entire reason for this
  rewrite.
- Future maintainers tempted to "drop the per-feature loop and use
  vectorized matrix math across features" should read this ADR —
  doing so re-introduces the (N, N, F) intermediate and undoes the
  memory win. Per-feature streaming is the right shape.
- Future maintainers re-introducing `_zscore_normalize` as a
  reusable primitive should write tests for it independently — the
  current cost-matrix path doesn't need it.

## References

- PRD #196 — per-feature streaming + float32 in
  `HuMomentTracking._get_cost_matrix`
- Slice 1 #197 — pin current behavior with synthetic-test suite +
  this ADR (no production code changes)
- Slice 2 #198 — refactor + delete helpers + perf numbers
- ADR 0008 — `_calculate_normalized_moments` matmul rewrite
  (precedent for approx-equivalence test bar + intra-platform
  equivalence test on yeast 3D)
- ADRs 0005 / 0006 / 0007 — preceding optimizations in
  `nellie/segmentation/` adopted the same approx-equivalence test
  bar pattern
- `hu_tracking.py:807` — existing MPS-float64-coercion comment
  (precedent for accepting precision loss in this domain)
- `wiki/outputs/dechaos-hu-tracking.md` Pass 6 — note about
  `_zscore_normalize` as a deferred-extraction primitive
