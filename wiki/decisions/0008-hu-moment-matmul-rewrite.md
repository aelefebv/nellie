---
created: 2026-05-11
modified: 2026-05-11
---

# `HuMomentTracking._calculate_normalized_moments` switches from broadcast (N, H, W, 4, 4) tensor to two-step batched matmul

`HuMomentTracking._calculate_normalized_moments` (in
`nellie/tracking/hu_tracking.py`) builds normalized moments
`eta_{pq}` for Hu invariants. The current implementation
materializes `(N, H, W, 4, 4)` broadcast tensors twice per call (raw
moments + central moments) before reducing over the spatial axes.
For typical N=1000 markers with H=W=21 (the yeast 3D fixture's
max_radius), each broadcast tensor is ~28 MB; two allocations per
call × 3 calls per 3D frame = ~168 MB peak transient memory per
frame for the moment math alone. The function runs once per frame in
2D mode and 3× per frame in 3D mode (one call per orthogonal
projection in `_get_hu_moments`).

PRD #191 replaces the broadcast pattern with a two-step batched
matmul. Three coupled decisions in the design need to be pinned
ahead of the rewrite in Slice 2 (#193): (1) the reformulation uses
`xp.matmul` / `@` rather than `xp.einsum`, (2) the test bar is
approx-equivalence (`rtol=1e-5`), not byte-equal, and (3) no
cross-platform snapshot test. Status: **Accepted (preventive)** —
documents non-obvious decisions ahead of the rewrite in Slice 2
(#193).

## Considered Options

- **Two-step batched matmul (chosen).** Reformulate
  `M[n, p, q] = sum_{h, w} I[n, h, w] * w^p * h^q` as a 2-step
  contraction:
    - `M_inter[n, h, p] = sum_w I[n, h, w] * x_powers[w, p]` via
      `(N, H, W) @ (W, P) → (N, H, P)`
    - `M[n, p, q] = sum_h M_inter[n, h, p] * y_powers[h, q]` via
      `(N, P, H) @ (H, Q) → (N, P, Q)`
  Central moments use the same shape but with per-N power tables
  (`x_centered_powers[n, w, p]`) — `xp.matmul` broadcasts
  `(N, H, W) @ (N, W, P) → (N, H, P)` natively. Peak intermediate
  drops from `(N, H, W, 4, 4)` ≈ 28 MB to `(N, max(H, W), 4)` ≈
  336 KB at typical N=1000, H=W=21 — ~80× memory reduction. matmul
  dispatches to BLAS via `@`; expected 5–10× speedup, validated by
  the perf microbenchmark added in Slice 1 (`tests/test_hu_tracking_perf.py`,
  `test_calculate_normalized_moments_scaling` and
  `test_get_hu_moments_3d_scaling`). Backend-uniform: `xp.matmul`
  / `@` works identically on numpy, cupy, and torch with predictable
  BLAS dispatch.
- **`xp.einsum('nhw,wp,hq->npq', ...)` with `optimize=True`
  (rejected).** numpy and cupy support `optimize=True` (path
  optimization to enable BLAS dispatch). `torch.einsum` does NOT
  have an `optimize` parameter — it uses an internal heuristic which
  is not guaranteed to choose the same contraction order across
  backends. Two-step matmul keeps backend behavior identical and
  reviewable; the contraction order is explicit in the source rather
  than implementation-defined.
- **Keep the broadcast `(N, H, W, 4, 4)` formulation (rejected).**
  The memory blowup is the entire problem — this is what PRD #191
  exists to fix.
- **Per-marker Python loop with N=1 batched matmul (rejected).**
  Would lose all vectorization. Only useful if the batched matmul
  path were memory-bound, which it's not — the largest intermediate
  is `(N, max(H, W), 4)` = 84K float32 elements = 336 KB at N=1000.
- **Cross-platform snapshot test on yeast 3D (rejected).** BLAS
  implementations differ across platforms (Accelerate on macOS vs
  OpenBLAS / MKL on Linux vs cuBLAS on CUDA). Same root cause as
  ADRs 0005 / 0006 / 0007 (cross-platform `gaussian_laplace` is
  SIMD-sensitive). Intra-platform serial-vs-matmul equivalence test
  (added by Slice 2 #193) is the right granularity — same process,
  same BLAS, deterministic by construction.

## Consequences

- The Slice 1 synthetic suite (#192) covers the high-level contracts:
  output shape, empty-input boundary, all-zeros finiteness,
  single-pixel analytical eta, and translation invariance. All
  hand-derived from the algorithmic spec; platform-stable in shape
  (count + shape + allclose-with-tolerance + analytical-value
  assertions).
- **Test bar: approx-equivalence (`rtol=1e-5`), not byte-equal.**
  BLAS reorders the reduction axes (different from the broadcast
  `xp.sum(..., axis=(1, 2))` order), so the matmul result is not
  bit-equal to the current implementation. At float32 precision the
  relative error is ≤ 1e-5 on realistic Hu moment magnitudes.
  Acceptable per existing precedent: ADRs 0005 / 0006 / 0007
  (cross-platform `gaussian_laplace` SIMD-sensitive) and the
  existing `hu_tracking.py:807` MPS-float64-coercion comment that
  already tolerates this precision loss downstream in
  `_get_difference_matrix`.
- **No cross-platform snapshot test.** BLAS implementations differ
  across platforms. The Slice 1 synthetic suite + the Slice 2
  serial-vs-matmul intra-platform equivalence test on the yeast 3D
  fixture (deterministic by construction) are sufficient.
- The serial-vs-matmul equivalence test added by Slice 2 pins the
  approx-equivalence claim end-to-end on realistic
  `_get_orthogonal_projections` output: both paths run in the same
  process so BLAS implementation is constant. If a future
  maintainer accidentally introduces a behavioral change (e.g.,
  swaps `transpose(0, 2, 1)` for `transpose(0, 1, 2)`, or flips the
  contraction order so x_powers and y_powers are wired into the
  wrong axes), this test fails.
- Future maintainers should not switch from `@` / `xp.matmul` back
  to `xp.einsum` without verifying that the active backend's einsum
  supports a contraction-order optimization flag uniformly.
  `torch.einsum` is the current limitation; this could change in a
  future torch release, but the change must be re-validated.
- Future maintainers tempted to "preserve the broadcast formulation
  for clarity" should read this ADR — the broadcast formulation has
  an 80× memory penalty that's the whole reason for the rewrite.

## References

- PRD #191 — matmul rewrite of
  `HuMomentTracking._calculate_normalized_moments`
- Slice 1 #192 — pin current behavior with synthetic-test suite +
  perf benchmark backfill + this ADR (no production code changes)
- Slice 2 #193 — matmul rewrite + serial-vs-matmul approx-
  equivalence test on yeast 3D
- ADR 0005 — `Network._relabel_objects` writeback serialization
  (precedent for approx-equivalence test bar)
- ADR 0006 — `Markers._remove_close_peaks` sparse cKDTree
  (precedent for "no cross-platform snapshot")
- ADR 0007 — `Markers._local_max_peak` per-sigma threading
  (precedent for "no cross-platform snapshot" + intra-platform
  equivalence test)
- `hu_tracking.py:807` — existing MPS-float64-coercion comment
  (precedent for accepting precision loss in this domain)
