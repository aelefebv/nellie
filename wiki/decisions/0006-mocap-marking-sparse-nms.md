---
created: 2026-05-11
modified: 2026-05-11
---

# `Markers._remove_close_peaks` switches from morphological max-filter to sparse cKDTree

`Markers._remove_close_peaks` (in `nellie/segmentation/mocap_marking.py`)
performs non-max suppression on the post-LoG peak coordinates. The
original implementation allocated a full-volume float32 `score_img`,
sparsely assigned peak intensities into it, and ran
`maximum_filter(size=2*peak_min_distance + 1)` over the entire volume
— cost `O(volume × size^d)`, regardless of how many peaks survived
`_local_max_peak`. On a `(200,200,200)` frame with the default
`peak_min_distance=2`, that's ~1B ops per frame for a few thousand
peaks.

PRD #179 replaces the morphological path with a sparse-coordinate
approach using `scipy.spatial.cKDTree(coords).query_pairs(r, p=∞)`
(Chebyshev metric). Memory becomes `O(P)` instead of `O(N)`; cost
becomes `O(P log P + |pairs|)` rather than `O(N × size^d)`. Two
coupled decisions in the design need to be pinned ahead of the
rewrite in Slice 2 (#181): (1) the chunked path
(`_remove_close_peaks_chunked` + `_nms_halo`) is deleted entirely
because sparse memory is `O(P)` and chunking served only the
full-volume score_img allocation, and (2) the wiki article's "chosen
for GPU-friendliness" framing (`wiki/segmentation/mocap-marking.md`
line 23) is reversed because the GPU rationale was theoretical
(Markers is not MPS-onboarded; the CUDA path is unvalidated per
`wiki/now.md` watch list). Status: **Accepted (preventive)** —
documents non-obvious decisions ahead of the rewrite in Slice 2 (#181).

## Considered Options

- **Sparse cKDTree with Chebyshev metric (chosen).** `cKDTree(coords)`
  build + `query_pairs(r=peak_min_distance, p=np.inf)` returns exactly
  the pairs the morphological max-filter would have considered
  competing — the Chebyshev box of width `2*peak_min_distance + 1` is
  the same neighborhood as the L∞ ball of radius `peak_min_distance`.
  For each pair, the lower-intensity peak is suppressed; ties are
  preserved on both sides (matches morphological "both equal the local
  max" behavior). Bit-equivalent semantics modulo coord ordering
  (sort row-major to match `xp.argwhere`).
- **Sparse with Euclidean metric (rejected).** Would change the
  neighborhood definition: a peak at offset `(2, 2, 0)` is at
  Chebyshev distance 2 (in the morphological window at
  `peak_min_distance=2`) but Euclidean distance √8 ≈ 2.83 (outside
  the L2 ball of radius 2). Rejected because preserving the
  morphological semantics is non-negotiable — existing
  characterization tests (`test_low_memory_matches_full_*`,
  `test_markers_inside_objects_*`) depend on it.
- **Bounding-box morphological (rejected).** Keep the morphological
  path but only apply `maximum_filter` within the bounding box of all
  peak coordinates. Saves the volume cost outside the bbox but is
  still `O(bbox_volume × size^d)` — no asymptotic improvement, and
  for a label-heavy frame the peak bbox covers most of the volume
  anyway.
- **Keep morphological as fallback for very dense peaks (rejected).**
  Adds a code path with no realistic trigger. cKDTree's per-pair cost
  scales with `|pairs|`, which on densely-packed peaks could in
  principle exceed `O(N)`, but `peak_min_distance` is small (default
  2) and post-LoG peaks are spatially separated by construction.
  Rejected to keep the implementation single-path; if a future
  workload demonstrates morphological wins, that's a separate PRD
  with its own justification.
- **Drop `_remove_close_peaks_chunked` + `_nms_halo` (chosen).** The
  chunked variant existed solely because the full-volume `score_img`
  was the dominant memory cost. Sparse memory is `O(P)`, so chunking
  is moot. `low_memory=True` becomes a no-op for NMS specifically;
  the LoG chunked variant (`_local_max_peak_chunked` +
  `_log_halo`) still gates on `low_memory` and is untouched by this
  PRD.
- **Reverse the "GPU-friendliness" intent (chosen).** The wiki article
  at `wiki/segmentation/mocap-marking.md` line 23 documented the
  morphological choice as "Chosen for GPU-friendliness, not
  theoretical purity." This rationale is theoretical: Markers is not
  MPS-onboarded; the CUDA path exists but is not perf-validated; the
  99% case is CPU. cKDTree is scipy/CPU-only — for backend coords we
  transfer to CPU once (size = peak count, not volume — negligible),
  run, transfer back. If/when Markers is MPS-onboarded later, the
  sparse approach still wins because the transfer is `O(P)` while
  the morphological cost is `O(N × size^d)` regardless of platform.
- **Always thread `query_pairs` (rejected).** `cKDTree.query_pairs` is
  a single C call that releases the GIL internally and is already
  fast on typical peak counts (microseconds for `P ≈ 10⁴`). Threading
  the post-pair `keep[i]` / `keep[j]` updates would race on shared
  state and is unwarranted given the negligible serial cost.

## Consequences

- The Slice 1 synthetic-test suite (#180) covers all morphological-NMS
  edge cases: empty input, single peak, two-peaks-far-apart,
  two-peaks-close-different-scores, two-peaks-close-equal-scores,
  Chebyshev-corner-peak (offset `(peak_min_distance,
  peak_min_distance)`), Chebyshev-just-outside (offset
  `(peak_min_distance + 1, 0)`), three-peak chain. All hand-derived
  from morphological semantics, platform-stable, parametrized 2D/3D.
  Slice 2's sparse rewrite must pass them byte-identical (modulo
  coord ordering).
- **No cross-platform snapshot test.** `_remove_close_peaks`'s input
  `coords` come from `_local_max_peak`'s `gaussian_laplace`, which is
  SIMD-sensitive across platforms (same root cause that forced PRD
  #168 to cache both input and output `.npy` files for
  `_remove_connected_label_pixels`). The synthetic suite is
  sufficient; a yeast-3D snapshot would either require
  input-caching infrastructure (not worth it for this scope) or
  drift cross-platform in a way that masks real regressions.
- Future maintainers should not re-introduce the chunked path. If a
  real workload demands memory-bounded NMS, that's a separate PRD
  that must address why sparse-O(P) is insufficient (and update or
  remove this ADR).
- Future maintainers should not re-introduce the "GPU-friendliness"
  framing without a real perf measurement on a backend-aware
  morphological path. The current expectation is that sparse cKDTree
  on CPU is faster than morphological max-filter on any backend for
  any realistic peak count.
- Future maintainers tempted to switch to Euclidean metric should
  read this ADR first — the morphological semantics are the contract,
  and Chebyshev is the only metric that preserves them.

## References

- PRD #179 — replace morphological NMS in `Markers._remove_close_peaks`
  with sparse `cKDTree`
- Slice 1 #180 — pin current behavior with synthetic-test suite + this
  ADR (no production code changes)
- Slice 2 #181 — sparse rewrite that preserves the synthetic suite
  byte-identical
- ADR 0005 — `Network._relabel_objects` writeback serialization
  (similar precedent: synthetic suite + intra-platform equivalence,
  no cross-platform snapshot, low_memory gating decisions)
- PRD #168 — `Network._remove_connected_label_pixels` sparse rewrite
  (precedent for sparse coordinate scan in this repo, including the
  input-caching infrastructure for snapshot tests)
