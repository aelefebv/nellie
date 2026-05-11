---
created: 2026-05-11
modified: 2026-05-11
---

# `_relabel_objects` writeback is intentionally serialized; threading is gated on `low_memory`

`Network._relabel_objects` runs a per-object EDT loop (one
`scipy.ndimage.distance_transform_edt` per label, on its bounding-box
crop). The EDT releases the GIL — empirically 3.0× from 4 worker threads
on `(200,200,200)` inputs — so the threading rewrite in PRD #173 Slice 2
(#175) parallelizes the per-object loop with a `ThreadPoolExecutor`. The
shared output array `relabelled_np` and the per-worker memory budget
both interact across iterations, and two coupled decisions in the design
need to be pinned ahead of that rewrite: (1) the writeback to
`relabelled_np` is serialized on the main thread (workers compute and
return `(slice, obj_mask, values)`; main thread consumes via
`as_completed` and writes), and (2) `self.low_memory=True` forces the
serial path. Worker count is capped at `min(cpu_count, len(work), 8)`.
Status: **Accepted (preventive)** — documents non-obvious decisions
ahead of the threading rewrite in Slice 2 (#175).

## Considered Options

- **Serialize the writeback (chosen).** Workers compute their per-object
  EDT and return `(slice, obj_mask, values)`; the main thread consumes
  via `as_completed` and writes `relabelled_np[sl][obj_mask] = values`.
  Bounding boxes overlap on label-heavy frames (real, common — one
  object's bbox can contain another object's voxels; pinned by
  `tests/test_networking.py::test_relabel_objects_two_overlapping_bboxes_3d`).
  Object **masks** do not overlap (each voxel has exactly one label or
  zero). Masked-write semantics are safe at the *mask* level — but the
  actual write pattern reads a slice view, modifies it, and writes the
  slice back, which IS a race surface when two workers fetch overlapping
  views simultaneously. Serializing the writeback eliminates the race
  entirely with negligible cost (one masked assignment per object). A
  future maintainer tempted to "parallelize the writeback for that last
  5%" would break correctness silently on overlapping bboxes.
- **Parallelize the writeback (rejected).** Each worker writes its own
  result directly into `relabelled_np`. Avoids the main-thread
  bottleneck of serialized writeback. Rejected because the read-modify-
  write through a slice view races on overlapping bboxes (see above) and
  the per-write cost is already negligible — workers spend ~all their
  time inside the EDT, not the writeback.
- **Lock per write (rejected).** A `threading.Lock` around each write.
  Correctness-equivalent to serializing on the main thread but adds
  per-call lock-acquire overhead with no upside.
- **Force serial when `low_memory=True` (chosen).** Threading raises peak
  working memory to roughly `K workers × max_bbox_voxels × ~5 floats`
  (the EDT's distance + indices arrays, sub-mask, sub-branch, sub-labels
  crops). For typical bboxes this is small, but `low_memory` exists
  precisely for cases where it isn't. Matches the existing pattern at
  `_skeletonize` line 345 (`low_memory=True` swaps in
  `_skeletonize_per_object`).
- **Always thread regardless of `low_memory` (rejected).** Subverts the
  user-facing escape hatch. A user who has explicitly opted into
  `low_memory=True` because their machine cannot hold the high-memory
  path should not pay the threading multiplier.
- **Add a `NetworkConfig.num_threads` field (rejected for this PR).**
  Finer worker-count control would be a separate user-facing knob. Out
  of scope; `low_memory` is the existing escape hatch. If a real
  workload needs finer control later, that is a follow-up PR with its
  own justification.
- **Cap workers at `min(cpu_count, len(work), 8)` (chosen).** EDT
  scaling is memory-bandwidth-bound past 4-8 threads on typical
  hardware (empirical: 3.00× from 4 workers, diminishing returns past
  that). Capping at 8 protects against 64-core boxes wasting context-
  switch overhead on already-saturated bandwidth. Capping at `len(work)`
  avoids spinning idle workers on trivial frames. Capping at
  `cpu_count` because cores aren't free.

## Consequences

- The Slice 1 synthetic-test suite includes
  `test_relabel_objects_two_overlapping_bboxes_3d`, which would fail if
  a future rewrite parallelized the writeback and assumed bbox non-
  overlap. The other five synthetic tests (empty, single object, two
  non-overlapping bboxes, sparse label IDs, object-with-no-seeds) pin
  the rest of the contract that Slice 2 (#175) will preserve byte-for-
  byte.
- The committed snapshot at `tests/fixtures/relabel_objects_3d_*.npy`
  (~10.8 MB total, three .npy files: input + branch + golden) is the
  cross-platform regression bar. All three are cached because the
  upstream Filter+Label+skeleton+pixel-class chain is platform-sensitive
  (Frangi vesselness is floating-point SIMD-sensitive); caching the
  inputs lets the snapshot test exercise only the platform-deterministic
  `_relabel_objects` algorithm. PRD #168 Slice 1 hit this exact lesson
  by caching only the output and watching CI fail on Linux + Windows.
- Future maintainers should not parallelize the writeback. If a real
  perf measurement on a future workload shows the serialized writeback
  is meaningful overhead, that is a separate PRD that must address the
  overlapping-bbox race condition explicitly (and update or remove this
  ADR).
- Future maintainers should not remove the `low_memory` gating. The
  threading branch raises peak memory by `K × max_bbox_voxels × ~5
  floats`; `low_memory` is the user's signal that this is unacceptable.
- Future maintainers who need finer worker-count control should add a
  `NetworkConfig.num_threads` field with explicit justification, not
  bury it as an environment variable or hardcode.

## References

- PRD #173 — thread `_relabel_objects` per-object EDTs
- Slice 1 #174 — pin current behavior with snapshot + 6 synthetic tests
  + this ADR (no production code changes)
- Slice 2 #175 — threading rewrite that preserves the serialized
  writeback and `low_memory` gating
