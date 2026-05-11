---
created: 2026-05-09
modified: 2026-05-11
---


# Architecture Decision Records

Decisions worth recording per [[CLAUDE|wiki conventions]] — hard to reverse, surprising without context, the result of a real trade-off. See `repo-wiki/DECISIONS_FORMAT.md` for the format.

## Index

- [[decisions/0001-pytorch-mps-over-mlx]] — chose PyTorch+MPS over MLX for Mac GPU acceleration
- [[decisions/0002-adaptive-run-extension-over-static-torch-xp]] — wired MPS through `adaptive_run` rather than reviving the commented-out static `torch_xp` block
- [[decisions/0003-device-gpu-platform-aware]] — `device="gpu"` becomes platform-aware (MPS on Darwin, CUDA elsewhere)
- [[decisions/0004-skel-boundary-preservation]] — boundary voxels in `Network._remove_connected_label_pixels` are exempt from ambiguity cleanup (root cause unknown; pinned by test)
- [[decisions/0005-relabel-objects-serialized-writeback]] — `Network._relabel_objects` writeback is intentionally serialized; threading is gated on `low_memory` (preventive, ahead of PRD #173 Slice 2 threading rewrite)
- [[decisions/0006-mocap-marking-sparse-nms]] — `Markers._remove_close_peaks` switches from morphological max-filter to sparse `cKDTree` with Chebyshev metric; chunked variant + `_nms_halo` are dropped (preventive, ahead of PRD #179 Slice 2 rewrite)
- [[decisions/0007-mocap-log-peak-threading]] — `Markers._local_max_peak` per-sigma loop is threaded with `as_completed` (reduction is order-independent); chunked path stays serial; threading gated on `low_memory` (preventive, ahead of PRD #184 Slice 2 threading rewrite)
- [[decisions/0008-hu-moment-matmul-rewrite]] — `HuMomentTracking._calculate_normalized_moments` switches from broadcast `(N, H, W, 4, 4)` to two-step batched matmul; `@` over `einsum` for backend-uniform BLAS dispatch; approx-equivalence (`rtol=1e-5`) test bar (preventive, ahead of PRD #191 Slice 2 rewrite)
- [[decisions/0009-hu-cost-matrix-streaming]] — `HuMomentTracking._get_cost_matrix` switches from broadcast `(N, N, F)` float64 tensors to per-feature streaming float32; `_get_difference_matrix` + `_zscore_normalize` deleted; approx-equivalence (`rtol=1e-3`) test bar — looser than 0008 due to float64→float32 drop (preventive, ahead of PRD #196 Slice 2 rewrite)
- [[decisions/0010-flow-nearby-coords-single-query]] — `FlowInterpolator._get_nearby_coords` switches from double `query_ball_point` + `query(k=max_k)` traversal to single `query_ball_point` + per-coord `linalg.norm`; bundled with NaN-alignment bug fix (`for pos, i in enumerate(good_coords)`); bit-identical for no-NaN, corrected for NaN (preventive, ahead of PRD #204 Slice 2 rewrite)
- [[decisions/0011-voxel-assign-unique-matches-round-based]] — `VoxelReassigner._assign_unique_matches` switches from Python greedy loop to round-based per-prev/per-next argmin intersection; rejected first-occurrence-of-both heuristic as not equivalent to greedy; set-equality (not ordered-equality) test bar (preventive, ahead of PRD #222 Slice 2 rewrite)
