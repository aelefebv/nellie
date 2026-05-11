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
