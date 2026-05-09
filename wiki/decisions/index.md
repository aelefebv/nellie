---
created: 2026-05-09
modified: 2026-05-09
---

# Architecture Decision Records

Decisions worth recording per [[CLAUDE|wiki conventions]] — hard to reverse, surprising without context, the result of a real trade-off. See `repo-wiki/DECISIONS_FORMAT.md` for the format.

## Index

- [[decisions/0001-pytorch-mps-over-mlx]] — chose PyTorch+MPS over MLX for Mac GPU acceleration
- [[decisions/0002-adaptive-run-extension-over-static-torch-xp]] — wired MPS through `adaptive_run` rather than reviving the commented-out static `torch_xp` block
- [[decisions/0003-device-gpu-platform-aware]] — `device="gpu"` becomes platform-aware (MPS on Darwin, CUDA elsewhere)
