---
created: 2026-05-09
modified: 2026-05-09
---

# MPS dispatch via adaptive_run, not static torch_xp module-level shim

For wiring PyTorch+MPS into nellie (see PRD #140 and [[decisions/0001-pytorch-mps-over-mlx]]), we extended `adaptive_run.resolve_backend` with a `device="mps"` arm rather than resurrecting the previously-attempted static `torch_xp` import in `nellie/__init__.py` (now a commented-out block). The per-stage adaptive_run path is the canonical device-resolution surface in the codebase today; module-level pinning would conflict with per-stage `device=` choices and re-introduce the coordination bug between the two device-detection systems described in [[gpu-runtime]].

## Considered Options

- **Resurrect static `torch_xp` in `__init__.py`** — the original commented-out approach. Sets `xp = torch_xp` at import time on Darwin. Rejected because it conflicts with per-stage selection: a user who passes `device="cpu"` to a single stage would still get torch_xp from the module-level import, and the two device systems would disagree on which backend is active.
- **Hybrid: torch only inside `Filter`** — only the recently-perf-tuned filtering stage gets MPS-aware paths; other stages stay numpy-or-cupy. Rejected because the shim cost is roughly fixed regardless of how many stages use it; restricting scope just makes the GPU-on-Mac promise a half-truth.

## Consequences

- The commented-out `torch_xp` block in `nellie/__init__.py` is removed (superseded), not preserved as historical reference. This ADR carries the historical context.
- A new `adaptive_run.device_cascade(device) → list[str]` helper replaces inline `device_order = [...]` constructions across cascade-using stages. Adding any future backend is a one-line change to `device_cascade` rather than an N-stage refactor.
- Non-onboarded stages (`hierarchical`, `voxel_reassignment`, `mocap_marking`) still need to handle a Mac user passing `device="mps"` — they fall back to CPU via the cascade rather than crashing.
