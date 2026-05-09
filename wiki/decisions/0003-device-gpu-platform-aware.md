---
created: 2026-05-09
modified: 2026-05-09
---

# `device="gpu"` is platform-aware: MPS on Darwin, CUDA elsewhere

The user-facing `device=` API on pipeline stages now treats `"gpu"` as a platform-aware request — MPS on Darwin (if torch+MPS is available), CUDA elsewhere (if cupy is available). Explicit overrides remain: `device="mps"` and `device="cuda"` force a specific backend. See PRD #140 and [[gpu-runtime]] for the broader context.

This is the user-visible API choice that makes the "researchers on Macs get GPU acceleration" story work: a research user who copies a `device="gpu"` snippet from a Linux-CUDA tutorial gets MPS on their Mac without code changes, and the napari plugin's existing `["auto", "cpu", "gpu"]` combo dispatches correctly on every platform.

## Considered Options

- **`"gpu"` keeps meaning CUDA only; Mac users must write `"mps"` explicitly.** Cleaner mental model but worse DX for the target audience — Mac users have to know the magic string and copy-paste tutorials don't work.
- **No new explicit value; `"gpu"` learns platform dispatch and `"mps"` is not exposed.** Simpler API but loses the escape hatch for users who want to force one specific backend (e.g., to test the MPS path on a Mac that also has eGPU CUDA — rare but real).

## Consequences

- `adaptive_run.normalize_device` and `resolve_backend` accept `"mps"` as a valid value alongside `"auto" | "cpu" | "gpu" | "cuda"`.
- `device="auto"` continues to resolve to the platform's preferred GPU first, falling back to CPU.
- Users on a Mac without `pip install 'nellie[mps]'` who pass `device="gpu"` still get CPU (graceful fallback), since the resolver detects torch is missing.
