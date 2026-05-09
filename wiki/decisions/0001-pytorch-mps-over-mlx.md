---
created: 2026-05-09
modified: 2026-05-09
---

# PyTorch+MPS over MLX for Mac GPU acceleration

For adding Apple Silicon GPU support to nellie's pipeline (see PRD #140 and [[gpu-runtime]]), we chose **PyTorch with MPS** over MLX, JAX-Metal, or custom Metal kernels. PyTorch wins on three axes that matter for our user base of researchers: (1) it's already installed in most scientific Python environments, (2) `torch.nn.functional.conv*` and pool variants cover the convolutional `scipy.ndimage` ops directly, and (3) it preserves a future "torch on both Mac and Linux/CUDA" consolidation path that MLX (Mac-only) would foreclose.

## Considered Options

- **MLX** — Apple-native, smaller install (~50 MB vs PyTorch's ~600 MB), better unified-memory zero-copy semantics, lazy evaluation suits pipeline code. Rejected because the ndimage-shaped op coverage is much thinner (more shim code to write and maintain), the ecosystem is younger and less stable, and researchers are less likely to already have it installed than PyTorch.
- **JAX with Metal backend** — experimental and less stable than the alternatives; rejected on maturity grounds.
- **Custom Metal kernels via PyObjC / coremltools** — out of scope for general array ops; would essentially mean writing our own GPU runtime.

## Consequences

- ~600 MB install footprint when users opt into `pip install 'nellie[mps]'`. Mitigated by making it an optional extra rather than a base dependency.
- MPS doesn't support float64; the shim silently coerces with a one-time log message. See [[gpu-runtime]] gotchas.
- Three structural ndimage ops (`binary_fill_holes`, `binary_opening`, `label`) have no MPS equivalent and must round-trip to scipy on CPU. Cost is small on unified memory but documented as a stage-level perf consideration.
