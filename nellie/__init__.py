# Static module-level dispatch — keeps the legacy ``nellie.xp`` / ``nellie.ndi``
# attributes that early callers still import. This is *not* the canonical
# device-resolution surface — see :mod:`nellie.utils.adaptive_run` and
# wiki/decisions/0002-adaptive-run-extension-over-static-torch-xp.md
# for why per-stage dispatch is the path forward.
#
# macOS stays pinned to numpy here. MPS is selected per-stage via
# ``adaptive_run.resolve_backend("mps")`` when the user opts in via
# ``device="auto"`` / ``"gpu"`` / ``"mps"``.
import platform


device_type = "cpu"
xp_bk = None
is_gpu = False

if platform.system() == "Darwin":
    import numpy as xp
    import scipy.ndimage as ndi
    from skimage import filters, morphology, measure
else:
    try:
        import cupy as xp
        import cupy_backends as xp_bk
        import cupyx.scipy.ndimage as ndi

        is_gpu = True
        device_type = "cuda"
    except ModuleNotFoundError:
        import numpy as xp
        import scipy.ndimage as ndi
        from skimage import filters, morphology, measure

        xp_bk = None
