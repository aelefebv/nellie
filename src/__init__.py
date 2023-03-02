from src.utils.base_logger import logger
try:
    import cupy as xp
    import cupy_backends as xp_bk
    import cupyx.scipy.ndimage as ndi
    from cucim.skimage import filters, morphology, measure
    is_gpu = True
    logger.info('GPU packages detected, running via GPU.')
except ModuleNotFoundError:
    import numpy as xp
    xp_bk = None
    import scipy.ndimage as ndi
    from skimage import filters, morphology
    is_gpu = False
    logger.warning('GPU packages not detected, running via CPU.')
