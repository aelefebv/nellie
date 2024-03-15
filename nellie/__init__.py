from nellie.utils.base_logger import logger
import platform

device_type = 'cpu'
# if it's a mac
if platform.system() == 'Darwin':
    # if it has pytorch
    # try:
    #     import torch
    #     # For Mac GPUs with MPS support
    #     if torch.backends.mps.is_available():
    #         import nellie.utils.torch_xp as xp
    #         device_type = 'mps'
    #         logger.warning('GPU packages detected, running via GPU.')
    #     else:
    #         import numpy as xp
    #         device_type = 'cpu'
    #         logger.warning('GPU packages not detected, running via CPU.')
    # except ModuleNotFoundError:
    import numpy as xp
    device_type = 'cpu'
    logger.warning('GPU packages not detected, running via CPU.')

    xp_bk = None
    import scipy.ndimage as ndi
    from skimage import filters, morphology, measure

    is_gpu = False


# if it's an NVIDIA GPU
else:
    try:
        import cupy as xp
        import cupy_backends as xp_bk
        import cupyx.scipy.ndimage as ndi
        is_gpu = True
        logger.info('GPU packages detected, running via GPU.')
        device_type = 'cuda'
    except ModuleNotFoundError:
        import numpy as xp

        xp_bk = None
        import scipy.ndimage as ndi
        from skimage import filters, morphology, measure

        is_gpu = False
        logger.warning('GPU packages not detected, running via CPU.')
        device_type = 'cpu'

