from src.utils.base_logger import logger
try:
    import cupy as xp
    is_gpu = True
    logger.info('CUPY detected, running via GPU.')
except ModuleNotFoundError:
    import numpy as xp
    is_gpu = False
    logger.warning('CUPY not detected, running via CPU.')
