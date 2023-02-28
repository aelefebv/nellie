from src.io.im_info import ImInfo
from src import xp, is_gpu
from src.utils.base_logger import logger
import numpy as np


class FrangiFilter:
    def __init__(
            self, im_info: ImInfo,
            alpha: float = 0.5, beta: float = 0.5,
            num_sigma: int = 10,
            sigma_min_max: tuple = (None, None),
    ):
        self.im_info = im_info
        self.im_memmap = im_info.get_im_memmap(self.im_info.im_path)
        self.alpha = alpha
        self.beta = beta
        self.num_sigma = num_sigma
        self.sigma_min, self.sigma_max = sigma_min_max
        if (self.sigma_min is None) or (self.sigma_max is None):
            self._set_default_sigmas()

    def _set_default_sigmas(self):
        logger.debug('No sigma values provided, setting to defaults.')
        self.sigma_min = self.im_info.dim_sizes['X'] * 10
        self.sigma_max = self.im_info.dim_sizes['X'] * 15
        min_sigma_step_size = 0.2

        sigma_step_size_calculated = (self.sigma_max - self.sigma_min) / self.num_sigma
        sigma_step_size = max(min_sigma_step_size, sigma_step_size_calculated)  # Avoid taking too small of steps.
        logger.debug(f'Calculated sigma step size = {sigma_step_size_calculated}. Using {sigma_step_size}')
        self.sigmas = list(xp.arange(self.sigma_min, self.sigma_max, sigma_step_size))

    def run_filter(self):
        logger.info('Allocating memory for frangi filtered image.')
        self.im_info.allocate_memory(
            self.im_info.path_im_frangi, shape=self.im_memmap.shape, dtype='double', description='Frangi image.',
        )


if __name__ == "__main__":
    filepath = r"D:\test_files\nelly\deskewed-single.ome.tif"
    test = ImInfo(filepath)
    frangi = FrangiFilter(test)
    frangi.run_filter()
