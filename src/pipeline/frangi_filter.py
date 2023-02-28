import tifffile

from src.io.im_info import ImInfo
from src import xp, is_gpu, ndi, filters
from src.utils.base_logger import logger
import numpy as np


class FrangiFilter:
    def __init__(
            self, im_info: ImInfo,
            alpha: float = 0.5, beta: float = 0.5,
            num_sigma: int = 10,
            sigma_min_max: tuple = (None, None),
            gamma: float = None,
            frobenius_thresh: float = None,
    ):
        self.im_info = im_info
        self.im_memmap = im_info.get_im_memmap(self.im_info.im_path)
        self.im_frangi = None
        self.frangi_sigma = None
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.frobenius_thresh = frobenius_thresh
        self.num_sigma = num_sigma
        self.sigma_min, self.sigma_max = sigma_min_max
        if (self.sigma_min is None) or (self.sigma_max is None):
            self._set_default_sigmas()
        min_sigma_step_size = 0.2
        sigma_step_size_calculated = (self.sigma_max - self.sigma_min) / self.num_sigma
        sigma_step_size = max(min_sigma_step_size, sigma_step_size_calculated)  # Avoid taking too small of steps.
        logger.debug(f'Calculated sigma step size = {sigma_step_size_calculated}. Using {sigma_step_size}')
        self.sigmas = list(xp.arange(self.sigma_min, self.sigma_max, sigma_step_size))

    def _set_default_sigmas(self):
        logger.debug('No sigma values provided, setting to defaults.')
        self.sigma_min = self.im_info.dim_sizes['X'] * 10
        self.sigma_max = self.im_info.dim_sizes['X'] * 15

    def _gaussian_filter(self, sigma, t_num):
        if len(self.frangi_sigma.shape) == 3:
            z_ratio = self.im_info.dim_sizes['Z'] / self.im_info.dim_sizes['X']
            sigma_vec = (sigma / z_ratio, sigma, sigma)
        elif len(self.frangi_sigma.shape) == 2:
            sigma_vec = (sigma, sigma)
        else:
            sigma_vec = None
            logger.error('Frangi filter supported only for 2D and 3D arrays')
            exit(1)
        self.frangi_sigma = xp.asarray(self.im_memmap[t_num, ...])
        self.frangi_sigma = ndi.gaussian_filter(self.frangi_sigma, sigma=sigma_vec,
                                                mode='reflect', cval=0.0, truncate=3).astype('double')

    def _find_gamma(self):
        if self.gamma is None:
            gamma_tri = filters.threshold_triangle(self.frangi_sigma[self.frangi_sigma > 0])
            gamma_otsu = filters.threshold_otsu(self.frangi_sigma[self.frangi_sigma > 0])
            if is_gpu:
                gamma = (gamma_tri.get() + gamma_otsu.get())/2
            else:
                gamma = (gamma_tri + gamma_otsu)/2
        else:
            gamma = self.gamma
        logger.debug(f'{gamma=}')

    def _filter_with_sigma(self, sigma, t_num):
        self._gaussian_filter(sigma, t_num)
        self._find_gamma()

    def run_filter(self):
        logger.info('Allocating memory for frangi filtered image.')
        im_path = self.im_info.path_im_frangi
        self.im_info.allocate_memory(
            im_path, shape=self.im_memmap.shape, dtype='double', description='Frangi image.',
        )
        self.im_frangi = tifffile.memmap(im_path, mode='r')
        # for each T, go through each sigma, update the im_frangi's T at each step for the max value rather than
        # constructing large stack and finding max at the end (for memory's-sake)
        self.frangi_sigma = xp.zeros(shape=self.im_memmap.shape[1:], dtype='double')
        for t_num in range(self.im_memmap.shape[0]):
            for sigma in self.sigmas:
                self._filter_with_sigma(sigma, t_num)


if __name__ == "__main__":
    filepath = r"D:\test_files\nelly\deskewed-single.ome.tif"
    test = ImInfo(filepath)
    frangi = FrangiFilter(test)
    frangi.run_filter()
