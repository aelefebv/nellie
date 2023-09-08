import os

from src import logger
from src_2.io.im_info import ImInfo
from src_2.utils.general import get_reshaped_image
from src import xp, ndi
from skimage import filters

from src_2.utils.gpu_functions import triangle_threshold, otsu_threshold


class FrangiFilter:
    def __init__(self, im_info: ImInfo,
                 num_t=None, remove_edges=True,
                 min_radius_um=0.20, max_radius_um=1):
        self.im_info = im_info
        self.num_t = num_t
        if num_t is None:
            self.num_t = im_info.shape[im_info.axes.index('T')]
        self.remove_edges = remove_edges
        # either (roughly) diffraction limit, or pixel size, whichever is larger
        self.min_radius_um = max(min_radius_um, self.im_info.dim_sizes['X'])
        self.max_radius_um = max_radius_um

        self.min_radius_px = self.min_radius_um / self.im_info.dim_sizes['X']
        self.max_radius_px = self.max_radius_um / self.im_info.dim_sizes['X']

        self.im_memmap = None
        self.frangi_memmap = None

        self.chunk_size = 128

        self.sigma_vec = None
        self.sigmas = None

    def _get_t(self):
        if self.num_t is None:
            if self.im_info.no_t:
                self.num_t = 1
            else:
                self.num_t = self.im_info.shape[self.im_info.axes.index('T')]
        else:
            return

    def _allocate_memory(self):
        logger.debug('Allocating memory for frangi filter.')
        im_memmap = self.im_info.get_im_memmap(self.im_info.im_path)
        self.im_memmap = get_reshaped_image(im_memmap, self.num_t, self.im_info)
        self.shape = self.im_memmap.shape
        im_frangi_path = self.im_info.create_output_path('im_frangi')
        self.frangi_memmap = self.im_info.allocate_memory(im_frangi_path, shape=self.shape, dtype='double',
                                                          description='frangi filtered im',
                                                          return_memmap=True)

    def _get_sigma_vec(self, sigma):
        if self.im_info.no_z:
            self.sigma_vec = (sigma, sigma)
        else:
            z_ratio = self.im_info.dim_sizes['Z'] / self.im_info.dim_sizes['X']
            self.sigma_vec = (sigma / z_ratio, sigma, sigma)

    def _set_default_sigmas(self):
        logger.debug('Setting to sigma values.')
        min_sigma_step_size = 0.2
        num_sigma = 5

        self.sigma_min = self.min_radius_px/2
        self.sigma_max = self.max_radius_px/3


        sigma_step_size_calculated = (self.sigma_max - self.sigma_min) / num_sigma
        sigma_step_size = max(min_sigma_step_size, sigma_step_size_calculated)  # Avoid taking too small of steps.

        logger.debug(f'Calculated sigma step size = {sigma_step_size_calculated}. Using {sigma_step_size}')
        self.sigmas = list(xp.arange(self.sigma_min, self.sigma_max, sigma_step_size))

    def _gauss_filter(self, sigma, t=None):
        if self.sigma_vec is None:
            self._get_sigma_vec(sigma)
        gauss_volume = xp.asarray(self.im_memmap[t, ...]).astype('double')
        logger.debug(f'Gaussian filtering {t=} with {self.sigma_vec=}.')
        gauss_volume = ndi.gaussian_filter(gauss_volume, sigma=self.sigma_vec,
                                           mode='reflect', cval=0.0, truncate=3).astype('double')
        return gauss_volume

    def _calculate_gamma(self, gauss_volume):
        gamma_tri = triangle_threshold(gauss_volume[gauss_volume > 0])
        gamma_otsu = otsu_threshold(gauss_volume[gauss_volume > 0])
        gamma = (gamma_tri + gamma_otsu) / 2
        return gamma

    def _run_filter(self):
        for t in range(self.num_t):
            for sigma_num, sigma in enumerate(self.sigmas):
                logger.info(f'Running frangi filter on {t=} for {sigma=}.')
                gauss_volume = self._gauss_filter(sigma, t)
                gamma = self._calculate_gamma(gauss_volume)
                logger.debug(f'Gamma calculated as {gamma}.')


    def run(self):
        logger.info('Running frangi filter.')
        self._get_t()
        self._allocate_memory()
        self._set_default_sigmas()
        self._run_filter()

if __name__ == "__main__":
    test_folder = r"D:\test_files\nelly_tests"
    all_files = os.listdir(test_folder)
    all_files = [file for file in all_files if not os.path.isdir(os.path.join(test_folder, file))]
    im_infos = []
    for file in all_files:
        im_path = os.path.join(test_folder, file)
        im_info = ImInfo(im_path)
        im_infos.append(im_info)

    frangis = []
    for im_info in im_infos:
        frangi = FrangiFilter(im_info)
        frangi.run()
        frangis.append(frangi)