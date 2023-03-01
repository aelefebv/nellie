import tifffile

from src.io.im_info import ImInfo
from src import xp, is_gpu, ndi, filters, xp_bk
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
        self.in_mem_1 = None
        self.in_mem_2 = None
        self.in_mem_z = None
        self.chunk_size = 20
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
        if len(self.im_memmap[t_num, ...].shape) == 3:
            z_ratio = self.im_info.dim_sizes['Z'] / self.im_info.dim_sizes['X']
            sigma_vec = (sigma / z_ratio, sigma, sigma)
        elif len(self.im_memmap[t_num, ...].shape) == 2:
            sigma_vec = (sigma, sigma)
        else:
            sigma_vec = None
            logger.error('Frangi filter supported only for 2D and 3D arrays')
            exit(1)
        gauss_volume = xp.asarray(self.im_memmap[t_num, ...])
        gauss_volume = ndi.gaussian_filter(gauss_volume, sigma=sigma_vec,
                                           mode='reflect', cval=0.0, truncate=3).astype('double')
        return gauss_volume

    def _find_gamma(self, gauss_volume):
        if self.gamma is None:
            gamma_tri = filters.threshold_triangle(gauss_volume[gauss_volume > 0])
            gamma_otsu = filters.threshold_otsu(gauss_volume[gauss_volume > 0])
            if is_gpu:
                gamma = (gamma_tri.get() + gamma_otsu.get())/2
            else:
                gamma = (gamma_tri + gamma_otsu)/2
        else:
            gamma = self.gamma
        return gamma

    def _get_h(self, gauss_slice):
        dz, dy, dx = xp.gradient(gauss_slice)
        dzz, _, _ = xp.gradient(dz)
        dyz, dyy, _ = xp.gradient(dy)
        dxz, dxy, dxx = xp.gradient(dx)
        h_matrix = xp.array([[dxx, dxy, dxz],
                             [dxy, dyy, dyz],
                             [dxz, dyz, dzz]])
        return h_matrix

    def _get_h_mask(self, h_matrix):
        frobenius_norm = xp.linalg.norm(
            xp.concatenate([h_matrix[0], h_matrix[1], h_matrix[2]], axis=0),
            axis=0)
        frobenius_norm[xp.isinf(frobenius_norm)] = xp.nanmax(frobenius_norm[~xp.isinf(frobenius_norm)])
        if self.frobenius_thresh is None:
            frobenius_threshold = xp.sqrt(xp.nanmax(frobenius_norm))
        else:
            frobenius_threshold = self.frobenius_thresh
        mask = frobenius_norm > frobenius_threshold
        return mask

    def _get_filtered_im(self, h_matrix, gamma):
        alpha_sq = 2 * self.alpha ** 2
        beta_sq = 2 * self.beta ** 2
        gamma_sq = 2 * gamma ** 2

        eigs_thresh, _ = xp.linalg.eigh(h_matrix)
        sort_order = xp.argsort(xp.abs(eigs_thresh), axis=1)
        eig_sort = xp.take_along_axis(eigs_thresh, sort_order, axis=1)
        ra_sq = (xp.abs(eig_sort[:, 1]) / xp.abs(eig_sort[:, 2])) ** 2
        rb_sq = (xp.abs(eig_sort[:, 1]) / xp.sqrt(xp.abs(eig_sort[:, 1] * eig_sort[:, 2]))) ** 2
        s_sq = (xp.sqrt((eig_sort[:, 0] ** 2) + (eig_sort[:, 1] ** 2) + (eig_sort[:, 2] ** 2))) ** 2

        filtered_im = (1 - xp.exp(-(ra_sq / alpha_sq))) * (xp.exp(-(rb_sq / beta_sq))) * \
                      (1 - xp.exp(-(s_sq / gamma_sq)))
        filtered_im[eig_sort[:, 2] > 0] = 0
        filtered_im[eig_sort[:, 1] > 0] = 0
        filtered_im = xp.nan_to_num(filtered_im, False, 1)
        return filtered_im, eig_sort

    def _mask_h_matrix(self, h_matrix):
        h_mask = self._get_h_mask(h_matrix)
        h_matrix = xp.transpose(h_matrix[:, :, h_mask], (2, 1, 0))
        return h_matrix, h_mask

    def _filter_with_sigma(self, sigma, t_num):
        gauss_volume = self._gaussian_filter(sigma, t_num)
        gamma = self._find_gamma(gauss_volume)
        num_z = gauss_volume.shape[0]
        filtered_volume = xp.zeros_like(gauss_volume, dtype='double')
        # todo, get this to work for 2d
        while self.chunk_size > 0:
            try:
                for z in range(0, num_z, self.chunk_size):
                    z_radius = 1  # 1 should be fine since truncation occurs at 3 sigma?
                    start = max(0, z-z_radius)
                    gauss_slice = gauss_volume[start:z + self.chunk_size + z_radius, ...]
                    h_matrix = self._get_h(gauss_slice)
                    h_matrix, h_mask = self._mask_h_matrix(h_matrix)
                    if not len(h_matrix):  # means nothing above threshold got through, so keep filtered as blank
                        continue
                    filtered_im, _ = self._get_filtered_im(h_matrix, gamma)
                    mask_filtered_im = xp.zeros(gauss_slice.shape, dtype='double')
                    mask_filtered_im[h_mask] = filtered_im
                    if start == 0:
                        z_radius = 0
                    filtered_volume[z:z + self.chunk_size, ...] = mask_filtered_im[z_radius:z_radius + self.chunk_size]
                break
            except (xp.cuda.memory.OutOfMemoryError, xp_bk.cuda.libs.cusolver.CUSOLVERError):
                if self.chunk_size == 1:
                    logger.error('GPU memory not large enough, try again using the CPU or downsample your image')
                    exit(1)
                logger.warning(f'Out of memory with chunk size {self.chunk_size}, '
                               f'retrying with a smaller chunk size...')
                xp.get_default_memory_pool().free_all_blocks()
                self.chunk_size = max(self.chunk_size//2, 1)
        return filtered_volume

    def run_filter(self, num_t: int = None):
        logger.info('Allocating memory for frangi filtered image.')
        im_path = self.im_info.path_im_frangi
        if num_t is not None:
            self.im_memmap = self.im_memmap[:num_t, ...]
        num_t = self.im_memmap.shape[0]
        self.im_info.allocate_memory(
            im_path, shape=self.im_memmap.shape, dtype='double', description='Frangi image.',
        )
        self.im_frangi = tifffile.memmap(im_path, mode='r+')
        # allocates memory for a single volume
        for t_num in range(num_t):
            for sigma_number, sigma in enumerate(self.sigmas):
                frangi_in_mem = xp.asarray(self.im_frangi[t_num, ...])
                logger.info(f'Running sigma {sigma_number}/{len(self.sigmas)-1}, volume {t_num}/{num_t-1}')
                filtered_volume = self._filter_with_sigma(sigma, t_num)
                if is_gpu:
                    self.im_frangi[t_num, ...] = xp.amax(xp.stack((frangi_in_mem, filtered_volume)), axis=0).get()
                else:
                    self.im_frangi[t_num, ...] = xp.amax(xp.stack((frangi_in_mem, filtered_volume)), axis=0)


if __name__ == "__main__":
    filepath = r"D:\test_files\nelly\deskewed-single.ome.tif"
    test = ImInfo(filepath, ch=1)
    frangi = FrangiFilter(test)
    frangi.run_filter(2)
    print('hi')
