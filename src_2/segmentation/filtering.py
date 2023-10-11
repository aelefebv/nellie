import os
from itertools import combinations_with_replacement

from src import logger
from src_2.io.im_info import ImInfo
from src_2.utils.general import get_reshaped_image, bbox
from src import xp, ndi

from src_2.utils.gpu_functions import triangle_threshold, otsu_threshold


class Filter:
    def __init__(self, im_info: ImInfo,
                 num_t=None, remove_edges=True,
                 min_radius_um=0.20, max_radius_um=1):
        self.im_info = im_info
        self.z_ratio = self.im_info.dim_sizes['Z'] / self.im_info.dim_sizes['X']
        self.num_t = num_t
        if num_t is None and not self.im_info.no_t:
            self.num_t = im_info.shape[im_info.axes.index('T')]
        self.remove_edges = remove_edges
        # either (roughly) diffraction limit, or pixel size, whichever is larger
        self.min_radius_um = max(min_radius_um, self.im_info.dim_sizes['X'])
        self.max_radius_um = max_radius_um

        self.min_radius_px = self.min_radius_um / self.im_info.dim_sizes['X']
        self.max_radius_px = self.max_radius_um / self.im_info.dim_sizes['X']

        self.im_memmap = None
        self.frangi_memmap = None

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
            self.sigma_vec = (sigma / self.z_ratio, sigma, sigma)
        return self.sigma_vec

    def _set_default_sigmas(self):
        logger.debug('Setting to sigma values.')
        min_sigma_step_size = 0.2
        num_sigma = 5

        self.sigma_min = self.min_radius_px/2
        self.sigma_max = self.max_radius_px/3


        sigma_step_size_calculated = (self.sigma_max - self.sigma_min) / num_sigma
        sigma_step_size = max(min_sigma_step_size, sigma_step_size_calculated)  # Avoid taking too small of steps.

        self.sigmas = list(xp.arange(self.sigma_min, self.sigma_max, sigma_step_size))
        logger.debug(f'Calculated sigma step size = {sigma_step_size_calculated}. Sigmas = {self.sigmas}')

    def _gauss_filter(self, sigma, t=None):
        # if self.sigma_vec is None:
        self._get_sigma_vec(sigma)
        gauss_volume = xp.asarray(self.im_memmap[t, ...]).astype('double')
        logger.debug(f'Gaussian filtering {t=} with {self.sigma_vec=}.')
        gauss_volume = ndi.gaussian_filter(gauss_volume, sigma=self.sigma_vec,
                                           mode='reflect', cval=0.0, truncate=3).astype('double')
        return gauss_volume

    def _calculate_gamma(self, gauss_volume):
        gamma_tri = triangle_threshold(gauss_volume[gauss_volume > 0])
        gamma_otsu, _ = otsu_threshold(gauss_volume[gauss_volume > 0])
        gamma = (gamma_tri + gamma_otsu) / 2
        return gamma

    def _compute_hessian(self, image):
        gradients = xp.gradient(image)
        axes = range(image.ndim)

        h_elems = xp.array([xp.gradient(gradients[ax0], axis=ax1)
                   for ax0, ax1 in combinations_with_replacement(axes, 2)])
        h_mask = self._get_frob_mask(h_elems)

        hxx, hxy, hxz, hyy, hyz, hzz = [elem[..., xp.newaxis, xp.newaxis] for elem in h_elems[:, h_mask]]

        hessian_matrices = xp.concatenate([
            xp.concatenate([hxx, hxy, hxz], axis=-1),
            xp.concatenate([hxy, hyy, hyz], axis=-1),
            xp.concatenate([hxz, hyz, hzz], axis=-1)
        ], axis=-2)

        return h_mask, hessian_matrices

    def _get_frob_mask(self, hessian_matrices):
        frobenius_norm = xp.linalg.norm(hessian_matrices, axis=0)
        frobenius_threshold = triangle_threshold(frobenius_norm[frobenius_norm > 0])
        mask = frobenius_norm > frobenius_threshold
        return mask

    def _compute_chunkwise_eigenvalues(self, hessian_matrices, chunk_size=1E6):
        total_voxels = len(hessian_matrices)

        eigenvalues_list = []

        if chunk_size is None:  # chunk size is entire vector
            chunk_size = total_voxels

        # Iterate over chunks
        for start_idx in range(0, total_voxels, int(chunk_size)):
            end_idx = min(start_idx + chunk_size, total_voxels)
            chunk_eigenvalues = xp.linalg.eigvalsh(
                hessian_matrices[start_idx:end_idx]
            )
            eigenvalues_list.append(chunk_eigenvalues)

        # Concatenate all the eigenvalue chunks and reshape to the original spatial structure
        eigenvalues_flat = xp.concatenate(eigenvalues_list, axis=0)
        sort_order = xp.argsort(xp.abs(eigenvalues_flat), axis=1)
        eigenvalues_flat = xp.take_along_axis(eigenvalues_flat, sort_order, axis=1)

        return eigenvalues_flat

    def _filter_hessian(self, eigenvalues, gamma_sq):
        alpha_sq = 0.5
        beta_sq = 0.5

        ra_sq = (xp.abs(eigenvalues[:, 1]) / xp.abs(eigenvalues[:, 2])) ** 2
        rb_sq = (xp.abs(eigenvalues[:, 1]) / xp.sqrt(xp.abs(eigenvalues[:, 1] * eigenvalues[:, 2]))) ** 2
        s_sq = (xp.sqrt((eigenvalues[:, 0] ** 2) + (eigenvalues[:, 1] ** 2) + (eigenvalues[:, 2] ** 2))) ** 2

        filtered_im = (1 - xp.exp(-(ra_sq / alpha_sq))) * (xp.exp(-(rb_sq / beta_sq))) * \
                      (1 - xp.exp(-(s_sq / gamma_sq)))
        filtered_im[eigenvalues[:, 2] > 0] = 0
        filtered_im[eigenvalues[:, 1] > 0] = 0
        filtered_im = xp.nan_to_num(filtered_im, False, 1)
        return filtered_im

    def _filter_log(self, frame, mask):
        lapofg = xp.empty(((len(self.sigmas),) + frame.shape), dtype=float)
        for i, s in enumerate(self.sigmas):
            sigma_vec = self._get_sigma_vec(s)
            current_lapofg = -ndi.gaussian_laplace(frame, sigma_vec) * xp.mean(s) ** 2
            current_lapofg = current_lapofg * mask
            current_lapofg[current_lapofg < 0] = 0
            lapofg[i] = current_lapofg
        lapofg_min_proj = xp.min(lapofg, axis=0)
        return lapofg_min_proj

    def _run_frame(self, t):
        logger.info(f'Running frangi filter on {t=}.')

        vesselness = xp.zeros_like(self.im_memmap[t, ...], dtype='double')
        temp = xp.zeros_like(self.im_memmap[t, ...], dtype='double')
        masks = xp.ones_like(self.im_memmap[t, ...])

        for sigma_num, sigma in enumerate(self.sigmas):
            gauss_volume = self._gauss_filter(sigma, t)

            gamma = self._calculate_gamma(gauss_volume)
            gamma_sq = 2 * gamma ** 2

            h_mask, hessian_matrices = self._compute_hessian(gauss_volume)
            eigenvalues = self._compute_chunkwise_eigenvalues(hessian_matrices)

            temp[h_mask] = self._filter_hessian(eigenvalues, gamma_sq=gamma_sq)

            max_indices = temp > vesselness
            vesselness[max_indices] = temp[max_indices]
            masks = xp.where(~h_mask, 0, masks)

        vesselness = vesselness * masks
        return vesselness

    def _mask_volume(self, frangi_frame):
        frangi_threshold = xp.percentile(frangi_frame[frangi_frame > 0], 1)
        frangi_mask = frangi_frame > frangi_threshold
        frangi_mask = ndi.binary_opening(frangi_mask)
        frangi_frame = frangi_frame * frangi_mask
        return frangi_frame

    def _remove_edges(self, frangi_frame):
        for z_idx, z_slice in enumerate(frangi_frame):
            rmin, rmax, cmin, cmax = bbox(z_slice)
            frangi_frame[z_idx, rmin:rmin + 10, ...] = 0
            frangi_frame[z_idx, rmax - 10:rmax + 1, ...] = 0
            frangi_frame[z_idx, :, cmin:cmin + 10] = 0
            frangi_frame[z_idx, :, cmax - 10:cmax + 1] = 0
        return frangi_frame

    def _run_filter(self):
        for t in range(self.num_t):
            frangi_frame = self._run_frame(t)
            if self.remove_edges:
                frangi_frame = self._remove_edges(frangi_frame)
            frangi_frame = self._mask_volume(frangi_frame)#.get()
            log_frame = self._filter_log(frangi_frame, frangi_frame > 0)
            if self.im_info.no_t:
                self.frangi_memmap[:] = log_frame.get()[:]
            else:
                self.frangi_memmap[t, ...] = log_frame.get()

    def run(self):
        logger.info('Running frangi filter.')
        self._get_t()
        self._allocate_memory()
        self._set_default_sigmas()
        self._run_filter()

if __name__ == "__main__":
    # test_folder = r"D:\test_files\nelly_tests"
    test_folder = r"D:\test_files\julius_examples"
    all_files = os.listdir(test_folder)
    all_files = [file for file in all_files if not os.path.isdir(os.path.join(test_folder, file))]
    im_infos = []
    for file in all_files:
        im_path = os.path.join(test_folder, file)
        # im_info = ImInfo(im_path)
        im_info = ImInfo(im_path, dim_sizes={'T': 0, 'X': 0.11, 'Y': 0.11, 'Z': 0.1})
        im_infos.append(im_info)

    frangis = []
    for im_info in im_infos:
        frangi = Filter(im_info, remove_edges=False)
        # frangi = Filter(im_info, num_t=2)
        frangi.run()
        frangis.append(frangi)