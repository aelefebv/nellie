from itertools import combinations_with_replacement

from nellie import logger
from nellie.im_info.im_info import ImInfo
from nellie.utils.general import get_reshaped_image, bbox
import numpy as np
from nellie import ndi, xp, device_type

from nellie.utils.gpu_functions import triangle_threshold, otsu_threshold


class Filter:
    def __init__(self, im_info: ImInfo,
                 num_t=None, remove_edges=True,
                 min_radius_um=0.20, max_radius_um=1, alpha_sq=0.5, beta_sq=0.5):
        self.im_info = im_info
        if not self.im_info.no_z:
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

        self.alpha_sq = alpha_sq
        self.beta_sq = beta_sq

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
        im_frangi_path = self.im_info.pipeline_paths['im_frangi']
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

        self.sigma_min = self.min_radius_px / 2
        self.sigma_max = self.max_radius_px / 3

        sigma_step_size_calculated = (self.sigma_max - self.sigma_min) / num_sigma
        sigma_step_size = max(min_sigma_step_size, sigma_step_size_calculated)  # Avoid taking too small of steps.

        self.sigmas = list(np.arange(self.sigma_min, self.sigma_max, sigma_step_size))
        logger.debug(f'Calculated sigma step size = {sigma_step_size_calculated}. Sigmas = {self.sigmas}')

    def _gauss_filter(self, sigma, t=None):
        self._get_sigma_vec(sigma)
        gauss_volume = xp.asarray(self.im_memmap[t, ...], dtype='double')
        logger.debug(f'Gaussian filtering {t=} with {self.sigma_vec=}.')

        gauss_volume = ndi.gaussian_filter(gauss_volume, sigma=self.sigma_vec,
                                           mode='reflect', cval=0.0, truncate=3).astype('double')
        return gauss_volume

    def _calculate_gamma(self, gauss_volume):
        gamma_tri = triangle_threshold(gauss_volume[gauss_volume > 0])
        gamma_otsu, _ = otsu_threshold(gauss_volume[gauss_volume > 0])
        gamma = min(gamma_tri, gamma_otsu)
        return gamma

    def _compute_hessian(self, image, mask=True):
        gradients = xp.gradient(image)
        axes = range(image.ndim)
        h_elems = xp.array([xp.gradient(gradients[ax0], axis=ax1).astype('float16')
                            for ax0, ax1 in combinations_with_replacement(axes, 2)])
        if mask:
            h_mask = self._get_frob_mask(h_elems)
        else:
            h_mask = xp.ones_like(image, dtype='bool')
        if self.remove_edges:
            h_mask = self._remove_edges(h_mask)

        if self.im_info.no_z:
            if device_type == 'cuda':
                hxx, hxy, hyy = [elem[..., np.newaxis, np.newaxis] for elem in h_elems[:, h_mask].get()]
            else:
                hxx, hxy, hyy = [elem[..., np.newaxis, np.newaxis] for elem in h_elems[:, h_mask]]
            hessian_matrices = np.concatenate([
                np.concatenate([hxx, hxy], axis=-1),
                np.concatenate([hxy, hyy], axis=-1)
            ], axis=-2)
        else:
            if device_type == 'cuda':
                hxx, hxy, hxz, hyy, hyz, hzz = [elem[..., np.newaxis, np.newaxis] for elem in h_elems[:, h_mask].get()]
            else:
                hxx, hxy, hxz, hyy, hyz, hzz = [elem[..., np.newaxis, np.newaxis] for elem in h_elems[:, h_mask]]
            hessian_matrices = np.concatenate([
                np.concatenate([hxx, hxy, hxz], axis=-1),
                np.concatenate([hxy, hyy, hyz], axis=-1),
                np.concatenate([hxz, hyz, hzz], axis=-1)
            ], axis=-2)

        return h_mask, hessian_matrices

    def _get_frob_mask(self, hessian_matrices):
        # todo, can we avoid rescaling? slowing down..
        rescaled_hessian = hessian_matrices / xp.max(xp.abs(hessian_matrices))
        frobenius_norm = xp.linalg.norm(rescaled_hessian, axis=0)
        frobenius_norm[xp.isinf(frobenius_norm)] = 0
        frob_triangle_thresh = triangle_threshold(frobenius_norm[frobenius_norm > 0])
        frob_otsu_thresh, _ = otsu_threshold(frobenius_norm[frobenius_norm > 0])
        frobenius_threshold = min(frob_triangle_thresh, frob_otsu_thresh)
        mask = frobenius_norm > frobenius_threshold
        return mask

    def _compute_chunkwise_eigenvalues(self, hessian_matrices, chunk_size=1E6):
        chunk_size = int(chunk_size)
        total_voxels = len(hessian_matrices)

        eigenvalues_list = []

        if chunk_size is None:  # chunk size is entire vector
            chunk_size = total_voxels

        # iterate over chunks
        for start_idx in range(0, total_voxels, int(chunk_size)):
            end_idx = min(start_idx + chunk_size, total_voxels)
            gpu_chunk = xp.array(hessian_matrices[start_idx:end_idx])
            chunk_eigenvalues = xp.linalg.eigvalsh(gpu_chunk)
            eigenvalues_list.append(chunk_eigenvalues)

        # concatenate all the eigval chunks and reshape to the original spatial structure
        eigenvalues_flat = xp.concatenate(eigenvalues_list, axis=0)
        sort_order = xp.argsort(xp.abs(eigenvalues_flat), axis=1)
        eigenvalues_flat = xp.take_along_axis(eigenvalues_flat, sort_order, axis=1)

        return eigenvalues_flat

    def _filter_hessian(self, eigenvalues, gamma_sq):
        if self.im_info.no_z:
            rb_sq = (xp.abs(eigenvalues[:, 0]) / xp.abs(eigenvalues[:, 1])) ** 2
            s_sq = (eigenvalues[:, 0] ** 2) + (eigenvalues[:, 1] ** 2)
            filtered_im = (xp.exp(-(rb_sq / self.beta_sq))) * (1 - xp.exp(-(s_sq / gamma_sq)))
        else:
            ra_sq = (xp.abs(eigenvalues[:, 1]) / xp.abs(eigenvalues[:, 2])) ** 2
            rb_sq = (xp.abs(eigenvalues[:, 1]) / xp.sqrt(xp.abs(eigenvalues[:, 1] * eigenvalues[:, 2]))) ** 2
            s_sq = (xp.sqrt((eigenvalues[:, 0] ** 2) + (eigenvalues[:, 1] ** 2) + (eigenvalues[:, 2] ** 2))) ** 2
            filtered_im = (1 - xp.exp(-(ra_sq / self.alpha_sq))) * (xp.exp(-(rb_sq / self.beta_sq))) * \
                          (1 - xp.exp(-(s_sq / gamma_sq)))
        if not self.im_info.no_z:
            filtered_im[eigenvalues[:, 2] > 0] = 0
        filtered_im[eigenvalues[:, 1] > 0] = 0
        filtered_im = xp.nan_to_num(filtered_im, False, 1)
        return filtered_im

    def _filter_log(self, frame, mask):
        lapofg = xp.zeros_like(frame, dtype='double')
        for i, s in enumerate(self.sigmas):
            sigma_vec = self._get_sigma_vec(s)
            current_lapofg = -ndi.gaussian_laplace(frame, sigma_vec) * xp.mean(s) ** 2
            current_lapofg = current_lapofg * mask
            min_indices = current_lapofg < lapofg
            lapofg[min_indices] = current_lapofg[min_indices]
            if i == 0:
                lapofg = current_lapofg
        lapofg_min_proj = lapofg
        return lapofg_min_proj

    def _run_frame(self, t, mask=True):
        logger.info(f'Running frangi filter on {t=}.')
        vesselness = xp.zeros_like(self.im_memmap[t, ...], dtype='float64')
        temp = xp.zeros_like(self.im_memmap[t, ...], dtype='float64')
        masks = xp.ones_like(self.im_memmap[t, ...], dtype='bool')
        for sigma_num, sigma in enumerate(self.sigmas):
            gauss_volume = self._gauss_filter(sigma, t)  # * xp.mean(sigma) ** 2

            gamma = self._calculate_gamma(gauss_volume)
            gamma_sq = 2 * gamma ** 2

            h_mask, hessian_matrices = self._compute_hessian(gauss_volume, mask=mask)
            eigenvalues = self._compute_chunkwise_eigenvalues(hessian_matrices.astype('float'))

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
        if self.im_info.no_z:
            num_z = 1
        else:
            num_z = self.im_info.shape[self.im_info.axes.index('Z')]
        for z_idx in range(num_z):
            if self.im_info.no_z:
                rmin, rmax, cmin, cmax = bbox(frangi_frame)
            else:
                rmin, rmax, cmin, cmax = bbox(frangi_frame[z_idx, ...])
            frangi_frame[z_idx, rmin:rmin + 15, ...] = 0
            frangi_frame[z_idx, rmax - 15:rmax + 1, ...] = 0
        return frangi_frame

    def _run_filter(self, mask=True):
        for t in range(self.num_t):
            frangi_frame = self._run_frame(t, mask=mask)
            frangi_frame = self._mask_volume(frangi_frame)
            if not self.im_info.no_z:  # helps with z anisotropy
                log_frame = self._filter_log(frangi_frame, frangi_frame > 0)
                log_frame[log_frame < 0] = 0
                filtered_im = log_frame
            else:
                filtered_im = frangi_frame

            if device_type == 'cuda':
                filtered_im = filtered_im.get()

            if self.im_info.no_t:
                self.frangi_memmap[:] = filtered_im[:]
            else:
                self.frangi_memmap[t, ...] = filtered_im
            self.frangi_memmap.flush()

    def run(self, mask=True):
        logger.info('Running frangi filter.')
        self._get_t()
        self._allocate_memory()
        self._set_default_sigmas()
        self._run_filter(mask=mask)


if __name__ == "__main__":
    im_path = r"D:\test_files\nelly_gav_tests\fibro_3.nd2"
    im_info = ImInfo(im_path)
    filter_im = Filter(im_info, num_t=2)
    filter_im.run()
