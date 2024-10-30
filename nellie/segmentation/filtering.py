from itertools import combinations_with_replacement

from nellie import logger
from nellie.im_info.verifier import ImInfo
from nellie.utils.general import bbox
import numpy as np
from nellie import ndi, xp, device_type

from nellie.utils.gpu_functions import triangle_threshold, otsu_threshold


class Filter:
    """
    A class that applies the Frangi vesselness filter to 3D or 4D microscopy image data for vessel-like structure detection.

    Attributes
    ----------
    im_info : ImInfo
        An object containing image metadata and memory-mapped image data.
    z_ratio : float
        Ratio of Z to X resolution for scaling Z-axis.
    num_t : int
        Number of timepoints in the image.
    remove_edges : bool
        Flag to remove edges from the processed image.
    min_radius_um : float
        Minimum radius of detected objects in micrometers.
    max_radius_um : float
        Maximum radius of detected objects in micrometers.
    min_radius_px : float
        Minimum radius of detected objects in pixels.
    max_radius_px : float
        Maximum radius of detected objects in pixels.
    im_memmap : np.ndarray or None
        Memory-mapped image data.
    frangi_memmap : np.ndarray or None
        Memory-mapped data for the Frangi-filtered image.
    sigma_vec : tuple or None
        Sigma vector used for Gaussian filtering.
    sigmas : list or None
        List of sigma values for multiscale Frangi filtering.
    alpha_sq : float
        Alpha squared parameter for Frangi filter's sensitivity to vesselness.
    beta_sq : float
        Beta squared parameter for Frangi filter's sensitivity to blobness.
    frob_thresh : float or None
        Threshold for Frobenius norm-based masking of the Hessian matrix.
    viewer : object or None
        Viewer object for displaying status during processing.

    Methods
    -------
    _get_t()
        Determines the number of timepoints in the image.
    _allocate_memory()
        Allocates memory for the Frangi-filtered image.
    _get_sigma_vec(sigma)
        Computes the sigma vector based on image dimensions (Z, Y, X).
    _set_default_sigmas()
        Sets the default sigma values for the Frangi filter.
    _gauss_filter(sigma, t=None)
        Applies a Gaussian filter to a single timepoint of the image.
    _calculate_gamma(gauss_volume)
        Calculates gamma values for vesselness thresholding using triangle and Otsu methods.
    _compute_hessian(image, mask=True)
        Computes the Hessian matrix of the input image and applies masking.
    _get_frob_mask(hessian_matrices)
        Creates a Frobenius norm mask for the Hessian matrix based on a threshold.
    _compute_chunkwise_eigenvalues(hessian_matrices, chunk_size=1E6)
        Computes eigenvalues of the Hessian matrix in chunks to avoid memory overflow.
    _filter_hessian(eigenvalues, gamma_sq)
        Applies the Frangi filter to the Hessian eigenvalues to detect vessel-like structures.
    _filter_log(frame, mask)
        Applies the Laplacian of Gaussian (LoG) filter to enhance vessel structures.
    _run_frame(t, mask=True)
        Runs the Frangi filter for a single timepoint in the image.
    _mask_volume(frangi_frame)
        Creates a binary mask of vessel-like structures in the image based on a threshold.
    _remove_edges(frangi_frame)
        Removes edges from the detected structures in the image.
    _run_filter(mask=True)
        Runs the Frangi filter over all timepoints in the image.
    run(mask=True)
        Main method to execute the Frangi filter process over the image data.
    """
    def __init__(self, im_info: ImInfo,
                 num_t=None, remove_edges=False,
                 min_radius_um=0.20, max_radius_um=1, alpha_sq=0.5, beta_sq=0.5, frob_thresh=None, viewer=None):
        """
        Initializes the Filter object with image metadata and filter parameters.

        Parameters
        ----------
        im_info : ImInfo
            An instance of the ImInfo class, containing image metadata and file paths.
        num_t : int, optional
            Number of timepoints to process. If None, defaults to the number of timepoints in the image.
        remove_edges : bool, optional
            Whether to remove edges from the detected structures (default is False).
        min_radius_um : float, optional
            Minimum radius of detected objects in micrometers (default is 0.20).
        max_radius_um : float, optional
            Maximum radius of detected objects in micrometers (default is 1).
        alpha_sq : float, optional
            Alpha squared parameter for the Frangi filter (default is 0.5).
        beta_sq : float, optional
            Beta squared parameter for the Frangi filter (default is 0.5).
        frob_thresh : float or None, optional
            Threshold for the Frobenius norm-based mask (default is None).
        viewer : object or None, optional
            Viewer object for displaying status during processing (default is None).
        """
        self.im_info = im_info
        if not self.im_info.no_z:
            self.z_ratio = self.im_info.dim_res['Z'] / self.im_info.dim_res['X']
        self.num_t = num_t
        if num_t is None and not self.im_info.no_t:
            self.num_t = im_info.shape[im_info.axes.index('T')]
        self.remove_edges = remove_edges
        # either (roughly) diffraction limit, or pixel size, whichever is larger
        self.min_radius_um = max(min_radius_um, self.im_info.dim_res['X'])
        self.max_radius_um = max_radius_um

        self.min_radius_px = self.min_radius_um / self.im_info.dim_res['X']
        self.max_radius_px = self.max_radius_um / self.im_info.dim_res['X']

        self.im_memmap = None
        self.frangi_memmap = None

        self.sigma_vec = None
        self.sigmas = None

        self.alpha_sq = alpha_sq
        self.beta_sq = beta_sq

        self.frob_thresh = frob_thresh

        self.viewer = viewer

    def _get_t(self):
        """
        Determines the number of timepoints to process.

        If `num_t` is not set and the image contains a temporal dimension, it sets `num_t` to the number of timepoints.
        """
        if self.num_t is None:
            if self.im_info.no_t:
                self.num_t = 1
            else:
                self.num_t = self.im_info.shape[self.im_info.axes.index('T')]
        else:
            return

    def _allocate_memory(self):
        """
        Allocates memory for the Frangi-filtered image.

        This method creates memory-mapped arrays for both the original image and the Frangi-filtered image.
        """
        logger.debug('Allocating memory for frangi filter.')
        self.im_memmap = self.im_info.get_memmap(self.im_info.im_path)
        self.shape = self.im_memmap.shape
        im_frangi_path = self.im_info.pipeline_paths['im_preprocessed']
        self.frangi_memmap = self.im_info.allocate_memory(im_frangi_path, dtype='double',
                                                          description='frangi filtered im',
                                                          return_memmap=True)

    def _get_sigma_vec(self, sigma):
        """
        Generates the sigma vector for Gaussian filtering based on the image dimensions (Z, Y, X).

        Parameters
        ----------
        sigma : float
            The sigma value to use for Gaussian filtering.

        Returns
        -------
        tuple
            Sigma vector for Gaussian filtering.
        """
        if self.im_info.no_z:
            self.sigma_vec = (sigma, sigma)
        else:
            self.sigma_vec = (sigma / self.z_ratio, sigma, sigma)
        return self.sigma_vec

    def _set_default_sigmas(self):
        """
        Sets the default sigma values for the Frangi filter, based on the minimum and maximum radius in pixels.
        """
        logger.debug('Setting to sigma values.')
        min_sigma_step_size = 0.2
        num_sigma = 5

        sigma_1 = self.min_radius_px / 2
        sigma_2 = self.max_radius_px / 3
        self.sigma_min = min(sigma_1, sigma_2)
        self.sigma_max = max(sigma_1, sigma_2)


        sigma_step_size_calculated = (self.sigma_max - self.sigma_min) / num_sigma
        sigma_step_size = max(min_sigma_step_size, sigma_step_size_calculated)  # Avoid taking too small of steps.

        self.sigmas = list(np.arange(self.sigma_min, self.sigma_max, sigma_step_size))

        logger.debug(f'Calculated sigma step size = {sigma_step_size_calculated}. Sigmas = {self.sigmas}')

    def _gauss_filter(self, sigma, t=None):
        """
        Applies a Gaussian filter to a single timepoint in the image.

        Parameters
        ----------
        sigma : float
            Sigma value for Gaussian filtering.
        t : int or None, optional
            Timepoint index to filter (default is None, for a single frame).

        Returns
        -------
        np.ndarray
            Gaussian-filtered volume.
        """
        self._get_sigma_vec(sigma)
        gauss_volume = xp.asarray(self.im_memmap[t, ...], dtype='double')
        logger.debug(f'Gaussian filtering {t=} with {self.sigma_vec=}.')

        gauss_volume = ndi.gaussian_filter(gauss_volume, sigma=self.sigma_vec,
                                           mode='reflect', cval=0.0, truncate=3).astype('double')
        return gauss_volume

    def _calculate_gamma(self, gauss_volume):
        """
        Calculates gamma values for vesselness thresholding using the triangle and Otsu methods.

        Parameters
        ----------
        gauss_volume : np.ndarray
            The Gaussian-filtered volume.

        Returns
        -------
        float
            The minimum gamma value for thresholding.
        """
        gamma_tri = triangle_threshold(gauss_volume[gauss_volume > 0])
        gamma_otsu, _ = otsu_threshold(gauss_volume[gauss_volume > 0])
        gamma = min(gamma_tri, gamma_otsu)
        return gamma

    def _compute_hessian(self, image, mask=True):
        """
        Computes the Hessian matrix of the input image and applies optional masking.

        Parameters
        ----------
        image : np.ndarray
            The input image.
        mask : bool, optional
            Whether to apply Frobenius norm masking (default is True).

        Returns
        -------
        tuple
            Mask and Hessian matrix.
        """
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
        """
        Creates a Frobenius norm mask for the Hessian matrix based on a threshold.

        Parameters
        ----------
        hessian_matrices : np.ndarray
            The Hessian matrix for which to generate a mask.

        Returns
        -------
        np.ndarray
            The Frobenius norm mask.
        """
        rescaled_hessian = hessian_matrices / xp.max(xp.abs(hessian_matrices))
        frobenius_norm = xp.linalg.norm(rescaled_hessian, axis=0)
        frobenius_norm[xp.isinf(frobenius_norm)] = xp.max(frobenius_norm[~xp.isinf(frobenius_norm)])
        if self.frob_thresh is None:
            non_zero_frobenius = frobenius_norm[frobenius_norm > 0]
            if len(non_zero_frobenius) == 0:
                frobenius_threshold = 0
            else:
                frob_triangle_thresh = triangle_threshold(non_zero_frobenius)
                frob_otsu_thresh, _ = otsu_threshold(non_zero_frobenius)
                frobenius_threshold = min(frob_triangle_thresh, frob_otsu_thresh)
        else:
            frobenius_threshold = self.frob_thresh
        mask = frobenius_norm > frobenius_threshold
        print('hi')
        return mask

    def _compute_chunkwise_eigenvalues(self, hessian_matrices, chunk_size=1E6):
        """
        Computes eigenvalues of the Hessian matrix in chunks to avoid memory overflow.

        Parameters
        ----------
        hessian_matrices : np.ndarray
            Hessian matrices to compute eigenvalues for.
        chunk_size : float, optional
            Size of the chunks to process at a time (default is 1E6).

        Returns
        -------
        np.ndarray
            Array of eigenvalues.
        """
        chunk_size = int(chunk_size)
        total_voxels = len(hessian_matrices)

        eigenvalues_list = []

        if chunk_size is None:  # chunk size is entire vector
            chunk_size = total_voxels

        # iterate over chunks
        # todo make chunk size dynamic based on available memory
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
        """
        Applies the Frangi filter to the Hessian eigenvalues to detect vessel-like structures.

        Parameters
        ----------
        eigenvalues : np.ndarray
            Eigenvalues of the Hessian matrix.
        gamma_sq : float
            Squared gamma value for vesselness calculation.

        Returns
        -------
        np.ndarray
            Filtered vesselness image.
        """
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
        """
        Applies the Laplacian of Gaussian (LoG) filter to enhance vessel structures.

        Parameters
        ----------
        frame : np.ndarray
            Input image frame to filter.
        mask : np.ndarray
            Mask to apply during filtering.

        Returns
        -------
        np.ndarray
            Filtered image.
        """
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
        """
        Runs the Frangi filter for a single timepoint in the image.

        Parameters
        ----------
        t : int
            Timepoint index.
        mask : bool, optional
            Whether to apply masking during processing (default is True).

        Returns
        -------
        np.ndarray
            Vesselness-enhanced image for the given timepoint.
        """
        logger.info(f'Running frangi filter on {t=}.')
        vesselness = xp.zeros_like(self.im_memmap[t, ...], dtype='float64')
        temp = xp.zeros_like(self.im_memmap[t, ...], dtype='float64')
        masks = xp.ones_like(self.im_memmap[t, ...], dtype='bool')
        for sigma_num, sigma in enumerate(self.sigmas):
            gauss_volume = self._gauss_filter(sigma, t)  # * xp.mean(sigma) ** 2

            gamma = self._calculate_gamma(gauss_volume)
            gamma_sq = 2 * gamma ** 2

            h_mask, hessian_matrices = self._compute_hessian(gauss_volume, mask=mask)
            if len(hessian_matrices) == 0:
                continue
            eigenvalues = self._compute_chunkwise_eigenvalues(hessian_matrices.astype('float'))

            temp[h_mask] = self._filter_hessian(eigenvalues, gamma_sq=gamma_sq)

            max_indices = temp > vesselness
            vesselness[max_indices] = temp[max_indices]
            masks = xp.where(~h_mask, 0, masks)

        vesselness = vesselness * masks
        return vesselness

    def _mask_volume(self, frangi_frame):
        """
        Creates a binary mask of vessel-like structures in the image based on a threshold.

        Parameters
        ----------
        frangi_frame : np.ndarray
            Image processed by the Frangi filter.

        Returns
        -------
        np.ndarray
            Masked image.
        """
        frangi_threshold = xp.percentile(frangi_frame[frangi_frame > 0], 1)
        frangi_mask = frangi_frame > frangi_threshold
        frangi_mask = ndi.binary_opening(frangi_mask)
        frangi_frame = frangi_frame * frangi_mask
        return frangi_frame

    def _remove_edges(self, frangi_frame):
        """
        Removes edges from the detected structures in the image.

        Parameters
        ----------
        frangi_frame : np.ndarray
            The Frangi-filtered image.

        Returns
        -------
        np.ndarray
            Image with edges removed.
        """
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
        """
        Runs the Frangi filter over all timepoints in the image.

        Parameters
        ----------
        mask : bool, optional
            Whether to apply masking during processing (default is True).
        """
        for t in range(self.num_t):
            if self.viewer is not None:
                self.viewer.status = f'Preprocessing. Frame: {t + 1} of {self.num_t}.'
            frangi_frame = self._run_frame(t, mask=mask)
            if not xp.sum(frangi_frame):
                frangi_frame = self._mask_volume(frangi_frame)
            filtered_im = frangi_frame

            if device_type == 'cuda':
                filtered_im = filtered_im.get()

            if self.im_info.no_t or self.num_t == 1:
                self.frangi_memmap[:] = filtered_im[:]
            else:
                self.frangi_memmap[t, ...] = filtered_im
            self.frangi_memmap.flush()

    def run(self, mask=True):
        """
        Main method to execute the Frangi filter process over the image data.

        Parameters
        ----------
        mask : bool, optional
            Whether to apply masking during processing (default is True).
        """
        logger.info('Running frangi filter.')
        self._get_t()
        self._allocate_memory()
        self._set_default_sigmas()
        self._run_filter(mask=mask)


if __name__ == "__main__":
    im_path = r"F:\2024_06_26_SD_ExM_nhs_u2OS_488+578_cropped.tif"
    im_info = ImInfo(im_path, dim_res={'T': 1, 'Z': 0.2, 'Y': 0.1, 'X': 0.1}, dimension_order='ZYX')
    filter_im = Filter(im_info)
    filter_im.run()
