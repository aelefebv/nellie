import numpy as np
from scipy.spatial import cKDTree, distance

from nellie import xp, ndi, logger, device_type
from nellie.im_info.im_info import ImInfo
from nellie.utils.general import get_reshaped_image


class Markers:
    def __init__(self, im_info: ImInfo, num_t=None,
                 min_radius_um=0.20, max_radius_um=1, use_im='distance', num_sigma=5):
        self.im_info = im_info
        self.num_t = num_t
        if self.im_info.no_t:
            self.num_t = 1
        elif num_t is None:  # and not self.im_info.no_t:
            self.num_t = im_info.shape[im_info.axes.index('T')]
        if not self.im_info.no_z:
            self.z_ratio = self.im_info.dim_sizes['Z'] / self.im_info.dim_sizes['X']
        self.min_radius_um = max(min_radius_um, self.im_info.dim_sizes['X'])
        self.max_radius_um = max_radius_um

        self.min_radius_px = self.min_radius_um / self.im_info.dim_sizes['X']
        self.max_radius_px = self.max_radius_um / self.im_info.dim_sizes['X']
        self.use_im = use_im
        self.num_sigma = num_sigma

        self.shape = ()

        self.im_memmap = None
        self.im_frangi_memmap = None
        self.label_memmap = None
        self.im_marker_memmap = None
        self.im_distance_memmap = None
        self.im_border_memmap = None

        self.debug = None

    def _get_sigma_vec(self, sigma):
        if self.im_info.no_z:
            sigma_vec = (sigma, sigma)
        else:
            sigma_vec = (sigma / self.z_ratio, sigma, sigma)
        return sigma_vec

    def _set_default_sigmas(self):
        logger.debug('Setting sigma values.')
        min_sigma_step_size = 0.2

        self.sigma_min = self.min_radius_px / 2
        self.sigma_max = self.max_radius_px / 3

        sigma_step_size_calculated = (self.sigma_max - self.sigma_min) / self.num_sigma
        sigma_step_size = max(min_sigma_step_size, sigma_step_size_calculated)  # Avoid taking too small of steps.

        self.sigmas = list(xp.arange(self.sigma_min, self.sigma_max, sigma_step_size))
        logger.debug(f'Calculated sigma step size = {sigma_step_size_calculated}. Sigmas = {self.sigmas}')

    def _get_t(self):
        if self.num_t is None:
            if self.im_info.no_t:
                self.num_t = 1
            else:
                self.num_t = self.im_info.shape[self.im_info.axes.index('T')]
        else:
            return

    def _allocate_memory(self):
        logger.debug('Allocating memory for mocap marking.')

        label_memmap = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_instance_label'])
        self.label_memmap = get_reshaped_image(label_memmap, self.num_t, self.im_info)

        im_memmap = self.im_info.get_im_memmap(self.im_info.im_path)
        self.im_memmap = get_reshaped_image(im_memmap, self.num_t, self.im_info)

        im_frangi_memmap = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_frangi'])
        self.im_frangi_memmap = get_reshaped_image(im_frangi_memmap, self.num_t, self.im_info)
        self.shape = self.label_memmap.shape

        im_marker_path = self.im_info.pipeline_paths['im_marker']
        self.im_marker_memmap = self.im_info.allocate_memory(im_marker_path, shape=self.shape,
                                                             dtype='uint8',
                                                             description='mocap marker image',
                                                             return_memmap=True)

        im_distance_path = self.im_info.pipeline_paths['im_distance']
        self.im_distance_memmap = self.im_info.allocate_memory(im_distance_path, shape=self.shape,
                                                               dtype='float',
                                                               description='distance transform image',
                                                               return_memmap=True)

        im_border_path = self.im_info.pipeline_paths['im_border']
        self.im_border_memmap = self.im_info.allocate_memory(im_border_path, shape=self.shape,
                                                             dtype='uint8',
                                                             description='border image',
                                                             return_memmap=True)

    def _distance_im(self, mask):
        border_mask = ndi.binary_dilation(mask, iterations=1) ^ mask

        if device_type == 'cuda':
            mask_coords = xp.argwhere(mask).get()
            border_mask_coords = xp.argwhere(border_mask).get()
        else:
            mask_coords = xp.argwhere(mask)
            border_mask_coords = xp.argwhere(border_mask)

        border_tree = cKDTree(border_mask_coords)
        dist, _ = border_tree.query(mask_coords, k=1, distance_upper_bound=self.max_radius_px * 2)
        distances_im_frame = xp.zeros_like(mask, dtype='float32')
        if self.im_info.no_z:
            distances_im_frame[mask_coords[:, 0], mask_coords[:, 1]] = dist
        else:
            distances_im_frame[mask_coords[:, 0], mask_coords[:, 1], mask_coords[:, 2]] = dist
        # any inf pixels get set to upper bound
        distances_im_frame[distances_im_frame == xp.inf] = self.max_radius_px * 2
        return distances_im_frame, border_mask

    def _remove_close_peaks(self, coord, check_im):
        check_im_max = ndi.maximum_filter(check_im, size=3, mode='nearest')
        if not self.im_info.no_z:
            intensities = check_im_max[coord[:, 0], coord[:, 1], coord[:, 2]]
        else:
            intensities = check_im_max[coord[:, 0], coord[:, 1]]

        # sort to remove peaks that are too close by keeping the brightest peak
        idx_maxsort = np.argsort(-intensities)

        if device_type == 'cuda':
            coord_sorted = coord[idx_maxsort].get()
        else:
            coord_sorted = coord[idx_maxsort]

        tree = cKDTree(coord_sorted)
        min_dist = 2
        indices = tree.query_ball_point(coord_sorted, r=min_dist, p=2, workers=-1)
        rejected_peaks_indices = set()
        naccepted = 0
        for idx, candidates in enumerate(indices):
            if idx not in rejected_peaks_indices:
                # keep current point and the points at exactly spacing from it
                candidates.remove(idx)
                dist = distance.cdist([coord_sorted[idx]],
                                      coord_sorted[candidates],
                                      distance.minkowski,
                                      p=2).reshape(-1)
                candidates = [c for c, d in zip(candidates, dist)
                              if d < min_dist]

                rejected_peaks_indices.update(candidates)
                naccepted += 1

        cleaned_coords = np.delete(coord_sorted, tuple(rejected_peaks_indices), axis=0)

        return cleaned_coords

    def _local_max_peak(self, use_im, mask, distance_im):
        lapofg = xp.empty(((len(self.sigmas),) + use_im.shape), dtype=float)
        for i, s in enumerate(self.sigmas):
            sigma_vec = self._get_sigma_vec(s)
            current_lapofg = -ndi.gaussian_laplace(use_im, sigma_vec) * xp.mean(s) ** 2
            current_lapofg[current_lapofg < 0] = 0
            lapofg[i] = current_lapofg

        filt_footprint = xp.ones((3,) * (use_im.ndim + 1))
        max_filt = ndi.maximum_filter(lapofg, footprint=filt_footprint, mode='nearest')
        peaks = xp.empty(lapofg.shape, dtype=bool)
        for filt_slice, max_filt_slice in enumerate(max_filt):
            peaks[filt_slice] = (xp.asarray(lapofg[filt_slice]) == xp.asarray(max_filt_slice))  # * max_filt_mask
        distance_mask = distance_im > 0
        peaks = peaks * mask * distance_mask
        # get the coordinates of all true pixels in peaks
        coords = xp.max(peaks, axis=0)
        coords_idx = xp.argwhere(coords)
        return coords_idx

    def _run_frame(self, t):
        logger.info(f'Running motion capture marking, volume {t}/{self.num_t - 1}')
        intensity_frame = xp.asarray(self.im_memmap[t])
        mask_frame = xp.asarray(self.label_memmap[t] > 0)
        distance_im, border_mask = self._distance_im(mask_frame)
        if self.use_im == 'distance':
            peak_coords = self._local_max_peak(distance_im, mask_frame, distance_im)
        elif self.use_im == 'frangi':
            peak_coords = self._local_max_peak(xp.asarray(self.im_frangi_memmap[t]), mask_frame, distance_im)
        peak_coords = self._remove_close_peaks(peak_coords, intensity_frame)
        peak_im = xp.zeros_like(mask_frame)
        peak_im[tuple(peak_coords.T)] = 1
        if device_type == "cuda":
            return peak_im.get(), distance_im.get(), border_mask.get()
        else:
            return peak_im, distance_im, border_mask

    def _run_mocap_marking(self):
        for t in range(self.num_t):
            marker_frame = self._run_frame(t)
            if self.im_marker_memmap.shape != self.shape and self.im_info.no_t:
                self.im_marker_memmap[:], self.im_distance_memmap[:], self.im_border_memmap[:] = marker_frame
            else:
                self.im_marker_memmap[t], self.im_distance_memmap[t], self.im_border_memmap[t] = marker_frame
            self.im_marker_memmap.flush()
            self.im_distance_memmap.flush()
            self.im_border_memmap.flush()

    def run(self):
        # if self.im_info.no_t:
        #     return
        self._get_t()
        self._allocate_memory()
        self._set_default_sigmas()
        self._run_mocap_marking()


if __name__ == "__main__":
    im_path = r"D:\test_files\nelly_smorgasbord\deskewed-iono_pre.ome.tif"
    im_info = ImInfo(im_path)
    num_t = 3
    markers = Markers(im_info, num_t=num_t)
    markers.run()
