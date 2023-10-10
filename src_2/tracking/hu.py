from src_2.io.im_info import ImInfo
from src import xp, ndi, logger
from src_2.utils.general import get_reshaped_image
import numpy as np


class HuMomentTracking:
    def __init__(self, im_info: ImInfo, num_t=None,
                 min_radius_um=0.20, max_radius_um=1):
        self.im_info = im_info
        self.num_t = num_t
        if num_t is None:
            self.num_t = im_info.shape[im_info.axes.index('T')]
        self.z_ratio = self.im_info.dim_sizes['Z'] / self.im_info.dim_sizes['X']

        self.shape = ()

        self.im_memmap = None
        self.im_frangi_memmap = None
        self.im_distance_memmap = None
        self.im_marker_memmap = None

        self.debug = None

    def _calculate_normalized_moments(self, image):
        # raw moments
        x, y = xp.meshgrid(xp.arange(image.shape[1]), xp.arange(image.shape[0]))
        M = {}
        for i in range(4):
            for j in range(4):
                M[str(i) + str(j)] = xp.sum(x ** i * y ** j * image)

        # central moments
        x, y = xp.meshgrid(xp.arange(image.shape[1]), xp.arange(image.shape[0]))
        x_bar = M['10'] / M['00']
        y_bar = M['01'] / M['00']
        mu = {}
        for i in range(4):
            for j in range(4):
                mu[str(i) + str(j)] = xp.sum((x - x_bar) ** i * (y - y_bar) ** j * image)

        # normalized moments
        eta = {}
        for i in range(4):
            for j in range(4):
                eta[str(i) + str(j)] = mu[str(i) + str(j)] / (M['00'] ** ((i + j + 2) / 2))
        return eta

    def _calculate_hu_moments(self, eta):
        eta_20, eta_02, eta_11 = eta['20'], eta['02'], eta['11']
        eta_30, eta_12, eta_21, eta_03 = eta['30'], eta['12'], eta['21'], eta['03']

        hu = np.zeros(7)
        hu[0] = eta_20 + eta_02
        hu[1] = (eta_20 - eta_02) ** 2 + 4 * eta_11 ** 2
        hu[2] = (eta_30 - 3 * eta_12) ** 2 + (3 * eta_21 - eta_03) ** 2
        hu[3] = (eta_30 + eta_12) ** 2 + (eta_21 + eta_03) ** 2
        hu[4] = (eta_30 - 3 * eta_12) * (eta_30 + eta_12) * ((eta_30 + eta_12) ** 2 - 3 * (eta_21 + eta_03) ** 2) + \
                (3 * eta_21 - eta_03) * (eta_21 + eta_03) * (3 * (eta_30 + eta_12) ** 2 - (eta_21 + eta_03) ** 2)
        hu[5] = (eta_20 - eta_02) * ((eta_30 + eta_12) ** 2 - (eta_21 + eta_03) ** 2) + \
                4 * eta_11 * (eta_30 + eta_12) * (eta_21 + eta_03)
        return hu[:5]  # Return the first 5 Hu moments

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

        im_marker_memmap = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_marker'])
        self.im_marker_memmap = get_reshaped_image(im_marker_memmap, self.num_t, self.im_info)

        im_distance_memmap = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_distance'])
        self.im_distance_memmap = get_reshaped_image(im_distance_memmap, self.num_t, self.im_info)

    def _run_frame(self, t):
        intensity_frame = xp.array(self.im_memmap[t])
        distance_frame = ndi.maximum_filter(xp.array(self.im_distance_memmap[t]), size=3)
        marker_frame = xp.array(self.im_marker_memmap[t]) > 0
        marker_indices = xp.argwhere(marker_frame)
        # all_radii = distance_frame[marker_frame]

        test_marker = marker_indices[50]
        test_marker_radius = int(xp.ceil(distance_frame[test_marker[0], test_marker[1], test_marker[2]]))
        # NxNxN region of intensity im
        z_low = max(0, test_marker[0] - test_marker_radius)
        z_high = min(self.shape[1], test_marker[0] + test_marker_radius)
        y_low = max(0, test_marker[1] - test_marker_radius)
        y_high = min(self.shape[2], test_marker[1] + test_marker_radius)
        x_low = max(0, test_marker[2] - test_marker_radius)
        x_high = min(self.shape[3], test_marker[2] + test_marker_radius)

        test_region = intensity_frame[z_low:z_high, y_low:y_high, x_low:x_high]
        z_proj = xp.max(test_region, axis=0)
        y_proj = xp.max(test_region, axis=1)
        x_proj = xp.max(test_region, axis=2)

        z_moments = self._calculate_normalized_moments(z_proj)
        y_moments = self._calculate_normalized_moments(y_proj)
        x_moments = self._calculate_normalized_moments(x_proj)

        z_hu = self._calculate_hu_moments(z_moments)
        y_hu = self._calculate_hu_moments(y_moments)
        x_hu = self._calculate_hu_moments(x_moments)

        #rotate x 90 degress
        x_rotated = xp.rot90(x_proj, k=0, axes=(0, 1))
        x_rot_moments = self._calculate_normalized_moments(x_rotated)
        x_rot_hu = self._calculate_hu_moments(x_rot_moments)
        x_zoom2 = ndi.zoom(x_proj, zoom=2)
        x_zoom2_moments = self._calculate_normalized_moments(x_zoom2)
        x_zoom2_hu = self._calculate_hu_moments(x_zoom2_moments)

    def _run_pso(self):
        for t in range(self.num_t):
            logger.debug(f'Running hu-moment tracking for frame {t + 1} of {self.num_t}')
            self._run_frame(t)

    def run(self):
        self._get_t()
        self._allocate_memory()
        self._run_pso()


if __name__ == "__main__":
    import os
    test_folder = r"D:\test_files\nelly_tests"
    all_files = os.listdir(test_folder)
    all_files = [file for file in all_files if not os.path.isdir(os.path.join(test_folder, file))]
    im_infos = []
    for file in all_files:
        im_path = os.path.join(test_folder, file)
        im_info = ImInfo(im_path)
        im_info.create_output_path('im_instance_label')
        im_info.create_output_path('im_frangi')
        im_info.create_output_path('im_marker')
        im_info.create_output_path('im_distance')
        im_infos.append(im_info)

    hu_files = []
    for im_info in im_infos:
        hu = HuMomentTracking(im_info, num_t=2)
        hu.run()
        hu_files.append(hu)