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

    def _calculate_normalized_moments(self, images):
        # I know the broadcasting is super confusing but it makes it so much faster...

        # Assuming images is a 3D numpy array of shape (num_images, height, width)
        num_images, height, width = images.shape
        extended_images = images[:, :, :, None, None]  # shape (num_images, height, width, 1, 1)

        # Pre-compute meshgrid
        x, y = xp.meshgrid(xp.arange(width), xp.arange(height))

        # Reshape for broadcasting
        x = x[None, :, :, None, None]  # shape (1, height, width, 1, 1)
        y = y[None, :, :, None, None]  # shape (1, height, width, 1, 1)

        # Raw Moments
        M = xp.sum(extended_images * (x ** xp.arange(4)[None, None, None, :, None]) *
                   (y ** xp.arange(4)[None, None, None, None, :]), axis=(1, 2))  # shape (num_images, 4, 4)

        # Central Moments; compute x_bar and y_bar
        x_bar = M[:, 1, 0] / M[:, 0, 0]  # shape (num_images,)
        y_bar = M[:, 0, 1] / M[:, 0, 0]  # shape (num_images,)

        x_bar = x_bar[:, None, None, None, None]  # shape (num_images, 1, 1, 1, 1)
        y_bar = y_bar[:, None, None, None, None]  # shape (num_images, 1, 1, 1, 1)

        # Calculate mu using broadcasting
        mu = xp.sum(extended_images * (x - x_bar) ** xp.arange(4)[None, None, None, :, None] *
                    (y - y_bar) ** xp.arange(4)[None, None, None, None, :], axis=(1, 2))  # shape (num_images, 4, 4)

        # Normalized moments
        i_plus_j = xp.arange(4)[:, None] + xp.arange(4)[None, :]
        eta = mu / (M[:, 0, 0][:, None, None] ** ((i_plus_j[None, :, :] + 2) / 2))

        return eta

    def _calculate_hu_moments(self, eta):
        num_images = eta.shape[0]
        hu = xp.zeros((num_images, 5))  # initialize Hu moments for each image

        hu[:, 0] = eta[:, 2, 0] + eta[:, 0, 2]
        hu[:, 1] = (eta[:, 2, 0] - eta[:, 0, 2]) ** 2 + 4 * eta[:, 1, 1] ** 2
        hu[:, 2] = (eta[:, 3, 0] - 3 * eta[:, 1, 2]) ** 2 + (3 * eta[:, 2, 1] - eta[:, 0, 3]) ** 2
        hu[:, 3] = (eta[:, 3, 0] + eta[:, 1, 2]) ** 2 + (eta[:, 2, 1] + eta[:, 0, 3]) ** 2
        hu[:, 4] = (eta[:, 3, 0] - 3 * eta[:, 1, 2]) * (eta[:, 3, 0] + eta[:, 1, 2]) * \
                   ((eta[:, 3, 0] + eta[:, 1, 2]) ** 2 - 3 * (eta[:, 2, 1] + eta[:, 0, 3]) ** 2) + \
                   (3 * eta[:, 2, 1] - eta[:, 0, 3]) * (eta[:, 2, 1] + eta[:, 0, 3]) * \
                   (3 * (eta[:, 3, 0] + eta[:, 1, 2]) ** 2 - (eta[:, 2, 1] + eta[:, 0, 3]) ** 2)

        return hu  # return the first 5 Hu moments for each image

    def _get_im_bounds(self, markers, distance_frame):
        radii = distance_frame[markers[:, 0], markers[:, 1], markers[:, 2]]
        marker_radii = xp.ceil(radii)
        z_low = xp.clip(markers[:, 0] - marker_radii, 0, self.shape[1])
        z_high = xp.clip(markers[:, 0] + (marker_radii + 1), 0, self.shape[1])
        y_low = xp.clip(markers[:, 1] - marker_radii, 0, self.shape[2])
        y_high = xp.clip(markers[:, 1] + (marker_radii + 1), 0, self.shape[2])
        x_low = xp.clip(markers[:, 2] - marker_radii, 0, self.shape[3])
        x_high = xp.clip(markers[:, 2] + (marker_radii + 1), 0, self.shape[3])
        return z_low, z_high, y_low, y_high, x_low, x_high

    def _get_orthogonal_projections(self, im_frame, im_bounds, max_radius):
        z_low, z_high, y_low, y_high, x_low, x_high = im_bounds
        # z_projs = []
        # y_projs = []
        # x_projs = []

        z_projections = xp.ones((len(z_low), max_radius, max_radius))# * xp.nan
        y_projections = xp.ones((len(y_low), max_radius, max_radius))# * xp.nan
        x_projections = xp.ones((len(x_low), max_radius, max_radius))# * xp.nan

        for i in range(len(x_low)):
            zl, zh, yl, yh, xl, xh = z_low[i], z_high[i], y_low[i], y_high[i], x_low[i], x_high[i]
            test_region = im_frame[zl:zh, yl:yh, xl:xh]

            # max projections along each axis
            z_projections[i, :yh - yl, :xh - xl] = xp.max(test_region, axis=0)
            y_projections[i, :zh - zl, :xh - xl] = xp.max(test_region, axis=1)
            x_projections[i, :zh - zl, :yh - yl] = xp.max(test_region, axis=2)

            # z_projs.append(z_proj)
            # y_projs.append(y_proj)
            # x_projs.append(x_proj)

        # proj_stacks = [xp.array(z_proj, y_proj, x_proj) for z_proj, y_proj, x_proj in zip(z_projs, y_projs, x_projs)]
        # proj_stacks = xp.array(proj_stacks)
        return z_projections, y_projections, x_projections
        # return z_projs, y_projs, x_projs

    def _get_hu_moments(self, projections):
        z_moments = self._calculate_normalized_moments(projections[0])
        y_moments = self._calculate_normalized_moments(projections[1])
        x_moments = self._calculate_normalized_moments(projections[2])

        z_hu = self._calculate_hu_moments(z_moments)
        y_hu = self._calculate_hu_moments(y_moments)
        x_hu = self._calculate_hu_moments(x_moments)

        return xp.concatenate((z_hu, y_hu, x_hu))

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
        region_bounds = self._get_im_bounds(marker_indices, distance_frame)
        max_radius = int(xp.ceil(xp.max(distance_frame[marker_frame])))*2+1
        projections = self._get_orthogonal_projections(intensity_frame, region_bounds, max_radius)
        logger.debug(f'Calculating Hu moments')
        etas_z = self._calculate_normalized_moments(projections[0])
        etas_y = self._calculate_normalized_moments(projections[1])
        etas_x = self._calculate_normalized_moments(projections[2])
        logger.debug('done')
        hu_moments_z = self._calculate_hu_moments(etas_z)
        hu_moments_y = self._calculate_hu_moments(etas_y)
        hu_moments_x = self._calculate_hu_moments(etas_x)
        # z, y, and x at the same index should be concatenated into one array of hu_moments for each index
        hu_moments = xp.concatenate((hu_moments_z, hu_moments_y, hu_moments_x), axis=1)


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
    for im_info in im_infos[-1:]:
        hu = HuMomentTracking(im_info, num_t=2)
        hu.run()
        hu_files.append(hu)