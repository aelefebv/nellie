from src import xp, ndi, logger
from src_2.io.im_info import ImInfo
from src_2.utils.general import get_reshaped_image
import skimage.morphology as morph
import numpy as np
import scipy.ndimage

from src_2.utils.gpu_functions import triangle_threshold


class Network:
    def __init__(self, im_info: ImInfo, num_t=None,
                 min_radius_um=0.20, max_radius_um=1):
        self.im_info = im_info
        self.num_t = num_t
        if num_t is None:
            self.num_t = im_info.shape[im_info.axes.index('T')]
        self.z_ratio = self.im_info.dim_sizes['Z'] / self.im_info.dim_sizes['X']
        # either (roughly) diffraction limit, or pixel size, whichever is larger
        self.min_radius_um = max(min_radius_um, self.im_info.dim_sizes['X'])
        self.max_radius_um = max_radius_um

        self.min_radius_px = self.min_radius_um / self.im_info.dim_sizes['X']
        self.max_radius_px = self.max_radius_um / self.im_info.dim_sizes['X']

        self.shape = ()

        self.im_memmap = None
        self.label_memmap = None
        self.network_memmap = None

        self.debug = None

    def _remove_connected_label_pixels(self, skel_labels):
        depth, height, width = skel_labels.shape

        true_coords = np.argwhere(skel_labels>0)

        pixels_to_delete = []
        for z, y, x in true_coords:
            if z == 0 or z == depth - 1 or y == 0 or y == height - 1 or x == 0 or x == width - 1:
                continue  # Skip boundary voxels

            # Extract 3x3x3 neighborhood
            label_neighborhood = skel_labels[z - 1:z + 2, y - 1:y + 2, x - 1:x + 2]

            # Get labels of set voxels in the neighborhood
            labels_in_neighborhood = label_neighborhood[label_neighborhood > 0]

            if len(set(labels_in_neighborhood)) > 1:
                pixels_to_delete.append((z, y, x))

        for z, y, x in pixels_to_delete:
            skel_labels[z, y, x] = 0

        return skel_labels

    def _add_missing_skeleton_labels(self, skel_frame, label_frame):
        logger.debug('Adding missing skeleton labels.')

        # identify unique labels and find missing ones
        unique_labels = np.unique(label_frame)
        unique_skel_labels = np.unique(skel_frame)

        missing_labels = set(unique_labels) - set(unique_skel_labels)

        # for each missing label, find the centroid and mark it in the skeleton
        for label in missing_labels:
            if label == 0:  # ignore bg label
                continue

            label_coords = np.argwhere(label_frame == label)
            centroid = label_coords.mean(axis=0).astype(int)

            skel_frame[tuple(centroid)] = label

        return skel_frame

    def _skeletonize(self, frame):
        cpu_frame = np.array(frame)
        # gpu_frame = xp.array(frame)
        # test = self._remove_connected_label_pixels(cpu_frame)

        skel = morph.skeletonize(cpu_frame > 0).astype('bool')
        # skel = morph.skeletonize(test > 0).astype('bool')

        skel_labels = skel * cpu_frame
        skel_labels = self._remove_connected_label_pixels(skel_labels)

        return skel_labels

    def _get_sigma_vec(self, sigma):
        if self.im_info.no_z:
            sigma_vec = (sigma, sigma)
        else:
            sigma_vec = (sigma / self.z_ratio, sigma, sigma)
        return sigma_vec

    def _set_default_sigmas(self):
        logger.debug('Setting to sigma values.')
        min_sigma_step_size = 0.2
        num_sigma = 5

        self.sigma_min = self.min_radius_px / 2
        self.sigma_max = self.max_radius_px / 3

        sigma_step_size_calculated = (self.sigma_max - self.sigma_min) / num_sigma
        sigma_step_size = max(min_sigma_step_size, sigma_step_size_calculated)  # Avoid taking too small of steps.

        self.sigmas = list(xp.arange(self.sigma_min, self.sigma_max, sigma_step_size))
        logger.debug(f'Calculated sigma step size = {sigma_step_size_calculated}. Sigmas = {self.sigmas}')

    def _local_max_peak(self, frame, mask):
        lapofg = xp.empty(((len(self.sigmas),) + frame.shape), dtype=float)
        for i, s in enumerate(self.sigmas):
            sigma_vec = self._get_sigma_vec(s)
            current_lapofg = -ndi.gaussian_laplace(frame, sigma_vec) * xp.mean(s) ** 2
            current_lapofg = current_lapofg * mask
            current_lapofg[current_lapofg < 0] = 0
            lapofg[i] = current_lapofg

        filt_footprint = xp.ones((3,) * (frame.ndim + 1))
        max_filt = ndi.maximum_filter(lapofg, footprint=filt_footprint, mode='nearest')
        peaks = xp.empty(lapofg.shape, dtype=bool)
        for filt_slice, max_filt_slice in enumerate(max_filt):
            thresh = triangle_threshold(max_filt_slice[max_filt_slice > 0]) / 2
            max_filt_mask = xp.asarray(max_filt_slice > thresh) * mask
            peaks[filt_slice] = (xp.asarray(lapofg[filt_slice]) == xp.asarray(max_filt_slice)) * max_filt_mask
        # get the coordinates of all true pixels in peaks
        coords = xp.max(peaks, axis=0)
        coords_3d = xp.argwhere(coords)

    def _get_t(self):
        if self.num_t is None:
            if self.im_info.no_t:
                self.num_t = 1
            else:
                self.num_t = self.im_info.shape[self.im_info.axes.index('T')]
        else:
            return

    def _allocate_memory(self):
        logger.debug('Allocating memory for semantic segmentation.')
        label_memmap = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_instance_label'])
        self.label_memmap = get_reshaped_image(label_memmap, self.num_t, self.im_info)

        im_memmap = self.im_info.get_im_memmap(self.im_info.im_path)
        self.im_memmap = get_reshaped_image(im_memmap, self.num_t, self.im_info)

        im_frangi_memmap = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_frangi'])
        self.im_frangi_memmap = get_reshaped_image(im_frangi_memmap, self.num_t, self.im_info)
        self.shape = self.label_memmap.shape

        im_skel_path = self.im_info.create_output_path('im_skel')
        self.skel_memmap = self.im_info.allocate_memory(im_skel_path, shape=self.shape,
                                                                  dtype='uint16',
                                                                  description='skeleton image',
                                                                  return_memmap=True)

    def _run_frame(self, t):
        logger.info(f'Running network analysis, volume {t}/{self.num_t - 1}')
        label_frame = self.label_memmap[t]
        skel_frame = self._skeletonize(label_frame)
        skel = self._add_missing_skeleton_labels(skel_frame, label_frame)
        return skel

    def _run_networking(self):
        for t in range(self.num_t):
            skel = self._run_frame(t)
            self.skel_memmap[t] = skel
            # intensity_frame = xp.asarray(self.im_frangi_memmap[t])
            # intensity_frame = xp.asarray(self.im_memmap[t])
            # peaks = self._local_max_peak(intensity_frame, xp.asarray(label_frame > 0))
            # self.network_memmap[t] = frame
            # self.debug = frame
            # break

    def run(self):
        self._get_t()
        self._allocate_memory()
        self._set_default_sigmas()
        self._run_networking()


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
        im_infos.append(im_info)

    skeletonis = []
    for im_info in im_infos:
        skel = Network(im_info, num_t=2)
        skel.run()
        skeletonis.append(skel)

    # # check if viewer exists as a variable
    # if 'viewer' not in locals():
    #     import napari
    #     viewer = napari.Viewer()
    # # viewer.add_points(skeletonis[0].debug.get(), name='debug', size=1, face_color='red')
    # # viewer.add_points(skeletonis[1].debug.get(), name='debug', size=1, face_color='red')
    # viewer.add_image(skeletonis[0].debug.get(), name='im')
    # viewer.add_image(skeletonis[1].debug.get(), name='im')
