from src_2.im_info.im_info import ImInfo
from src import xp, ndi, logger
from src_2.utils.general import get_reshaped_image
from src_2.utils.gpu_functions import otsu_threshold, triangle_threshold


class Label:
    def __init__(self, im_info: ImInfo,
                 num_t=None,
                 threshold: float = 0,
                 # min_radius_um: float = 0.2,
                 # max_radius_um=xp.inf,
                 snr_cleaning=False):
        self.im_info = im_info
        self.num_t = num_t
        if num_t is None and not self.im_info.no_t:
            self.num_t = im_info.shape[im_info.axes.index('T')]
        self.threshold = threshold
        # self.min_radius_um = max(min_radius_um, self.im_info.dim_sizes['X'])
        # self.max_radius_um = max_radius_um
        self.snr_cleaning = snr_cleaning

        self.im_memmap = None
        self.frangi_memmap = None

        self.min_size_threshold_px = 0
        self.max_size_threshold_px = xp.inf

        self.remove_in_2d = False

        self.max_label_num = 0
        # if any(xp.array(
        #         [self.im_info.dim_sizes['Z'], self.im_info.dim_sizes['Y'], self.im_info.dim_sizes['X']]
        # ) > (self.min_radius_um*2)):
        #     logger.warning(f"One of the dimensions' voxel sizes is greater than the minimum radius of the structure in "
        #                    f"question so object removal will be conducted based on 2D parameters instead of 3D. "
        #                    f"This may result in objects being kept that should not be.")
        #     self.remove_in_2d = True
        #
        # # convert min radius um to a min area / volume
        # if not self.im_info.no_z and not self.remove_in_2d:
        #     # volume of sphere of radius min_width/2 in pixels cubed
        #     self.min_size_threshold_px = (4 / 3 * xp.pi * (min_radius_um / 2) ** 2) / (
        #             self.im_info.dim_sizes['X'] ** 2 * self.im_info.dim_sizes['Z']
        #     )
        #     # min area of 3*3*3 pixels
        #     self.min_size_threshold_px = max(self.min_size_threshold_px, 27)
        #     self.max_size_threshold_px = (4 / 3 * xp.pi * (max_radius_um / 2) ** 2) / (
        #             self.im_info.dim_sizes['X'] ** 2 * self.im_info.dim_sizes['Z']
        #     )
        # else:
        #     self.min_size_threshold_px = (xp.pi * (min_radius_um / 2) ** 2) / (self.im_info.dim_sizes['X'] ** 2)
        #     # min area of 3*3 pixels
        #     self.min_size_threshold_px = max(self.min_size_threshold_px, 9)
        #     self.max_size_threshold_px = (xp.pi * (max_radius_um / 2) ** 2) / (self.im_info.dim_sizes['X'] ** 2)

        self.semantic_mask_memmap = None
        self.instance_label_memmap = None
        self.shape = ()

        self.debug = {}

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
        im_memmap = self.im_info.get_im_memmap(self.im_info.im_path)
        self.im_memmap = get_reshaped_image(im_memmap, self.num_t, self.im_info)

        frangi_memmap = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_frangi'])
        self.frangi_memmap = get_reshaped_image(frangi_memmap, self.num_t, self.im_info)
        self.shape = self.frangi_memmap.shape

        # im_semantic_mask_path = self.im_info.create_output_path('im_semantic_mask')
        # self.semantic_mask_memmap = self.im_info.allocate_memory(im_semantic_mask_path, shape=self.shape, dtype='int8',
        #                                                          description='semantic segmentation',
        #                                                          return_memmap=True)

        im_instance_label_path = self.im_info.create_output_path('im_instance_label')
        self.instance_label_memmap = self.im_info.allocate_memory(im_instance_label_path, shape=self.shape, dtype='int32',
                                                                  description='instance segmentation',
                                                                  return_memmap=True)

    def _get_labels(self, frame):
        ndim = 2 if self.remove_in_2d or self.im_info.no_z else 3
        footprint = ndi.generate_binary_structure(ndim, 1)

        # thresh, _ = otsu_threshold(xp.log10(frame[frame > 0]))
        # thresh = 10**thresh
        if self.im_info.no_z:
            thresh = 0
        else:
            thresh = 10**triangle_threshold(xp.log10(frame[frame > 0]))
        # print(thresh, triangle_thresh)

        mask = frame > thresh
        if not self.im_info.no_z:
            mask = ndi.binary_fill_holes(mask)
            structure = xp.ones((2, 2, 2))
            mask = ndi.binary_opening(mask, structure=structure)
        # else:
        #     structure = xp.ones((2, 2))

        labels, _ = ndi.label(mask, structure=footprint)
        return mask, labels

    def _remove_bad_sized_objects(self, labels):
        ndim = 2 if self.remove_in_2d or self.im_info.no_z else 3
        footprint = ndi.generate_binary_structure(ndim, 1)

        label_sizes = xp.bincount(labels.ravel())

        above_threshold = label_sizes > self.min_size_threshold_px
        below_threshold = label_sizes < self.max_size_threshold_px
        mask_sizes = above_threshold * below_threshold

        mask = xp.zeros_like(labels, dtype=bool)
        mask[mask_sizes[labels]] = True
        mask[labels == 0] = False

        labels, _ = ndi.label(mask, structure=footprint)

        return mask, labels

    # def _trim_labels(self, frangi_frame, labels):
    #     # todo this is slow af.
    #     # for each label, get the median intensity, set all pixels below it to 0
    #     gauss_frangi = ndi.gaussian_filter(frangi_frame, sigma=1)
    #     unique_labels = xp.unique(labels)
    #     trimmed_labels = labels.copy()
    #     for label in unique_labels:
    #         median_intensity = triangle_threshold(gauss_frangi[labels == label])
    #         trimmed_labels[(labels == label) & (gauss_frangi < median_intensity)] = 0
    #     trimmed_labels, _ = ndi.label(trimmed_labels > 0)
    #     return trimmed_labels

    def _get_subtraction_mask(self, original_frame, labels_frame):
        subtraction_mask = original_frame.copy()
        subtraction_mask[labels_frame > 0] = 0
        return subtraction_mask

    def _get_object_snrs(self, original_frame, labels_frame):
        logger.debug('Calculating object SNRs.')
        subtraction_mask = self._get_subtraction_mask(original_frame, labels_frame)
        unique_labels = xp.unique(labels_frame)
        extend_bbox_by = 1
        keep_labels = []
        for label in unique_labels:
            if label == 0:
                continue
            coords = xp.nonzero(labels_frame == label)
            z_coords, r_coords, c_coords = coords

            zmin, zmax = xp.min(z_coords), xp.max(z_coords)
            rmin, rmax = xp.min(r_coords), xp.max(r_coords)
            cmin, cmax = xp.min(c_coords), xp.max(c_coords)

            zmin, zmax = xp.clip(zmin - extend_bbox_by, 0, labels_frame.shape[0]), xp.clip(zmax + extend_bbox_by, 0,
                                                                                           labels_frame.shape[0])
            rmin, rmax = xp.clip(rmin - extend_bbox_by, 0, labels_frame.shape[1]), xp.clip(rmax + extend_bbox_by, 0,
                                                                                           labels_frame.shape[1])
            cmin, cmax = xp.clip(cmin - extend_bbox_by, 0, labels_frame.shape[2]), xp.clip(cmax + extend_bbox_by, 0,
                                                                                           labels_frame.shape[2])

            # only keep objects over 1 std from its surroundings
            local_intensity = subtraction_mask[zmin:zmax, rmin:rmax, cmin:cmax]
            local_intensity_mean = local_intensity[local_intensity > 0].mean()
            local_intensity_std = local_intensity[local_intensity > 0].std()
            label_intensity_mean = original_frame[coords].mean()
            intensity_cutoff = label_intensity_mean / (local_intensity_mean + local_intensity_std)
            if intensity_cutoff > 1:
                keep_labels.append(label)

        keep_labels = xp.asarray(keep_labels)
        labels_frame = xp.where(xp.isin(labels_frame, keep_labels), labels_frame, 0)
        return labels_frame


    def _run_frame(self, t):
        logger.info(f'Running semantic segmentation, volume {t}/{self.num_t - 1}')
        original_in_mem = xp.asarray(self.im_memmap[t, ...])
        frangi_in_mem = xp.asarray(self.frangi_memmap[t, ...])
        _, labels = self._get_labels(frangi_in_mem)
        # _, labels = self._remove_bad_sized_objects(labels)
        if self.snr_cleaning:
            labels = self._get_object_snrs(original_in_mem, labels)
        labels[labels>0] += self.max_label_num
        self.max_label_num = xp.max(labels)
        return labels

    def _run_segmentation(self):
        for t in range(self.num_t):
            labels = self._run_frame(t)
            if self.im_info.no_t:
                self.instance_label_memmap[:] = labels.get()[:]
            else:
                self.instance_label_memmap[t, ...] = labels.get()

    def run(self):
        logger.info('Running semantic segmentation.')
        self._get_t()
        self._allocate_memory()
        self._run_segmentation()


if __name__ == "__main__":
    im_path = r"D:\test_files\nelly_gav_tests\fibro_3.nd2"
    im_info = ImInfo(im_path)
    im_info.create_output_path('im_frangi')
    segment_unique = Label(im_info, num_t=2)
    segment_unique.run()

    # import os
    # test_folder = r"D:\test_files\nelly_tests"
    # # test_folder = r"D:\test_files\beading"
    # # test_folder = r"D:\test_files\julius_examples"
    # all_files = os.listdir(test_folder)
    # all_files = [file for file in all_files if not os.path.isdir(os.path.join(test_folder, file))]
    # im_infos = []
    # for file in all_files:
    #     im_path = os.path.join(test_folder, file)
    #     im_info = ImInfo(im_path)
    #     # im_info = ImInfo(im_path, dim_sizes={'T': 0, 'X': 0.11, 'Y': 0.11, 'Z': 0.1})
    #     im_info.create_output_path('im_frangi')
    #     im_infos.append(im_info)
    #
    # segmentations = []
    # for im_info in im_infos[:1]:
    #     # segment_unique = Label(im_info, snr_cleaning=False)
    #     # segment_unique = Label(im_info, num_t=4, snr_cleaning=False)
    #     segment_unique = Label(im_info, snr_cleaning=False)
    #     segment_unique.run()
    #     segmentations.append(segment_unique)
