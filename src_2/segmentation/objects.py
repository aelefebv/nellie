import tifffile

from src_2.io.im_info import ImInfo
from src import xp, morphology, ndi, is_gpu, logger, filters
from src_2.utils.general import get_reshaped_image
from src_2.utils.gpu_functions import otsu_threshold, triangle_threshold


class Segment:
    def __init__(self, im_info: ImInfo,
                 num_t=None,
                 threshold: float = 0,
                 min_radius_um: float = 0.2,
                 max_radius_um=xp.inf):
        self.im_info = im_info
        self.num_t = num_t
        if num_t is None:
            self.num_t = im_info.shape[im_info.axes.index('T')]
        self.threshold = threshold
        self.min_radius_um = max(min_radius_um, self.im_info.dim_sizes['X'])
        self.max_radius_um = max_radius_um

        self.frangi_memmap = None

        self.remove_in_2d = False
        if any(xp.array(
                [self.im_info.dim_sizes['Z'], self.im_info.dim_sizes['Y'], self.im_info.dim_sizes['X']]
        ) > (self.min_radius_um*2)):
            logger.warning(f"One of the dimensions' voxel sizes is greater than the minimum radius of the structure in "
                           f"question so object removal will be conducted based on 2D parameters instead of 3D. "
                           f"This may result in objects being kept that should not be.")
            self.remove_in_2d = True

        # convert min radius um to a min area / volume
        if not self.im_info.no_z and not self.remove_in_2d:
            # volume of sphere of radius min_width/2 in pixels cubed
            self.min_size_threshold_px = (4 / 3 * xp.pi * (min_radius_um / 2) ** 2) / (
                    self.im_info.dim_sizes['X'] ** 2 * self.im_info.dim_sizes['Z']
            )
            # min area of 3*3*3 pixels
            self.min_size_threshold_px = max(self.min_size_threshold_px, 27)
            self.max_size_threshold_px = (4 / 3 * xp.pi * (max_radius_um / 2) ** 2) / (
                    self.im_info.dim_sizes['X'] ** 2 * self.im_info.dim_sizes['Z']
            )
        else:
            self.min_size_threshold_px = (xp.pi * (min_radius_um / 2) ** 2) / (self.im_info.dim_sizes['X'] ** 2)
            # min area of 3*3 pixels
            self.min_size_threshold_px = max(self.min_size_threshold_px, 9)
            self.max_size_threshold_px = (xp.pi * (max_radius_um / 2) ** 2) / (self.im_info.dim_sizes['X'] ** 2)

        self.semantic_mask_memmap = None
        self.instance_label_memmap = None
        self.shape = ()

        self.debug = None

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
        frangi_memmap = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_frangi'])
        self.frangi_memmap = get_reshaped_image(frangi_memmap, self.num_t, self.im_info)
        self.shape = self.frangi_memmap.shape

        im_semantic_mask_path = self.im_info.create_output_path('im_semantic_mask')
        self.semantic_mask_memmap = self.im_info.allocate_memory(im_semantic_mask_path, shape=self.shape, dtype='int8',
                                                                 description='semantic segmentation',
                                                                 return_memmap=True)

        im_instance_label_path = self.im_info.create_output_path('im_instance_label')
        self.instance_label_memmap = self.im_info.allocate_memory(im_instance_label_path, shape=self.shape, dtype='int16',
                                                                  description='instance segmentation',
                                                                  return_memmap=True)

    def _remove_bad_sized_objects(self, frame):
        ndim = 2 if self.remove_in_2d else 3
        footprint = ndi.generate_binary_structure(ndim, 1)
        labels, _ = ndi.label(frame>0, structure=footprint)
        label_sizes = xp.bincount(labels.ravel())
        above_threshold = label_sizes > self.min_size_threshold_px
        below_threshold = label_sizes < self.max_size_threshold_px
        mask_sizes = above_threshold * below_threshold
        mask = xp.zeros_like(labels, dtype=bool)
        mask[mask_sizes[labels]] = True
        mask[labels == 0] = False
        labels = labels * mask
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

    def _run_frame(self, t):
        logger.info(f'Running semantic segmentation, volume {t}/{self.num_t - 1}')
        frame_in_mem = xp.asarray(self.frangi_memmap[t, ...])
        # threshold_mask, threshold_labels = self._remove_bad_sized_objects(frame_in_mem)
        # trimmed_labels = self._trim_labels(frame_in_mem, threshold_labels)
        return self._remove_bad_sized_objects(frame_in_mem)

    def _run_segmentation(self):
        for t in range(self.num_t):
            semantic_mask, instance_mask = self._run_frame(t)
            self.semantic_mask_memmap[t, ...] = semantic_mask.get()
            self.instance_label_memmap[t, ...] = instance_mask.get()

    def run(self):
        logger.info('Running semantic segmentation.')
        self._get_t()
        self._allocate_memory()
        self._run_segmentation()


if __name__ == "__main__":
    import os
    test_folder = r"D:\test_files\nelly_tests"
    all_files = os.listdir(test_folder)
    all_files = [file for file in all_files if not os.path.isdir(os.path.join(test_folder, file))]
    im_infos = []
    for file in all_files:
        im_path = os.path.join(test_folder, file)
        im_info = ImInfo(im_path)
        im_info.create_output_path('im_frangi')
        im_infos.append(im_info)

    segmentations = []
    for im_info in im_infos:
        segment_unique = Segment(im_info, num_t=2)
        segment_unique.run()
        segmentations.append(segment_unique)
