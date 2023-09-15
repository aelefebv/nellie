import tifffile

from src_2.io.im_info import ImInfo
from src import xp, morphology, ndi, is_gpu, logger, filters
from src_2.utils.general import get_reshaped_image


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
        self.instance_mask_memmap = None
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
        im_instance_mask_path = self.im_info.create_output_path('im_instance_mask')
        self.instance_mask_memmap = self.im_info.allocate_memory(im_instance_mask_path, shape=self.shape, dtype='int16',
                                                                 description='instance segmentation',
                                                                 return_memmap=True)

    def _remove_bad_sized_objects(self, frame):
        ndim = 2 if self.remove_in_2d else 3
        footprint = ndi.generate_binary_structure(ndim, 1)
        labels, _ = ndi.label(frame, structure=footprint)
        label_sizes = xp.bincount(labels.ravel())
        above_threshold = label_sizes > self.min_size_threshold_px
        below_threshold = label_sizes < self.max_size_threshold_px
        mask_sizes = above_threshold * below_threshold
        mask = xp.zeros_like(labels, dtype=bool)
        mask[mask_sizes[labels]] = True
        mask[labels == 0] = False
        return mask

    def _run_semantic_frame(self, t):
        logger.info(f'Running semantic segmentation, volume {t}/{self.num_t - 1}')
        frame_in_mem = xp.asarray(self.frangi_memmap[t, ...]) > 0
        return self._remove_bad_sized_objects(frame_in_mem)

    def _run_instance_frame(self, t, semantic_mask):


    def _run_segmentation(self):
        for t in range(self.num_t):
            semantic_mask = self._run_semantic_frame(t)
            instance_mask = self._run_instance_frame(t, semantic_mask)
            self.semantic_mask_memmap[t, ...] = semantic_mask.get()
            # if self.remove_edges:
            #     frangi_frame = self._remove_edges(frangi_frame)
            # self.frangi_memmap[t, ...] = self._mask_volume(frangi_frame).get()

    def run(self):
        logger.info('Running semantic segmentation.')
        self._get_t()
        self._allocate_memory()
        self._run_segmentation()
    # def semantic(self, num_t: int = None):
    #     """
    #     Run semantic segmentation on the frangi filtered image.
    #
    #     Args:
    #         num_t (int, optional): Number of timepoints to process. Defaults to None, which processes all timepoints.
    #     """
    #     frangi_memmap = tifffile.memmap(self.im_info.path_im_frangi, mode='r')
    #     frangi_memmap = get_reshaped_image(frangi_memmap, num_t, self.im_info)
    #     shape = frangi_memmap.shape
    #
    #     self.im_info.allocate_memory(
    #         self.im_info.path_im_mask, shape=shape, dtype='uint8', description='Semantic mask image.',
    #     )
    #
    #     self.semantic_mask_memmap = tifffile.memmap(self.im_info.path_im_mask, mode='r+')
    #     if len(self.semantic_mask_memmap.shape) == len(shape)-1:
    #         self.semantic_mask_memmap = self.semantic_mask_memmap[None, ...]
    #
    #     for frame_num, frame in enumerate(frangi_memmap):
    #         logger.info(f'Running semantic segmentation, volume {frame_num}/{len(frangi_memmap) - 1}')
    #         frame_in_mem = xp.asarray(frame)
    #         test_triangle = filters.threshold_triangle(frame_in_mem[frame_in_mem > 0])
    #         frame_in_mem = frame_in_mem > test_triangle
    #         if self.remove_in_2d:
    #             struct = ndi.generate_binary_structure(2, 1)
    #             for z in range(frame_in_mem.shape[0]):
    #                 frame_in_mem[z] = ndi.binary_opening(frame_in_mem[z], structure=struct)
    #         else:
    #             frame_in_mem = ndi.binary_opening(frame_in_mem)
    #             # frame_in_mem = ndi.grey_opening(frame_in_mem)
    #         frame_in_mem = morphology.remove_small_objects(frame_in_mem, self.min_size_threshold_px)
    #         if is_gpu:
    #             self.semantic_mask_memmap[frame_num] = frame_in_mem.get()
    #         else:
    #             self.semantic_mask_memmap[frame_num] = frame_in_mem
    #
    # def instance(self, num_t: int = None, dtype: str = 'uint32'):
    #     """
    #     Run instance segmentation on the semantic segmentation.
    #
    #     Args:
    #         num_t (int, optional): Number of timepoints to process. Defaults to None, which processes all timepoints.
    #         dtype (str, optional): Data type of the output instance mask. Defaults to 'uint32'.
    #     """
    #     self.semantic_mask_memmap = tifffile.memmap(self.im_info.path_im_mask, mode='r')
    #     self.semantic_mask_memmap = get_reshaped_image(self.semantic_mask_memmap, num_t, self.im_info)
    #     self.shape = self.semantic_mask_memmap.shape
    #
    #     self.im_info.allocate_memory(
    #         self.im_info.path_im_label_obj, shape=self.shape, dtype=dtype, description='Instance mask image.',
    #     )
    #     self.instance_mask_memmap = tifffile.memmap(self.im_info.path_im_label_obj, mode='r+')
    #
    #     if len(self.instance_mask_memmap.shape) == len(self.shape)-1:
    #         self.instance_mask_memmap = self.instance_mask_memmap[None, ...]
    #
    #     if self.im_info.is_3d:
    #         structure = xp.ones((3, 3, 3))
    #     else:
    #         structure = xp.ones((3, 3))
    #     for frame_num, frame in enumerate(self.semantic_mask_memmap):
    #         logger.info(f'Running instance segmentation, volume {frame_num}/{len(self.semantic_mask_memmap) - 1}')
    #         label_im = xp.asarray(frame).astype(bool)
    #         label_im, _ = ndi.label(label_im, structure=structure)
    #         # remove objects that have a mean intensity less that a threshold
    #         if self.min_mean_intensity is not None:
    #             unique_labels = xp.unique(label_im)
    #             # print('unique_labels', unique_labels)
    #             # print('memmap_shape', self.im_memmap[frame_num, ...].shape)
    #
    #             mean_intensities = ndi.labeled_comprehension(self.im_memmap[frame_num, ...], label_im, unique_labels,
    #                                                          xp.mean, float, xp.nan)
    #             # print(mean_intensities)
    #             label_mean_intensity = dict(zip(unique_labels.tolist(), mean_intensities.tolist()))
    #             filtered_label_mean_intensity = {label: mean_intensity
    #                                              for label, mean_intensity in label_mean_intensity.items()
    #                                              if mean_intensity >= self.min_mean_intensity}
    #             label_voxel_count = {label: xp.sum(label_im == label) for label in filtered_label_mean_intensity.keys()}
    #             if self.max_size_threshold_px is not None:
    #                 filtered_label_voxel_count = {label: voxel_count for label, voxel_count in label_voxel_count.items() if
    #                                               voxel_count < self.max_size_threshold_px}
    #                 # Create a new label image with only the remaining labels
    #                 filtered_label_img = xp.zeros_like(label_im)
    #                 for label in filtered_label_voxel_count.keys():
    #                     filtered_label_img[label_im == label] = label
    #             else:
    #                 # Create a new label image with only the remaining labels
    #                 filtered_label_img = xp.zeros_like(label_im)
    #                 for label in filtered_label_mean_intensity.keys():
    #                     filtered_label_img[label_im == label] = label
    #             label_im = filtered_label_img
    #
    #         if is_gpu:
    #             self.instance_mask_memmap[frame_num] = label_im.get()
    #         else:
    #             self.instance_mask_memmap[frame_num] = label_im


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
