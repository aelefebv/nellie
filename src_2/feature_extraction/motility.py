from src import logger, xp
from src_2.im_info.im_info import ImInfo
from src_2.tracking.flow_interpolation import FlowInterpolator
from src_2.utils.general import get_reshaped_image


class LabelFeatures:
    def __init__(self, label_num):
        self.label_num = label_num
        self.coords = {}
        self.flow_vectors = {}
        self.label_values = {}

        self.features = {}

    def set_values(self, coords, flow_vectors, label_values, frame_num):
        self.coords[frame_num] = coords
        self.flow_vectors[frame_num] = flow_vectors
        self.label_values[frame_num] = label_values

class MotilityFeatures:
    def __init__(self, im_info: ImInfo):
        self.im_info = im_info
        if self.im_info.no_z:
            self.spacing = xp.array((self.im_info.dim_sizes['Y'], self.im_info.dim_sizes['X']))
        else:
            self.spacing = xp.array((self.im_info.dim_sizes['Z'], self.im_info.dim_sizes['Y'], self.im_info.dim_sizes['X']))

        self.im_memmap = None
        self.label_memmap = None
        self.relabelled_memmap = None
        self.flow_vectors = None
        self.flow_interpolator = FlowInterpolator(im_info, forward=True)

        self.features = {}

    def _get_memmaps(self):
        logger.debug('Allocating memory for spatial feature extraction.')
        im_memmap = self.im_info.get_im_memmap(self.im_info.im_path)
        self.im_memmap = get_reshaped_image(im_memmap, None, self.im_info)

        label_memmap = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_instance_label'])
        self.label_memmap = get_reshaped_image(label_memmap, None, self.im_info)

        relabelled_memmap = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_instance_label_reassigned'])
        self.relabelled_memmap = get_reshaped_image(relabelled_memmap, None, self.im_info)

        self.im_info.create_output_path('motility_features', ext='.csv')
        self.motility_features_path = self.im_info.pipeline_paths['motility_features']

        self.shape = self.label_memmap.shape

    def _get_label_coords(self):
        # for frame_num in range(len(self.relabelled_memmap)-1):
        for frame_num in range(2):
            logger.debug(f'Getting label coords for frame {frame_num}.')
            relabelled_frame_gpu = xp.array(self.relabelled_memmap[frame_num])
            label_frame_gpu = xp.array(self.label_memmap[frame_num])

            labels_coords = xp.argwhere(relabelled_frame_gpu)
            relabelled_values = relabelled_frame_gpu[relabelled_frame_gpu > 0]
            labelled_values = label_frame_gpu[relabelled_frame_gpu > 0]

            unique_labels = xp.unique(relabelled_values)
            flow_interpolation = xp.array(self.flow_interpolator.interpolate_coord(labels_coords.get(), frame_num))

            if frame_num == 0:
                self.unique_labels = {label: LabelFeatures(label) for label in unique_labels.tolist()}
            for unique_label, label_obj in self.unique_labels.items():
                label_idxs = relabelled_values == unique_label
                label_obj.set_values(
                    labels_coords[label_idxs], flow_interpolation[label_idxs], labelled_values[label_idxs],
                    frame_num
                )

    def _get_label_motility_features(self, label_obj):
        for frame, flow_vector in label_obj.flow_vectors.items():
            if len(flow_vector) == 0:
                continue
            sum_flow = xp.nansum(flow_vector, axis=0) * self.spacing
            mean_flow = xp.nanmean(flow_vector, axis=0) * self.spacing
            # todo how to describe rotation? differential motion? directionality?
            if frame == 0:
                label_obj.features['sum_flow_vector'] = [sum_flow]
                label_obj.features['mean_flow_vector'] = [mean_flow]
            else:
                label_obj.features['sum_flow_vector'].append(sum_flow)
                label_obj.features['mean_flow_vector'].append(mean_flow)

    def _get_all_motility_features(self):
        for label_num, label_obj in self.unique_labels.items():
            logger.debug(f'Getting motility features for label {label_num}.')
            self._get_label_motility_features(label_obj)
            # todo, could get multi-object features for each label


    def _get_frame_motility(self):
        self._get_label_coords()
        self._get_all_motility_features()
        print('hi')

    def run(self):
        self._get_memmaps()
        self._get_frame_motility()

if __name__ == "__main__":
    im_path = r"D:\test_files\nelly_tests\deskewed-2023-07-13_14-58-28_000_wt_0_acquire.ome.tif"
    im_info = ImInfo(im_path)
    im_info.create_output_path('im_instance_label')
    im_info.create_output_path('im_instance_label_reassigned')
    im_info.create_output_path('flow_vector_array', ext='.npy')

    motility_features = MotilityFeatures(im_info)
    motility_features.run()

    # im_info = ImInfo(im_path)