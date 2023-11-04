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

    def set_coords(self, coords, frame_num):
        self.coords[frame_num] = coords

    def set_flow(self, flow_vectors, frame_num):
        self.flow_vectors[frame_num] = flow_vectors

    def set_label_values(self, label_values, frame_num):
        self.label_values[frame_num] = label_values

class MotilityFeatures:
    def __init__(self, im_info: ImInfo):
        self.im_info = im_info
        if self.im_info.no_z:
            self.spacing = (self.im_info.dim_sizes['Y'], self.im_info.dim_sizes['X'])
        else:
            self.spacing = (self.im_info.dim_sizes['Z'], self.im_info.dim_sizes['Y'], self.im_info.dim_sizes['X'])

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
        for frame_num in range(len(self.relabelled_memmap)-1):
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
                label_obj.set_coords(labels_coords[label_idxs], frame_num)
                label_obj.set_flow(flow_interpolation[label_idxs], frame_num)
                label_obj.set_label_values(labelled_values[label_idxs], frame_num)

            print('hi')


    # def _get_flow_vectors(self):
    #     for frame_num, frame in enumerate


    def _get_frame_motility(self):
        self._get_label_coords()
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