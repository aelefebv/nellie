from src_2.io.im_info import ImInfo
from src import xp, ndi, logger
from src_2.utils.general import get_reshaped_image
import numpy as np


class FlowInterpolation:
    def __init__(self, im_info: ImInfo, num_t=None):
        self.im_info = im_info
        self.num_t = num_t
        if num_t is None:
            self.num_t = im_info.shape[im_info.axes.index('T')]
        self.scaling = (im_info.dim_sizes['Z'], im_info.dim_sizes['Y'], im_info.dim_sizes['X'])

        # self.max_distance_um = max_distance_um

        # self.vector_start_coords = []
        # self.vectors = []
        # self.vector_magnitudes = []

        self.shape = ()

        self.im_memmap = None
        self.im_distance_memmap = None
        self.flow_vector_array = None

        self.debug = None

    def _allocate_memory(self):
        logger.debug('Allocating memory for mocap marking.')

        im_memmap = self.im_info.get_im_memmap(self.im_info.im_path)
        self.im_memmap = get_reshaped_image(im_memmap, self.num_t, self.im_info)

        im_distance_memmap = self.im_info.get_im_memmap(self.im_info.pipeline_paths['im_distance'])
        self.im_distance_memmap = get_reshaped_image(im_distance_memmap, self.num_t, self.im_info)
        self.shape = self.im_distance_memmap.shape

        flow_vector_array_path = self.im_info.pipeline_paths['flow_vector_array']
        self.flow_vector_array = np.load(flow_vector_array_path)

    def _get_t(self):
        if self.num_t is None:
            if self.im_info.no_t:
                self.num_t = 1
            else:
                self.num_t = self.im_info.shape[self.im_info.axes.index('T')]
        else:
            return

    def interpolate_coord(self, coord, t, forward=True):
        # interpolate the flow vector at the coordinate at time t, either forward in time or backward in time.
        # For forward, simply find nearby LMPs, interpolate based on distance-weighted vectors
        # For backward, get coords from t-1 + vector, then find nearby coords from that, and interpolate based on distance-weighted vectors


    def run(self):
        self._get_t()
        self._allocate_memory()


if __name__ == "__main__":
    import os
    test_folder = r"D:\test_files\nelly_tests"
    all_files = os.listdir(test_folder)
    all_files = [file for file in all_files if not os.path.isdir(os.path.join(test_folder, file))]
    im_infos = []
    for file in all_files:
        im_path = os.path.join(test_folder, file)
        im_info = ImInfo(im_path)
        im_info.create_output_path('im_distance')
        im_info.create_output_path('flow_vector_array', ext='.npy')
        im_infos.append(im_info)

    flow_interpx = FlowInterpolation(im_infos[0])
    flow_interpx.run()

