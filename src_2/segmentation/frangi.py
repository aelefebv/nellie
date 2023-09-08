import os

from src import logger
from src_2.io.im_info import ImInfo
from src_2.utils.general import get_reshaped_image


class FrangiFilter:
    def __init__(self, im_info: ImInfo,
                 num_t=None, remove_edges=True,
                 min_radius_um=0.20, max_radius_um=1):
        self.im_info = im_info
        self.num_t = num_t
        self.remove_edges = remove_edges
        # either (roughly) diffraction limit, or pixel size, whichever is larger
        self.min_radius_um = max(min_radius_um, self.im_info.dim_sizes['X'])
        self.max_radius_um = max_radius_um

        self.im_memmap = None
        self.frangi_memmap = None

    def _get_t(self):
        if self.num_t is None:
            if self.im_info.no_t:
                self.num_t = 1
            else:
                self.num_t = self.im_info.shape[self.im_info.axes.index('T')]
        else:
            return

    def _allocate_memory(self):
        logger.debug('Allocating memory for frangi filter.')
        im_memmap = self.im_info.get_im_memmap(self.im_info.im_path)
        self.im_memmap = get_reshaped_image(im_memmap, self.num_t, self.im_info)
        self.shape = self.im_memmap.shape
        im_frangi_path = self.im_info.create_output_path('im_frangi')
        self.frangi_memmap = self.im_info.allocate_memory(im_frangi_path, shape=self.shape, dtype='double',
                                                          description='frangi filtered im',
                                                          return_memmap=True)

    def run(self):
        logger.info('Running frangi filter.')
        self._get_t()
        self._allocate_memory()

if __name__ == "__main__":
    test_folder = r"D:\test_files\nelly_tests"
    all_files = os.listdir(test_folder)
    all_files = [file for file in all_files if not os.path.isdir(os.path.join(test_folder, file))]
    im_infos = []
    for file in all_files:
        im_path = os.path.join(test_folder, file)
        im_info = ImInfo(im_path)
        im_infos.append(im_info)

    frangis = []
    for im_info in im_infos:
        frangi = FrangiFilter(im_info)
        frangi.run()
        frangis.append(frangi)