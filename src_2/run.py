from src_2.io.im_info import ImInfo
from src_2.segmentation.filtering import Filter
from src_2.segmentation.labelling import Label
from src_2.segmentation.mocap_marking import Markers
from src_2.segmentation.networking import Network
from src_2.tracking.hu_tracking import HuMomentTracking


def run(im_path):
    im_info = ImInfo(im_path)

    preprocessing = Filter(im_info)
    preprocessing.run()

    segmenting = Label(im_info, snr_cleaning=False)
    segmenting.run()

    networking = Network(im_info)
    networking.run()

    mocap_marking = Markers(im_info)
    mocap_marking.run()

    hu_tracking = HuMomentTracking(im_info)
    hu_tracking.run()

    print('done')
    return im_info

if __name__ == "__main__":
    im_path = r"D:\test_files\nelly_tests\test_2\deskewed-2023-07-13_14-58-28_000_wt_0_acquire.ome.tif"
    im_info = run(im_path)
