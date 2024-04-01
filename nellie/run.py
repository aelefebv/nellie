from nellie.feature_extraction.hierarchical import Hierarchy
from nellie.im_info.im_info import ImInfo
from nellie.segmentation.filtering import Filter
from nellie.segmentation.labelling import Label
from nellie.segmentation.mocap_marking import Markers
from nellie.segmentation.networking import Network
from nellie.tracking.hu_tracking import HuMomentTracking
from nellie.tracking.voxel_reassignment import VoxelReassigner


def run(im_path, num_t=None, remove_edges=True, ch=0):
    im_info = ImInfo(im_path, ch=ch)

    preprocessing = Filter(im_info, num_t, remove_edges=remove_edges)
    preprocessing.run()

    segmenting = Label(im_info, num_t)
    segmenting.run()

    networking = Network(im_info, num_t)
    networking.run()

    mocap_marking = Markers(im_info, num_t)
    mocap_marking.run()

    hu_tracking = HuMomentTracking(im_info, num_t)
    hu_tracking.run()

    vox_reassign = VoxelReassigner(im_info, num_t)
    vox_reassign.run()

    hierarchy = Hierarchy(im_info, num_t)
    hierarchy.run()

    return im_info


if __name__ == "__main__":
    # Single file run
    im_path = r"/Users/austin/Downloads/test.tif"
    im_info = run(im_path, remove_edges=False, ch=0)

    # # Directory bactch run
    # import os
    # top_dir = r"D:\test_files\nellie_longer_smorgasbord"
    # ch = 0
    # num_t = None
    # # get all non-folder files
    # all_files = os.listdir(top_dir)
    # all_files = [os.path.join(top_dir, file) for file in all_files if not os.path.isdir(os.path.join(top_dir, file))]
    # for file_num, tif_file in enumerate(all_files):
    #     # for ch in range(1):
    #     print(f'Processing file {file_num + 1} of {len(all_files)}, channel {ch + 1} of 1')
    #     im_info = run(tif_file, remove_edges=True, ch=ch, num_t=num_t)
