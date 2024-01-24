from src.feature_extraction.graph_frame import GraphBuilder
from src.feature_extraction.morphology_labels import MorphologyLabelFeatures
from src.feature_extraction.morphology_skeletons import MorphologySkeletonFeatures
from src.feature_extraction.motility_labels import CoordMovement
from src.im_info.im_info import ImInfo
from src.segmentation.filtering import Filter
from src.segmentation.labelling import Label
from src.segmentation.mocap_marking import Markers
from src.segmentation.networking import Network
from src.tracking.hu_tracking import HuMomentTracking
from src.tracking.voxel_reassignment import VoxelReassigner


def run(im_path, num_t=None, remove_edges=True, ch=0):
    im_info = ImInfo(im_path, ch=ch)

    # preprocessing = Filter(im_info, num_t, remove_edges=remove_edges)
    # preprocessing.run()
    #
    # segmenting = Label(im_info, num_t)
    # segmenting.run()
    #
    # networking = Network(im_info, num_t)
    # networking.run()
    #
    # mocap_marking = Markers(im_info, num_t)
    # mocap_marking.run()
    #
    # hu_tracking = HuMomentTracking(im_info, num_t)
    # hu_tracking.run()

    vox_reassign = VoxelReassigner(im_info, num_t)
    vox_reassign.run()

    # morphology_skeleton_features = MorphologySkeletonFeatures(im_info, num_t)
    # morphology_skeleton_features.run()
    #
    # morphology_label_features = MorphologyLabelFeatures(im_info, num_t)
    # morphology_label_features.run()

    # motility_label_features = CoordMovement(im_info, num_t)
    # motility_label_features.run()
    #
    # graph_builder = GraphBuilder(im_info, num_t)
    # graph_builder.run()

    return im_info


if __name__ == "__main__":
    im_path = "/Users/austin/test_files/nelly/ND Stimulation Parallel 12.nd2"
    im_info = run(im_path, remove_edges=False, ch=0, num_t=3)

    # # im_path = r"D:\test_files\nelly_smorgasbord\deskewed-iono_pre.ome.tif"
    # # im_info = run(im_path, remove_edges=True, num_t=3)
    # import os
    # # top_dir = r"D:\test_files\nelly_smorgasbord"
    # top_dir = r"D:\test_files\gav_aging_2"
    # ch = 1
    # # top_dir = r"D:\test_files\nelly_iono\full_2"
    # # top_dir = r"D:\test_files\stress_granules"
    # # top_dir = r"D:\test_files\nelly_multichannel"
    # # get all non-folder files
    # all_files = os.listdir(top_dir)
    # all_files = [os.path.join(top_dir, file) for file in all_files if not os.path.isdir(os.path.join(top_dir, file))]
    # # all_files = [r"D:\test_files\nelly_tests\deskewed-2023-07-13_14-58-28_000_wt_0_acquire.ome.tif"]
    # for file_num, tif_file in enumerate(all_files):
    #     # for ch in range(1):
    #     print(f'Processing file {file_num + 1} of {len(all_files)}, channel {ch + 1} of 1')
    #     im_info = run(tif_file, remove_edges=True, ch=ch)
