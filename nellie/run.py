from nellie.feature_extraction.hierarchical import Hierarchy
from nellie.im_info.im_info import ImInfo
from nellie.segmentation.filtering import Filter
from nellie.segmentation.labelling import Label
from nellie.segmentation.mocap_marking import Markers
from nellie.segmentation.networking import Network
from nellie.tracking.hu_tracking import HuMomentTracking
from nellie.tracking.voxel_reassignment import VoxelReassigner


def run(im_path, num_t=None, remove_edges=True, ch=0, output_dirpath=None, otsu_thresh_intensity=False, dim_sizes=None, threshold=None, dim_order=""):
    im_info = ImInfo(im_path, num_t=num_t, ch=ch, output_dirpath=output_dirpath, dim_sizes=dim_sizes, dimension_order=dim_order)
    preprocessing = Filter(im_info, num_t, remove_edges=remove_edges)
    preprocessing.run()

    segmenting = Label(im_info, num_t, otsu_thresh_intensity=otsu_thresh_intensity, threshold=threshold)
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
    # # Single file run
    im_path = r"/Users/austin/test_files/nellie_all_tests/ND Stimulation Parallel 12.nd2"
    im_info = run(im_path, remove_edges=False, num_t=5)
    # im_info = run(im_path, remove_edges=False, ch=1, dim_sizes={'T': 1, 'Z': 0.1, 'Y': 0.1, 'X': 0.1}, otsu_thresh_intensity=True)

    # Directory bactch run
    # import os
    # top_dirs = [
    #     r"C:\Users\austin\GitHub\nellie-supplemental\comparisons\simulations\multi_grid\outputs",
    #     r"C:\Users\austin\GitHub\nellie-supplemental\comparisons\simulations\separation\outputs",
    #     r"C:\Users\austin\GitHub\nellie-supplemental\comparisons\simulations\px_sizes\outputs",
    #     ]
    # ch = 0
    # num_t = 1
    # # get all non-folder files
    # for top_dir in top_dirs:
    #     all_files = os.listdir(top_dir)
    #     all_files = [os.path.join(top_dir, file) for file in all_files if not os.path.isdir(os.path.join(top_dir, file))]
    #     all_files = [file for file in all_files if file.endswith('.tif')]
    #     for file_num, tif_file in enumerate(all_files):
    #         # for ch in range(1):
    #         print(f'Processing file {file_num + 1} of {len(all_files)}, channel {ch + 1} of 1')
    #         im_info = ImInfo(tif_file, ch=ch)
    #         if os.path.exists(im_info.pipeline_paths['im_skel_relabelled']):
    #             print(f'Already exists, skipping.')
    #             continue
    #         im_info = run(tif_file, remove_edges=False, ch=ch, num_t=num_t)
