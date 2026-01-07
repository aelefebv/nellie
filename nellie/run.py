"""
Main entry point for the Nellie image analysis pipeline.

This module provides the run function, which orchestrates the complete Nellie pipeline
including filtering, segmentation, tracking, and feature extraction.
"""
from nellie.feature_extraction.hierarchical import Hierarchy
from nellie.im_info.verifier import FileInfo, ImInfo
from nellie.segmentation.filtering import Filter
from nellie.segmentation.labelling import Label
from nellie.segmentation.mocap_marking import Markers
from nellie.segmentation.networking import Network
from nellie.tracking.hu_tracking import HuMomentTracking
from nellie.tracking.voxel_reassignment import VoxelReassigner

import time

def run(
    file_info,
    remove_edges=False,
    otsu_thresh_intensity=False,
    threshold=None,
    timeit=False,
    device="auto",
    low_memory=False,
):
    """
    Main entry point for the Nellie pipeline.

    Parameters
    ----------
    file_info : FileInfo
        FileInfo object containing metadata about the input image.
    remove_edges : bool, optional
        Whether to remove edges during filtering (default is False).
    otsu_thresh_intensity : bool, optional
        Whether to use Otsu thresholding for intensity (default is False).
    threshold : float, optional
        Manual threshold value (default is None).
    timeit : bool, optional
        Whether to time each step of the pipeline (default is False).
    device : {"auto", "cpu", "gpu"}, optional
        Backend selection for preprocessing, labeling, and feature extraction.
    low_memory : bool, optional
        Whether to prefer lower-memory (slower) implementations where available.

    Returns
    -------
    ImInfo
        ImInfo object containing processed image data and paths.
    """
    im_info = ImInfo(file_info)

    if timeit:
        start_time = time.perf_counter()
    preprocessing = Filter(
        im_info, remove_edges=remove_edges, device=device, low_memory=low_memory
    )
    preprocessing.run()
    if timeit:
        end_time = time.perf_counter()
        preprocessing_time = end_time - start_time

    if timeit:
        start_time = time.perf_counter()
    segmenting = Label(
        im_info,
        otsu_thresh_intensity=otsu_thresh_intensity,
        threshold=threshold,
        device=device,
        low_memory=low_memory,
    )
    segmenting.run()
    if timeit:
        end_time = time.perf_counter()
        segmenting_time = end_time - start_time

    if timeit:
        start_time = time.perf_counter()
    networking = Network(im_info, device=device)
    networking.run()
    if timeit:
        end_time = time.perf_counter()
        networking_time = end_time - start_time

    if timeit:
        start_time = time.perf_counter()
    mocap_marking = Markers(im_info, device=device)
    mocap_marking.run()
    if timeit:
        end_time = time.perf_counter()
        mocap_marking_time = end_time - start_time

    if timeit:
        start_time = time.perf_counter()
    hu_tracking = HuMomentTracking(im_info, device=device, low_memory=low_memory)
    hu_tracking.run()
    if timeit:
        end_time = time.perf_counter()
        hu_tracking_time = end_time - start_time

    if timeit:
        start_time = time.perf_counter()
    vox_reassign = VoxelReassigner(im_info, device=device)
    vox_reassign.run()
    if timeit:
        end_time = time.perf_counter()
        vox_reassign_time = end_time - start_time

    if timeit:
        start_time = time.perf_counter()
    hierarchy = Hierarchy(
        im_info, skip_nodes=False, device=device, low_memory=low_memory
    )
    hierarchy.run()
    if timeit:
        end_time = time.perf_counter()
        hierarchy_time = end_time - start_time

    if timeit:
        print(f"Nellie Pipeline: Filter step took {preprocessing_time:.4f} seconds")
        print(f"Nellie Pipeline: Label step took {segmenting_time:.4f} seconds")
        print(f"Nellie Pipeline: Network step took {networking_time:.4f} seconds")
        print(f"Nellie Pipeline: Markers step took {mocap_marking_time:.4f} seconds")
        print(f"Nellie Pipeline: HuMomentTracking step took {hu_tracking_time:.4f} seconds")
        print(f"Nellie Pipeline: VoxelReassigner step took {vox_reassign_time:.4f} seconds")
        print(f"Nellie Pipeline: Hierarchy step took {hierarchy_time:.4f} seconds")
        print(f"Nellie Pipeline: Total time took {preprocessing_time + segmenting_time + networking_time + mocap_marking_time + hu_tracking_time + vox_reassign_time + hierarchy_time:.4f} seconds")

    return im_info

if __name__ == "__main__":
    # # Single file run
    # im_path = r"/Users/austin/test_files/nellie_all_tests/ND Stimulation Parallel 12.nd2"
    # im_info = run(im_path, remove_edges=False, num_t=5)
    # im_info = run(im_path, remove_edges=False, ch=1, dim_sizes={'T': 1, 'Z': 0.1, 'Y': 0.1, 'X': 0.1}, otsu_thresh_intensity=True)

    # Directory batch run
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

    # test_file = '/Users/austin/test_files/nellie_all_tests/yeast_3d_mitochondria.ome.tif'
    # test_file = '/Users/austin/Downloads/26598942-Pos213-t_008-y_1744-x_0329.ome.tif'
    test_file = "sample_data/yeast_3d_mitochondria.ome.tif"
    # test_file = r"D:\test_files\nellie_all_tests\ND Stimulation Parallel 12.nd2"
    # test_file = "/Users/austin/test_files/nellie_all_tests/test_2.nd2"
    # test_file = all_paths[1]
    file_info = FileInfo(test_file)
    file_info.find_metadata()
    file_info.load_metadata()
    # print(f'{file_info.metadata_type=}')
    # print(f'{file_info.axes=}')
    # print(f'{file_info.shape=}')
    # print(f'{file_info.dim_res=}')
    # print(f'{file_info.good_axes=}')
    # print(f'{file_info.good_dims=}')
    # print('\n')

    # file_info.change_axes('TZYX')
    # print('Axes changed')
    # print(f'{file_info.axes=}')
    # print(f'{file_info.dim_res=}')
    # print(f'{file_info.good_axes=}')
    # print(f'{file_info.good_dims=}')
    # print('\n')
    #
    # file_info.change_dim_res('T', 1)
    # file_info.change_dim_res('Z', 0.5)
    # file_info.change_dim_res('Y', 0.2)
    # file_info.change_dim_res('X', 0.2)
    #
    # print('Dimension resolutions changed')
    # print(f'{file_info.axes=}')
    # print(f'{file_info.dim_res=}')
    # print(f'{file_info.good_axes=}')
    # print(f'{file_info.good_dims=}')
    # print('\n')
    #
    # # print(f'{file_info.ch=}')
    # file_info.change_selected_channel(2)
    # # print('Channel changed')
    # # print(f'{file_info.ch=}')
    #
    # print(f'{file_info.t_start=}')
    # print(f'{file_info.t_end=}')
    # file_info.select_temporal_range(1, 3)
    # print('Temporal range selected')
    # print(f'{file_info.t_start=}')
    # print(f'{file_info.t_end=}')
    #
    # # file_info.save_ome_tiff()
    # # im_info = ImInfo(file_info)
    run(file_info)
