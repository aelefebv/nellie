#%%

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
import os

def run(file_info, remove_edges=False, otsu_thresh_intensity=False, threshold=None, timeit=False):
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

    Returns
    -------
    ImInfo
        ImInfo object containing processed image data and paths.
    """
    im_info = ImInfo(file_info)

    if timeit:
        start_time = time.perf_counter()
    preprocessing = Filter(im_info, remove_edges=remove_edges)
    preprocessing.run()
    if timeit:
        end_time = time.perf_counter()
        preprocessing_time = end_time - start_time

    if timeit:
        start_time = time.perf_counter()
    segmenting = Label(im_info, otsu_thresh_intensity=otsu_thresh_intensity, threshold=threshold)
    segmenting.run()
    if timeit:
        end_time = time.perf_counter()
        segmenting_time = end_time - start_time

    if timeit:
        start_time = time.perf_counter()
    networking = Network(im_info)
    networking.run()
    if timeit:
        end_time = time.perf_counter()
        networking_time = end_time - start_time

    if timeit:
        start_time = time.perf_counter()
    mocap_marking = Markers(im_info)
    mocap_marking.run()
    if timeit:
        end_time = time.perf_counter()
        mocap_marking_time = end_time - start_time

    if timeit:
        start_time = time.perf_counter()
    hu_tracking = HuMomentTracking(im_info)
    hu_tracking.run()
    if timeit:
        end_time = time.perf_counter()
        hu_tracking_time = end_time - start_time

    if timeit:
        start_time = time.perf_counter()
    vox_reassign = VoxelReassigner(im_info)
    vox_reassign.run()
    if timeit:
        end_time = time.perf_counter()
        vox_reassign_time = end_time - start_time

    if timeit:
        start_time = time.perf_counter()
    hierarchy = Hierarchy(im_info, skip_nodes=False)
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

#%%
device = 'gpu'
low_memory = True
top_dir = r'D:\test_files\nellie_all_tests\yeast_3d_mitochondria.ome_variants'
# top_dir = '/Users/austin/test_files/nellie_all_tests/yeast_3d_mitochondria.ome_variants/'
# find all the files ending in .tif
tif_files = [os.path.join(top_dir, f) for f in os.listdir(top_dir) if f.endswith('.tif')]
for test_file in tif_files:
    # test_file = '/Users/austin/test_files/nellie_all_tests/yeast_3d_mitochondria.ome_variants/variant_YX_firstT_maxZ.ome.tif'
    file_info = FileInfo(test_file)
    file_info.find_metadata()
    file_info.load_metadata()
    im_info = ImInfo(file_info)
    preprocessing = Filter(im_info, device=device, low_memory=low_memory)
    preprocessing.run()
    labelling = Label(im_info, device=device, low_memory=low_memory)
    labelling.run()
    networking = Network(im_info, device=device, low_memory=low_memory)
    networking.run()
    mocap_marking = Markers(im_info, device=device, low_memory=low_memory)
    mocap_marking.run()
    hu_tracking = HuMomentTracking(im_info, device=device, low_memory=low_memory)
    hu_tracking.run()
    vox_reassign = VoxelReassigner(im_info, device=device, low_memory=low_memory)
    vox_reassign.run()
    hierarchy = Hierarchy(im_info, device=device, low_memory=low_memory)
    hierarchy.run()

# %%
