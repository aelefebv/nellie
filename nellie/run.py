import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import multiprocessing

from nellie.feature_extraction.hierarchical import Hierarchy
from nellie.im_info.verifier import FileInfo, ImInfo
from nellie.segmentation.filtering import Filter
from nellie.segmentation.labelling import Label
from nellie.segmentation.mocap_marking import Markers
from nellie.segmentation.networking import Network
from nellie.tracking.hu_tracking import HuMomentTracking
from nellie.tracking.voxel_reassignment import VoxelReassigner
import os


def run(file_info, remove_edges=False, otsu_thresh_intensity=False, threshold=None):
    im_info = ImInfo(file_info)
    preprocessing = Filter(im_info, remove_edges=remove_edges)
    preprocessing.run()

    segmenting = Label(im_info, otsu_thresh_intensity=otsu_thresh_intensity, threshold=threshold)
    segmenting.run()

    networking = Network(im_info)
    networking.run()

    mocap_marking = Markers(im_info)
    mocap_marking.run()

    hu_tracking = HuMomentTracking(im_info)
    hu_tracking.run()

    vox_reassign = VoxelReassigner(im_info)
    vox_reassign.run()

    hierarchy = Hierarchy(im_info, skip_nodes=False)
    hierarchy.run()

    return im_info


def run_folders_multiproc(sub_dir, substring, output_dir):
    all_files = sorted(
        [os.path.join(sub_dir, f) for f in os.listdir(sub_dir) if substring in f and f.endswith('.tiff')])
    error_files = []
    for file_num, tif_file in enumerate(all_files):
        print(f'Processing file {file_num + 1} of {len(all_files)}')
        try:
            file_info = FileInfo(tif_file, output_dir=output_dir)
            file_info.find_metadata()
            file_info.load_metadata()
            im_info = run(file_info, otsu_thresh_intensity=True)
            im_info.close_all_memmaps()
        except Exception as e:
            print(f'Failed to run {tif_file}: {e}')
            error_files.append((tif_file, e))
            continue
    print(f'Error files: {error_files}')
    print(f'Number of error files: {len(error_files)}')
    # save error files to a text file
    with open(os.path.join(output_dir, f'error_files_{os.path.basename(sub_dir)}.txt'), 'w') as f:
        f.write(f'Number of error files: {len(error_files)}\n')
        f.write(f'Error files:\n')
        for error_file in error_files:
            f.write(f'\n{error_file[0]}\n')
            f.write(f'{error_file[1]}\n')


def process_directory(args):
    sub_dir, substring, output_dir = args
    print(f"Processing directory: {sub_dir}")
    run_folders_multiproc(sub_dir, substring, output_dir)


def run_all_directories_parallel(top_dir, substring, output_dir, num_processes=None):
    sub_dirs = [os.path.join(top_dir, f) for f in os.listdir(top_dir) if os.path.isdir(os.path.join(top_dir, f))]

    if not num_processes:
        num_processes = max(multiprocessing.cpu_count(), 20)

    with multiprocessing.Pool(processes=num_processes) as pool:
        args_list = [(sub_dir, substring, output_dir) for sub_dir in sub_dirs]
        pool.map(process_directory, args_list)


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
    # # test_file = all_paths[1]
    # file_info = FileInfo(test_file)
    # file_info.find_metadata()
    # file_info.load_metadata()
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
    # # file_info.change_selected_channel(3)
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
    # run(file_info)

    pass
