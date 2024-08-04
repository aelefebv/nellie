import os
from .im_info.verifier import FileInfo


if __name__ == "__main__":
    test_dir = '/Users/austin/test_files/nellie_all_tests'
    all_paths = os.listdir(test_dir)
    all_paths = [os.path.join(test_dir, path) for path in all_paths if path.endswith('.tiff') or path.endswith('.tif') or path.endswith('.nd2')]
    # for filepath in all_paths:
    #     file_info = FileInfo(filepath)
    #     file_info.find_metadata()
    #     file_info.load_metadata()
    #     print(file_info.metadata_type)
    #     print(file_info.axes)
    #     print(file_info.shape)
    #     print(file_info.dim_res)
    #     print('\n\n')

    test_file = all_paths[1]
    file_info = FileInfo(test_file)
    file_info.find_metadata()
    file_info.load_metadata()
    print(f'{file_info.metadata_type=}')
    print(f'{file_info.axes=}')
    print(f'{file_info.shape=}')
    print(f'{file_info.dim_res=}')
    print(f'{file_info.good_axes=}')
    print(f'{file_info.good_dims=}')
    print('\n')

    file_info.change_axes('TZYX')
    print('Axes changed')
    print(f'{file_info.axes=}')
    print(f'{file_info.dim_res=}')
    print(f'{file_info.good_axes=}')
    print(f'{file_info.good_dims=}')
    print('\n')

    file_info.change_dim_res('T', 0.5)
    file_info.change_dim_res('Z', 0.2)

    print('Dimension resolutions changed')
    print(f'{file_info.axes=}')
    print(f'{file_info.dim_res=}')
    print(f'{file_info.good_axes=}')
    print(f'{file_info.good_dims=}')
    print('\n')

    # print(f'{file_info.ch=}')
    # file_info.change_selected_channel(3)
    # print('Channel changed')
    # print(f'{file_info.ch=}')

    # print(f'{file_info.t_start=}')
    # print(f'{file_info.t_end=}')
    # file_info.select_temporal_range(1, 3)
    # print('Temporal range selected')
    # print(f'{file_info.t_start=}')
    # print(f'{file_info.t_end=}')

    # file_info.save_ome_tiff()
