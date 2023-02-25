import os
import numpy as np
from src.io.im_info import ImInfo
import tifffile


def test_im_info():
    # create a temporary tif file for testing
    data = np.zeros((3, 3, 3, 3, 3), dtype=np.uint8)
    tif_file = "./data/test_5d.tif"
    tifffile.imwrite(tif_file, data, imagej=True,
                     metadata={'axes': 'TZCYX', 'physicalsizex': 0.1, 'physicalsizey': 0.2,
                               'spacing': 0.5, 'finterval': 0.1})

    # test creating an ImInfo object
    im_info = ImInfo(tif_file)
    assert im_info.im_path == tif_file
    assert im_info.ch == 0
    assert im_info.extension == "tif"
    assert im_info.filename == "test_5d"
    assert im_info.dirname == "data"
    assert im_info.input_dirpath == "./data"
    assert im_info.metadata is not None
    assert im_info.axes == "TZCYX"
    assert im_info.shape == (3, 3, 3, 3, 3)
    assert im_info.output_dirpath is not None
    assert im_info.output_images_dirpath is not None
    assert im_info.output_pickles_dirpath is not None
    assert im_info.output_csv_dirpath is not None
    assert im_info.path_im_frangi is not None
    assert im_info.path_im_mask is not None
    assert im_info.path_im_skeleton is not None
    assert im_info.path_im_label_obj is not None
    assert im_info.path_im_label_seg is not None
    assert im_info.path_im_network is not None
    assert im_info.path_im_event is not None
    assert im_info.path_pickle_obj is not None
    assert im_info.path_pickle_seg is not None
    assert im_info.path_pickle_track is not None

    # test creating an ImInfo object with output directory
    output_dir = "test_output"
    im_info = ImInfo(tif_file, output_dirpath=output_dir)
    assert im_info.output_dirpath == os.path.join(output_dir, "output")

    # test creating an ImInfo object with channel index and dimension sizes
    dim_sizes = {'x': 0.1, 'y': 0.2, 'z': 0.5, 't': 1.0}
    im_info = ImInfo(tif_file, ch=1, dim_sizes=dim_sizes)
    assert im_info.ch == 1
    assert im_info.dim_sizes == dim_sizes

    # test creating an ImInfo object with invalid tif file
    invalid_tif_file = "invalid.tif"
    try:
        im_info = ImInfo(invalid_tif_file)
    except SystemExit:
        pass  # expected exit
    else:
        assert False, "SystemExit not raised for invalid tif file"

    # clean up temporary tif file
    os.remove(tif_file)
