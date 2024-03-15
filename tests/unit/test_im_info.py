import os
import shutil
import tempfile
from nellie import ImInfo
# todo tests for different filetypes or input types... Think of others

def test_iminfo_get_metadata_5d():
    im_path = 'data/tczyx_ex.tif'
    output_dir = tempfile.TemporaryDirectory().name
    iminfo = ImInfo(im_path, output_dir)
    assert iminfo.axes == 'TCZYX'
    assert iminfo.shape == (7, 3, 5, 167, 439)
    shutil.rmtree(output_dir)


def test_iminfo_get_metadata_3d():
    im_path = 'data/tyx_ex.tif'
    output_dir = tempfile.TemporaryDirectory().name
    iminfo = ImInfo(im_path, output_dir)
    assert iminfo.axes == 'TYX'
    assert iminfo.shape == (7, 167, 439)
    shutil.rmtree(output_dir)


def test_iminfo_get_dim_sizes_5d():
    im_path = 'data/tczyx_ex.tif'
    output_dir = tempfile.TemporaryDirectory().name
    iminfo = ImInfo(im_path, output_dir)
    assert iminfo.dim_sizes['X'] is None
    assert iminfo.dim_sizes['Y'] is None
    assert iminfo.dim_sizes['Z'] is None
    assert iminfo.dim_sizes['T'] is None
    assert iminfo.dim_sizes['C'] == 1
    shutil.rmtree(output_dir)


def test_iminfo_get_dim_sizes_3d():
    im_path = 'data/tyx_ex.tif'
    output_dir = tempfile.TemporaryDirectory().name
    iminfo = ImInfo(im_path, output_dir)
    assert iminfo.dim_sizes['X'] is None
    assert iminfo.dim_sizes['Y'] is None
    assert iminfo.dim_sizes['Z'] is None
    assert iminfo.dim_sizes['T'] is None
    assert iminfo.dim_sizes['C'] == 1
    shutil.rmtree(output_dir)


def test_iminfo_create_output_dirs_5d():
    im_path = 'data/tczyx_ex.tif'
    output_dir = tempfile.TemporaryDirectory().name
    iminfo = ImInfo(im_path, output_dir)
    assert os.path.isdir(iminfo.output_dirpath)
    assert os.path.isdir(iminfo.output_images_dirpath)
    assert os.path.isdir(iminfo.output_pickles_dirpath)
    assert os.path.isdir(iminfo.output_csv_dirpath)
    shutil.rmtree(output_dir)


def test_iminfo_create_output_dirs_3d():
    im_path = 'data/tyx_ex.tif'
    output_dir = tempfile.TemporaryDirectory().name
    iminfo = ImInfo(im_path, output_dir)
    assert os.path.isdir(iminfo.output_dirpath)
    assert os.path.isdir(iminfo.output_images_dirpath)
    assert os.path.isdir(iminfo.output_pickles_dirpath)
    assert os.path.isdir(iminfo.output_csv_dirpath)
    shutil.rmtree(output_dir)


def test_iminfo_init_5d():
    im_path = 'data/tczyx_ex.tif'
    output_dir = tempfile.TemporaryDirectory().name
    iminfo = ImInfo(im_path, output_dir, ch=1, dim_sizes={'X': 0.1, 'Y': 0.2, 'Z': 0.3, 'T': 0.4, 'C': 1})

    # test init method
    assert iminfo.im_path == im_path
    assert iminfo.ch == 1
    assert iminfo.dim_sizes['X'] == 0.1
    assert iminfo.dim_sizes['Y'] == 0.2
    assert iminfo.dim_sizes['Z'] == 0.3
    assert iminfo.dim_sizes['T'] == 0.4
    assert iminfo.dim_sizes['C'] == 1
    shutil.rmtree(output_dir)


def test_iminfo_init_3d():
    im_path = 'data/tyx_ex.tif'
    output_dir = tempfile.TemporaryDirectory().name
    iminfo = ImInfo(im_path, output_dir, ch=1, dim_sizes={'X': 0.1, 'Y': 0.2, 'T': 0.4, 'C': 1})

    # test init method
    assert iminfo.im_path == im_path
    assert iminfo.ch == 1
    assert iminfo.dim_sizes['X'] == 0.1
    assert iminfo.dim_sizes['Y'] == 0.2
    assert iminfo.dim_sizes['T'] == 0.4
    assert iminfo.dim_sizes['C'] == 1
    shutil.rmtree(output_dir)
