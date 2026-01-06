import numpy as np
import ome_types
from tifffile import tifffile

from nellie.im_info.verifier import FileInfo, ImInfo


def test_allocate_memory_updates_ome_metadata(tmp_path):
    base_path = tmp_path / "base.ome.tif"
    base_data = np.zeros((1, 2, 3), dtype=np.uint16)
    tifffile.imwrite(
        str(base_path),
        base_data,
        metadata={"axes": "TYX"},
        photometric="minisblack",
    )

    file_info = FileInfo(str(tmp_path / "input.tif"))
    file_info.axes = "TYX"
    file_info.shape = base_data.shape
    file_info.dim_res = {"X": 1.0, "Y": 1.0, "Z": None, "T": 1.0}
    file_info.t_start = 0
    file_info.t_end = 0
    file_info.ch = 0
    file_info._get_output_path()
    file_info.ome_output_path = str(base_path)

    im_info = ImInfo(file_info)

    new_data = np.zeros((1, 2, 4), dtype=np.float32)
    output_path = tmp_path / "allocated.ome.tif"
    im_info.allocate_memory(str(output_path), data=new_data, description="allocated")

    ome = ome_types.from_xml(tifffile.tiffcomment(str(output_path)))
    pixels = ome.images[0].pixels
    assert pixels.size_x == 4
    assert pixels.size_y == 2
    assert pixels.size_t == 1
    assert pixels.type.value == "float"
    assert ome.images[0].description == "allocated"


def test_iminfo_adds_t_axis_when_missing(tmp_path):
    path = tmp_path / "no_t_axis.ome.tif"
    data = np.zeros((2, 3), dtype=np.uint16)
    tifffile.imwrite(
        str(path),
        data,
        metadata={"axes": "YX"},
        photometric="minisblack",
    )

    file_info = FileInfo(str(tmp_path / "input.tif"))
    file_info.axes = "YX"
    file_info.shape = data.shape
    file_info.dim_res = {"X": 1.0, "Y": 1.0, "Z": None, "T": None}
    file_info.t_start = 0
    file_info.t_end = 0
    file_info.ch = 0
    file_info._get_output_path()
    file_info.ome_output_path = str(path)

    im_info = ImInfo(file_info)

    assert im_info.new_axes == "TYX"
    assert im_info.shape == (1, 2, 3)
