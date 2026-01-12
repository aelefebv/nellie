import numpy as np
import pytest
from tifffile import tifffile

from nellie.im_info.verifier import FileInfo


class DummyTag:
    def __init__(self, name, value):
        self.name = name
        self.value = value


class DummyVolume:
    def __init__(self, axes_calibration):
        self.axesCalibration = axes_calibration


class DummyChannel:
    def __init__(self, axes_calibration):
        self.volume = DummyVolume(axes_calibration)


class DummyNd2Metadata:
    def __init__(self, axes_calibration=None, channels=None):
        self.volume = DummyVolume(axes_calibration) if axes_calibration is not None else None
        self.channels = channels or []


@pytest.mark.parametrize(
    ("unit", "expected_scale"),
    [
        (tifffile.RESUNIT.CENTIMETER, 1e4),
        (tifffile.RESUNIT.INCH, 25400),
    ],
)
def test_tif_tag_resolution_unit_scaling(tmp_path, unit, expected_scale):
    file_info = FileInfo(str(tmp_path / "dummy.tif"))
    file_info.axes = "YX"
    file_info.dim_res = {"X": None, "Y": None, "Z": None, "T": None}

    metadata = {
        282: DummyTag("XResolution", (2, 1)),
        283: DummyTag("YResolution", (4, 1)),
        296: DummyTag("ResolutionUnit", unit),
    }

    file_info._get_tif_tags_metadata(metadata)

    assert file_info.dim_res["X"] == pytest.approx((1 / 2) * expected_scale)
    assert file_info.dim_res["Y"] == pytest.approx((1 / 4) * expected_scale)


def test_tif_tag_resolution_unit_missing_no_scaling(tmp_path):
    file_info = FileInfo(str(tmp_path / "dummy.tif"))
    file_info.axes = "YX"
    file_info.dim_res = {"X": None, "Y": None, "Z": None, "T": None}

    metadata = {
        282: DummyTag("XResolution", (2, 1)),
        283: DummyTag("YResolution", (4, 1)),
    }

    file_info._get_tif_tags_metadata(metadata)

    assert file_info.dim_res["X"] == pytest.approx(1 / 2)
    assert file_info.dim_res["Y"] == pytest.approx(1 / 4)


def test_nd2_time_increment_uses_median_diff(tmp_path):
    file_info = FileInfo(str(tmp_path / "dummy.nd2"))
    file_info.axes = "TZYX"
    file_info.dim_res = {"X": None, "Y": None, "Z": None, "T": None}

    timestamps = [0.0, 1.0, 2.2, 3.1]
    metadata = {
        "root": DummyNd2Metadata(axes_calibration=[0.2, 0.2, 0.5]),
        "recorded_data": {"Time [s]": timestamps},
    }

    file_info._get_nd2_metadata(metadata)

    assert file_info.dim_res["T"] == pytest.approx(np.median(np.diff(timestamps)))
    assert file_info.dim_res["X"] == pytest.approx(0.2)
    assert file_info.dim_res["Y"] == pytest.approx(0.2)
    assert file_info.dim_res["Z"] == pytest.approx(0.5)


def test_nd2_time_increment_missing_or_single_timepoint(tmp_path):
    file_info = FileInfo(str(tmp_path / "dummy.nd2"))
    file_info.axes = "TYX"
    file_info.dim_res = {"X": None, "Y": None, "Z": None, "T": None}

    metadata = {
        "root": DummyNd2Metadata(axes_calibration=[0.2, 0.2, 0.5]),
        "recorded_data": {"Time [s]": [0.0]},
    }

    file_info._get_nd2_metadata(metadata)

    assert file_info.dim_res["T"] is None
    assert file_info.dim_res["X"] == pytest.approx(0.2)
    assert file_info.dim_res["Y"] == pytest.approx(0.2)
    assert file_info.dim_res["Z"] == pytest.approx(0.5)


def test_nd2_time_increment_missing_time_key(tmp_path):
    file_info = FileInfo(str(tmp_path / "dummy.nd2"))
    file_info.axes = "TYX"
    file_info.dim_res = {"X": None, "Y": None, "Z": None, "T": None}

    metadata = {
        "root": DummyNd2Metadata(axes_calibration=[0.2, 0.2, 0.5]),
        "recorded_data": {},
    }

    file_info._get_nd2_metadata(metadata)

    assert file_info.dim_res["T"] is None
    assert file_info.dim_res["X"] == pytest.approx(0.2)
    assert file_info.dim_res["Y"] == pytest.approx(0.2)
    assert file_info.dim_res["Z"] == pytest.approx(0.5)


def test_nd2_axes_calibration_fallback_to_channel(tmp_path):
    file_info = FileInfo(str(tmp_path / "dummy.nd2"))
    file_info.axes = "ZYX"
    file_info.dim_res = {"X": None, "Y": None, "Z": None, "T": None}

    metadata = {
        "root": DummyNd2Metadata(channels=[DummyChannel([0.1, 0.2, 0.3])]),
        "recorded_data": {},
    }

    file_info._get_nd2_metadata(metadata)

    assert file_info.dim_res["X"] == pytest.approx(0.1)
    assert file_info.dim_res["Y"] == pytest.approx(0.2)
    assert file_info.dim_res["Z"] == pytest.approx(0.3)


def test_change_dim_res_invalid_dimension_raises(tmp_path):
    file_info = FileInfo(str(tmp_path / "dummy.tif"))
    file_info.dim_res = {"X": None, "Y": None, "Z": None, "T": None}

    with pytest.raises(ValueError, match="Invalid dimension"):
        file_info.change_dim_res("Q", 1.0)


def test_select_temporal_range_requires_t_axis(tmp_path):
    file_info = FileInfo(str(tmp_path / "dummy.tif"))
    file_info.axes = "YX"
    file_info.shape = (2, 2)

    with pytest.raises(KeyError, match="No time dimension"):
        file_info.select_temporal_range(0, 1)


def test_validate_preserves_time_range(tmp_path):
    file_info = FileInfo(str(tmp_path / "dummy.tif"))
    file_info.axes = "TYX"
    file_info.shape = (5, 2, 2)
    file_info.dim_res = {"X": 1.0, "Y": 1.0, "Z": None, "T": 1.0}
    file_info.t_start = 1
    file_info.t_end = 3

    file_info._validate()

    assert file_info.t_start == 1
    assert file_info.t_end == 3
