"""Per-extractor + factory tests for ``nellie.im_info.extractors``.

Slice 6 of the verifier dechaos refactor (see
``wiki/outputs/dechaos-verifier.md`` Pass 8 Slice 6). The legacy
6-method per-format dispatch on ``FileInfo`` was extracted into 5
``MetadataExtractor`` dataclasses + a ``detect_extractor`` factory.

Test layout:

- **Per-extractor unit tests** (sections 1-5): construct each extractor
  with **synthetic dataclass arguments** (no filesystem needed) and
  pin the per-format ``parse_dim_res`` outputs. This is the
  pure-logic surface — adding a new ND2 calibration fallback or
  changing the ResolutionUnit scaling table should be testable here
  without writing a TIFF/ND2 file.
- **Factory tests** (section 6): one per fixture in conftest's
  ``verifier_fixture_paths`` dict, asserting both the returned class
  type AND the ``metadata_type`` discriminator string. These exercise
  the file-classification branches end-to-end against real fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from tifffile import tifffile

from nellie.im_info.extractors import (
    ImageJExtractor,
    ImageJTifTagExtractor,
    Nd2Extractor,
    OmeExtractor,
    RawTiffTagExtractor,
    detect_extractor,
)


# ============================================================
# Test helpers — minimal stand-ins for the format-specific raw shapes
# ============================================================


@dataclass
class _FakeTag:
    """Stand-in for a ``tifffile.TiffTag`` — only ``name`` and ``value`` matter."""
    name: str
    value: Any


def _build_tags(entries: dict[str, Any]) -> dict[int, _FakeTag]:
    """Build a ``tags`` dict of the shape ``RawTiffTagExtractor.tags`` expects.

    The legacy ``_get_tif_tags_metadata`` builds a ``tag_names`` index
    by iterating ``metadata.items()`` and reading ``.name`` off the
    value, then looks values back up via the ``code → tag`` mapping.
    The codes themselves don't matter for the projection — they are an
    opaque key — so we synthesize sequential ints.
    """
    return {
        100 + i: _FakeTag(name=name, value=value)
        for i, (name, value) in enumerate(entries.items())
    }


@dataclass
class _FakeOmePixels:
    physical_size_x: float | None
    physical_size_y: float | None
    physical_size_z: float | None
    time_increment: float | None


@dataclass
class _FakeOmeImage:
    pixels: _FakeOmePixels


@dataclass
class _FakeOme:
    images: list[_FakeOmeImage]


def _build_ome(x: float | None, y: float | None, z: float | None, t: float | None) -> _FakeOme:
    """Build a fake OME object whose ``images[0].pixels`` reads cleanly."""
    return _FakeOme(images=[_FakeOmeImage(pixels=_FakeOmePixels(x, y, z, t))])


@dataclass
class _FakeNd2Volume:
    axesCalibration: tuple | list | None


@dataclass
class _FakeNd2Channel:
    volume: Any


@dataclass
class _FakeNd2Root:
    volume: Any = None
    channels: Any = None


# ============================================================
# 1. OmeExtractor
# ============================================================


def test_ome_extractor_parses_all_four_dims() -> None:
    """All four pixels fields populated → all four dim_res entries set."""
    ome = _build_ome(x=0.0655, y=0.0655, z=0.25, t=4.5)
    extractor = OmeExtractor(ome=ome, axes='TZYX', shape=(2, 16, 192, 279))
    assert extractor.metadata_type == 'ome'
    assert extractor.parse_dim_res() == {
        'X': 0.0655, 'Y': 0.0655, 'Z': 0.25, 'T': 4.5,
    }


def test_ome_extractor_propagates_none_z() -> None:
    """A 2D OME (``physical_size_z`` is None) → dim_res['Z'] is None."""
    ome = _build_ome(x=0.1, y=0.1, z=None, t=2.0)
    extractor = OmeExtractor(ome=ome, axes='TYX', shape=(2, 192, 279))
    result = extractor.parse_dim_res()
    assert result['Z'] is None
    assert result == {'X': 0.1, 'Y': 0.1, 'Z': None, 'T': 2.0}


def test_ome_extractor_against_real_fixture() -> None:
    """End-to-end smoke: real OME-TIFF → factory → parse_dim_res matches conftest fixture."""
    fixture = Path(__file__).resolve().parent / 'fixtures' / 'yeast_3d_t0_to_1.ome.tif'
    extractor = detect_extractor(str(fixture))
    assert isinstance(extractor, OmeExtractor)
    assert extractor.parse_dim_res() == pytest.approx(
        {'X': 0.0655, 'Y': 0.0655, 'Z': 0.25, 'T': 4.535566806793213}
    )


# ============================================================
# 2. ImageJExtractor
# ============================================================


def test_imagej_extractor_full_dict_parses_all_four() -> None:
    """All four imagej keys present → all four dim_res entries set."""
    meta = {'physicalsizex': 0.108, 'physicalsizey': 0.108, 'spacing': 0.5, 'finterval': 2.0}
    extractor = ImageJExtractor(imagej_meta=meta, axes='YX', shape=(32, 32))
    assert extractor.metadata_type == 'imagej'
    assert extractor.parse_dim_res() == {
        'X': 0.108, 'Y': 0.108, 'Z': 0.5, 'T': 2.0,
    }


def test_imagej_extractor_missing_physicalsizex_returns_none_for_x() -> None:
    """``physicalsizex`` missing → dim_res['X'] is None (rest pulled normally)."""
    meta = {'physicalsizey': 0.108, 'spacing': 0.5, 'finterval': 2.0}
    extractor = ImageJExtractor(imagej_meta=meta, axes='YX', shape=(32, 32))
    result = extractor.parse_dim_res()
    assert result['X'] is None
    assert result['Y'] == 0.108


def test_imagej_extractor_missing_spacing_returns_none_for_z() -> None:
    """``spacing`` missing → dim_res['Z'] is None."""
    meta = {'physicalsizex': 0.108, 'physicalsizey': 0.108, 'finterval': 2.0}
    extractor = ImageJExtractor(imagej_meta=meta, axes='YX', shape=(32, 32))
    assert extractor.parse_dim_res()['Z'] is None


def test_imagej_extractor_missing_finterval_returns_none_for_t() -> None:
    """``finterval`` missing → dim_res['T'] is None."""
    meta = {'physicalsizex': 0.108, 'physicalsizey': 0.108, 'spacing': 0.5}
    extractor = ImageJExtractor(imagej_meta=meta, axes='YX', shape=(32, 32))
    assert extractor.parse_dim_res()['T'] is None


def test_imagej_extractor_empty_dict_returns_all_none() -> None:
    """Empty imagej_meta → all four dim_res entries are None."""
    extractor = ImageJExtractor(imagej_meta={}, axes='YX', shape=(32, 32))
    assert extractor.parse_dim_res() == {'X': None, 'Y': None, 'Z': None, 'T': None}


# ============================================================
# 3. ImageJTifTagExtractor
# ============================================================


def test_imagej_tif_tags_extractor_layers_imagej_then_tags() -> None:
    """imagej dict carries Y, raw tags carry X — both end up in dim_res."""
    imagej_meta = {'physicalsizey': 0.108}
    tif_tags = _build_tags({
        'XResolution': (10000, 1),  # → 1 / 10000 = 0.0001
        'ResolutionUnit': tifffile.RESUNIT.NONE,
    })
    extractor = ImageJTifTagExtractor(
        imagej_meta=imagej_meta, tif_tags=tif_tags, axes='YX', shape=(32, 32),
    )
    result = extractor.parse_dim_res()
    assert extractor.metadata_type == 'imagej_tif_tags'
    assert result['X'] == pytest.approx(0.0001)
    assert result['Y'] == 0.108
    assert result['Z'] is None
    assert result['T'] is None


def test_imagej_tif_tags_extractor_tags_overwrite_imagej_xy() -> None:
    """If both imagej AND tif tags carry X/Y, the raw-tag values win (legacy ordering)."""
    imagej_meta = {'physicalsizex': 999.0, 'physicalsizey': 999.0}
    tif_tags = _build_tags({
        'XResolution': (10000, 1),
        'YResolution': (10000, 1),
        'ResolutionUnit': tifffile.RESUNIT.CENTIMETER,
    })
    extractor = ImageJTifTagExtractor(
        imagej_meta=imagej_meta, tif_tags=tif_tags, axes='YX', shape=(32, 32),
    )
    result = extractor.parse_dim_res()
    # CENTIMETER scaling: 1/10000 * 1e4 = 1.0
    assert result['X'] == pytest.approx(1.0)
    assert result['Y'] == pytest.approx(1.0)


def test_imagej_tif_tags_extractor_against_real_fixture() -> None:
    """End-to-end: real ImageJ TIFF without ``physicalsizex`` → tif-tags fallback fires."""
    fixture = Path(__file__).resolve().parent / 'fixtures' / 'imagej_no_physicalsize.tif'
    extractor = detect_extractor(str(fixture))
    assert isinstance(extractor, ImageJTifTagExtractor)
    assert extractor.parse_dim_res() == {'X': 1.0, 'Y': 1.0, 'Z': None, 'T': None}


# ============================================================
# 4. Nd2Extractor
# ============================================================


def test_nd2_extractor_volume_axes_calibration_path() -> None:
    """root_meta.volume.axesCalibration → X/Y/Z populated."""
    root = _FakeNd2Root(volume=_FakeNd2Volume(axesCalibration=(0.1, 0.2, 0.5)))
    recorded = {'Time [s]': [0.0, 1.0, 2.0]}
    extractor = Nd2Extractor(
        root_meta=root, recorded_data=recorded, axes='TZYX', shape=(3, 4, 32, 32),
    )
    result = extractor.parse_dim_res()
    assert extractor.metadata_type == 'nd2'
    assert result['X'] == 0.1
    assert result['Y'] == 0.2
    assert result['Z'] == 0.5
    # Median diff of [0, 1, 2] is 1.0
    assert result['T'] == pytest.approx(1.0)


def test_nd2_extractor_channel_volume_fallback() -> None:
    """root_meta.volume.axesCalibration absent → fall back to channels[0].volume.axesCalibration."""
    fallback_volume = _FakeNd2Volume(axesCalibration=(0.3, 0.4, 0.7))
    root = _FakeNd2Root(volume=None, channels=[_FakeNd2Channel(volume=fallback_volume)])
    extractor = Nd2Extractor(
        root_meta=root, recorded_data={}, axes='ZYX', shape=(4, 32, 32),
    )
    result = extractor.parse_dim_res()
    assert result['X'] == 0.3
    assert result['Y'] == 0.4
    assert result['Z'] == 0.7
    # No timestamps → T is None
    assert result['T'] is None


def test_nd2_extractor_median_of_diffs_uses_median_not_mean() -> None:
    """Robust to outliers: timestamps [0, 1, 2, 100] have median diff 1.0 (mean would be 33.0)."""
    root = _FakeNd2Root()  # no axes calibration anywhere
    recorded = {'Time [s]': [0.0, 1.0, 2.0, 100.0]}
    extractor = Nd2Extractor(
        root_meta=root, recorded_data=recorded, axes='TYX', shape=(4, 32, 32),
    )
    assert extractor.parse_dim_res()['T'] == pytest.approx(1.0)


def test_nd2_extractor_single_timepoint_returns_none_for_t() -> None:
    """Only one timestamp → cannot compute diff → T is None."""
    extractor = Nd2Extractor(
        root_meta=_FakeNd2Root(),
        recorded_data={'Time [s]': [42.0]},
        axes='TYX',
        shape=(1, 32, 32),
    )
    assert extractor.parse_dim_res()['T'] is None


def test_nd2_extractor_missing_recorded_data_returns_all_none_for_t() -> None:
    """``recorded_data`` empty → ``Time [s]`` lookup returns None → T stays None."""
    extractor = Nd2Extractor(
        root_meta=_FakeNd2Root(),
        recorded_data={},
        axes='TYX',
        shape=(1, 32, 32),
    )
    result = extractor.parse_dim_res()
    assert result == {'X': None, 'Y': None, 'Z': None, 'T': None}


# ============================================================
# 5. RawTiffTagExtractor
# ============================================================


def test_raw_tiff_no_resunit_no_scaling() -> None:
    """RESUNIT.NONE: 1/10000 = 0.0001 (no scaling)."""
    tags = _build_tags({
        'XResolution': (10000, 1),
        'YResolution': (10000, 1),
        'ResolutionUnit': tifffile.RESUNIT.NONE,
    })
    extractor = RawTiffTagExtractor(tags=tags, axes='YX', shape=(32, 32))
    assert extractor.metadata_type is None
    result = extractor.parse_dim_res()
    assert result['X'] == pytest.approx(0.0001)
    assert result['Y'] == pytest.approx(0.0001)


def test_raw_tiff_centimeter_scales_by_1e4() -> None:
    """RESUNIT.CENTIMETER: (1/10000) * 1e4 = 1.0."""
    tags = _build_tags({
        'XResolution': (10000, 1),
        'YResolution': (10000, 1),
        'ResolutionUnit': tifffile.RESUNIT.CENTIMETER,
    })
    extractor = RawTiffTagExtractor(tags=tags, axes='YX', shape=(32, 32))
    result = extractor.parse_dim_res()
    assert result['X'] == pytest.approx(1.0)
    assert result['Y'] == pytest.approx(1.0)


def test_raw_tiff_inch_scales_by_25400() -> None:
    """RESUNIT.INCH: (1/10000) * 25400 = 2.54."""
    tags = _build_tags({
        'XResolution': (10000, 1),
        'YResolution': (10000, 1),
        'ResolutionUnit': tifffile.RESUNIT.INCH,
    })
    extractor = RawTiffTagExtractor(tags=tags, axes='YX', shape=(32, 32))
    result = extractor.parse_dim_res()
    assert result['X'] == pytest.approx(2.54)
    assert result['Y'] == pytest.approx(2.54)


def test_raw_tiff_z_resolution_only_when_z_in_axes() -> None:
    """ZResolution present but axes lack 'Z' → Z stays None."""
    tags = _build_tags({
        'XResolution': (10000, 1),
        'ZResolution': (4, 1),  # would yield Z=0.25 if gated open
    })
    yx_extractor = RawTiffTagExtractor(tags=tags, axes='YX', shape=(32, 32))
    assert yx_extractor.parse_dim_res()['Z'] is None
    zyx_extractor = RawTiffTagExtractor(tags=tags, axes='ZYX', shape=(4, 32, 32))
    assert zyx_extractor.parse_dim_res()['Z'] == pytest.approx(0.25)


def test_raw_tiff_frame_rate_only_when_t_in_axes() -> None:
    """FrameRate present but axes lack 'T' → T stays None."""
    tags = _build_tags({
        'XResolution': (10000, 1),
        'FrameRate': (2, 1),  # would yield T=0.5 if gated open
    })
    yx_extractor = RawTiffTagExtractor(tags=tags, axes='YX', shape=(32, 32))
    assert yx_extractor.parse_dim_res()['T'] is None
    tyx_extractor = RawTiffTagExtractor(tags=tags, axes='TYX', shape=(2, 32, 32))
    assert tyx_extractor.parse_dim_res()['T'] == pytest.approx(0.5)


# ============================================================
# 6. detect_extractor factory — one test per verifier_fixture_paths entry
# ============================================================


def test_factory_unsupported_extension_raises(tmp_path: Path) -> None:
    """Non-TIFF/ND2 extension raises ValueError."""
    bogus = tmp_path / 'foo.png'
    bogus.write_bytes(b'\x89PNG\r\n\x1a\n')
    with pytest.raises(ValueError, match='File type not supported'):
        detect_extractor(str(bogus))


def test_factory_returns_ome_for_3d_yeast(verifier_fixture_paths) -> None:
    extractor = detect_extractor(str(verifier_fixture_paths['ome']))
    assert isinstance(extractor, OmeExtractor)
    assert extractor.metadata_type == 'ome'


def test_factory_returns_ome_for_2d_yeast(verifier_fixture_paths) -> None:
    extractor = detect_extractor(str(verifier_fixture_paths['ome_2d']))
    assert isinstance(extractor, OmeExtractor)
    assert extractor.metadata_type == 'ome'


def test_factory_returns_imagej_for_imagej_with_physicalsize(verifier_fixture_paths) -> None:
    extractor = detect_extractor(str(verifier_fixture_paths['imagej']))
    assert isinstance(extractor, ImageJExtractor)
    assert extractor.metadata_type == 'imagej'


def test_factory_returns_imagej_tif_tags_for_imagej_no_physicalsize(verifier_fixture_paths) -> None:
    extractor = detect_extractor(str(verifier_fixture_paths['imagej_tif_tags']))
    assert isinstance(extractor, ImageJTifTagExtractor)
    assert extractor.metadata_type == 'imagej_tif_tags'


def test_factory_returns_raw_tiff_for_no_resunit(verifier_fixture_paths) -> None:
    extractor = detect_extractor(str(verifier_fixture_paths['raw_no_resunit']))
    assert isinstance(extractor, RawTiffTagExtractor)
    assert extractor.metadata_type is None


def test_factory_returns_raw_tiff_for_centimeter(verifier_fixture_paths) -> None:
    extractor = detect_extractor(str(verifier_fixture_paths['raw_centimeter']))
    assert isinstance(extractor, RawTiffTagExtractor)
    assert extractor.metadata_type is None


def test_factory_returns_raw_tiff_for_inch(verifier_fixture_paths) -> None:
    extractor = detect_extractor(str(verifier_fixture_paths['raw_inch']))
    assert isinstance(extractor, RawTiffTagExtractor)
    assert extractor.metadata_type is None
