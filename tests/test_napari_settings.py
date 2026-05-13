"""Tests for the napari Settings widget's intermediate-retention surface.

Slice 3 of #245 replaces the legacy single ``remove_intermediates_checkbox``
with a per-output ``IntermediateRetentionGroup`` and switches the
``SettingsConfig`` schema field from ``remove_intermediates: bool`` to
``cleanup_drop_keys: frozenset[str]``.

These tests pin the contract via the public ``IntermediateRetentionGroup``
API (``drop_keys()`` / ``set_drop_keys(...)``) and the
``SettingsConfig`` round-trip. We instantiate the widget directly (no
napari viewer needed) to keep tests fast and isolated. Each test
constructs its own widget instance — the widget is cheap to create
once a single ``QApplication`` exists for the test session.
"""

from __future__ import annotations

import pytest

# Skip the whole module when the napari/Qt stack isn't installed
# (e.g., a headless `uv sync` without the `gui` extra) or when Qt's
# display libs aren't available (typical of headless Linux CI without
# libEGL/libgl). The widget tests need ``QApplication``; the
# dataclass-only tests transitively import qtpy via
# ``nellie_napari.nellie_settings``.
pytest.importorskip("qtpy.QtWidgets")
pytest.importorskip("nellie_napari.nellie_settings")

from nellie.im_info.verifier import (
    CSVS_ONLY_PRESET,
    DROPPABLE_KEYS,
    KEEP_EVERYTHING_PRESET,
    MASKS_AND_CSVS_PRESET,
)


@pytest.fixture(scope="session")
def qapp():
    """Provide a QApplication for the test session.

    pytest-qt is not a project dependency; this fixture is the minimum
    needed to instantiate QWidgets in isolation.
    """
    from qtpy.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app
    # Don't quit — other tests in the session may need the app.


@pytest.fixture
def retention_group(qapp):
    from nellie_napari.nellie_settings import IntermediateRetentionGroup
    return IntermediateRetentionGroup()


# ---------------------------------------------------------------------------
# IntermediateRetentionGroup
# ---------------------------------------------------------------------------


def test_default_drop_keys_is_empty_frozenset(retention_group) -> None:
    """Fresh widget defaults to "Keep everything" (empty drop set)."""
    assert retention_group.drop_keys() == frozenset()


def test_default_preset_radio_is_keep_everything(retention_group) -> None:
    """The "Keep everything" preset radio is the default selection."""
    assert retention_group._preset_buttons['Keep everything'].isChecked()
    assert not retention_group._preset_buttons['Custom'].isChecked()


def test_widget_has_one_checkbox_per_droppable_key(retention_group) -> None:
    """Every key in DROPPABLE_KEYS has a checkbox; no extras, no missing."""
    assert set(retention_group._key_checkboxes.keys()) == DROPPABLE_KEYS


def test_widget_checkboxes_carry_tooltips(retention_group) -> None:
    """Each checkbox tooltip is non-empty (PRD User Story 10)."""
    for key, cb in retention_group._key_checkboxes.items():
        assert cb.toolTip(), f"checkbox for {key!r} has no tooltip"


@pytest.mark.parametrize('preset_name,preset_set', [
    ('Keep everything', KEEP_EVERYTHING_PRESET),
    ('Masks + CSVs only', MASKS_AND_CSVS_PRESET),
    ('CSVs only', CSVS_ONLY_PRESET),
])
def test_clicking_preset_radio_sets_drop_keys(
    retention_group, preset_name: str, preset_set: frozenset[str],
) -> None:
    """Clicking each named preset snaps the drop set to the preset's value."""
    retention_group._preset_buttons[preset_name].setChecked(True)
    assert retention_group.drop_keys() == preset_set


@pytest.mark.parametrize('preset_set', [
    KEEP_EVERYTHING_PRESET,
    MASKS_AND_CSVS_PRESET,
    CSVS_ONLY_PRESET,
])
def test_set_drop_keys_round_trip_preserves_preset(
    retention_group, preset_set: frozenset[str],
) -> None:
    """``set_drop_keys`` then ``drop_keys`` round-trips the preset."""
    retention_group.set_drop_keys(preset_set)
    assert retention_group.drop_keys() == preset_set


def test_set_drop_keys_with_named_preset_selects_matching_radio(
    retention_group,
) -> None:
    """Loading a named preset's drop set selects the corresponding radio."""
    retention_group.set_drop_keys(MASKS_AND_CSVS_PRESET)
    assert retention_group._preset_buttons['Masks + CSVs only'].isChecked()
    assert not retention_group._preset_buttons['Custom'].isChecked()


def test_set_drop_keys_with_non_preset_set_selects_custom_radio(
    retention_group,
) -> None:
    """A drop set that doesn't match any named preset selects "Custom"."""
    retention_group.set_drop_keys(frozenset({'im_preprocessed'}))
    assert retention_group._preset_buttons['Custom'].isChecked()
    for name in ('Keep everything', 'Masks + CSVs only', 'CSVs only'):
        assert not retention_group._preset_buttons[name].isChecked(), (
            f"named preset {name!r} should not be selected for a custom set"
        )


def test_toggling_checkbox_after_preset_switches_to_custom(
    retention_group,
) -> None:
    """Toggling any checkbox after a preset selection flips to "Custom"."""
    retention_group.set_drop_keys(KEEP_EVERYTHING_PRESET)
    assert retention_group._preset_buttons['Keep everything'].isChecked()

    retention_group._key_checkboxes['im_preprocessed'].setChecked(True)
    assert retention_group._preset_buttons['Custom'].isChecked()
    assert not retention_group._preset_buttons['Keep everything'].isChecked()


def test_toggling_checkbox_back_to_preset_state_re_selects_preset(
    retention_group,
) -> None:
    """Toggling all the way back to a preset's drop set re-selects that preset."""
    retention_group.set_drop_keys(KEEP_EVERYTHING_PRESET)
    cb = retention_group._key_checkboxes['im_preprocessed']
    cb.setChecked(True)
    assert retention_group._preset_buttons['Custom'].isChecked()
    cb.setChecked(False)
    assert retention_group._preset_buttons['Keep everything'].isChecked()


def test_set_drop_keys_rejects_unknown_keys(retention_group) -> None:
    """Unknown keys raise AssertionError (CSV keys also rejected)."""
    with pytest.raises(AssertionError):
        retention_group.set_drop_keys(frozenset({'not_a_real_key'}))
    with pytest.raises(AssertionError):
        retention_group.set_drop_keys(frozenset({'features_voxels'}))


# ---------------------------------------------------------------------------
# SettingsConfig schema (no widget needed)
# ---------------------------------------------------------------------------


def test_settings_config_has_cleanup_drop_keys_field() -> None:
    """``SettingsConfig`` exposes ``cleanup_drop_keys: frozenset[str]``."""
    from dataclasses import fields
    from nellie_napari.nellie_settings import SettingsConfig

    field_names = {f.name for f in fields(SettingsConfig)}
    assert 'cleanup_drop_keys' in field_names
    # Legacy field is gone.
    assert 'remove_intermediates' not in field_names


def test_settings_config_cleanup_drop_keys_accepts_frozenset() -> None:
    """Construction with a frozenset[str] cleanup_drop_keys works."""
    # Build a minimal config; only assert the cleanup_drop_keys field
    # round-trips (other fields are unrelated to this slice).
    from dataclasses import fields
    from nellie_napari.nellie_settings import SettingsConfig

    # All field names except cleanup_drop_keys (we'll set those to dummies).
    other_kwargs = {}
    for f in fields(SettingsConfig):
        if f.name == 'cleanup_drop_keys':
            continue
        # Best-effort dummy by type annotation.
        ann = f.type
        if ann == 'bool' or ann is bool:
            other_kwargs[f.name] = False
        elif ann == 'int' or ann is int:
            other_kwargs[f.name] = 0
        elif ann == 'float' or ann is float:
            other_kwargs[f.name] = 0.0
        elif ann == 'str' or ann is str:
            other_kwargs[f.name] = ''
        else:
            other_kwargs[f.name] = None
    cfg = SettingsConfig(cleanup_drop_keys=CSVS_ONLY_PRESET, **other_kwargs)
    assert cfg.cleanup_drop_keys == CSVS_ONLY_PRESET
    assert isinstance(cfg.cleanup_drop_keys, frozenset)
