from dataclasses import dataclass

from qtpy.QtWidgets import (
    QWidget,
    QCheckBox,
    QSpinBox,
    QLabel,
    QVBoxLayout,
    QGroupBox,
    QHBoxLayout,
    QFormLayout,
)


@dataclass
class SettingsConfig:
    """
    Structured representation of the Settings UI state.
    """
    remove_edges: bool
    remove_intermediates: bool
    voxel_reassign: bool
    analyze_node_level: bool
    track_all_frames: bool
    subsample_voxels: bool
    skip_vox: int


class Settings(QWidget):
    """
    The Settings class provides a user interface for configuring various options and
    settings for the Nellie pipeline and visualizations. Users can enable or disable
    specific processing options, control track visualization settings, and configure
    voxel visualization parameters.

    Attributes
    ----------
    nellie : object
        Reference to the Nellie instance managing the pipeline.
    viewer : napari.viewer.Viewer
        Reference to the napari viewer instance.
    remove_edges_checkbox : QCheckBox
        Checkbox for enabling or disabling the removal of image edges.
    remove_intermediates_checkbox : QCheckBox
        Checkbox for enabling or disabling the removal of intermediate files
        after processing.
    voxel_reassign : QCheckBox
        Checkbox to enable or disable the automatic voxel reassignment step
        after tracking.
    analyze_node_level : QCheckBox
        Checkbox to enable or disable node-level analysis during feature extraction.
    track_all_frames : QCheckBox
        Checkbox to enable or disable the visualization of voxel tracks for all frames.
    subsample_voxels_checkbox : QCheckBox
        Checkbox to enable or disable subsampling of voxel tracks.
    skip_vox_label : QLabel
        Label describing the setting for skipping voxels during track visualization.
    skip_vox : QSpinBox
        Spin box for selecting the value of N to visualize tracks for every Nth voxel.
    initialized : bool
        Flag to indicate whether the settings interface has been initialized.
    """

    def __init__(self, napari_viewer: "napari.viewer.Viewer", nellie, parent=None):
        """
        Initialize the Settings class, setting up the user interface and options
        for configuring processing and track visualization.

        Parameters
        ----------
        napari_viewer : napari.viewer.Viewer
            Reference to the napari viewer instance.
        nellie : object
            Reference to the Nellie instance that manages the pipeline.
        parent : QWidget, optional
            Optional parent widget (default is None).
        """
        super().__init__(parent)
        self.nellie = nellie
        self.viewer = napari_viewer

        # Processor settings
        self.remove_edges_checkbox = QCheckBox("Remove image edges")
        self.remove_edges_checkbox.setChecked(False)
        self.remove_edges_checkbox.setEnabled(True)
        self.remove_edges_checkbox.setToolTip(
            "Originally for Snouty deskewed images. Enable this if you see "
            "weird image edge artifacts."
        )

        self.remove_intermediates_checkbox = QCheckBox("Remove intermediate files")
        self.remove_intermediates_checkbox.setChecked(False)
        self.remove_intermediates_checkbox.setEnabled(True)
        self.remove_intermediates_checkbox.setToolTip(
            "Remove intermediate files after processing. This means only CSV files "
            "will be saved and intermediate data cannot be recovered."
        )

        self.voxel_reassign = QCheckBox("Auto-run voxel reassignment")
        self.voxel_reassign.setChecked(True)
        self.voxel_reassign.setEnabled(True)
        self.voxel_reassign.setToolTip(
            "Automatically run the voxel reassignment step after tracking to refine "
            "track-voxel associations."
        )

        # Analyze node level
        self.analyze_node_level = QCheckBox("Analyze node level (slow)")
        self.analyze_node_level.setChecked(True)
        self.analyze_node_level.setEnabled(True)
        self.analyze_node_level.setToolTip(
            "Compute node-level features (e.g., per-timepoint statistics) for each "
            "track. This provides more detailed analysis but can be significantly "
            "slower for large datasets."
        )

        # Track visualization settings
        self.track_all_frames = QCheckBox("Visualize all frames' voxel tracks")
        self.track_all_frames.setChecked(True)
        self.track_all_frames.setEnabled(True)
        self.track_all_frames.setToolTip(
            "If checked, show voxel tracks for all time points. Disable this if you "
            "only want to visualize tracks for a subset of frames elsewhere."
        )

        # Subsampling of voxel tracks
        self.subsample_voxels_checkbox = QCheckBox("Subsample voxel tracks")
        self.subsample_voxels_checkbox.setChecked(True)
        self.subsample_voxels_checkbox.setEnabled(True)
        self.subsample_voxels_checkbox.setToolTip(
            "If enabled, only every Nth voxel track will be visualized to reduce "
            "visual clutter and improve performance."
        )

        # Label above the spin box
        self.skip_vox_label = QLabel(
            "Only visualize tracks for every Nth voxel (N =)"
        )

        self.skip_vox = QSpinBox()
        self.skip_vox.setRange(1, 10000)
        self.skip_vox.setValue(5)
        self.skip_vox.setEnabled(False)
        self.skip_vox.setToolTip(
            "Visualize only every Nth voxel track. Larger values reduce clutter and "
            "can improve performance."
        )

        self.initialized = False

        self.set_ui()
        self._connect_signals()

    def post_init(self):
        """
        Post-initialization method that sets the initialized flag to True.

        This can be called by the plugin infrastructure once all external components
        are set up and the Settings widget is fully integrated.
        """
        self.initialized = True

    def set_ui(self):
        """
        Initialize and set the layout and UI components for the Settings class.
        This includes checkboxes for configuring the processing pipeline and track
        visualization options, as well as a spin box for setting voxel track
        visualization parameters.
        """
        main_layout = QVBoxLayout()

        # Processor settings group
        processor_group = QGroupBox("Processor settings")
        processor_layout = QVBoxLayout()

        subprocessor_layout1 = QHBoxLayout()
        subprocessor_layout1.addWidget(self.remove_intermediates_checkbox)
        subprocessor_layout1.addWidget(self.remove_edges_checkbox)

        subprocessor_layout2 = QHBoxLayout()
        subprocessor_layout2.addWidget(self.analyze_node_level)
        subprocessor_layout2.addWidget(self.voxel_reassign)

        processor_layout.addLayout(subprocessor_layout1)
        processor_layout.addLayout(subprocessor_layout2)
        processor_group.setLayout(processor_layout)

        # Tracking settings group
        tracking_group = QGroupBox("Track visualization settings")
        tracking_layout = QFormLayout()
        tracking_layout.addRow(self.track_all_frames)
        tracking_layout.addRow(self.subsample_voxels_checkbox)
        tracking_layout.addRow(self.skip_vox_label, self.skip_vox)
        tracking_group.setLayout(tracking_layout)

        main_layout.addWidget(processor_group)
        main_layout.addWidget(tracking_group)
        main_layout.addStretch(1)

        self.setLayout(main_layout)

    def _connect_signals(self):
        """
        Connect internal signals/slots to keep the UI consistent.
        """
        self.subsample_voxels_checkbox.toggled.connect(
            self._update_skip_vox_enabled
        )

        # Ensure initial enabled/disabled state is consistent with the checkbox
        self._update_skip_vox_enabled(self.subsample_voxels_checkbox.isChecked())

    def _update_skip_vox_enabled(self, checked: bool):
        """
        Enable or disable the skip_vox spin box based on the subsample checkbox.

        Parameters
        ----------
        checked : bool
            Whether subsampling is enabled.
        """
        self.skip_vox.setEnabled(checked)

    # -------------------------------------------------------------------------
    # Public helpers for integration with the rest of the plugin
    # -------------------------------------------------------------------------

    def to_config(self) -> SettingsConfig:
        """
        Return the current UI state as a SettingsConfig dataclass.

        Returns
        -------
        SettingsConfig
            Dataclass capturing the current configuration of the settings widget.
        """
        return SettingsConfig(
            remove_edges=self.remove_edges_checkbox.isChecked(),
            remove_intermediates=self.remove_intermediates_checkbox.isChecked(),
            voxel_reassign=self.voxel_reassign.isChecked(),
            analyze_node_level=self.analyze_node_level.isChecked(),
            track_all_frames=self.track_all_frames.isChecked(),
            subsample_voxels=self.subsample_voxels_checkbox.isChecked(),
            skip_vox=self.skip_vox.value(),
        )

    def apply_config(self, config: SettingsConfig):
        """
        Apply a SettingsConfig instance to the UI.

        Parameters
        ----------
        config : SettingsConfig
            Configuration to apply to the UI.
        """
        self.remove_edges_checkbox.setChecked(config.remove_edges)
        self.remove_intermediates_checkbox.setChecked(config.remove_intermediates)
        self.voxel_reassign.setChecked(config.voxel_reassign)
        self.analyze_node_level.setChecked(config.analyze_node_level)
        self.track_all_frames.setChecked(config.track_all_frames)
        self.subsample_voxels_checkbox.setChecked(config.subsample_voxels)
        self.skip_vox.setValue(config.skip_vox)
        # Ensure the skip_vox enabled state matches the incoming config
        self._update_skip_vox_enabled(config.subsample_voxels)


if __name__ == "__main__":
    import napari

    viewer = napari.Viewer()
    # You can pass a real Nellie instance instead of None when integrating.
    settings = Settings(viewer, nellie=None)
    viewer.window.add_dock_widget(settings, area="right")
    napari.run()