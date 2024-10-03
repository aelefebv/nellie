from qtpy.QtWidgets import QWidget, QCheckBox, QSpinBox, QLabel, QVBoxLayout, QGroupBox, QHBoxLayout
import napari


class Settings(QWidget):
    """
    The Settings class provides a user interface for configuring various options and settings for the Nellie pipeline
    and visualizations. Users can enable or disable specific processing options, control track visualization settings,
    and configure voxel visualization parameters.

    Attributes
    ----------
    nellie : object
        Reference to the Nellie instance managing the pipeline.
    viewer : napari.viewer.Viewer
        Reference to the napari viewer instance.
    remove_edges_checkbox : QCheckBox
        Checkbox for enabling or disabling the removal of image edges.
    remove_intermediates_checkbox : QCheckBox
        Checkbox for enabling or disabling the removal of intermediate files after processing.
    voxel_reassign : QCheckBox
        Checkbox to enable or disable the automatic voxel reassignment step after tracking.
    analyze_node_level : QCheckBox
        Checkbox to enable or disable node-level analysis during feature extraction.
    track_all_frames : QCheckBox
        Checkbox to enable or disable the visualization of voxel tracks for all frames.
    skip_vox_label : QLabel
        Label describing the setting for skipping voxels during track visualization.
    skip_vox : QSpinBox
        Spin box for selecting the value of N to visualize tracks for every Nth voxel.
    initialized : bool
        Flag to indicate whether the settings interface has been initialized.

    Methods
    -------
    post_init()
        Post-initialization method that sets the initialized flag to True.
    set_ui()
        Initializes and sets the layout and UI components for the Settings class.
    """
    def __init__(self, napari_viewer: 'napari.viewer.Viewer', nellie, parent=None):
        """
        Initializes the Settings class, setting up the user interface and options for configuring
        processing and track visualization.

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

        # Checkbox for 'Remove edges'
        self.remove_edges_checkbox = QCheckBox("Remove image edges")
        self.remove_edges_checkbox.setChecked(False)
        self.remove_edges_checkbox.setEnabled(True)
        self.remove_edges_checkbox.setToolTip(
            "Originally for Snouty deskewed images. If you see weird image edge artifacts, enable this.")

        self.remove_intermediates_checkbox = QCheckBox("Remove intermediate files")
        self.remove_intermediates_checkbox.setChecked(False)
        self.remove_intermediates_checkbox.setEnabled(True)
        self.remove_intermediates_checkbox.setToolTip(
            "Remove intermediate files after processing. This means only csv files will be saved.")

        self.voxel_reassign = QCheckBox("Auto-run voxel reassignment")
        self.voxel_reassign.setChecked(False)
        self.voxel_reassign.setEnabled(True)

        # Analyze node level
        self.analyze_node_level = QCheckBox("Analyze node level (slow)")
        self.analyze_node_level.setChecked(False)
        self.analyze_node_level.setEnabled(True)

        # Track all frames
        self.track_all_frames = QCheckBox("Visualize all frames' voxel tracks")
        self.track_all_frames.setChecked(True)
        self.track_all_frames.setEnabled(True)

        # Label above the spinner box
        self.skip_vox_label = QLabel("Visualize tracks for every N voxel. N=")

        self.skip_vox = QSpinBox()
        self.skip_vox.setRange(1, 10000)
        self.skip_vox.setValue(5)
        self.skip_vox.setEnabled(False)

        self.set_ui()

        self.initialized = False

    def post_init(self):
        """
        Post-initialization method that sets the initialized flag to True.
        """
        self.initialized = True

    def set_ui(self):
        """
        Initializes and sets the layout and UI components for the Settings class. This includes checkboxes for
        configuring the processing pipeline and track visualization options, as well as a spin box for setting
        voxel track visualization parameters.
        """
        main_layout = QVBoxLayout()

        # Processor settings
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

        # Tracking settings
        tracking_group = QGroupBox("Track visualization settings")
        tracking_layout = QVBoxLayout()
        tracking_layout.addWidget(self.track_all_frames)
        skip_vox_layout = QHBoxLayout()
        skip_vox_layout.addWidget(self.skip_vox_label)
        skip_vox_layout.addWidget(self.skip_vox)
        tracking_layout.addLayout(skip_vox_layout)
        tracking_group.setLayout(tracking_layout)

        main_layout.addWidget(processor_group)
        main_layout.addWidget(tracking_group)
        self.setLayout(main_layout)


if __name__ == "__main__":
    import napari
    viewer = napari.Viewer()
    napari.run()
