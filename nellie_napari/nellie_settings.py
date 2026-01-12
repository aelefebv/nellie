from dataclasses import dataclass
from typing import Optional

from qtpy.QtWidgets import (
    QWidget,
    QCheckBox,
    QSpinBox,
    QLabel,
    QVBoxLayout,
    QGroupBox,
    QHBoxLayout,
    QFormLayout,
    QTabWidget,
    QDoubleSpinBox,
    QComboBox,
    QScrollArea,
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

    preprocessing_num_t: Optional[int]
    preprocessing_min_radius_um: float
    preprocessing_max_radius_um: float
    preprocessing_alpha_sq: float
    preprocessing_beta_sq: float
    preprocessing_frob_thresh: Optional[float]
    preprocessing_frob_thresh_division: int
    preprocessing_device: str
    preprocessing_low_memory: bool
    preprocessing_max_chunk_voxels: int
    preprocessing_max_threshold_samples: int

    segmentation_label_num_t: Optional[int]
    segmentation_label_threshold: Optional[float]
    segmentation_label_otsu_thresh_intensity: bool
    segmentation_label_chunk_z: Optional[int]
    segmentation_label_flush_interval: int
    segmentation_label_min_radius_um: float
    segmentation_label_threshold_sampling_pixels: int
    segmentation_label_histogram_nbins: int
    segmentation_label_device: str
    segmentation_label_low_memory: bool
    segmentation_label_max_chunk_voxels: int

    segmentation_network_num_t: Optional[int]
    segmentation_network_min_radius_um: float
    segmentation_network_max_radius_um: float
    segmentation_network_device: str
    segmentation_network_low_memory: bool
    segmentation_network_max_chunk_voxels: int

    mocap_num_t: Optional[int]
    mocap_min_radius_um: float
    mocap_max_radius_um: float
    mocap_use_im: str
    mocap_num_sigma: int
    mocap_prefer_gpu: bool
    mocap_peak_min_distance: int
    mocap_device: str
    mocap_low_memory: bool
    mocap_max_chunk_voxels: int

    tracking_num_t: Optional[int]
    tracking_max_distance_um: float
    tracking_device: str
    tracking_mode: str
    tracking_max_dense_pairs: int
    tracking_max_dense_roi_voxels_cpu: int
    tracking_max_dense_roi_voxels_gpu: int
    tracking_low_memory: bool

    reassign_num_t: Optional[int]
    reassign_store_running_matches: bool
    reassign_max_refine_iterations: int
    reassign_device: str
    reassign_low_memory: bool
    reassign_max_query_points: int
    reassign_max_bruteforce_pairs: int

    feature_skip_nodes: Optional[bool]
    feature_use_gpu: bool
    feature_low_memory: bool
    feature_enable_motility: bool
    feature_enable_adjacency: bool
    feature_device: str
    feature_node_chunk_size: Optional[int]
    feature_max_node_mask_elems: int


class Settings(QWidget):
    """
    The Settings class provides a user interface for configuring various options and
    settings for the Nellie pipeline and visualizations. Users can enable or disable
    specific processing options, control track visualization settings, and configure
    voxel visualization parameters.
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

        max_int = 2_000_000_000

        # Preprocessing (Filter)
        self.preprocessing_num_t = self._make_int_spinbox(1, max_value=max_int)
        self.preprocessing_num_t_override, self.preprocessing_num_t_row = (
            self._make_optional_spinbox(self.preprocessing_num_t)
        )

        self.preprocessing_min_radius_um = self._make_double_spinbox(0.25, max_value=100.0)
        self.preprocessing_max_radius_um = self._make_double_spinbox(1.0, max_value=100.0)
        self.preprocessing_alpha_sq = self._make_double_spinbox(0.5, max_value=10.0)
        self.preprocessing_beta_sq = self._make_double_spinbox(0.5, max_value=10.0)

        self.preprocessing_frob_thresh = self._make_double_spinbox(0.0, max_value=1_000_000.0)
        self.preprocessing_frob_thresh_override, self.preprocessing_frob_thresh_row = (
            self._make_optional_spinbox(self.preprocessing_frob_thresh)
        )

        self.preprocessing_frob_thresh_division = self._make_int_spinbox(2, min_value=1, max_value=100)
        self.preprocessing_device = self._make_device_combo("auto")
        self.preprocessing_low_memory = QCheckBox("Low memory")
        self.preprocessing_low_memory.setChecked(False)
        self.preprocessing_max_chunk_voxels = self._make_int_spinbox(1_000_000, max_value=max_int)
        self.preprocessing_max_threshold_samples = self._make_int_spinbox(1_000_000, max_value=max_int)

        # Segmentation label (Label)
        self.segmentation_label_num_t = self._make_int_spinbox(1, max_value=max_int)
        self.segmentation_label_num_t_override, self.segmentation_label_num_t_row = (
            self._make_optional_spinbox(self.segmentation_label_num_t)
        )

        self.segmentation_label_threshold = self._make_double_spinbox(0.0, max_value=1_000_000.0)
        self.segmentation_label_threshold_override, self.segmentation_label_threshold_row = (
            self._make_optional_spinbox(self.segmentation_label_threshold)
        )

        self.segmentation_label_otsu_thresh_intensity = QCheckBox("Use Otsu threshold")
        self.segmentation_label_otsu_thresh_intensity.setChecked(False)

        self.segmentation_label_chunk_z = self._make_int_spinbox(1, max_value=max_int)
        self.segmentation_label_chunk_z_override, self.segmentation_label_chunk_z_row = (
            self._make_optional_spinbox(self.segmentation_label_chunk_z)
        )

        self.segmentation_label_flush_interval = self._make_int_spinbox(1, max_value=1000)
        self.segmentation_label_min_radius_um = self._make_double_spinbox(0.25, max_value=100.0)
        self.segmentation_label_threshold_sampling_pixels = self._make_int_spinbox(1_000_000, max_value=max_int)
        self.segmentation_label_histogram_nbins = self._make_int_spinbox(256, min_value=1, max_value=4096)
        self.segmentation_label_device = self._make_device_combo("auto")
        self.segmentation_label_low_memory = QCheckBox("Low memory")
        self.segmentation_label_low_memory.setChecked(False)
        self.segmentation_label_max_chunk_voxels = self._make_int_spinbox(1_000_000, max_value=max_int)

        # Segmentation network (Network)
        self.segmentation_network_num_t = self._make_int_spinbox(1, max_value=max_int)
        self.segmentation_network_num_t_override, self.segmentation_network_num_t_row = (
            self._make_optional_spinbox(self.segmentation_network_num_t)
        )

        self.segmentation_network_min_radius_um = self._make_double_spinbox(0.2, max_value=100.0)
        self.segmentation_network_max_radius_um = self._make_double_spinbox(1.0, max_value=100.0)
        self.segmentation_network_device = self._make_device_combo("auto")
        self.segmentation_network_low_memory = QCheckBox("Low memory")
        self.segmentation_network_low_memory.setChecked(False)
        self.segmentation_network_max_chunk_voxels = self._make_int_spinbox(1_000_000, max_value=max_int)

        # Mocap marking (Markers)
        self.mocap_num_t = self._make_int_spinbox(1, max_value=max_int)
        self.mocap_num_t_override, self.mocap_num_t_row = (
            self._make_optional_spinbox(self.mocap_num_t)
        )

        self.mocap_min_radius_um = self._make_double_spinbox(0.2, max_value=100.0)
        self.mocap_max_radius_um = self._make_double_spinbox(1.0, max_value=100.0)

        self.mocap_use_im = QComboBox()
        self.mocap_use_im.addItems(["distance", "frangi"])
        self.mocap_use_im.setCurrentText("distance")

        self.mocap_num_sigma = self._make_int_spinbox(5, min_value=1, max_value=100)
        self.mocap_prefer_gpu = QCheckBox("Prefer GPU")
        self.mocap_prefer_gpu.setChecked(True)
        self.mocap_peak_min_distance = self._make_int_spinbox(2, min_value=1, max_value=1000)
        self.mocap_device = self._make_device_combo("auto")
        self.mocap_low_memory = QCheckBox("Low memory")
        self.mocap_low_memory.setChecked(False)
        self.mocap_max_chunk_voxels = self._make_int_spinbox(1_000_000, max_value=max_int)

        # Tracking (HuMomentTracking)
        self.tracking_num_t = self._make_int_spinbox(1, max_value=max_int)
        self.tracking_num_t_override, self.tracking_num_t_row = (
            self._make_optional_spinbox(self.tracking_num_t)
        )

        self.tracking_max_distance_um = self._make_double_spinbox(1.0, max_value=1000.0)
        self.tracking_device = self._make_device_combo("auto")

        self.tracking_mode = QComboBox()
        self.tracking_mode.addItems(["auto", "dense", "sparse"])
        self.tracking_mode.setCurrentText("auto")

        self.tracking_max_dense_pairs = self._make_int_spinbox(10_000_000, max_value=max_int)
        self.tracking_max_dense_roi_voxels_cpu = self._make_int_spinbox(50_000_000, max_value=max_int)
        self.tracking_max_dense_roi_voxels_gpu = self._make_int_spinbox(20_000_000, max_value=max_int)
        self.tracking_low_memory = QCheckBox("Low memory")
        self.tracking_low_memory.setChecked(False)

        # Voxel reassignment (VoxelReassigner)
        self.reassign_num_t = self._make_int_spinbox(1, max_value=max_int)
        self.reassign_num_t_override, self.reassign_num_t_row = (
            self._make_optional_spinbox(self.reassign_num_t)
        )

        self.reassign_store_running_matches = QCheckBox("Store running matches")
        self.reassign_store_running_matches.setChecked(True)
        self.reassign_max_refine_iterations = self._make_int_spinbox(3, min_value=1, max_value=100)
        self.reassign_device = self._make_device_combo("auto")
        self.reassign_low_memory = QCheckBox("Low memory")
        self.reassign_low_memory.setChecked(False)
        self.reassign_max_query_points = self._make_int_spinbox(1_000_000, max_value=max_int)
        self.reassign_max_bruteforce_pairs = self._make_int_spinbox(10_000_000, max_value=max_int)

        # Feature export (Hierarchy)
        self.feature_skip_nodes = QCheckBox("Enabled")
        self.feature_skip_nodes.setChecked(False)
        self.feature_skip_nodes_override, self.feature_skip_nodes_row = (
            self._make_optional_checkbox(self.feature_skip_nodes)
        )

        self.feature_use_gpu = QCheckBox("Use GPU")
        self.feature_use_gpu.setChecked(True)
        self.feature_low_memory = QCheckBox("Low memory")
        self.feature_low_memory.setChecked(False)
        self.feature_enable_motility = QCheckBox("Enable motility")
        self.feature_enable_motility.setChecked(True)
        self.feature_enable_adjacency = QCheckBox("Enable adjacency")
        self.feature_enable_adjacency.setChecked(True)
        self.feature_device = self._make_device_combo("auto")

        self.feature_node_chunk_size = self._make_int_spinbox(1_000_000, max_value=max_int)
        self.feature_node_chunk_size_override, self.feature_node_chunk_size_row = (
            self._make_optional_spinbox(self.feature_node_chunk_size)
        )

        self.feature_max_node_mask_elems = self._make_int_spinbox(50_000_000, max_value=max_int)

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

    def _make_device_combo(self, default):
        combo = QComboBox()
        combo.addItems(["auto", "cpu", "gpu"])
        combo.setCurrentText(default)
        return combo

    def _make_int_spinbox(self, value, min_value=1, max_value=1000000):
        spinbox = QSpinBox()
        spinbox.setRange(min_value, max_value)
        spinbox.setValue(int(value))
        return spinbox

    def _make_double_spinbox(
        self,
        value,
        min_value=0.0,
        max_value=1000.0,
        decimals=4,
        step=0.05,
    ):
        spinbox = QDoubleSpinBox()
        spinbox.setRange(min_value, max_value)
        spinbox.setDecimals(decimals)
        spinbox.setSingleStep(step)
        spinbox.setValue(float(value))
        return spinbox

    def _make_optional_spinbox(self, spinbox):
        override = QCheckBox("Override")
        spinbox.setEnabled(False)
        override.toggled.connect(spinbox.setEnabled)

        container = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(override)
        layout.addWidget(spinbox)
        container.setLayout(layout)
        return override, container

    def _make_optional_checkbox(self, checkbox):
        override = QCheckBox("Override")
        checkbox.setEnabled(False)
        override.toggled.connect(checkbox.setEnabled)

        container = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(override)
        layout.addWidget(checkbox)
        container.setLayout(layout)
        return override, container

    def _wrap_scroll(self, widget):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)
        return scroll

    def _optional_spinbox_value(self, override, spinbox):
        return spinbox.value() if override.isChecked() else None

    def _prune_none(self, params: dict) -> dict:
        return {key: value for key, value in params.items() if value is not None}

    def set_ui(self):
        """
        Initialize and set the layout and UI components for the Settings class.
        This includes checkboxes for configuring the processing pipeline and track
        visualization options, as well as advanced per-step parameters.
        """
        main_layout = QVBoxLayout()

        tabs = QTabWidget()

        # Basic tab
        basic_tab = QWidget()
        basic_layout = QVBoxLayout()

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

        basic_layout.addWidget(processor_group)
        basic_layout.addWidget(tracking_group)
        basic_layout.addStretch(1)
        basic_tab.setLayout(basic_layout)

        # Advanced tab
        advanced_tab = QWidget()
        advanced_layout = QVBoxLayout()

        note = QLabel(
            "Adjust per-step parameters for the pipeline. Use Override to enable optional values."
        )
        note.setWordWrap(True)
        advanced_layout.addWidget(note)

        advanced_tabs = QTabWidget()
        advanced_tabs.addTab(self._build_preprocessing_tab(), "Preprocessing")
        advanced_tabs.addTab(self._build_segmentation_tab(), "Segmentation")
        advanced_tabs.addTab(self._build_mocap_tab(), "Mocap")
        advanced_tabs.addTab(self._build_tracking_tab(), "Tracking")
        advanced_tabs.addTab(self._build_reassign_tab(), "Reassignment")
        advanced_tabs.addTab(self._build_feature_tab(), "Feature export")

        advanced_layout.addWidget(advanced_tabs)
        advanced_tab.setLayout(advanced_layout)

        tabs.addTab(basic_tab, "Basic")
        tabs.addTab(advanced_tab, "Advanced")

        main_layout.addWidget(tabs)

        self.setLayout(main_layout)

    def _build_preprocessing_tab(self):
        content = QWidget()
        layout = QFormLayout()
        layout.addRow("Limit timepoints (num_t)", self.preprocessing_num_t_row)
        layout.addRow("Min radius (um)", self.preprocessing_min_radius_um)
        layout.addRow("Max radius (um)", self.preprocessing_max_radius_um)
        layout.addRow("Alpha sq", self.preprocessing_alpha_sq)
        layout.addRow("Beta sq", self.preprocessing_beta_sq)
        layout.addRow("Frobenius threshold", self.preprocessing_frob_thresh_row)
        layout.addRow("Frobenius thresh division", self.preprocessing_frob_thresh_division)
        layout.addRow("Device", self.preprocessing_device)
        layout.addRow("Low memory", self.preprocessing_low_memory)
        layout.addRow("Max chunk voxels", self.preprocessing_max_chunk_voxels)
        layout.addRow("Max threshold samples", self.preprocessing_max_threshold_samples)
        content.setLayout(layout)
        return self._wrap_scroll(content)

    def _build_segmentation_tab(self):
        content = QWidget()
        layout = QVBoxLayout()

        label_group = QGroupBox("Label")
        label_layout = QFormLayout()
        label_layout.addRow("Limit timepoints (num_t)", self.segmentation_label_num_t_row)
        label_layout.addRow("Threshold", self.segmentation_label_threshold_row)
        label_layout.addRow("Use Otsu threshold", self.segmentation_label_otsu_thresh_intensity)
        label_layout.addRow("Chunk Z", self.segmentation_label_chunk_z_row)
        label_layout.addRow("Flush interval", self.segmentation_label_flush_interval)
        label_layout.addRow("Min radius (um)", self.segmentation_label_min_radius_um)
        label_layout.addRow("Threshold sampling pixels", self.segmentation_label_threshold_sampling_pixels)
        label_layout.addRow("Histogram bins", self.segmentation_label_histogram_nbins)
        label_layout.addRow("Device", self.segmentation_label_device)
        label_layout.addRow("Low memory", self.segmentation_label_low_memory)
        label_layout.addRow("Max chunk voxels", self.segmentation_label_max_chunk_voxels)
        label_group.setLayout(label_layout)

        network_group = QGroupBox("Network")
        network_layout = QFormLayout()
        network_layout.addRow("Limit timepoints (num_t)", self.segmentation_network_num_t_row)
        network_layout.addRow("Min radius (um)", self.segmentation_network_min_radius_um)
        network_layout.addRow("Max radius (um)", self.segmentation_network_max_radius_um)
        network_layout.addRow("Device", self.segmentation_network_device)
        network_layout.addRow("Low memory", self.segmentation_network_low_memory)
        network_layout.addRow("Max chunk voxels", self.segmentation_network_max_chunk_voxels)
        network_group.setLayout(network_layout)

        layout.addWidget(label_group)
        layout.addWidget(network_group)
        layout.addStretch(1)
        content.setLayout(layout)
        return self._wrap_scroll(content)

    def _build_mocap_tab(self):
        content = QWidget()
        layout = QFormLayout()
        layout.addRow("Limit timepoints (num_t)", self.mocap_num_t_row)
        layout.addRow("Min radius (um)", self.mocap_min_radius_um)
        layout.addRow("Max radius (um)", self.mocap_max_radius_um)
        layout.addRow("Use image", self.mocap_use_im)
        layout.addRow("Num sigma", self.mocap_num_sigma)
        layout.addRow("Prefer GPU", self.mocap_prefer_gpu)
        layout.addRow("Peak min distance", self.mocap_peak_min_distance)
        layout.addRow("Device", self.mocap_device)
        layout.addRow("Low memory", self.mocap_low_memory)
        layout.addRow("Max chunk voxels", self.mocap_max_chunk_voxels)
        content.setLayout(layout)
        return self._wrap_scroll(content)

    def _build_tracking_tab(self):
        content = QWidget()
        layout = QFormLayout()
        layout.addRow("Limit timepoints (num_t)", self.tracking_num_t_row)
        layout.addRow("Max distance (um)", self.tracking_max_distance_um)
        layout.addRow("Device", self.tracking_device)
        layout.addRow("Mode", self.tracking_mode)
        layout.addRow("Max dense pairs", self.tracking_max_dense_pairs)
        layout.addRow("Max dense ROI voxels (CPU)", self.tracking_max_dense_roi_voxels_cpu)
        layout.addRow("Max dense ROI voxels (GPU)", self.tracking_max_dense_roi_voxels_gpu)
        layout.addRow("Low memory", self.tracking_low_memory)
        content.setLayout(layout)
        return self._wrap_scroll(content)

    def _build_reassign_tab(self):
        content = QWidget()
        layout = QFormLayout()
        layout.addRow("Limit timepoints (num_t)", self.reassign_num_t_row)
        layout.addRow("Store running matches", self.reassign_store_running_matches)
        layout.addRow("Max refine iterations", self.reassign_max_refine_iterations)
        layout.addRow("Device", self.reassign_device)
        layout.addRow("Low memory", self.reassign_low_memory)
        layout.addRow("Max query points", self.reassign_max_query_points)
        layout.addRow("Max bruteforce pairs", self.reassign_max_bruteforce_pairs)
        content.setLayout(layout)
        return self._wrap_scroll(content)

    def _build_feature_tab(self):
        content = QWidget()
        layout = QVBoxLayout()

        note = QLabel(
            "Skip node-level features uses the Basic setting unless Override is enabled."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        form = QFormLayout()
        form.addRow("Skip node-level features", self.feature_skip_nodes_row)
        form.addRow("Use GPU", self.feature_use_gpu)
        form.addRow("Low memory", self.feature_low_memory)
        form.addRow("Enable motility", self.feature_enable_motility)
        form.addRow("Enable adjacency", self.feature_enable_adjacency)
        form.addRow("Device", self.feature_device)
        form.addRow("Node chunk size", self.feature_node_chunk_size_row)
        form.addRow("Max node mask elems", self.feature_max_node_mask_elems)

        layout.addLayout(form)
        layout.addStretch(1)
        content.setLayout(layout)
        return self._wrap_scroll(content)

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
            preprocessing_num_t=self._optional_spinbox_value(
                self.preprocessing_num_t_override, self.preprocessing_num_t
            ),
            preprocessing_min_radius_um=self.preprocessing_min_radius_um.value(),
            preprocessing_max_radius_um=self.preprocessing_max_radius_um.value(),
            preprocessing_alpha_sq=self.preprocessing_alpha_sq.value(),
            preprocessing_beta_sq=self.preprocessing_beta_sq.value(),
            preprocessing_frob_thresh=self._optional_spinbox_value(
                self.preprocessing_frob_thresh_override, self.preprocessing_frob_thresh
            ),
            preprocessing_frob_thresh_division=self.preprocessing_frob_thresh_division.value(),
            preprocessing_device=self.preprocessing_device.currentText(),
            preprocessing_low_memory=self.preprocessing_low_memory.isChecked(),
            preprocessing_max_chunk_voxels=self.preprocessing_max_chunk_voxels.value(),
            preprocessing_max_threshold_samples=self.preprocessing_max_threshold_samples.value(),
            segmentation_label_num_t=self._optional_spinbox_value(
                self.segmentation_label_num_t_override, self.segmentation_label_num_t
            ),
            segmentation_label_threshold=self._optional_spinbox_value(
                self.segmentation_label_threshold_override, self.segmentation_label_threshold
            ),
            segmentation_label_otsu_thresh_intensity=self.segmentation_label_otsu_thresh_intensity.isChecked(),
            segmentation_label_chunk_z=self._optional_spinbox_value(
                self.segmentation_label_chunk_z_override, self.segmentation_label_chunk_z
            ),
            segmentation_label_flush_interval=self.segmentation_label_flush_interval.value(),
            segmentation_label_min_radius_um=self.segmentation_label_min_radius_um.value(),
            segmentation_label_threshold_sampling_pixels=self.segmentation_label_threshold_sampling_pixels.value(),
            segmentation_label_histogram_nbins=self.segmentation_label_histogram_nbins.value(),
            segmentation_label_device=self.segmentation_label_device.currentText(),
            segmentation_label_low_memory=self.segmentation_label_low_memory.isChecked(),
            segmentation_label_max_chunk_voxels=self.segmentation_label_max_chunk_voxels.value(),
            segmentation_network_num_t=self._optional_spinbox_value(
                self.segmentation_network_num_t_override, self.segmentation_network_num_t
            ),
            segmentation_network_min_radius_um=self.segmentation_network_min_radius_um.value(),
            segmentation_network_max_radius_um=self.segmentation_network_max_radius_um.value(),
            segmentation_network_device=self.segmentation_network_device.currentText(),
            segmentation_network_low_memory=self.segmentation_network_low_memory.isChecked(),
            segmentation_network_max_chunk_voxels=self.segmentation_network_max_chunk_voxels.value(),
            mocap_num_t=self._optional_spinbox_value(
                self.mocap_num_t_override, self.mocap_num_t
            ),
            mocap_min_radius_um=self.mocap_min_radius_um.value(),
            mocap_max_radius_um=self.mocap_max_radius_um.value(),
            mocap_use_im=self.mocap_use_im.currentText(),
            mocap_num_sigma=self.mocap_num_sigma.value(),
            mocap_prefer_gpu=self.mocap_prefer_gpu.isChecked(),
            mocap_peak_min_distance=self.mocap_peak_min_distance.value(),
            mocap_device=self.mocap_device.currentText(),
            mocap_low_memory=self.mocap_low_memory.isChecked(),
            mocap_max_chunk_voxels=self.mocap_max_chunk_voxels.value(),
            tracking_num_t=self._optional_spinbox_value(
                self.tracking_num_t_override, self.tracking_num_t
            ),
            tracking_max_distance_um=self.tracking_max_distance_um.value(),
            tracking_device=self.tracking_device.currentText(),
            tracking_mode=self.tracking_mode.currentText(),
            tracking_max_dense_pairs=self.tracking_max_dense_pairs.value(),
            tracking_max_dense_roi_voxels_cpu=self.tracking_max_dense_roi_voxels_cpu.value(),
            tracking_max_dense_roi_voxels_gpu=self.tracking_max_dense_roi_voxels_gpu.value(),
            tracking_low_memory=self.tracking_low_memory.isChecked(),
            reassign_num_t=self._optional_spinbox_value(
                self.reassign_num_t_override, self.reassign_num_t
            ),
            reassign_store_running_matches=self.reassign_store_running_matches.isChecked(),
            reassign_max_refine_iterations=self.reassign_max_refine_iterations.value(),
            reassign_device=self.reassign_device.currentText(),
            reassign_low_memory=self.reassign_low_memory.isChecked(),
            reassign_max_query_points=self.reassign_max_query_points.value(),
            reassign_max_bruteforce_pairs=self.reassign_max_bruteforce_pairs.value(),
            feature_skip_nodes=(
                self.feature_skip_nodes.isChecked()
                if self.feature_skip_nodes_override.isChecked()
                else None
            ),
            feature_use_gpu=self.feature_use_gpu.isChecked(),
            feature_low_memory=self.feature_low_memory.isChecked(),
            feature_enable_motility=self.feature_enable_motility.isChecked(),
            feature_enable_adjacency=self.feature_enable_adjacency.isChecked(),
            feature_device=self.feature_device.currentText(),
            feature_node_chunk_size=self._optional_spinbox_value(
                self.feature_node_chunk_size_override, self.feature_node_chunk_size
            ),
            feature_max_node_mask_elems=self.feature_max_node_mask_elems.value(),
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

        self.preprocessing_num_t_override.setChecked(config.preprocessing_num_t is not None)
        if config.preprocessing_num_t is not None:
            self.preprocessing_num_t.setValue(config.preprocessing_num_t)
        self.preprocessing_min_radius_um.setValue(config.preprocessing_min_radius_um)
        self.preprocessing_max_radius_um.setValue(config.preprocessing_max_radius_um)
        self.preprocessing_alpha_sq.setValue(config.preprocessing_alpha_sq)
        self.preprocessing_beta_sq.setValue(config.preprocessing_beta_sq)
        self.preprocessing_frob_thresh_override.setChecked(config.preprocessing_frob_thresh is not None)
        if config.preprocessing_frob_thresh is not None:
            self.preprocessing_frob_thresh.setValue(config.preprocessing_frob_thresh)
        self.preprocessing_frob_thresh_division.setValue(config.preprocessing_frob_thresh_division)
        self.preprocessing_device.setCurrentText(config.preprocessing_device)
        self.preprocessing_low_memory.setChecked(config.preprocessing_low_memory)
        self.preprocessing_max_chunk_voxels.setValue(config.preprocessing_max_chunk_voxels)
        self.preprocessing_max_threshold_samples.setValue(config.preprocessing_max_threshold_samples)

        self.segmentation_label_num_t_override.setChecked(config.segmentation_label_num_t is not None)
        if config.segmentation_label_num_t is not None:
            self.segmentation_label_num_t.setValue(config.segmentation_label_num_t)
        self.segmentation_label_threshold_override.setChecked(config.segmentation_label_threshold is not None)
        if config.segmentation_label_threshold is not None:
            self.segmentation_label_threshold.setValue(config.segmentation_label_threshold)
        self.segmentation_label_otsu_thresh_intensity.setChecked(config.segmentation_label_otsu_thresh_intensity)
        self.segmentation_label_chunk_z_override.setChecked(config.segmentation_label_chunk_z is not None)
        if config.segmentation_label_chunk_z is not None:
            self.segmentation_label_chunk_z.setValue(config.segmentation_label_chunk_z)
        self.segmentation_label_flush_interval.setValue(config.segmentation_label_flush_interval)
        self.segmentation_label_min_radius_um.setValue(config.segmentation_label_min_radius_um)
        self.segmentation_label_threshold_sampling_pixels.setValue(config.segmentation_label_threshold_sampling_pixels)
        self.segmentation_label_histogram_nbins.setValue(config.segmentation_label_histogram_nbins)
        self.segmentation_label_device.setCurrentText(config.segmentation_label_device)
        self.segmentation_label_low_memory.setChecked(config.segmentation_label_low_memory)
        self.segmentation_label_max_chunk_voxels.setValue(config.segmentation_label_max_chunk_voxels)

        self.segmentation_network_num_t_override.setChecked(config.segmentation_network_num_t is not None)
        if config.segmentation_network_num_t is not None:
            self.segmentation_network_num_t.setValue(config.segmentation_network_num_t)
        self.segmentation_network_min_radius_um.setValue(config.segmentation_network_min_radius_um)
        self.segmentation_network_max_radius_um.setValue(config.segmentation_network_max_radius_um)
        self.segmentation_network_device.setCurrentText(config.segmentation_network_device)
        self.segmentation_network_low_memory.setChecked(config.segmentation_network_low_memory)
        self.segmentation_network_max_chunk_voxels.setValue(config.segmentation_network_max_chunk_voxels)

        self.mocap_num_t_override.setChecked(config.mocap_num_t is not None)
        if config.mocap_num_t is not None:
            self.mocap_num_t.setValue(config.mocap_num_t)
        self.mocap_min_radius_um.setValue(config.mocap_min_radius_um)
        self.mocap_max_radius_um.setValue(config.mocap_max_radius_um)
        self.mocap_use_im.setCurrentText(config.mocap_use_im)
        self.mocap_num_sigma.setValue(config.mocap_num_sigma)
        self.mocap_prefer_gpu.setChecked(config.mocap_prefer_gpu)
        self.mocap_peak_min_distance.setValue(config.mocap_peak_min_distance)
        self.mocap_device.setCurrentText(config.mocap_device)
        self.mocap_low_memory.setChecked(config.mocap_low_memory)
        self.mocap_max_chunk_voxels.setValue(config.mocap_max_chunk_voxels)

        self.tracking_num_t_override.setChecked(config.tracking_num_t is not None)
        if config.tracking_num_t is not None:
            self.tracking_num_t.setValue(config.tracking_num_t)
        self.tracking_max_distance_um.setValue(config.tracking_max_distance_um)
        self.tracking_device.setCurrentText(config.tracking_device)
        self.tracking_mode.setCurrentText(config.tracking_mode)
        self.tracking_max_dense_pairs.setValue(config.tracking_max_dense_pairs)
        self.tracking_max_dense_roi_voxels_cpu.setValue(config.tracking_max_dense_roi_voxels_cpu)
        self.tracking_max_dense_roi_voxels_gpu.setValue(config.tracking_max_dense_roi_voxels_gpu)
        self.tracking_low_memory.setChecked(config.tracking_low_memory)

        self.reassign_num_t_override.setChecked(config.reassign_num_t is not None)
        if config.reassign_num_t is not None:
            self.reassign_num_t.setValue(config.reassign_num_t)
        self.reassign_store_running_matches.setChecked(config.reassign_store_running_matches)
        self.reassign_max_refine_iterations.setValue(config.reassign_max_refine_iterations)
        self.reassign_device.setCurrentText(config.reassign_device)
        self.reassign_low_memory.setChecked(config.reassign_low_memory)
        self.reassign_max_query_points.setValue(config.reassign_max_query_points)
        self.reassign_max_bruteforce_pairs.setValue(config.reassign_max_bruteforce_pairs)

        self.feature_skip_nodes_override.setChecked(config.feature_skip_nodes is not None)
        if config.feature_skip_nodes is not None:
            self.feature_skip_nodes.setChecked(config.feature_skip_nodes)
        self.feature_use_gpu.setChecked(config.feature_use_gpu)
        self.feature_low_memory.setChecked(config.feature_low_memory)
        self.feature_enable_motility.setChecked(config.feature_enable_motility)
        self.feature_enable_adjacency.setChecked(config.feature_enable_adjacency)
        self.feature_device.setCurrentText(config.feature_device)
        self.feature_node_chunk_size_override.setChecked(config.feature_node_chunk_size is not None)
        if config.feature_node_chunk_size is not None:
            self.feature_node_chunk_size.setValue(config.feature_node_chunk_size)
        self.feature_max_node_mask_elems.setValue(config.feature_max_node_mask_elems)

    def get_preprocessing_params(self) -> dict:
        params = {
            "num_t": self._optional_spinbox_value(
                self.preprocessing_num_t_override, self.preprocessing_num_t
            ),
            "min_radius_um": self.preprocessing_min_radius_um.value(),
            "max_radius_um": self.preprocessing_max_radius_um.value(),
            "alpha_sq": self.preprocessing_alpha_sq.value(),
            "beta_sq": self.preprocessing_beta_sq.value(),
            "frob_thresh": self._optional_spinbox_value(
                self.preprocessing_frob_thresh_override, self.preprocessing_frob_thresh
            ),
            "frob_thresh_division": self.preprocessing_frob_thresh_division.value(),
            "device": self.preprocessing_device.currentText(),
            "low_memory": self.preprocessing_low_memory.isChecked(),
            "max_chunk_voxels": self.preprocessing_max_chunk_voxels.value(),
            "max_threshold_samples": self.preprocessing_max_threshold_samples.value(),
        }
        return self._prune_none(params)

    def get_segmentation_label_params(self) -> dict:
        params = {
            "num_t": self._optional_spinbox_value(
                self.segmentation_label_num_t_override, self.segmentation_label_num_t
            ),
            "threshold": self._optional_spinbox_value(
                self.segmentation_label_threshold_override, self.segmentation_label_threshold
            ),
            "otsu_thresh_intensity": self.segmentation_label_otsu_thresh_intensity.isChecked(),
            "chunk_z": self._optional_spinbox_value(
                self.segmentation_label_chunk_z_override, self.segmentation_label_chunk_z
            ),
            "flush_interval": self.segmentation_label_flush_interval.value(),
            "min_radius_um": self.segmentation_label_min_radius_um.value(),
            "threshold_sampling_pixels": self.segmentation_label_threshold_sampling_pixels.value(),
            "histogram_nbins": self.segmentation_label_histogram_nbins.value(),
            "device": self.segmentation_label_device.currentText(),
            "low_memory": self.segmentation_label_low_memory.isChecked(),
            "max_chunk_voxels": self.segmentation_label_max_chunk_voxels.value(),
        }
        return self._prune_none(params)

    def get_segmentation_network_params(self) -> dict:
        params = {
            "num_t": self._optional_spinbox_value(
                self.segmentation_network_num_t_override, self.segmentation_network_num_t
            ),
            "min_radius_um": self.segmentation_network_min_radius_um.value(),
            "max_radius_um": self.segmentation_network_max_radius_um.value(),
            "device": self.segmentation_network_device.currentText(),
            "low_memory": self.segmentation_network_low_memory.isChecked(),
            "max_chunk_voxels": self.segmentation_network_max_chunk_voxels.value(),
        }
        return self._prune_none(params)

    def get_mocap_params(self) -> dict:
        params = {
            "num_t": self._optional_spinbox_value(
                self.mocap_num_t_override, self.mocap_num_t
            ),
            "min_radius_um": self.mocap_min_radius_um.value(),
            "max_radius_um": self.mocap_max_radius_um.value(),
            "use_im": self.mocap_use_im.currentText(),
            "num_sigma": self.mocap_num_sigma.value(),
            "prefer_gpu": self.mocap_prefer_gpu.isChecked(),
            "peak_min_distance": self.mocap_peak_min_distance.value(),
            "device": self.mocap_device.currentText(),
            "low_memory": self.mocap_low_memory.isChecked(),
            "max_chunk_voxels": self.mocap_max_chunk_voxels.value(),
        }
        return self._prune_none(params)

    def get_tracking_params(self) -> dict:
        params = {
            "num_t": self._optional_spinbox_value(
                self.tracking_num_t_override, self.tracking_num_t
            ),
            "max_distance_um": self.tracking_max_distance_um.value(),
            "device": self.tracking_device.currentText(),
            "mode": self.tracking_mode.currentText(),
            "max_dense_pairs": self.tracking_max_dense_pairs.value(),
            "max_dense_roi_voxels_cpu": self.tracking_max_dense_roi_voxels_cpu.value(),
            "max_dense_roi_voxels_gpu": self.tracking_max_dense_roi_voxels_gpu.value(),
            "low_memory": self.tracking_low_memory.isChecked(),
        }
        return self._prune_none(params)

    def get_reassign_params(self) -> dict:
        params = {
            "num_t": self._optional_spinbox_value(
                self.reassign_num_t_override, self.reassign_num_t
            ),
            "store_running_matches": self.reassign_store_running_matches.isChecked(),
            "max_refine_iterations": self.reassign_max_refine_iterations.value(),
            "device": self.reassign_device.currentText(),
            "low_memory": self.reassign_low_memory.isChecked(),
            "max_query_points": self.reassign_max_query_points.value(),
            "max_bruteforce_pairs": self.reassign_max_bruteforce_pairs.value(),
        }
        return self._prune_none(params)

    def get_feature_params(self) -> dict:
        params = {
            "use_gpu": self.feature_use_gpu.isChecked(),
            "low_memory": self.feature_low_memory.isChecked(),
            "enable_motility": self.feature_enable_motility.isChecked(),
            "enable_adjacency": self.feature_enable_adjacency.isChecked(),
            "device": self.feature_device.currentText(),
            "max_node_mask_elems": self.feature_max_node_mask_elems.value(),
        }

        if self.feature_skip_nodes_override.isChecked():
            params["skip_nodes"] = self.feature_skip_nodes.isChecked()
        if self.feature_node_chunk_size_override.isChecked():
            params["node_chunk_size"] = self.feature_node_chunk_size.value()

        return params


if __name__ == "__main__":
    import napari

    viewer = napari.Viewer()
    # You can pass a real Nellie instance instead of None when integrating.
    settings = Settings(viewer, nellie=None)
    viewer.window.add_dock_widget(settings, area="right")
    napari.run()
