import os
import pickle

import numpy as np
from qtpy.QtWidgets import (
    QComboBox,
    QCheckBox,
    QPushButton,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QSpinBox,
    QDoubleSpinBox,
    QWidget,
    QVBoxLayout,
    QGroupBox,
    QHBoxLayout,
)
from qtpy.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from napari.utils.notifications import show_info
import pandas as pd
import datetime


class NellieAnalysis(QWidget):
    """
    A class for analyzing and visualizing multi-dimensional microscopy data using histograms, graphs,
    and overlays in the napari viewer.
    """

    # known statistic suffixes and their normalized keys
    STAT_SUFFIXES = [
        ("_raw", "raw"),
        ("_std_dev", "std_dev"),
        ("_std", "std_dev"),
        ("_mean", "mean"),
        ("_median", "median"),
        ("_min", "min"),
        ("_max", "max"),
        ("_sum", "sum"),
    ]

    STAT_LABELS = {
        "value": "Value",
        "raw": "Raw",
        "mean": "Mean",
        "median": "Median",
        "std_dev": "St. Dev.",
        "min": "Min",
        "max": "Max",
        "sum": "Sum",
    }

    STAT_ORDER = {
        "value": 0,
        "raw": 1,
        "mean": 2,
        "median": 3,
        "std_dev": 4,
        "min": 5,
        "max": 6,
        "sum": 7,
    }

    def __init__(self, napari_viewer: "napari.viewer.Viewer", nellie, parent=None):
        """
        Initialize the NellieAnalysis widget.

        Parameters
        ----------
        napari_viewer : napari.viewer.Viewer
            The napari viewer instance.
        nellie : object
            The main Nellie plugin instance.
        parent : QWidget, optional
            The parent widget, by default None.
        """
        super().__init__(parent)
        self.nellie = nellie
        self.viewer = napari_viewer

        # plotting canvas
        self.canvas = FigureCanvasQTAgg()
        self.canvas.figure.set_layout_engine("constrained")

        # viewer scale
        self.scale = (1, 1, 1)

        # histogram / stats flags
        self.log_scale = False
        self.is_median = False
        self.match_t = False
        self.hist_reset = True

        # UI widgets (initialized in post_init)
        self.mean_median_toggle = None
        self.overlay_button = None
        self.match_t_toggle = None
        self.hist_min = None
        self.hist_max = None
        self.num_bins = None
        self.export_data_button = None
        self.save_graph_button = None
        self.log_scale_checkbox = None
        self.dropdown = None          # level (voxel/node/...)
        self.dropdown_attr = None     # feature (e.g. divergence)
        self.dropdown_stat = None     # statistic/form (e.g. mean)
        self.click_match_table = None
        self.click_match_group = None

        # dataframes
        self.voxel_df: pd.DataFrame | None = None
        self.node_df: pd.DataFrame | None = None
        self.branch_df: pd.DataFrame | None = None
        self.organelle_df: pd.DataFrame | None = None
        self.image_df: pd.DataFrame | None = None
        self.df: pd.DataFrame | None = None  # currently selected level DataFrame

        # current selection
        self.selected_level: str | None = None
        self.selected_attr: str | None = None       # concrete column name
        self.selected_feature: str | None = None    # base feature name
        self.selected_form: str | None = None       # statistic/form key
        self.all_attr_data: pd.Series | None = None  # full series (all timepoints)
        self.attr_data: pd.Series | None = None      # current plotted series (all or per-t)
        self.time_col: pd.Series | None = None

        # attribute grouping (feature -> form -> column name)
        self.feature_map: dict[str, dict[str, str]] = {}

        # adjacency / overlay
        self.adjacency_maps: dict | None = None
        self.label_mask = None
        self.label_mask_layer = None
        self.label_coords: list[np.ndarray] = []

        # stats
        self.data_to_plot: pd.Series | None = None
        self.mean = np.nan
        self.std = np.nan
        self.median = np.nan
        self.iqr = np.nan
        self.perc75 = np.nan
        self.perc25 = np.nan

        self.initialized = False

    # -------------------------------------------------------------------------
    # lifecycle helpers
    # -------------------------------------------------------------------------
    def reset(self):
        """
        Reset internal state so the widget can be reused for a new dataset.
        """
        self.initialized = False

        # dataframes
        self.voxel_df = None
        self.node_df = None
        self.branch_df = None
        self.organelle_df = None
        self.image_df = None
        self.df = None

        # selections
        self.selected_level = None
        self.selected_attr = None
        self.selected_feature = None
        self.selected_form = None
        self.all_attr_data = None
        self.attr_data = None
        self.time_col = None

        # attribute grouping
        self.feature_map = {}

        # adjacency & overlay
        self.adjacency_maps = None
        self.label_mask = None
        self.label_mask_layer = None
        self.label_coords = []

        # stats / plotting
        self.data_to_plot = None
        self.mean = self.std = self.median = self.iqr = self.perc75 = self.perc25 = np.nan
        self.log_scale = False
        self.is_median = False
        self.match_t = False
        self.hist_reset = True

        # UI state
        self._clear_canvas()
        self._disable_hist_controls()
        if self.click_match_group is not None:
            self.click_match_group.setVisible(False)
        if self.dropdown_attr is not None:
            self.dropdown_attr.clear()
            self.dropdown_attr.addItem("None")
        if self.dropdown_stat is not None:
            self.dropdown_stat.clear()
            self.dropdown_stat.addItem("None")
            self.dropdown_stat.setEnabled(False)
        if self.dropdown is not None:
            idx = self.dropdown.findText("none")
            if idx >= 0:
                self.dropdown.setCurrentIndex(idx)

    def post_init(self):
        """
        Initialize UI elements and connect them to their event handlers.
        Must be called once after construction, when 'nellie' and 'viewer' are ready.
        """
        # checkboxes
        self.log_scale_checkbox = QCheckBox("Log scale")
        self.log_scale_checkbox.stateChanged.connect(self.on_log_scale)
        self.log_scale_checkbox.setEnabled(False)

        self.mean_median_toggle = QCheckBox("Median view")
        self.mean_median_toggle.stateChanged.connect(self.toggle_mean_med)
        self.mean_median_toggle.setEnabled(False)

        self.overlay_button = QPushButton("Overlay mask")
        self.overlay_button.clicked.connect(self.overlay)
        self.overlay_button.setEnabled(False)

        self.match_t_toggle = QCheckBox("Timepoint data")
        self.match_t_toggle.stateChanged.connect(self.toggle_match_t)
        self.match_t_toggle.setEnabled(False)
        self.viewer.dims.events.current_step.connect(self.on_t_change)

        # histogram spinboxes
        self.hist_min = QDoubleSpinBox()
        self.hist_min.setEnabled(False)
        self.hist_min.valueChanged.connect(self.on_hist_change)
        self.hist_min.setDecimals(4)

        self.hist_max = QDoubleSpinBox()
        self.hist_max.setEnabled(False)
        self.hist_max.valueChanged.connect(self.on_hist_change)
        self.hist_max.setDecimals(4)

        self.num_bins = QSpinBox()
        self.num_bins.setRange(1, 100)
        self.num_bins.setValue(10)
        self.num_bins.setEnabled(False)
        self.num_bins.valueChanged.connect(self.on_hist_change)

        # export buttons
        self.export_data_button = QPushButton("Export graph data")
        self.export_data_button.clicked.connect(self.export_data)

        self.save_graph_button = QPushButton("Save graph")
        self.save_graph_button.clicked.connect(self.save_graph)

        # scale bar configuration
        if self.nellie.im_info.no_z:
            self.scale = (
                self.nellie.im_info.dim_res["Y"],
                self.nellie.im_info.dim_res["X"],
            )
        else:
            self.scale = (
                self.nellie.im_info.dim_res["Z"],
                self.nellie.im_info.dim_res["Y"],
                self.nellie.im_info.dim_res["X"],
            )
        self.viewer.scale_bar.visible = True
        self.viewer.scale_bar.unit = "um"

        # selection dropdowns
        self._create_dropdown_selection()
        self.check_for_adjacency_map()

        # click mapping table
        self.click_match_table = QTableWidget()

        # build UI layout
        self._set_ui()

        self.initialized = True

    # -------------------------------------------------------------------------
    # UI construction helpers
    # -------------------------------------------------------------------------
    def _set_ui(self):
        """
        Set up the user interface layout: dropdowns, histogram controls, overlay controls, export buttons.
        """
        main_layout = QVBoxLayout()

        # Attribute dropdown group
        attr_group = QGroupBox("Select hierarchy level and attribute")
        attr_layout = QHBoxLayout()
        attr_layout.addWidget(self.dropdown)        # level
        attr_layout.addWidget(self.dropdown_attr)   # feature
        attr_layout.addWidget(self.dropdown_stat)   # form
        attr_group.setLayout(attr_layout)

        # Histogram group
        hist_group = QGroupBox("Histogram options")
        hist_layout = QVBoxLayout()

        # min / max controls
        sub_layout = QHBoxLayout()
        sub_layout.addWidget(QLabel("Min"), alignment=Qt.AlignRight)
        sub_layout.addWidget(self.hist_min)
        sub_layout.addWidget(self.hist_max)
        sub_layout.addWidget(QLabel("Max"), alignment=Qt.AlignLeft)
        hist_layout.addLayout(sub_layout)

        # canvas
        hist_layout.addWidget(self.canvas)

        # bins
        sub_layout = QHBoxLayout()
        sub_layout.addWidget(QLabel("Bins"), alignment=Qt.AlignRight)
        sub_layout.addWidget(self.num_bins)
        hist_layout.addLayout(sub_layout)

        # options row
        sub_layout = QHBoxLayout()
        sub_layout.addWidget(self.log_scale_checkbox)
        sub_layout.addWidget(self.mean_median_toggle)
        sub_layout.addWidget(self.match_t_toggle)
        sub_layout.addWidget(self.overlay_button)
        hist_layout.addLayout(sub_layout)

        hist_group.setLayout(hist_layout)

        # Save / export group
        save_group = QGroupBox("Export options")
        save_layout = QHBoxLayout()
        save_layout.addWidget(self.export_data_button)
        save_layout.addWidget(self.save_graph_button)
        save_group.setLayout(save_layout)

        # Click mapping group (hidden until first click)
        self.click_match_group = QGroupBox("Clicked voxel mapping")
        click_layout = QVBoxLayout()
        click_layout.addWidget(self.click_match_table)
        self.click_match_group.setLayout(click_layout)
        self.click_match_group.setVisible(False)

        main_layout.addWidget(attr_group)
        main_layout.addWidget(hist_group)
        main_layout.addWidget(save_group)
        main_layout.addWidget(self.click_match_group)
        self.setLayout(main_layout)

    def _create_dropdown_selection(self):
        """
        Create and configure the dropdown menus for selecting hierarchy levels,
        features, and statistic forms.
        """
        # hierarchy level (voxel / node / branch / organelle / image)
        self.dropdown = QComboBox()
        self.dropdown.currentIndexChanged.connect(self.on_level_selected)

        # base feature (e.g. divergence, lin_vel_mag_rel)
        self.dropdown_attr = QComboBox()
        self.dropdown_attr.currentIndexChanged.connect(self.on_attr_selected)

        # statistic / form (e.g. mean, std dev, min, max, sum)
        self.dropdown_stat = QComboBox()
        self.dropdown_stat.currentIndexChanged.connect(self.on_form_selected)
        self.dropdown_stat.setEnabled(False)

        self.rewrite_dropdown()
        self.set_default_dropdowns()

    def set_default_dropdowns(self):
        """
        Set default values for the hierarchy level and attribute dropdowns.
        """
        if self.dropdown is not None:
            organelle_idx = self.dropdown.findText("organelle")
            if organelle_idx >= 0:
                self.dropdown.setCurrentIndex(organelle_idx)

        if self.dropdown_attr is not None and self.df is not None:
            area_raw_idx = self.dropdown_attr.findText("organelle_area_raw")
            if area_raw_idx >= 0:
                self.dropdown_attr.setCurrentIndex(area_raw_idx)

    # -------------------------------------------------------------------------
    # small helpers
    # -------------------------------------------------------------------------
    def _clear_canvas(self):
        """
        Clear the plotting canvas.
        """
        self.canvas.figure.clear()
        self.canvas.draw()

    def _disable_hist_controls(self):
        """
        Disable all histogram control widgets.
        """
        for w in (
            self.hist_min,
            self.hist_max,
            self.num_bins,
            self.log_scale_checkbox,
            self.mean_median_toggle,
            self.match_t_toggle,
        ):
            if w is not None:
                w.setEnabled(False)

    def _enable_hist_controls(self):
        """
        Enable histogram control widgets.
        """
        for w in (self.hist_min, self.hist_max, self.num_bins):
            if w is not None:
                w.setEnabled(True)

        if self.log_scale_checkbox is not None:
            self.log_scale_checkbox.setEnabled(True)
        if self.mean_median_toggle is not None:
            self.mean_median_toggle.setEnabled(True)
        if self.match_t_toggle is not None and self.df is not None and "t" in self.df.columns:
            self.match_t_toggle.setEnabled(True)

    def _split_feature_form(self, col: str) -> tuple[str, str]:
        """
        Split a column name into (feature, form) using known suffixes.
        If no suffix matches, treat the entire name as the feature and form='value'.

        Parameters
        ----------
        col : str
            The column name to split.

        Returns
        -------
        tuple[str, str]
            A tuple containing the base feature name and the statistic form.
        """
        for suffix, key in self.STAT_SUFFIXES:
            if col.endswith(suffix):
                base = col[: -len(suffix)]
                return base, key
        return col, "value"

    def _form_label(self, form: str) -> str:
        """
        Get the human-readable label for a statistic form.

        Parameters
        ----------
        form : str
            The statistic form key.

        Returns
        -------
        str
            The human-readable label.
        """
        return self.STAT_LABELS.get(form, form)

    def _form_sort_key(self, form: str) -> int:
        """
        Get the sort key for a statistic form.

        Parameters
        ----------
        form : str
            The statistic form key.

        Returns
        -------
        int
            The sort key.
        """
        return self.STAT_ORDER.get(form, 100)

    def _populate_stat_dropdown(self, feature: str):
        """
        Populate the statistic dropdown for a given feature name.

        Parameters
        ----------
        feature : str
            The feature name to populate statistics for.
        """
        if self.dropdown_stat is None:
            return

        self.dropdown_stat.blockSignals(True)
        self.dropdown_stat.clear()
        self.dropdown_stat.addItem("None")

        forms_dict = self.feature_map.get(feature, {})
        if not forms_dict:
            self.dropdown_stat.setEnabled(False)
            self.dropdown_stat.blockSignals(False)
            return

        forms = sorted(forms_dict.keys(), key=self._form_sort_key)
        for form in forms:
            label = self._form_label(form)
            # store the canonical form key in userData
            self.dropdown_stat.addItem(label, userData=form)

        # default to first real form (index 1 because index 0 is "None")
        if self.dropdown_stat.count() > 1:
            self.dropdown_stat.setCurrentIndex(1)
        else:
            self.dropdown_stat.setCurrentIndex(0)

        # enable only if there is at least one real option
        self.dropdown_stat.setEnabled(self.dropdown_stat.count() > 1)
        self.dropdown_stat.blockSignals(False)

    def _current_feature_name(self) -> str | None:
        """
        Get the currently selected feature name.

        Returns
        -------
        str | None
            The selected feature name, or None if no valid selection.
        """
        if self.dropdown_attr is None or self.dropdown_attr.count() == 0:
            return None
        text = self.dropdown_attr.currentText()
        if text in ("", "None"):
            return None
        return text

    def _current_form_key(self) -> str | None:
        """
        Get the currently selected statistic form key.

        Returns
        -------
        str | None
            The selected form key, or None if no valid selection.
        """
        if self.dropdown_stat is None or self.dropdown_stat.count() == 0:
            return None
        text = self.dropdown_stat.currentText()
        if text in ("", "None"):
            return None
        data = self.dropdown_stat.currentData()
        if data is None:
            return text.lower()
        return data

    def _current_attr_name(self) -> str | None:
        """
        Resolve the currently selected (feature, form) back to a concrete column name.

        Returns
        -------
        str | None
            The concrete column name, or None if resolution fails.
        """
        feature = self._current_feature_name()
        if feature is None:
            return None

        forms_dict = self.feature_map.get(feature, {})

        # no grouping information: fall back to direct column name
        if not forms_dict:
            if self.df is not None and feature in self.df.columns:
                return feature
            return None

        form = self._current_form_key()

        # if form is not selected but there is exactly one choice, use it
        if form is None:
            if len(forms_dict) == 1:
                return next(iter(forms_dict.values()))
            # otherwise, pick the first one in a stable order
            form = sorted(forms_dict.keys(), key=self._form_sort_key)[0]

        if form in forms_dict:
            return forms_dict[form]

        # last resort: treat feature name as the column name
        if self.df is not None and feature in self.df.columns:
            return feature
        return None

    def _update_data_for_current_selection(self):
        """
        Update self.attr_data / self.all_attr_data / self.time_col based on
        current level, feature, statistic form, and match_t flag.
        """
        if self.df is None:
            self.selected_attr = None
            self.selected_feature = None
            self.selected_form = None
            self.all_attr_data = None
            self.attr_data = None
            self.time_col = None
            return

        # keep track of current selections
        self.selected_feature = self._current_feature_name()
        self.selected_form = self._current_form_key()

        attr = self._current_attr_name()
        self.selected_attr = attr
        if attr is None or attr not in self.df.columns:
            self.all_attr_data = None
            self.attr_data = None
            self.time_col = None
            return

        self.all_attr_data = self.df[attr]
        self.time_col = self.df["t"] if "t" in self.df.columns else None

        if self.match_t and self.time_col is not None:
            t = self.viewer.dims.current_step[0]
            mask = self.time_col == t
            self.attr_data = self.all_attr_data[mask]
        else:
            self.attr_data = self.all_attr_data

    def _refresh_plot(self, reset_hist: bool):
        """
        Recompute stats and redraw histogram for the current selection.

        Parameters
        ----------
        reset_hist : bool
            Whether to reset the histogram range and bins.
        """
        self.hist_reset = reset_hist
        self._update_data_for_current_selection()

        if self.attr_data is None or len(self.attr_data) == 0:
            self._clear_canvas()
            self._disable_hist_controls()
            return

        self._enable_hist_controls()
        self.get_stats()
        attr = self._current_attr_name()
        if attr is not None:
            self.plot_data(attr)

    # -------------------------------------------------------------------------
    # adjacency map / overlay
    # -------------------------------------------------------------------------
    def check_for_adjacency_map(self):
        """
        Check whether an adjacency map exists; if not, disable the overlay button.
        """
        if self.overlay_button is None:
            return
        self.overlay_button.setEnabled(False)
        if os.path.exists(self.nellie.im_info.pipeline_paths["adjacency_maps"]):
            self.overlay_button.setEnabled(True)

    def rewrite_dropdown(self):
        """
        Update the hierarchy level dropdown based on available data.
        """
        self.check_for_adjacency_map()

        self.dropdown.clear()
        if os.path.exists(self.nellie.im_info.pipeline_paths["features_nodes"]):
            options = ["none", "voxel", "node", "branch", "organelle", "image"]
        else:
            options = ["none", "voxel", "branch", "organelle", "image"]
        for option in options:
            self.dropdown.addItem(option)

        self.adjacency_maps = None

    # -------------------------------------------------------------------------
    # export helpers
    # -------------------------------------------------------------------------
    def export_data(self):
        """
        Export the current graph data as a CSV file to the graph directory.
        """
        if self.df is None or self._current_attr_name() is None:
            show_info("No data to export. Please select a level and attribute first.")
            return

        attr_name = self._current_attr_name()
        dt = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = self.nellie.im_info.graph_dir
        if not os.path.exists(export_dir):
            os.makedirs(export_dir, exist_ok=True)

        level = self.selected_level or "unknown"
        filename_root = f"{dt}-{level}-{attr_name}"
        subset = self.df

        if self.match_t and "t" in self.df.columns:
            current_t = self.viewer.dims.current_step[0]
            subset = subset[subset["t"] == current_t]
            filename_root += f"_T{current_t}"

        filename_root += f"_{self.nellie.im_info.file_info.filename_no_ext}"
        export_path = os.path.join(export_dir, f"{filename_root}.csv")

        if "t" in subset.columns:
            df_to_save = subset[["t", attr_name]].copy()
        else:
            df_to_save = subset[[attr_name]].copy()

        df_to_save.to_csv(export_path, index=False)
        show_info(f"Data exported to {export_path}")

    def save_graph(self):
        """
        Save the current graph as a PNG file to the graph directory.
        """
        if self.attr_data is None or len(self.attr_data) == 0:
            show_info("No graph to save. Please select a level and attribute first.")
            return

        attr_name = self._current_attr_name() or "attr"
        dt = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = self.nellie.im_info.graph_dir
        if not os.path.exists(export_dir):
            os.makedirs(export_dir, exist_ok=True)

        level = self.selected_level or "unknown"
        filename_root = f"{dt}-{level}-{attr_name}"
        if self.match_t and self.time_col is not None:
            filename_root += f"_T{self.viewer.dims.current_step[0]}"
        filename_root += f"_{self.nellie.im_info.file_info.filename_no_ext}"
        export_path = os.path.join(export_dir, f"{filename_root}.png")

        self.canvas.figure.savefig(export_path, dpi=300)
        show_info(f"Graph saved to {export_path}")

    # -------------------------------------------------------------------------
    # histogram / plotting
    # -------------------------------------------------------------------------
    def on_hist_change(self, event):
        """
        Called when histogram ranges or bin count change.

        Parameters
        ----------
        event : Any
            The event that triggered the change (unused).
        """
        if self.attr_data is None or self.data_to_plot is None:
            return
        self.plot_data(self._current_attr_name() or "")

    # -------------------------------------------------------------------------
    # click mapping
    # -------------------------------------------------------------------------
    def get_index(self, layer, event):
        """
        Retrieve indices of voxel and mapped features based on mouse position.

        Parameters
        ----------
        layer : napari.layers.Layer
            The layer that was clicked.
        event : Any
            The mouse event containing the position.
        """
        if self.label_coords is None or len(self.label_coords) == 0:
            return

        pos = self.viewer.cursor.position
        matched_row = None
        z = None

        if self.nellie.im_info.no_z:
            if len(pos) < 3:
                return
            t, y, x = int(np.round(pos[0])), int(np.round(pos[1])), int(np.round(pos[2]))
            if t < 0 or t >= len(self.label_coords):
                return
            t_coords = self.label_coords[t]
            if t_coords.size == 0:
                return
            match = np.where((t_coords[:, 0] == y) & (t_coords[:, 1] == x))[0]
            if len(match) == 0:
                return
            matched_row = int(match[0])
        else:
            if len(pos) < 4:
                return
            t, z, y, x = (
                int(np.round(pos[0])),
                int(np.round(pos[1])),
                int(np.round(pos[2])),
                int(np.round(pos[3])),
            )
            if t < 0 or t >= len(self.label_coords):
                return
            t_coords = self.label_coords[t]
            if t_coords.size == 0:
                return
            match = np.where(
                (t_coords[:, 0] == z)
                & (t_coords[:, 1] == y)
                & (t_coords[:, 2] == x)
            )[0]
            if len(match) == 0:
                return
            matched_row = int(match[0])

        if matched_row is None:
            return

        # ensure dataframes exist
        if self.voxel_df is None and os.path.exists(self.nellie.im_info.pipeline_paths["features_voxels"]):
            self.voxel_df = pd.read_csv(self.nellie.im_info.pipeline_paths["features_voxels"])
        if self.node_df is None and os.path.exists(self.nellie.im_info.pipeline_paths["features_nodes"]):
            self.node_df = pd.read_csv(self.nellie.im_info.pipeline_paths["features_nodes"])
        if self.branch_df is None and os.path.exists(self.nellie.im_info.pipeline_paths["features_branches"]):
            self.branch_df = pd.read_csv(self.nellie.im_info.pipeline_paths["features_branches"])
        if self.organelle_df is None and os.path.exists(self.nellie.im_info.pipeline_paths["features_organelles"]):
            self.organelle_df = pd.read_csv(self.nellie.im_info.pipeline_paths["features_organelles"])
        if self.image_df is None and os.path.exists(self.nellie.im_info.pipeline_paths["features_image"]):
            self.image_df = pd.read_csv(self.nellie.im_info.pipeline_paths["features_image"])

        t_val = int(t)

        # voxel index
        voxel_idx_str = ""
        if self.voxel_df is not None and "t" in self.voxel_df.columns:
            voxel_slice = self.voxel_df[self.voxel_df["t"] == t_val]
            if 0 <= matched_row < len(voxel_slice):
                voxel_idx = voxel_slice.iloc[matched_row, 0]
                voxel_idx_str = str(voxel_idx)

        node_str = ""
        branch_str = ""
        organelle_str = ""
        image_str = ""

        if self.adjacency_maps is not None:
            # nodes
            if "v_n" in self.adjacency_maps and self.node_df is not None:
                if t_val < len(self.adjacency_maps["v_n"]):
                    v_n = self.adjacency_maps["v_n"][t_val]
                    if v_n.size > 0 and matched_row < v_n.shape[0]:
                        node_rows = np.where(v_n[matched_row])[0]
                        node_slice = (
                            self.node_df[self.node_df["t"] == t_val]
                            if "t" in self.node_df.columns
                            else self.node_df
                        )
                        node_ids = []
                        for r in node_rows:
                            if 0 <= r < len(node_slice):
                                node_ids.append(str(node_slice.iloc[r, 0]))
                        node_str = ", ".join(node_ids)

            # branches
            if "v_b" in self.adjacency_maps and self.branch_df is not None:
                if t_val < len(self.adjacency_maps["v_b"]):
                    v_b = self.adjacency_maps["v_b"][t_val]
                    if v_b.size > 0 and matched_row < v_b.shape[0]:
                        branch_rows = np.where(v_b[matched_row])[0]
                        branch_slice = (
                            self.branch_df[self.branch_df["t"] == t_val]
                            if "t" in self.branch_df.columns
                            else self.branch_df
                        )
                        branch_ids = []
                        for r in branch_rows:
                            if 0 <= r < len(branch_slice):
                                branch_ids.append(str(branch_slice.iloc[r, 0]))
                        branch_str = ", ".join(branch_ids)

            # organelles
            if "v_o" in self.adjacency_maps and self.organelle_df is not None:
                if t_val < len(self.adjacency_maps["v_o"]):
                    v_o = self.adjacency_maps["v_o"][t_val]
                    if v_o.size > 0 and matched_row < v_o.shape[0]:
                        organelle_rows = np.where(v_o[matched_row])[0]
                        organelle_slice = (
                            self.organelle_df[self.organelle_df["t"] == t_val]
                            if "t" in self.organelle_df.columns
                            else self.organelle_df
                        )
                        organelle_ids = []
                        for r in organelle_rows:
                            if 0 <= r < len(organelle_slice):
                                organelle_ids.append(str(organelle_slice.iloc[r, 0]))
                        organelle_str = ", ".join(organelle_ids)

            # images
            if "v_i" in self.adjacency_maps and self.image_df is not None:
                if t_val < len(self.adjacency_maps["v_i"]):
                    v_i = self.adjacency_maps["v_i"][t_val]
                    if v_i.size > 0 and matched_row < v_i.shape[0]:
                        image_rows = np.where(v_i[matched_row])[0]
                        image_slice = (
                            self.image_df[self.image_df["t"] == t_val]
                            if "t" in self.image_df.columns
                            else self.image_df
                        )
                        image_ids = []
                        for r in image_rows:
                            if 0 <= r < len(image_slice):
                                image_ids.append(str(image_slice.iloc[r, 0]))
                        image_str = ", ".join(image_ids)
            elif self.image_df is not None and "t" in self.image_df.columns:
                image_slice = self.image_df[self.image_df["t"] == t_val]
                if len(image_slice) > 0:
                    image_str = str(image_slice.iloc[0, 0])

        # build table
        headers = ["Voxel"]
        values = [voxel_idx_str]
        if self.node_df is not None and node_str != "":
            headers.append("Nodes")
            values.append(node_str)
        if self.branch_df is not None and branch_str != "":
            headers.append("Branch")
            values.append(branch_str)
        if self.organelle_df is not None and organelle_str != "":
            headers.append("Organelle")
            values.append(organelle_str)
        if self.image_df is not None and image_str != "":
            headers.append("Image")
            values.append(image_str)

        if not headers:
            return

        self.click_match_table.clear()
        self.click_match_table.setRowCount(1)
        self.click_match_table.setColumnCount(len(headers))
        self.click_match_table.setHorizontalHeaderLabels(headers)
        for i, val in enumerate(values):
            self.click_match_table.setItem(0, i, QTableWidgetItem(val))

        # coordinate label
        if self.nellie.im_info.no_z:
            coord_str = f"(t={t_val}, y={y}, x={x})"
        else:
            coord_str = f"(t={t_val}, z={z}, y={y}, x={x})"
        self.click_match_table.setVerticalHeaderLabels([coord_str])

        if self.click_match_group is not None:
            self.click_match_group.setVisible(True)

    # -------------------------------------------------------------------------
    # overlay
    # -------------------------------------------------------------------------
    def overlay(self):
        """
        Apply an overlay of attribute data onto the image in the napari viewer,
        using adjacency maps to map higher-level features to voxels.
        """
        if self.selected_attr is None or self.df is None:
            show_info("Please select a hierarchy level and attribute before overlay.")
            return

        # build label mask / coords once
        if self.label_mask is None:
            label_mask = self.nellie.im_info.get_memmap(
                self.nellie.im_info.pipeline_paths["im_instance_label"]
            )
            self.label_mask = (label_mask > 0).astype(float)
            self.label_coords = []
            for t in range(self.nellie.im_info.shape[0]):
                coords = np.argwhere(self.label_mask[t])
                self.label_coords.append(coords)
                self.label_mask[t] *= np.nan

        # load adjacency maps if needed
        if self.adjacency_maps is None:
            pkl_path = self.nellie.im_info.pipeline_paths["adjacency_maps"]
            if not os.path.exists(pkl_path):
                show_info("Adjacency maps not found; cannot compute overlay.")
                return

            with open(pkl_path, "rb") as f:
                adjacency_slices = pickle.load(f)

            self.adjacency_maps = {
                "n_v": [],
                "b_v": [],
                "o_v": [],
                "v_n": [],
                "v_b": [],
                "v_o": [],
            }
            if "v_i" in adjacency_slices:
                self.adjacency_maps["i_v"] = []
                self.adjacency_maps["v_i"] = []

            # node adjacency
            if "v_n" in adjacency_slices:
                for t, adjacency_slice in enumerate(adjacency_slices["v_n"]):
                    if t >= len(self.label_coords):
                        self.adjacency_maps["n_v"].append(np.zeros((0, 0), dtype=bool))
                        self.adjacency_maps["v_n"].append(np.zeros((0, 0), dtype=bool))
                        continue
                    if (
                        len(self.label_coords[t]) == 0
                        or adjacency_slice is None
                        or adjacency_slice.size == 0
                    ):
                        self.adjacency_maps["n_v"].append(np.zeros((0, 0), dtype=bool))
                        self.adjacency_maps["v_n"].append(np.zeros((0, 0), dtype=bool))
                        continue
                    max_node = int(np.max(adjacency_slice[:, 1])) + 1
                    adjacency_matrix = np.zeros(
                        (len(self.label_coords[t]), max_node), dtype=bool
                    )
                    adjacency_matrix[
                        adjacency_slice[:, 0], adjacency_slice[:, 1]
                    ] = True
                    self.adjacency_maps["v_n"].append(adjacency_matrix)
                    self.adjacency_maps["n_v"].append(adjacency_matrix.T)

            # branch adjacency
            if "v_b" in adjacency_slices:
                for t, adjacency_slice in enumerate(adjacency_slices["v_b"]):
                    if t >= len(self.label_coords):
                        self.adjacency_maps["b_v"].append(np.zeros((0, 0), dtype=bool))
                        self.adjacency_maps["v_b"].append(np.zeros((0, 0), dtype=bool))
                        continue
                    if (
                        len(self.label_coords[t]) == 0
                        or adjacency_slice is None
                        or adjacency_slice.size == 0
                    ):
                        self.adjacency_maps["b_v"].append(np.zeros((0, 0), dtype=bool))
                        self.adjacency_maps["v_b"].append(np.zeros((0, 0), dtype=bool))
                        continue
                    max_branch = int(np.max(adjacency_slice[:, 1])) + 1
                    adjacency_matrix = np.zeros(
                        (len(self.label_coords[t]), max_branch), dtype=bool
                    )
                    adjacency_matrix[
                        adjacency_slice[:, 0], adjacency_slice[:, 1]
                    ] = True
                    self.adjacency_maps["v_b"].append(adjacency_matrix)
                    self.adjacency_maps["b_v"].append(adjacency_matrix.T)

            # organelle adjacency
            if "v_o" in adjacency_slices:
                for t, adjacency_slice in enumerate(adjacency_slices["v_o"]):
                    if t >= len(self.label_coords):
                        self.adjacency_maps["o_v"].append(np.zeros((0, 0), dtype=bool))
                        self.adjacency_maps["v_o"].append(np.zeros((0, 0), dtype=bool))
                        continue
                    if (
                        len(self.label_coords[t]) == 0
                        or adjacency_slice is None
                        or adjacency_slice.size == 0
                    ):
                        self.adjacency_maps["o_v"].append(np.zeros((0, 0), dtype=bool))
                        self.adjacency_maps["v_o"].append(np.zeros((0, 0), dtype=bool))
                        continue
                    # organelles indexed consecutively over the full timelapse
                    max_organelle = int(np.max(adjacency_slice[:, 1])) + 1
                    adjacency_matrix = np.zeros(
                        (len(self.label_coords[t]), max_organelle), dtype=bool
                    )
                    adjacency_matrix[
                        adjacency_slice[:, 0], adjacency_slice[:, 1]
                    ] = True
                    self.adjacency_maps["v_o"].append(adjacency_matrix)
                    self.adjacency_maps["o_v"].append(adjacency_matrix.T)

            # image adjacency (if present)
            if "v_i" in adjacency_slices:
                for t, adjacency_slice in enumerate(adjacency_slices["v_i"]):
                    if t >= len(self.label_coords):
                        self.adjacency_maps["i_v"].append(np.zeros((0, 0), dtype=bool))
                        self.adjacency_maps["v_i"].append(np.zeros((0, 0), dtype=bool))
                        continue
                    if (
                        len(self.label_coords[t]) == 0
                        or adjacency_slice is None
                        or adjacency_slice.size == 0
                    ):
                        self.adjacency_maps["i_v"].append(np.zeros((0, 0), dtype=bool))
                        self.adjacency_maps["v_i"].append(np.zeros((0, 0), dtype=bool))
                        continue
                    max_image = int(np.max(adjacency_slice[:, 1])) + 1
                    adjacency_matrix = np.zeros(
                        (len(self.label_coords[t]), max_image), dtype=bool
                    )
                    adjacency_matrix[
                        adjacency_slice[:, 0], adjacency_slice[:, 1]
                    ] = True
                    self.adjacency_maps["v_i"].append(adjacency_matrix)
                    self.adjacency_maps["i_v"].append(adjacency_matrix.T)

        # ensure attribute series across all timepoints
        if self.all_attr_data is None or self.time_col is None:
            self._update_data_for_current_selection()
            if self.all_attr_data is None or self.time_col is None:
                show_info("No attribute data available for overlay.")
                return

        num_t = self.nellie.im_info.shape[0]
        for t in range(num_t):
            if t >= len(self.label_coords):
                continue

            t_mask = self.time_col == t
            t_attr_data = self.all_attr_data[t_mask]
            if t_attr_data is None or len(t_attr_data) == 0:
                continue
            t_attr_data = t_attr_data.astype(float)

            if self.selected_level == "voxel":
                if len(self.label_coords[t]) == len(t_attr_data):
                    self.label_mask[t][tuple(self.label_coords[t].T)] = t_attr_data.values
                continue

            adjacency_mask = None
            if self.selected_level == "node" and self.adjacency_maps is not None:
                if "n_v" in self.adjacency_maps and t < len(self.adjacency_maps["n_v"]):
                    adjacency_mask = np.array(self.adjacency_maps["n_v"][t])
            elif self.selected_level == "branch" and self.adjacency_maps is not None:
                if "b_v" in self.adjacency_maps and t < len(self.adjacency_maps["b_v"]):
                    adjacency_mask = np.array(self.adjacency_maps["b_v"][t])
            elif self.selected_level == "organelle" and self.adjacency_maps is not None:
                if "o_v" in self.adjacency_maps and t < len(self.adjacency_maps["o_v"]):
                    adjacency_mask = np.array(self.adjacency_maps["o_v"][t])
            else:
                continue

            if adjacency_mask is None or adjacency_mask.size == 0:
                continue

            # Align attribute data to global label indices
            if "label" in self.df.columns:
                t_labels = self.df.loc[t_mask, "label"].values.astype(int)
                max_idx = max(adjacency_mask.shape[0], int(np.max(t_labels)) + 1)
            else:
                # Fallback if no label column (should not happen for standard files)
                t_labels = np.arange(len(t_attr_data))
                max_idx = max(adjacency_mask.shape[0], len(t_attr_data))

            if adjacency_mask.shape[0] < max_idx:
                padding = np.zeros(
                    (max_idx - adjacency_mask.shape[0], adjacency_mask.shape[1]),
                    dtype=bool,
                )
                adjacency_mask = np.vstack([adjacency_mask, padding])
            
            # Create sparse vector aligned to global indices
            attr_vec = np.full(max_idx, np.nan)
            # Filter labels that fit in the vector (safety check)
            valid_mask = t_labels < max_idx
            attr_vec[t_labels[valid_mask]] = t_attr_data.values[valid_mask]

            reshaped_t_attr = attr_vec.reshape(-1, 1)
            attributed_voxels = adjacency_mask * reshaped_t_attr
            attributed_voxels[~adjacency_mask] = np.nan
            voxel_attributes = np.nanmean(attributed_voxels, axis=0)

            if len(self.label_coords[t]) == len(voxel_attributes):
                self.label_mask[t][tuple(self.label_coords[t].T)] = voxel_attributes

        # create / update viewer layer
        layer_name = f"{self.selected_level} {self.selected_attr}"

        # determine contrast limits
        if "reassigned" not in (self.selected_attr or ""):
            real_vals = self.all_attr_data.replace([np.inf, -np.inf], np.nan).dropna()
            if len(real_vals) == 0:
                min_val = 0.0
                perc98 = 1.0
            else:
                perc98 = float(np.nanpercentile(real_vals, 98))
                min_val = float(np.nanmin(real_vals))
                min_val = min_val - (abs(min_val) * 0.01)
                if np.isnan(min_val):
                    min_val = float(np.nanmin(real_vals))
                if min_val == perc98:
                    perc98 = min_val + (abs(min_val) * 0.01)
                if np.isnan(perc98):
                    perc98 = float(np.nanmax(real_vals))
            contrast_limits = [min_val, perc98]

        if "reassigned" in (self.selected_attr or ""):
            layer = self.viewer.add_labels(
                self.label_mask.copy().astype("uint64"), scale=self.scale, name=layer_name
            )
        else:
            if not self.nellie.im_info.no_z:
                layer = self.viewer.add_image(
                    self.label_mask.copy(),
                    name=layer_name,
                    opacity=1,
                    colormap="turbo",
                    scale=self.scale,
                    contrast_limits=contrast_limits,
                    interpolation3d="nearest",
                )
            else:
                layer = self.viewer.add_image(
                    self.label_mask.copy(),
                    name=layer_name,
                    opacity=1,
                    colormap="turbo",
                    scale=self.scale,
                    contrast_limits=contrast_limits,
                    interpolation2d="nearest",
                )

        self.label_mask_layer = layer
        layer.mouse_drag_callbacks.append(self.get_index)

        self.viewer.reset_view()

    # -------------------------------------------------------------------------
    # napari dim events
    # -------------------------------------------------------------------------
    def on_t_change(self, event):
        """
        Called when the current timepoint changes in the viewer.

        Parameters
        ----------
        event : Any
            The event that triggered the change.
        """
        if self.match_t:
            self._refresh_plot(reset_hist=False)

    def toggle_match_t(self, state):
        """
        Toggle whether to pool across all timepoints or use the current one.

        Parameters
        ----------
        state : int
            The state of the checkbox (Qt.Checked or Qt.Unchecked).
        """
        self.match_t = state == Qt.Checked
        self._refresh_plot(reset_hist=True)

    def toggle_mean_med(self, state):
        """
        Toggle between mean/std view and median/quartiles.

        Parameters
        ----------
        state : int
            The state of the checkbox (Qt.Checked or Qt.Unchecked).
        """
        self.is_median = state == Qt.Checked
        self._refresh_plot(reset_hist=False)

    # -------------------------------------------------------------------------
    # CSV loading
    # -------------------------------------------------------------------------
    def get_csvs(self):
        """
        Load all feature CSVs into DataFrames.
        """
        self.voxel_df = pd.read_csv(self.nellie.im_info.pipeline_paths["features_voxels"])
        if os.path.exists(self.nellie.im_info.pipeline_paths["features_nodes"]):
            self.node_df = pd.read_csv(self.nellie.im_info.pipeline_paths["features_nodes"])
        self.branch_df = pd.read_csv(self.nellie.im_info.pipeline_paths["features_branches"])
        self.organelle_df = pd.read_csv(self.nellie.im_info.pipeline_paths["features_organelles"])
        self.image_df = pd.read_csv(self.nellie.im_info.pipeline_paths["features_image"])

    # -------------------------------------------------------------------------
    # dropdown handlers
    # -------------------------------------------------------------------------
    def on_level_selected(self, index):
        """
        Called when a hierarchy level is selected from the dropdown.

        Parameters
        ----------
        index : int
            The index of the selected item.
        """
        if self.dropdown is None:
            return

        level = self.dropdown.itemText(index)
        self.selected_level = level
        self.df = None

        if self.overlay_button is not None:
            self.overlay_button.setEnabled(True)

        try:
            if level == "voxel":
                if self.voxel_df is None:
                    self.voxel_df = pd.read_csv(
                        self.nellie.im_info.pipeline_paths["features_voxels"]
                    )
                self.df = self.voxel_df
            elif level == "node":
                if os.path.exists(self.nellie.im_info.pipeline_paths["features_nodes"]):
                    if self.node_df is None:
                        self.node_df = pd.read_csv(
                            self.nellie.im_info.pipeline_paths["features_nodes"]
                        )
                    self.df = self.node_df
                else:
                    self.df = None
            elif level == "branch":
                if self.branch_df is None:
                    self.branch_df = pd.read_csv(
                        self.nellie.im_info.pipeline_paths["features_branches"]
                    )
                self.df = self.branch_df
            elif level == "organelle":
                if self.organelle_df is None:
                    self.organelle_df = pd.read_csv(
                        self.nellie.im_info.pipeline_paths["features_organelles"]
                    )
                self.df = self.organelle_df
            elif level == "image":
                if self.overlay_button is not None:
                    self.overlay_button.setEnabled(False)
                if self.image_df is None:
                    self.image_df = pd.read_csv(
                        self.nellie.im_info.pipeline_paths["features_image"]
                    )
                self.df = self.image_df
            else:
                self.df = None
        except FileNotFoundError:
            self.df = None

        # reset feature/form selection and mapping
        self.feature_map = {}
        self.selected_attr = None
        self.selected_feature = None
        self.selected_form = None
        self.all_attr_data = None
        self.attr_data = None
        self.time_col = None

        # rebuild dropdowns
        self.dropdown_attr.blockSignals(True)
        self.dropdown_attr.clear()
        self.dropdown_attr.addItem("None")

        if self.dropdown_stat is not None:
            self.dropdown_stat.blockSignals(True)
            self.dropdown_stat.clear()
            self.dropdown_stat.addItem("None")
            self.dropdown_stat.setEnabled(False)

        if self.df is not None:
            base_order: list[str] = []
            for col in self.df.columns:
                if col == "t":
                    continue
                if not pd.api.types.is_numeric_dtype(self.df[col]):
                    continue
                base, form = self._split_feature_form(col)
                if base not in self.feature_map:
                    self.feature_map[base] = {}
                    base_order.append(base)
                self.feature_map[base][form] = col

            for base in base_order:
                self.dropdown_attr.addItem(base)

        self.dropdown_attr.blockSignals(False)
        if self.dropdown_stat is not None:
            self.dropdown_stat.blockSignals(False)

        self._clear_canvas()
        self._disable_hist_controls()

    def on_attr_selected(self, index):
        """
        Called when a feature (base attribute) is selected from the dropdown.

        Parameters
        ----------
        index : int
            The index of the selected item.
        """
        if self.dropdown_attr is None or self.df is None:
            return

        # index 0 is "None"
        if index <= 0 or index >= self.dropdown_attr.count():
            self.selected_feature = None
            self.selected_form = None
            self.selected_attr = None
            self.all_attr_data = None
            self.attr_data = None
            self.time_col = None
            self._clear_canvas()
            self._disable_hist_controls()
            if self.dropdown_stat is not None:
                self.dropdown_stat.blockSignals(True)
                self.dropdown_stat.clear()
                self.dropdown_stat.addItem("None")
                self.dropdown_stat.setEnabled(False)
                self.dropdown_stat.blockSignals(False)
            return

        feature = self.dropdown_attr.itemText(index)
        self.selected_feature = feature

        # update the form/statistic dropdown for this feature
        self._populate_stat_dropdown(feature)

        # update current selected_form from the populated dropdown
        self.selected_form = self._current_form_key()

        # recompute and redraw
        self._refresh_plot(reset_hist=True)

    def on_form_selected(self, index):
        """
        Called when a statistic form (mean, std, min, max, ...) is selected.

        Parameters
        ----------
        index : int
            The index of the selected item.
        """
        if self.dropdown_stat is None or self.df is None:
            return

        # index 0 is "None"
        if index <= 0 or index >= self.dropdown_stat.count():
            self.selected_form = None
            self.selected_attr = None
            self.all_attr_data = None
            self.attr_data = None
            self.time_col = None
            self._clear_canvas()
            self._disable_hist_controls()
            return

        self.selected_form = self._current_form_key()
        self._refresh_plot(reset_hist=True)

    # -------------------------------------------------------------------------
    # statistics
    # -------------------------------------------------------------------------
    def get_stats(self):
        """
        Compute basic statistics (mean/std or quartiles) for current attribute.
        """
        if self.attr_data is None or len(self.attr_data) == 0:
            self.data_to_plot = None
            self.mean = self.std = self.median = self.iqr = self.perc75 = self.perc25 = np.nan
            return

        data = self.attr_data.astype(float)

        if self.log_scale:
            positive_mask = data > 0
            data = data[positive_mask]
            data = np.log10(data)
        data = data.replace([np.inf, -np.inf], np.nan).dropna()

        self.data_to_plot = data

        if len(self.data_to_plot) == 0:
            self.mean = self.std = self.median = self.iqr = self.perc75 = self.perc25 = np.nan
            return

        if not self.is_median:
            self.mean = float(np.nanmean(self.data_to_plot))
            self.std = float(np.nanstd(self.data_to_plot))
        else:
            self.median = float(np.nanmedian(self.data_to_plot))
            self.perc25 = float(np.nanpercentile(self.data_to_plot, 25))
            self.perc75 = float(np.nanpercentile(self.data_to_plot, 75))
            self.iqr = self.perc75 - self.perc25

    def draw_stats(self):
        """
        Draw statistics on the histogram plot.
        """
        if self.data_to_plot is None or len(self.data_to_plot) == 0:
            return

        axes = self.canvas.figure.get_axes()
        if not axes:
            return
        ax = axes[0]

        if self.is_median:
            ax.axvline(self.perc25, color="r", linestyle="--", label="25th percentile")
            ax.axvline(self.median, color="m", linestyle="-", label="Median")
            ax.axvline(self.perc75, color="r", linestyle="--", label="75th percentile")
        else:
            ax.axvline(self.mean - self.std, color="b", linestyle="--", label="Mean - Std")
            ax.axvline(self.mean, color="c", linestyle="-", label="Mean")
            ax.axvline(self.mean + self.std, color="b", linestyle="--", label="Mean + Std")

        ax.legend()
        self.canvas.draw()

    def plot_data(self, title: str):
        """
        Plot the currently selected attribute data as a histogram.

        Parameters
        ----------
        title : str
            The title for the plot.
        """
        if self.data_to_plot is None or len(self.data_to_plot) == 0:
            self._clear_canvas()
            return

        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)

        data = self.data_to_plot

        try:
            if self.hist_reset:
                nbins = int(len(data) ** 0.5) or 1
                ax.hist(data, bins=nbins)
                hist_min, hist_max = ax.get_xlim()
            else:
                nbins = max(1, int(self.num_bins.value()))
                hist_min = self.hist_min.value()
                hist_max = self.hist_max.value()
                ax.hist(data, bins=nbins, range=(hist_min, hist_max))
        except ValueError:
            nbins = 10
            hist_min, hist_max = 0.0, 1.0
            ax.hist(data, bins=nbins, range=(hist_min, hist_max))

        if self.is_median:
            full_title = (
                f"{title}\n\n"
                f"Quartiles: {self.perc25:.4f}, {self.median:.4f}, {self.perc75:.4f}"
            )
        else:
            full_title = f"{title}\n\nMean: {self.mean:.4f}, Std: {self.std:.4f}"

        if self.match_t and self.time_col is not None:
            full_title += f"\nTimepoint: {self.viewer.dims.current_step[0]}"
        else:
            full_title += "\nTimepoint: all (pooled)"

        ax.set_title(full_title)
        ax.set_xlabel("Value (log10)" if self.log_scale else "Value")
        ax.set_ylabel("Frequency")

        self.canvas.draw()
        self.draw_stats()

        if self.hist_reset:
            self.hist_min.blockSignals(True)
            self.hist_min.setEnabled(True)
            self.hist_min.setRange(hist_min, hist_max)
            self.hist_min.setValue(hist_min)
            step = (hist_max - hist_min) / 100 if hist_max > hist_min else 1.0
            self.hist_min.setSingleStep(step)
            self.hist_min.blockSignals(False)

            self.hist_max.blockSignals(True)
            self.hist_max.setEnabled(True)
            self.hist_max.setRange(hist_min, hist_max)
            self.hist_max.setValue(hist_max)
            self.hist_max.setSingleStep(step)
            self.hist_max.blockSignals(False)

            self.num_bins.blockSignals(True)
            self.num_bins.setEnabled(True)
            self.num_bins.setRange(1, max(1, len(data)))
            self.num_bins.setValue(nbins)
            self.num_bins.blockSignals(False)

            self.hist_reset = False

    def on_log_scale(self, state):
        """
        Toggle logarithmic scaling for the histogram and refresh.

        Parameters
        ----------
        state : int
            The state of the checkbox (Qt.Checked or Qt.Unchecked).
        """
        self.log_scale = state == Qt.Checked
        self._refresh_plot(reset_hist=True)


if __name__ == "__main__":
    import napari

    viewer = napari.Viewer()
    napari.run()