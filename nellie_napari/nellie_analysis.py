import os
import pickle

import numpy as np
from qtpy.QtWidgets import QComboBox, QCheckBox, QPushButton, QLabel, QTableWidget, QTableWidgetItem, \
    QSpinBox, QDoubleSpinBox, QWidget, QVBoxLayout, QGroupBox, QHBoxLayout
from qtpy.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from napari.utils.notifications import show_info
import pandas as pd
import datetime


class NellieAnalysis(QWidget):
    """
    A class for analyzing and visualizing multi-dimensional microscopy data using histograms, graphs, and overlays in the napari viewer.

    Attributes
    ----------
    viewer : napari.viewer.Viewer
        An instance of the napari viewer.
    nellie : object
        Reference to the main Nellie object that contains the pipeline and analysis data.
    canvas : FigureCanvasQTAgg
        Canvas for rendering matplotlib plots.
    scale : tuple
        The scaling factors for X, Y, and Z dimensions, default is (1, 1, 1).
    log_scale : bool
        Boolean flag to toggle logarithmic scaling in histogram plots.
    is_median : bool
        Boolean flag to toggle between mean and median views.
    match_t : bool
        Boolean flag to enable/disable timepoint matching in data analysis.
    hist_reset : bool
        Boolean flag indicating whether the histogram settings are reset.
    voxel_df, node_df, branch_df, organelle_df, image_df : pd.DataFrame
        DataFrames containing features at different hierarchy levels (voxel, node, branch, organelle, image).
    attr_data : pd.Series or None
        Data for the selected attribute to be plotted.
    time_col : pd.Series or None
        Time column data from the currently selected level's DataFrame.
    adjacency_maps : dict or None
        Dictionary containing adjacency matrices for mapping hierarchy levels.
    mean, std, median, iqr, perc75, perc25 : float
        Statistical values for the selected attribute data (mean, std, median, interquartile range, 75th percentile, 25th percentile).

    Methods
    -------
    reset()
        Resets the internal state, including dataframes and initialization flags.
    post_init()
        Initializes UI elements such as checkboxes, buttons, and connects them to respective event handlers.
    set_ui()
        Sets up the user interface layout, including attribute selection, histogram, and export options.
    _create_dropdown_selection()
        Creates and configures the dropdowns for selecting hierarchy levels and attributes.
    set_default_dropdowns()
        Sets default values for the hierarchy level and attribute dropdowns.
    check_for_adjacency_map()
        Checks if an adjacency map is available and enables the overlay button accordingly.
    rewrite_dropdown()
        Rewrites the dropdown options based on the available data and features.
    export_data()
        Exports the current graph data to a CSV file.
    save_graph()
        Saves the current graph as a PNG file.
    on_hist_change(event)
        Event handler for histogram changes (e.g., adjusting bins or min/max values).
    get_index(layer, event)
        Gets the voxel index based on mouse hover coordinates in the napari viewer.
    overlay()
        Applies an overlay of selected attribute data onto the image, using adjacency maps to map voxel to higher-level features.
    on_t_change(event)
        Event handler that updates the graph when the timepoint changes in the napari viewer.
    toggle_match_t(state)
        Toggles timepoint matching and updates the graph accordingly.
    toggle_mean_med(state)
        Toggles between mean and median views and updates the graph.
    get_csvs()
        Loads the feature CSV files into DataFrames for voxels, nodes, branches, organelles, and images.
    on_level_selected(index)
        Event handler for when a hierarchy level is selected from the dropdown.
    on_attr_selected(index)
        Event handler for when an attribute is selected from the dropdown.
    get_stats()
        Computes basic statistics (mean, std, median, percentiles) for the selected attribute data.
    draw_stats()
        Draws the computed statistics on the histogram plot (e.g., mean, std, median, percentiles).
    plot_data(title)
        Plots the selected attribute data as a histogram, updates the UI, and displays statistical information.
    on_log_scale(state)
        Toggles logarithmic scaling for the histogram plot and refreshes the data.
    """
    def __init__(self, napari_viewer: 'napari.viewer.Viewer', nellie, parent=None):
        """
        Initializes the NellieAnalysis class.

        Parameters
        ----------
        napari_viewer : napari.viewer.Viewer
            Reference to the napari viewer instance.
        nellie : object
            Reference to the main Nellie object containing image and pipeline data.
        parent : QWidget, optional
            Optional parent widget (default is None).
        """
        super().__init__(parent)
        self.nellie = nellie
        self.viewer = napari_viewer

        self.canvas = FigureCanvasQTAgg()
        self.canvas.figure.set_layout_engine("constrained")

        self.scale = (1, 1, 1)

        self.log_scale = False
        self.is_median = False
        self.mean_median_toggle = None
        self.overlay_button = None
        self.match_t_toggle = None
        self.match_t = False
        self.hist_min = None
        self.hist_max = None
        self.num_bins = None
        self.hist_reset = True
        self.export_data_button = None
        self.save_graph_button = None

        self.click_match_table = None

        self.voxel_df = None
        self.node_df = None
        self.branch_df = None
        self.organelle_df = None
        self.image_df = None

        self.all_attr_data = None
        self.attr_data = None
        self.time_col = None
        self.adjacency_maps = None
        self.data_to_plot = None
        self.mean = np.nan
        self.std = np.nan
        self.median = np.nan
        self.iqr = np.nan
        self.perc75 = np.nan
        self.perc25 = np.nan

        # self.layout = QGridLayout()
        # self.setLayout(self.layout)

        self.dropdown = None
        self.dropdown_attr = None
        self.log_scale_checkbox = None
        self.selected_level = None

        self.label_mask = None
        self.label_mask_layer = None
        self.label_coords = []

        self.click_match_texts = []

        self.layout_anchors = {
            'dropdown': (0, 0),
            'canvas': (1, 0),
            'table': (50, 0)
        }

        self.df = None
        self.initialized = False

    def reset(self):
        """
        Resets the internal state, including the DataFrames and initialization flags.
        """
        self.initialized = False
        self.voxel_df = None
        self.node_df = None
        self.branch_df = None
        self.organelle_df = None
        self.image_df = None

    def post_init(self):
        """
        Initializes UI elements such as checkboxes, buttons, and connects them to their respective event handlers.
        """
        self.log_scale_checkbox = QCheckBox("Log scale")
        self.log_scale_checkbox.stateChanged.connect(self.on_log_scale)

        self.mean_median_toggle = QCheckBox("Median view")
        self.mean_median_toggle.stateChanged.connect(self.toggle_mean_med)

        self.overlay_button = QPushButton("Overlay mask")
        self.overlay_button.clicked.connect(self.overlay)

        self.match_t_toggle = QCheckBox("Timepoint data")
        self.match_t_toggle.stateChanged.connect(self.toggle_match_t)
        self.match_t_toggle.setEnabled(False)
        self.viewer.dims.events.current_step.connect(self.on_t_change)

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

        self.export_data_button = QPushButton("Export graph data")
        self.export_data_button.clicked.connect(self.export_data)

        self.save_graph_button = QPushButton("Save graph")
        self.save_graph_button.clicked.connect(self.save_graph)

        if self.nellie.im_info.no_z:
            self.scale = (self.nellie.im_info.dim_res['Y'], self.nellie.im_info.dim_res['X'])
        else:
            self.scale = (self.nellie.im_info.dim_res['Z'], self.nellie.im_info.dim_res['Y'], self.nellie.im_info.dim_res['X'])
        self.viewer.scale_bar.visible = True
        self.viewer.scale_bar.unit = 'um'

        self._create_dropdown_selection()
        self.check_for_adjacency_map()

        # # self.dropdown_attr = QComboBox()
        # self._create_dropdown_selection()
        # self.dropdown.setCurrentIndex(3)  # Organelle
        # self.dropdown_attr.currentIndexChanged.connect(self.on_attr_selected)
        # self.dropdown_attr.setCurrentIndex(6)  # Area

        self.set_ui()
        self.initialized = True

    def set_ui(self):
        """
        Sets up the user interface layout, including dropdowns for attribute selection, histogram controls, and export buttons.
        """
        main_layout = QVBoxLayout()

        # Attribute dropdown group
        attr_group = QGroupBox("Select hierarchy level and attribute")
        attr_layout = QHBoxLayout()
        attr_layout.addWidget(self.dropdown)
        attr_layout.addWidget(self.dropdown_attr)
        attr_group.setLayout(attr_layout)

        # Histogram group
        hist_group = QGroupBox("Histogram options")
        hist_layout = QVBoxLayout()

        sub_layout = QHBoxLayout()
        sub_layout.addWidget(QLabel("Min"), alignment=Qt.AlignRight)
        sub_layout.addWidget(self.hist_min)
        sub_layout.addWidget(self.hist_max)
        sub_layout.addWidget(QLabel("Max"), alignment=Qt.AlignLeft)
        hist_layout.addLayout(sub_layout)
        hist_layout.addWidget(self.canvas)

        sub_layout = QHBoxLayout()
        sub_layout.addWidget(QLabel("Bins"), alignment=Qt.AlignRight)
        sub_layout.addWidget(self.num_bins)
        hist_layout.addLayout(sub_layout)
        hist_group.setLayout(hist_layout)

        hist_layout.addWidget(self.canvas)

        sub_layout = QHBoxLayout()
        sub_layout.addWidget(self.log_scale_checkbox)
        sub_layout.addWidget(self.mean_median_toggle)
        sub_layout.addWidget(self.match_t_toggle)
        sub_layout.addWidget(self.overlay_button)
        hist_layout.addLayout(sub_layout)

        # Save options group
        save_group = QGroupBox("Export options")
        save_layout = QVBoxLayout()
        sub_layout = QHBoxLayout()
        sub_layout.addWidget(self.export_data_button)
        sub_layout.addWidget(self.save_graph_button)
        save_layout.addLayout(sub_layout)
        save_group.setLayout(save_layout)

        main_layout.addWidget(attr_group)
        main_layout.addWidget(hist_group)
        main_layout.addWidget(save_group)
        self.setLayout(main_layout)

    def _create_dropdown_selection(self):
        """
        Creates and configures the dropdown menus for selecting hierarchy levels and attributes.
        """
        # Create the dropdown menu
        self.dropdown = QComboBox()
        self.dropdown.currentIndexChanged.connect(self.on_level_selected)

        self.rewrite_dropdown()

        self.dropdown_attr = QComboBox()
        self.dropdown_attr.currentIndexChanged.connect(self.on_attr_selected)

        self.set_default_dropdowns()

    def set_default_dropdowns(self):
        """
        Sets the default values for the hierarchy level and attribute dropdowns.
        """
        organelle_idx = self.dropdown.findText('organelle')
        self.dropdown.setCurrentIndex(organelle_idx)
        area_raw_idx = self.dropdown_attr.findText('organelle_area_raw')
        self.dropdown_attr.setCurrentIndex(area_raw_idx)

    def check_for_adjacency_map(self):
        """
        Checks whether an adjacency map exists, and enables the overlay button if found.
        """
        self.overlay_button.setEnabled(False)
        if os.path.exists(self.nellie.im_info.pipeline_paths['adjacency_maps']):
            self.overlay_button.setEnabled(True)

    def rewrite_dropdown(self):
        """
        Updates the hierarchy level dropdown based on the available data, and checks for adjacency maps.
        """
        self.check_for_adjacency_map()

        self.dropdown.clear()
        if os.path.exists(self.nellie.im_info.pipeline_paths['features_nodes']):
            options = ['none', 'voxel', 'node', 'branch', 'organelle', 'image']
        else:
            options = ['none', 'voxel', 'branch', 'organelle', 'image']
        for option in options:
            self.dropdown.addItem(option)

        if self.dropdown_attr is not None:
            self.set_default_dropdowns()

        self.adjacency_maps = None

    def export_data(self):
        """
        Exports the current graph data as a CSV file to a specified directory.
        """
        dt = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = self.nellie.im_info.graph_dir
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        text = f"{dt}-{self.selected_level}-{self.dropdown_attr.currentText()}"
        if self.match_t:
            current_t = self.viewer.dims.current_step[0]
            text += f"_T{current_t}"
            timepoints = self.time_col[self.df['t'] == current_t]
        else:
            timepoints = self.time_col
        text += self.nellie.im_info.file_info.filename_no_ext
        export_path = os.path.join(export_dir, f"{text}.csv")
        df_to_save = pd.DataFrame({'t': timepoints, self.dropdown_attr.currentText(): self.attr_data})
        df_to_save.to_csv(export_path)

        # append time column
        show_info(f"Data exported to {export_path}")

    def save_graph(self):
        """
        Saves the current graph as a PNG file to a specified directory.
        """
        dt = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = self.nellie.im_info.graph_dir
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        text = f"{dt}-{self.selected_level}_{self.dropdown_attr.currentText()}"
        if self.match_t:
            text += f"_T{self.viewer.dims.current_step[0]}"
        text += self.nellie.im_info.file_info.filename_no_ext
        export_path = os.path.join(export_dir, f"{text}.png")
        self.canvas.figure.savefig(export_path, dpi=300)
        show_info(f"Graph saved to {export_path}")

    def on_hist_change(self, event):
        """
        Event handler for updating the histogram when changes are made (e.g., adjusting the number of bins, min/max values).
        """
        self.plot_data(self.dropdown_attr.currentText())

    def get_index(self, layer, event):
        """
        Retrieves the index of the voxel or feature based on mouse hover coordinates in the napari viewer.

        Parameters
        ----------
        layer : Layer
            The layer on which the event is triggered.
        event : Event
            The event triggered by mouse hover.
        """
        # get the coordinates of where the mouse is hovering
        pos = self.viewer.cursor.position
        matched_row = None
        if self.nellie.im_info.no_z:
            t, y, x = int(np.round(pos[0])), int(np.round(pos[1])), int(np.round(pos[2]))
        else:
            # todo, gonna be more complicated than this I think
            t, z, y, x = int(np.round(pos[0])), int(np.round(pos[1])), int(np.round(pos[2])), int(np.round(pos[3]))
        # find corresponding coord in self.label_coords
        if self.nellie.im_info.no_z:
            t_coords = self.label_coords[t]
            match = np.where((t_coords[:, 0] == y) & (t_coords[:, 1] == x))
            if len(match[0]) == 0:
                return
            else:
                matched_row = match[0][0]
            # show_info(f"Matched csv index: {matched_idx}, at T,Y,X: {t, y, x}")
        else:
            coord = self.label_coords[t][z]
        if matched_row is None:
            return
        # get the value of self.voxel_df_idxs[self.voxel_time_col == t] at the matched_idx row
        voxel_idx = self.voxel_df[self.voxel_df.columns[0]][self.voxel_df['t'] == t].iloc[matched_row]
        node_row = np.where(self.adjacency_maps['v_n'][t][matched_row])[0]
        node_idx = self.node_df[self.node_df.columns[0]][self.node_df['t'] == t].iloc[node_row].values
        branch_row = np.where(self.adjacency_maps['v_b'][t][matched_row])[0][0]
        branch_idx = self.branch_df[self.branch_df.columns[0]][self.branch_df['t'] == t].iloc[branch_row]
        organelle_row = np.where(self.adjacency_maps['v_o'][t][matched_row])[0][0]
        organelle_idx = self.organelle_df[self.organelle_df.columns[0]][self.organelle_df['t'] == t].iloc[organelle_row]
        image_row = np.where(self.adjacency_maps['v_i'][t][matched_row])[0][0]
        image_idx = self.image_df[self.image_df.columns[0]][self.image_df['t'] == t].iloc[image_row]

        self.click_match_table = QTableWidget()
        self.click_match_table.setRowCount(1)
        items = [f"{voxel_idx}", f"{node_idx}", f"{branch_idx}", f"{organelle_idx}", f"{image_idx}"]
        self.click_match_table.setColumnCount(len(items))
        if os.path.exists(self.nellie.im_info.pipeline_paths['features_nodes']):
            self.click_match_table.setHorizontalHeaderLabels(["Voxel", "Nodes", "Branch", "Organelle", "Image"])
        else:
            self.click_match_table.setHorizontalHeaderLabels(["Voxel", "Branch", "Organelle", "Image"])
        for i, item in enumerate(items):
            self.click_match_table.setItem(0, i, QTableWidgetItem(item))
        self.layout.addWidget(self.click_match_table, self.layout_anchors['table'][0], self.layout_anchors['table'][1], 1, 4)
        self.click_match_table.setVerticalHeaderLabels([f"{t, y, x}\nCSV row"])

    def overlay(self):
        """
        Applies an overlay of attribute data onto the image in the napari viewer, using adjacency maps for mapping between hierarchy levels.
        """
        if self.label_mask is None:
            label_mask = self.nellie.im_info.get_memmap(self.nellie.im_info.pipeline_paths['im_instance_label'])
            self.label_mask = (label_mask > 0).astype(float)
            for t in range(self.nellie.im_info.shape[0]):
                self.label_coords.append(np.argwhere(self.label_mask[t]))
                self.label_mask[t] *= np.nan

        if self.adjacency_maps is None:
            pkl_path = self.nellie.im_info.pipeline_paths['adjacency_maps']
            # load pkl file
            with open(pkl_path, 'rb') as f:
                adjacency_slices = pickle.load(f)
            self.adjacency_maps = {'n_v': [], 'b_v': [], 'o_v': []}
            for t in range(len(adjacency_slices['v_n'])):
                adjacency_slice = adjacency_slices['v_n'][t]
                # num_nodes = np.unique(adjacency_slice[:, 1]).shape[0]
                max_node = np.max(adjacency_slice[:, 1]) + 1
                min_node = np.min(adjacency_slice[:, 1])
                if len(self.label_coords[t]) == 0:
                    continue
                # adjacency_matrix = np.zeros((len(self.label_coords[t]), num_nodes), dtype=bool)
                adjacency_matrix = np.zeros((len(self.label_coords[t]), max_node), dtype=bool)
                adjacency_matrix[adjacency_slice[:, 0], adjacency_slice[:, 1]-min_node] = 1
                self.adjacency_maps['n_v'].append(adjacency_matrix.T)
            for t in range(len(adjacency_slices['v_b'])):
                adjacency_slice = adjacency_slices['v_b'][t]
                max_branch = np.max(adjacency_slice[:, 1]) + 1
                min_branch = np.min(adjacency_slice[:, 1])
                if len(self.label_coords[t]) == 0:
                    continue
                adjacency_matrix = np.zeros((len(self.label_coords[t]), max_branch), dtype=bool)
                adjacency_matrix[adjacency_slice[:, 0], adjacency_slice[:, 1]-min_branch] = 1
                self.adjacency_maps['b_v'].append(adjacency_matrix.T)
            for t in range(len(adjacency_slices['v_o'])):
                # organelles are indexed consecutively over the whole timelapse rather than per timepoint, so different
                #  way to construct the matrices
                adjacency_slice = adjacency_slices['v_o'][t]
                num_organelles = np.unique(adjacency_slice[:, 1]).shape[0]
                min_organelle = np.min(adjacency_slice[:, 1])
                if len(self.label_coords[t]) == 0:
                    continue
                # adjacency_matrix = np.zeros((np.max(adjacency_slice[:, 0])+1, np.max(adjacency_slice[:, 1])+1), dtype=bool)
                adjacency_matrix = np.zeros((len(self.label_coords[t]), num_organelles), dtype=bool)
                adjacency_matrix[adjacency_slice[:, 0], adjacency_slice[:, 1]-min_organelle] = 1
                self.adjacency_maps['o_v'].append(adjacency_matrix.T)
            # adjacency_slice = adjacency_slices['v_b'][t]
            # adjacency_matrix = np.zeros((adjacency_slice.shape[0], np.max(adjacency_slice[:, 1])+1))
            # adjacency_matrix[adjacency_slice[:, 0], adjacency_slice[:, 1]] = 1

        if self.attr_data is None:
            return

        for t in range(self.nellie.im_info.shape[0]):
            t_attr_data = self.all_attr_data[self.time_col == t].astype(float)
            if len(t_attr_data) == 0:
                continue
            if self.selected_level == 'voxel':
                self.label_mask[t][tuple(self.label_coords[t].T)] = t_attr_data
                continue
            elif self.selected_level == 'node' and len(self.adjacency_maps['n_v']) > 0:
                adjacency_mask = np.array(self.adjacency_maps['n_v'][t])
            elif self.selected_level == 'branch':
                adjacency_mask = np.array(self.adjacency_maps['b_v'][t])
            elif self.selected_level == 'organelle':
                adjacency_mask = np.array(self.adjacency_maps['o_v'][t])
            # elif self.selected_level == 'image':
            #     adjacency_mask = np.array(self.adjacency_maps['i_v'][t])
            else:
                return
            reshaped_t_attr = t_attr_data.values.reshape(-1, 1)
            # adjacency_mask = np.array(self.adjacency_maps['n_v'][t])
            if adjacency_mask.shape[0] == 0:
                continue
            attributed_voxels = adjacency_mask * reshaped_t_attr
            attributed_voxels[~adjacency_mask] = np.nan
            voxel_attributes = np.nanmean(attributed_voxels, axis=0)
            # if t > len(self.label_coords):
            #     continue
            # if self.selected_level == 'branch':
                # self.label_mask[t][tuple(self.skel_label_coords[t].T)] = voxel_attributes
            # else:
            self.label_mask[t][tuple(self.label_coords[t].T)] = voxel_attributes

        layer_name = f'{self.selected_level} {self.dropdown_attr.currentText()}'
        if 'reassigned' not in self.dropdown_attr.currentText():
            # label_mask_layer = self.viewer.add_image(self.label_mask, name=layer_name, opacity=1,
            #                                               colormap='turbo', scale=self.scale)
            perc98 = np.nanpercentile(self.attr_data, 98)
            # no nans, no infs
            real_vals = self.attr_data[~np.isnan(self.attr_data)]
            real_vals = real_vals[~np.isinf(real_vals)]
            min_val = np.min(real_vals) - (np.abs(np.min(real_vals)) * 0.01)
            if np.isnan(min_val):
                if len(real_vals) == 0:
                    min_val = 0
                else:
                    min_val = np.nanmin(real_vals)
            if min_val == perc98:
                perc98 = min_val + (np.abs(min_val) * 0.01)
            if np.isnan(perc98):
                if len(real_vals) == 0:
                    perc98 = 1
                else:
                    perc98 = np.nanmax(real_vals)
            contrast_limits = [min_val, perc98]
            # label_mask_layer.contrast_limits = contrast_limits
        # label_mask_layer.name = layer_name
        if not self.nellie.im_info.no_z:
            # if the layer isn't in 3D view, make it 3d view
            self.viewer.dims.ndisplay = 3
            # label_mask_layer.interpolation3d = 'nearest'
        # label_mask_layer.refresh()
        # self.label_mask_layer.mouse_drag_callbacks.append(self.get_index)
        if 'reassigned' in self.dropdown_attr.currentText():
            # make the label_mask_layer a label layer
                self.viewer.add_labels(self.label_mask.copy().astype('uint64'), scale=self.scale, name=layer_name)
        else:
            if not self.nellie.im_info.no_z:
                self.viewer.add_image(self.label_mask.copy(), name=layer_name, opacity=1,
                                      colormap='turbo', scale=self.scale, contrast_limits=contrast_limits,
                                      interpolation3d='nearest')
            else:
                self.viewer.add_image(self.label_mask.copy(), name=layer_name, opacity=1,
                                      colormap='turbo', scale=self.scale, contrast_limits=contrast_limits,
                                      interpolation2d='nearest')

        self.match_t_toggle.setEnabled(True)
        self.viewer.reset_view()

    def on_t_change(self, event):
        """
        Event handler for timepoint changes in the napari viewer. Updates the attribute data and refreshes the plot accordingly.
        """
        if self.match_t:
            self.on_attr_selected(self.dropdown_attr.currentIndex())

    def toggle_match_t(self, state):
        """
        Toggles timepoint matching and updates the graph and data accordingly.

        Parameters
        ----------
        state : int
            The state of the checkbox (checked or unchecked).
        """
        if state == 2:
            self.match_t = True
        else:
            self.match_t = False
        self.on_attr_selected(self.dropdown_attr.currentIndex())

    def toggle_mean_med(self, state):
        """
        Toggles between mean and median views for the histogram plot.

        Parameters
        ----------
        state : int
            The state of the checkbox (checked or unchecked).
        """
        if state == 2:
            self.is_median = True
        else:
            self.is_median = False
        self.on_attr_selected(self.dropdown_attr.currentIndex())

    def get_csvs(self):
        """
        Loads the CSV files containing voxel, node, branch, organelle, and image features into DataFrames.
        """
        self.voxel_df = pd.read_csv(self.nellie.im_info.pipeline_paths['features_voxels'])
        if os.path.exists(self.nellie.im_info.pipeline_paths['features_nodes']):
            self.node_df = pd.read_csv(self.nellie.im_info.pipeline_paths['features_nodes'])
        self.branch_df = pd.read_csv(self.nellie.im_info.pipeline_paths['features_branches'])
        self.organelle_df = pd.read_csv(self.nellie.im_info.pipeline_paths['features_organelles'])
        self.image_df = pd.read_csv(self.nellie.im_info.pipeline_paths['features_image'])

        # self.voxel_time_col = voxel_df['t']
        # self.voxel_df_idxs = voxel_df[voxel_df.columns[0]]

    def on_level_selected(self, index):
        """
        Event handler for when a hierarchy level is selected from the dropdown menu.

        Parameters
        ----------
        index : int
            The index of the selected item in the dropdown.
        """
        # This method is called whenever a radio button is selected
        # 'button' parameter is the clicked radio button
        self.selected_level = self.dropdown.itemText(index)
        self.overlay_button.setEnabled(True)
        if self.selected_level == 'voxel':
            if self.voxel_df is None:
                self.voxel_df = pd.read_csv(self.nellie.im_info.pipeline_paths['features_voxels'])
            self.df = self.voxel_df
        elif self.selected_level == 'node':
            if self.node_df is None:
                if os.path.exists(self.nellie.im_info.pipeline_paths['features_nodes']):
                    self.node_df = pd.read_csv(self.nellie.im_info.pipeline_paths['features_nodes'])
            self.df = self.node_df
        elif self.selected_level == 'branch':
            if self.branch_df is None:
                self.branch_df = pd.read_csv(self.nellie.im_info.pipeline_paths['features_branches'])
            self.df = self.branch_df
        elif self.selected_level == 'organelle':
            if self.organelle_df is None:
                self.organelle_df = pd.read_csv(self.nellie.im_info.pipeline_paths['features_organelles'])
            self.df = self.organelle_df
        elif self.selected_level == 'image':
            # turn off overlay button
            self.overlay_button.setEnabled(False)
            if self.image_df is None:
                self.image_df = pd.read_csv(self.nellie.im_info.pipeline_paths['features_image'])
            self.df = self.image_df
        else:
            return

        self.dropdown_attr.clear()
        # add a None option
        self.dropdown_attr.addItem("None")
        for col in self.df.columns[::-1]:
            # if "raw" not in col:
            #     continue
            # remove "_raw" from the column name
            # col = col[:-4]
            self.dropdown_attr.addItem(col)

    def on_attr_selected(self, index):
        """
        Event handler for when an attribute is selected from the dropdown menu.

        Parameters
        ----------
        index : int
            The index of the selected attribute in the dropdown.
        """
        self.hist_reset = True
        # if there are no items in dropdown_attr, return
        if self.dropdown_attr.count() == 0:
            return

        selected_attr = self.dropdown_attr.itemText(index)
        if selected_attr == '':
            return
        if selected_attr == "None":
            # clear the canvas
            self.canvas.figure.clear()
            self.canvas.draw()
            return

        self.all_attr_data = self.df[selected_attr]
        self.time_col = self.df['t']
        if self.match_t:
            t = self.viewer.dims.current_step[0]
            self.attr_data = self.df[self.df['t'] == t][selected_attr]
        else:
            self.attr_data = self.df[selected_attr]
        self.get_stats()
        self.plot_data(selected_attr)

    def get_stats(self):
        """
        Computes basic statistics (mean, std, median, percentiles) for the currently selected attribute data.
        """
        if self.attr_data is None:
            return
        if not self.log_scale:
            data = self.attr_data
        else:
            data = np.log10(self.attr_data)
            # convert non real numbers to nan
            data = data.replace([np.inf, -np.inf], np.nan)
        # only real data
        data = data.dropna()
        self.data_to_plot = data

        # todo only enable when mean is selected
        if not self.is_median:
            self.mean = np.nanmean(data)
            self.std = np.nanstd(data)

        # todo only enable when median is selected
        if self.is_median:
            self.median = np.nanmedian(data)
            self.perc75 = np.nanpercentile(data, 75)
            self.perc25 = np.nanpercentile(data, 25)
            self.iqr = self.perc75 - self.perc25

    def draw_stats(self):
        """
        Draws statistics on the histogram plot, including lines for mean, median, std, and percentiles.
        """
        if self.attr_data is None:
            return
        # draw lines for mean, median, std, percentiles on the canvas
        ax = self.canvas.figure.get_axes()[0]
        if self.is_median:
            ax.axvline(self.perc25, color='r', linestyle='--', label='25th percentile')
            ax.axvline(self.median, color='m', linestyle='-', label='Median')
            ax.axvline(self.perc75, color='r', linestyle='--', label='75th percentile')
        else:
            ax.axvline(self.mean - self.std, color='b', linestyle='--', label='Mean - Std')
            ax.axvline(self.mean, color='c', linestyle='-', label='Mean')
            ax.axvline(self.mean + self.std, color='b', linestyle='--', label='Mean + Std')
        ax.legend()
        self.canvas.draw()

    def plot_data(self, title):
        """
        Plots the selected attribute data as a histogram, updates the canvas, and displays the computed statistics.

        Parameters
        ----------
        title : str
            The title for the plot, usually the name of the selected attribute.
        """
        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)
        self.data_to_plot = self.data_to_plot.replace([np.inf, -np.inf], np.nan)
        try:
            if self.hist_reset:
                nbins = int(len(self.attr_data) ** 0.5)  # pretty nbins
                ax.hist(self.data_to_plot, bins=nbins)
                hist_min = ax.get_xlim()[0]
                hist_max = ax.get_xlim()[1]
            else:
                nbins = self.num_bins.value()
                hist_min = self.hist_min.value()
                hist_max = self.hist_max.value()
                ax.hist(self.data_to_plot, bins=nbins, range=(hist_min, hist_max))
        except ValueError:
            nbins = 10
            hist_min = 0
            hist_max = 1
        if self.is_median:
            full_title = f"{title}\n\nQuartiles: {self.perc25:.4f}, {self.median:.4f}, {self.perc75:.4f}"
        else:
            full_title = f"{title}\n\nMean: {self.mean:.4f}, Std: {self.std:.4f}"
        if self.match_t:
            full_title += f"\nTimepoint: {self.viewer.dims.current_step[0]}"
        else:
            full_title += f"\nTimepoint: all (pooled)"
        ax.set_title(full_title)
        if not self.log_scale:
            ax.set_xlabel("Value")
        else:
            ax.set_xlabel("Value (log10)")
        ax.set_ylabel("Frequency")
        self.canvas.draw()
        self.draw_stats()

        # if self.hist_min is not enabled
        if self.hist_reset:
            self.hist_min.setEnabled(True)
            self.hist_min.setValue(hist_min)
            self.hist_min.setRange(hist_min, hist_max)
            self.hist_min.setSingleStep((hist_max - hist_min) / 100)

            self.hist_max.setEnabled(True)
            self.hist_max.setValue(hist_max)
            self.hist_max.setRange(hist_min, hist_max)
            self.hist_max.setSingleStep((hist_max - hist_min) / 100)

            self.num_bins.setEnabled(True)
            self.num_bins.setValue(nbins)
            self.num_bins.setRange(1, len(self.attr_data))
            self.hist_reset = False

    def on_log_scale(self, state):
        """
        Toggles logarithmic scaling for the histogram plot and refreshes the data accordingly.

        Parameters
        ----------
        state : int
            The state of the checkbox (checked or unchecked).
        """
        self.hist_reset = True
        if state == 2:
            self.log_scale = True
        else:
            self.log_scale = False
        self.on_attr_selected(self.dropdown_attr.currentIndex())


if __name__ == "__main__":
    import napari
    viewer = napari.Viewer()
    napari.run()
