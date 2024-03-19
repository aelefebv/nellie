import os
import pickle

import numpy as np
from qtpy.QtWidgets import QGridLayout, QComboBox, QCheckBox, QPushButton, QLabel, QTableWidget, QTableWidgetItem, \
    QSpinBox, QDoubleSpinBox, QWidget
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from napari.utils.notifications import show_info
import pandas as pd
from nellie.utils.general import get_reshaped_image
import datetime


class NellieAnalysis(QWidget):
    def __init__(self, napari_viewer: 'napari.viewer.Viewer', nellie, parent=None):
        super().__init__(parent)
        self.nellie = nellie
        self.viewer = napari_viewer

        self.num_t = None
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

        self.layout = QGridLayout()
        self.setLayout(self.layout)

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

    def post_init(self):
        self.num_t = self.nellie.processor.time_input.value()

        self._create_dropdown_selection()

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

        canvas_width = 4
        canvas_height = 6
        self.layout.addWidget(self.canvas, self.layout_anchors['canvas'][0], self.layout_anchors['canvas'][1],
                              canvas_height, canvas_width)
        self.layout.addWidget(self.export_data_button, self.layout_anchors['canvas'][0]+canvas_height-2, 49, 1, 2)
        self.layout.addWidget(self.save_graph_button, self.layout_anchors['canvas'][0]+canvas_height-1, 49, 1, 2)
        self.layout.addWidget(QLabel("Max"), self.layout_anchors['canvas'][0], 49)
        self.layout.addWidget(self.hist_max, self.layout_anchors['canvas'][0], 50)
        self.layout.addWidget(QLabel("Min"), self.layout_anchors['canvas'][0]+1, 49)
        self.layout.addWidget(self.hist_min, self.layout_anchors['canvas'][0]+1, 50)
        self.layout.addWidget(QLabel("Bins"), self.layout_anchors['canvas'][0]+3, 49)
        self.layout.addWidget(self.num_bins, self.layout_anchors['canvas'][0]+3, 50)

        self.layout.addWidget(self.log_scale_checkbox, self.layout_anchors['table'][0]-1, 0)
        self.layout.addWidget(self.mean_median_toggle, self.layout_anchors['table'][0]-1, 1)
        self.layout.addWidget(self.overlay_button, self.layout_anchors['table'][0]-1, 2)
        self.layout.addWidget(self.match_t_toggle, self.layout_anchors['table'][0]-1, 3)

        if self.nellie.im_info.no_z:
            self.scale = (self.nellie.im_info.dim_sizes['Y'], self.nellie.im_info.dim_sizes['X'])
        else:
            self.scale = (self.nellie.im_info.dim_sizes['Z'], self.nellie.im_info.dim_sizes['Y'], self.nellie.im_info.dim_sizes['X'])
        self.viewer.scale_bar.visible = True
        self.viewer.scale_bar.unit = 'um'

        self.dropdown.setCurrentIndex(1)
        self.dropdown_attr.setCurrentIndex(1)


    def export_data(self):
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
        text += self.nellie.im_info.basename_no_ext
        export_path = os.path.join(export_dir, f"{text}.csv")
        df_to_save = pd.DataFrame({'t': timepoints, self.dropdown_attr.currentText(): self.attr_data})
        df_to_save.to_csv(export_path)

        # append time column
        show_info(f"Data exported to {export_path}")

    def save_graph(self):
        dt = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = self.nellie.im_info.graph_dir
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
        text = f"{dt}-{self.selected_level}_{self.dropdown_attr.currentText()}"
        if self.match_t:
            text += f"_T{self.viewer.dims.current_step[0]}"
        text += self.nellie.im_info.basename_no_ext
        export_path = os.path.join(export_dir, f"{text}.png")
        self.canvas.figure.savefig(export_path, dpi=300)
        show_info(f"Graph saved to {export_path}")

    def on_hist_change(self, event):
        self.plot_data(self.dropdown_attr.currentText())

    def get_index(self, layer, event):

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
        self.click_match_table.setHorizontalHeaderLabels(["Voxel", "Nodes", "Branch", "Organelle", "Image"])
        for i, item in enumerate(items):
            self.click_match_table.setItem(0, i, QTableWidgetItem(item))
        self.layout.addWidget(self.click_match_table, self.layout_anchors['table'][0], self.layout_anchors['table'][1], 1, 4)
        self.click_match_table.setVerticalHeaderLabels([f"{t, y, x}\nCSV row"])


    def overlay(self):
        if self.label_mask is None:
            label_mask = self.nellie.im_info.get_im_memmap(self.nellie.im_info.pipeline_paths['im_instance_label'])
            self.label_mask = (get_reshaped_image(label_mask, self.num_t, self.nellie.im_info) > 0).astype(float)
            # self.label_mask_layer = self.viewer.add_image(self.label_mask, name='objects', opacity=1, colormap='turbo')
            if self.num_t is None:
                self.num_t = self.label_mask.shape[0]
                t_max = self.num_t
            else:
                t_max = min(self.label_mask.shape[0], self.num_t)
            for t in range(t_max):
                self.label_coords.append(np.argwhere(self.label_mask[t]))
                self.label_mask[t] *= np.nan

        if self.adjacency_maps is None:
            pkl_path = self.nellie.im_info.pipeline_paths['adjacency_maps']
            # load pkl file
            with open(pkl_path, 'rb') as f:
                self.adjacency_maps = pickle.load(f)
        if self.attr_data is None:
            return

        for t in range(self.num_t):
            t_attr_data = self.all_attr_data[self.time_col == t].astype(float)
            if len(t_attr_data) == 0:
                continue
            if self.selected_level == 'voxel':
                self.label_mask[t][tuple(self.label_coords[t].T)] = t_attr_data
                continue
            elif self.selected_level == 'node':
                adjacency_mask = np.array(self.adjacency_maps['n_v'][t])
            elif self.selected_level == 'branch':
                adjacency_mask = np.array(self.adjacency_maps['b_v'][t])
            elif self.selected_level == 'organelle':
                adjacency_mask = np.array(self.adjacency_maps['o_v'][t])
            elif self.selected_level == 'image':
                adjacency_mask = np.array(self.adjacency_maps['i_v'][t])
            else:
                return
            reshaped_t_attr = t_attr_data.values.reshape(-1, 1)
            # adjacency_mask = np.array(self.adjacency_maps['n_v'][t])
            attributed_voxels = adjacency_mask * reshaped_t_attr
            attributed_voxels[~adjacency_mask] = np.nan
            voxel_attributes = np.nanmean(attributed_voxels, axis=0)
            self.label_mask[t][tuple(self.label_coords[t].T)] = voxel_attributes

        layer_name = f'{self.selected_level} {self.dropdown_attr.currentText()}'
        if 'reassigned' in self.dropdown_attr.currentText():
            # make the label_mask_layer a label layer
            self.label_mask_layer = self.viewer.add_labels(self.label_mask.astype('uint64'), scale=self.scale, name=layer_name)
        else:
            self.label_mask_layer = self.viewer.add_image(self.label_mask, name=layer_name, opacity=1,
                                                          colormap='turbo', scale=self.scale)
            perc98 = np.nanpercentile(self.attr_data, 98)
            min_val = np.nanmin(self.attr_data) - (np.abs(np.nanmin(self.attr_data)) * 0.01)
            if min_val == perc98:
                perc98 = min_val + (np.abs(min_val) * 0.01)
            contrast_limits = [min_val, perc98]
            self.label_mask_layer.contrast_limits = contrast_limits
        self.label_mask_layer.name = layer_name
        if not self.nellie.im_info.no_z:
            # if the layer isn't in 3D view, make it 3d view
            self.viewer.dims.ndisplay = 3
            self.label_mask_layer.interpolation3d = 'nearest'
        self.label_mask_layer.refresh()
        self.label_mask_layer.mouse_drag_callbacks.append(self.get_index)
        self.match_t_toggle.setEnabled(True)
        self.viewer.reset_view()

    def on_t_change(self, event):
        if self.match_t:
            self.on_attr_selected(self.dropdown_attr.currentIndex())

    def toggle_match_t(self, state):
        if state == 2:
            self.match_t = True
        else:
            self.match_t = False
        self.on_attr_selected(self.dropdown_attr.currentIndex())

    def toggle_mean_med(self, state):
        if state == 2:
            self.is_median = True
        else:
            self.is_median = False
        self.on_attr_selected(self.dropdown_attr.currentIndex())

    def _create_dropdown_selection(self):
        # Create the dropdown menu
        self.dropdown = QComboBox()

        # Add options to the dropdown
        options = ['none', 'voxel', 'node', 'branch', 'organelle', 'image']
        for option in options:
            self.dropdown.addItem(option)

        # Add the dropdown to the layout
        self.layout.addWidget(self.dropdown, self.layout_anchors['dropdown'][0], self.layout_anchors['dropdown'][1])

        # Connect the dropdown's signal to a method to handle selection changes
        self.dropdown.currentIndexChanged.connect(self.on_level_selected)

        self.get_csvs()

    def get_csvs(self):
        self.voxel_df = pd.read_csv(self.nellie.im_info.pipeline_paths['features_voxels'])
        self.node_df = pd.read_csv(self.nellie.im_info.pipeline_paths['features_nodes'])
        self.branch_df = pd.read_csv(self.nellie.im_info.pipeline_paths['features_branches'])
        self.organelle_df = pd.read_csv(self.nellie.im_info.pipeline_paths['features_components'])
        self.image_df = pd.read_csv(self.nellie.im_info.pipeline_paths['features_image'])

        # self.voxel_time_col = voxel_df['t']
        # self.voxel_df_idxs = voxel_df[voxel_df.columns[0]]

    def on_level_selected(self, index):
        # This method is called whenever a radio button is selected
        # 'button' parameter is the clicked radio button
        self.selected_level = self.dropdown.itemText(index)
        if self.selected_level == 'voxel':
            self.df = self.voxel_df
        elif self.selected_level == 'node':
            self.df = self.node_df
        elif self.selected_level == 'branch':
            self.df = self.branch_df
        elif self.selected_level == 'organelle':
            self.df = self.organelle_df
        elif self.selected_level == 'image':
            self.df = self.image_df
        else:
            return
        # self.df = pd.read_csv(csv_path)
        self.dropdown_attr = QComboBox()
        # add a None option
        self.dropdown_attr.addItem("None")
        for col in self.df.columns[::-1]:
            self.dropdown_attr.addItem(col)
        self.layout.addWidget(self.dropdown_attr, self.layout_anchors['dropdown'][0], self.layout_anchors['dropdown'][1] + 1)
        self.dropdown_attr.currentIndexChanged.connect(self.on_attr_selected)

    def on_attr_selected(self, index):
        self.hist_reset = True

        selected_attr = self.dropdown_attr.itemText(index)
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
        self.mean = np.nanmean(data)
        self.std = np.nanstd(data)
        self.median = np.nanmedian(data)
        self.perc75 = np.nanpercentile(data, 75)
        self.perc25 = np.nanpercentile(data, 25)
        self.iqr = self.perc75 - self.perc25

    def draw_stats(self):
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
        self.hist_reset = True
        if state == 2:
            self.log_scale = True
        else:
            self.log_scale = False
        # self.plot_data(self.dropdown_attr.currentText())
        self.on_attr_selected(self.dropdown_attr.currentIndex())

if __name__ == "__main__":
    import napari
    viewer = napari.Viewer()
    napari.run()
