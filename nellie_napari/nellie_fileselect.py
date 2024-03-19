import datetime
import os

import napari
from qtpy.QtWidgets import QWidget, QGridLayout, QLabel, QPushButton, QSpinBox, QCheckBox, QLineEdit, QMessageBox, \
    QFileDialog
from napari.utils.notifications import show_info
from tifffile import tifffile

from nellie.im_info.im_info import ImInfo
from nellie.utils.general import get_reshaped_image


class NellieFileSelect(QWidget):
    def __init__(self, napari_viewer: 'napari.viewer.Viewer', nellie, parent=None):
        super().__init__(parent)
        self.nellie = nellie
        self.filepath = None
        self.im_info = None
        self.im_info_valid = False

        self.num_frames_max = None
        self.num_channels = 1

        self.viewer = napari_viewer
        self.viewer.title = 'Nellie Napari'
        self.setLayout(QGridLayout())

        # Add text showing what filepath is selected
        self.filepath_text = QLabel(text="No file selected.")
        self.filepath_text.setWordWrap(True)

        # Filepath selector button
        self.filepath_button = QPushButton(text="Select File")
        self.filepath_button.clicked.connect(self.select_filepath)
        self.filepath_button.setEnabled(True)
        # Folder selector button
        self.folder_button = QPushButton(text="Select Folder")
        self.folder_button.clicked.connect(self.select_folder)
        self.folder_button.setEnabled(True)

        # reset button
        self.reset_button = QPushButton(text="Reset")
        self.reset_button.clicked.connect(self.reset)
        self.reset_button.setEnabled(True)

        # dims buttons
        self.label_t = QLabel("T resolution (s):")
        self.dim_t_button = QLineEdit(self)
        self.dim_t_button.textChanged.connect(self.handle_t_changed)
        self.dim_t = None
        self.label_z = QLabel("Z resolution (um):")
        self.dim_z_button = QLineEdit(self)
        self.dim_z_button.textChanged.connect(self.handle_z_changed)
        self.dim_z = None
        self.label_xy = QLabel("X & Y resolution (um):")
        self.dim_xy_button = QLineEdit(self)
        self.dim_xy_button.textChanged.connect(self.handle_xy_changed)
        self.dim_xy = None

        self.dim_sizes = {'T': None, 'Z': None, 'X': None, 'Y': None}
        self.dim_sizes_changed = False

        self.dim_order_button = QLineEdit(self)
        self.dim_order_button.setText("Select file.")
        self.dim_order_button.setToolTip("Accepted axes: ['TZYX', 'TYX', 'TZCYX', 'TCYX', 'TCZYX', 'ZYX', 'YX', 'CYX', 'CZYX', 'ZCYX']")
        self.dim_order = None
        self.dim_order_button.setEnabled(False)
        # self.dim_order_button.textChanged.connect(self.handleDimOrderChanged)

        self.set_dims_button = QPushButton(text="Set dimension order")
        self.set_dims_button.clicked.connect(self.handle_dim_order_changed)
        self.set_dims_button.setEnabled(False)

        self.set_res_button = QPushButton(text="Set new resolutions")
        self.set_res_button.clicked.connect(self.handle_res_changed)
        self.set_res_button.setEnabled(False)

        self.open_preview_button = QPushButton(text="Open preview")
        self.open_preview_button.setToolTip("Open two timepoints to check for valid dimension order and resolutions.")
        self.open_preview_button.clicked.connect(self.open_preview)
        self.open_preview_button.setEnabled(False)


        self.layout().addWidget(self.filepath_button, 1, 0, 1, 1)
        self.layout().addWidget(self.folder_button, 1, 1, 1, 1)
        self.layout().addWidget(self.filepath_text, 2, 0, 1, 2)
        self.layout().addWidget(QLabel("Dimension order: "), 3, 0, 1, 1)
        self.layout().addWidget(self.dim_order_button, 3, 1, 1, 1)
        self.layout().addWidget(self.set_dims_button, 4, 1, 1, 1)
        self.layout().addWidget(self.label_t, 5, 0, 1, 1)
        self.layout().addWidget(self.dim_t_button, 5, 1, 1, 1)
        self.layout().addWidget(self.label_z, 6, 0, 1, 1)
        self.layout().addWidget(self.dim_z_button, 6, 1, 1, 1)
        self.layout().addWidget(self.label_xy, 7, 0, 1, 1)
        self.layout().addWidget(self.dim_xy_button, 7, 1, 1, 1)
        self.layout().addWidget(self.set_res_button, 8, 1, 1, 1)
        self.layout().addWidget(self.open_preview_button, 49, 0, 1, 2)
        self.layout().addWidget(self.reset_button, 50, 0, 2, 2)

        self.nellie_viewer = None
        self.batch_mode = None
        self.nellie_analyzer = None

        self.single = None

    def open_preview(self):
        im_memmap = self.im_info.get_im_memmap(self.im_info.im_path)
        im_memmap_loaded = get_reshaped_image(im_memmap, 2, self.im_info)
        if self.im_info.no_z:
            scale = (self.im_info.dim_sizes['Y'], self.im_info.dim_sizes['X'])
        else:
            scale = (self.im_info.dim_sizes['Z'], self.im_info.dim_sizes['Y'], self.im_info.dim_sizes['X'])
        self.viewer.add_image(im_memmap_loaded, name=self.im_info.basename_no_ext,
                              scale=scale, blending='additive')
        self.viewer.scale_bar.visible = True
        self.viewer.scale_bar.unit = 'um'
        # set dims to 3d
        if self.im_info.no_z:
            self.viewer.dims.ndisplay = 2
        else:
            self.viewer.dims.ndisplay = 3

    def handle_res_changed(self, text):
        if self.dim_t is not None:
            self.dim_sizes['T'] = self.dim_t
        if self.dim_z is not None:
            self.dim_sizes['Z'] = self.dim_z
        if self.dim_xy is not None:
            self.dim_sizes['X'] = self.dim_xy
            self.dim_sizes['Y'] = self.dim_xy
        self.dim_sizes_changed = True
        if self.single:
            self.initialize_single_file()
        else:
            self.initialize_folder(self.filepath)

    def handle_dim_order_changed(self, text):
        self.dim_order = self.dim_order_button.text()
        if self.single:
            self.initialize_single_file()
        else:
            self.initialize_folder(self.filepath)

    def handle_t_changed(self, text):
        self.dim_t = self.handle_float(text)

    def handle_z_changed(self, text):
        self.dim_z = self.handle_float(text)

    def handle_xy_changed(self, text):
        self.dim_xy = self.handle_float(text)

    def handle_float(self, text):
        try:
            # Convert the text to float, store to respective dim
            if text is None:
                return None
            value = float(text)
            return value
        except ValueError:
            # Handle the case where conversion to float fails
            show_info("Please enter a valid number")
            return None

    def check_dims(self):
        self.set_dims_button.setEnabled(True)
        self.dim_order_button.setEnabled(True)
        self.dim_order_button.setText(self.im_info.axes)
        accepted_axes = ['TZYX', 'TYX', 'TZCYX', 'TCYX', 'TCZYX', 'ZYX', 'YX', 'CYX', 'CZYX', 'ZCYX']
        if self.dim_order not in accepted_axes:
            self.im_info.axes_valid = False
        else:
            self.im_info.axes_valid = True

        if not self.im_info.axes_valid:
            show_info("Invalid dimension order. Please check the dimension order.")
            self.dim_order_button.setStyleSheet("background-color: red")
            return
        self.dim_order_button.setStyleSheet("background-color: green")

    def check_axes(self):
        # by default enable all the res buttons
        self.set_res_button.setEnabled(True)
        self.dim_t_button.setEnabled(True)
        self.dim_z_button.setEnabled(True)
        self.dim_xy_button.setEnabled(True)

        self.dims_valid = True

        if self.im_info.no_t:
            self.dim_t_button.setEnabled(False)
        elif self.im_info.dim_sizes['T'] is not None:
            self.dim_t_button.setText(str(self.im_info.dim_sizes['T']))
            self.dim_t = self.im_info.dim_sizes['T']
            self.dim_t_button.setStyleSheet("background-color: green")
        else:
            self.dim_t_button.setStyleSheet("background-color: red")
            self.dims_valid = False

        if self.im_info.no_z:
            self.dim_z_button.setEnabled(False)
        elif self.im_info.dim_sizes['Z'] is not None:
            self.dim_z_button.setText(str(self.im_info.dim_sizes['Z']))
            self.dim_z = self.im_info.dim_sizes['Z']
            self.dim_z_button.setStyleSheet("background-color: green")
        else:
            self.dim_z_button.setStyleSheet("background-color: red")
            self.dims_valid = False

        if self.im_info.dim_sizes['X'] is not None:
            self.dim_xy_button.setText(str(self.im_info.dim_sizes['X']))
            self.dim_xy = self.im_info.dim_sizes['X']
            self.dim_xy_button.setStyleSheet("background-color: green")
        elif self.im_info.dim_sizes['Y'] is not None:
            self.dim_xy_button.setText(str(self.im_info.dim_sizes['Y']))
            self.dim_xy = self.im_info.dim_sizes['Y']
            self.dim_xy_button.setStyleSheet("background-color: green")
        else:
            self.dim_xy_button.setStyleSheet("background-color: red")
            self.dims_valid = False

    def check_manual_params(self):
        if self.dim_sizes_changed:
            dim_sizes = self.dim_sizes
        else:
            dim_sizes = None

        if self.dim_order is None:
            dim_order = ''
        else:
            dim_order = self.dim_order
        return dim_sizes, dim_order

    def check_im_info_valid(self):
        self.open_preview_button.setEnabled(False)
        self.check_dims()
        self.check_axes()
        if not self.dims_valid:
            show_info("Axes resolutions not set.")
            return False
        if not self.im_info.axes_valid:
            show_info("Dimension order is invalid.")
            return False
        self.open_preview_button.setEnabled(True)
        return True

    def set_max_frames(self):
        if self.im_info.no_t:
            self.num_frames_max = 1
        else:  # the index of self.im_info.shape that corresponds to the index of 'T' in self.im_info.axes
            self.num_frames_max = self.im_info.shape[self.im_info.axes.index('T')]

    def set_num_channels(self):
        if self.im_info.no_c:
            self.num_channels = 1
        else:  # the index of self.im_info.shape that corresponds to the index of 'C' in self.im_info.axes
            self.num_channels = self.im_info.shape[self.im_info.axes.index('C')]

    def initialize_single_file(self):
        dim_sizes, dim_order = self.check_manual_params()
        # open the file and load its info
        self.im_info = ImInfo(self.filepath, ch=self.nellie.processor.channel_input.value(), dimension_order=dim_order, dim_sizes=dim_sizes)
        self.nellie.im_info = self.im_info
        self.dim_order = self.im_info.axes

        # check if the dimension order and axes resolutions are valid
        self.im_info_valid = self.check_im_info_valid()

        self.post_file_selection()

    def initialize_folder(self, filepath):
        dim_sizes, dim_order = self.check_manual_params()
        # open the folder and load all the files it can
        filenames = os.listdir(filepath)
        self.nellie.valid_files = []
        for filename in filenames:
            try:
                im_info = ImInfo(os.path.join(filepath, filename), ch=self.nellie.processor.channel_input.value(),
                                 dim_sizes=dim_sizes, dimension_order=dim_order)
                self.nellie.valid_files.append(im_info)
            except:
                pass
        self.nellie.im_info = self.im_info

        self.im_info_valid = True
        for file in self.nellie.valid_files:
            self.im_info = file
            if not self.check_im_info_valid():
                self.im_info_valid = False
                break

        self.im_info = self.nellie.valid_files[0]
        self.post_file_selection()

    def check_analysis_valid(self):
        self.nellie.setTabEnabled(self.nellie.analysis_tab, False)
        if self.single and os.path.exists(self.im_info.pipeline_paths['adjacency_maps']):
            self.nellie.setTabEnabled(self.nellie.analysis_tab, True)
            self.nellie.analyzer.post_init()

    def disable_file_selection(self):
        self.filepath_button.setEnabled(False)
        self.folder_button.setEnabled(False)

    def enable_time_options(self):
        self.nellie.processor.time_input.setEnabled(False)
        self.nellie.batch_mode.time_input.setEnabled(False)
        if not self.im_info.no_t and self.num_frames_max is not None:
            if self.single:
                self.nellie.processor.time_input.setEnabled(True)
                self.nellie.processor.time_input.setRange(1, self.num_frames_max)
                self.nellie.processor.time_input.setValue(self.num_frames_max)
            else:
                self.nellie.batch_mode.time_input.setEnabled(True)
                self.nellie.batch_mode.time_input.setRange(1, self.num_frames_max)
                self.nellie.batch_mode.time_input.setValue(self.num_frames_max)
        else:
            self.nellie.processor.time_input.setRange(self.num_frames_max, self.num_frames_max)
            self.nellie.processor.time_input.setValue(self.num_frames_max)
            self.nellie.batch_mode.time_input.setRange(self.num_frames_max, self.num_frames_max)
            self.nellie.batch_mode.time_input.setValue(self.num_frames_max)

    def enable_channel_options(self):
        if not self.im_info.no_c and self.num_channels > 1:
            self.nellie.processor.channel_input.setEnabled(True)
            self.nellie.processor.channel_input.setRange(0, self.num_channels - 1)
            self.nellie.processor.channel_input.setValue(self.nellie.processor.channel_input.value())
        else:
            self.nellie.processor.channel_input.setEnabled(False)
            self.nellie.processor.channel_input.setRange(0, 0)
            self.nellie.processor.channel_input.setValue(0)

    def post_file_selection(self):
        self.set_max_frames()
        self.set_num_channels()

        # set valid tabs and buttons to be enabled
        self.check_analysis_valid()
        self.disable_file_selection()
        self.enable_channel_options()
        self.enable_time_options()
        if not self.im_info_valid:
            return
        if self.single:
            self.nellie.setTabEnabled(self.nellie.processor_tab, True)
            self.nellie.setTabEnabled(self.nellie.visualizer_tab, True)
            self.nellie.setTabEnabled(self.nellie.batch_tab, False)
        else:
            self.nellie.setTabEnabled(self.nellie.processor_tab, False)
            self.nellie.setTabEnabled(self.nellie.visualizer_tab, False)
            self.nellie.setTabEnabled(self.nellie.analysis_tab, False)
            self.nellie.setTabEnabled(self.nellie.batch_tab, True)
        self.nellie.file_ready()

    def select_folder(self):
        filepath = QFileDialog.getExistingDirectory(self, "Select folder")
        self.validate_path(filepath)
        if self.filepath is None:
            return
        self.single = False

        self.initialize_folder(filepath)
        filename = os.path.basename(self.filepath)
        show_info(f"Selected folder: {filename}")
        self.filepath_text.setText(f"Selected folder: {filename}")

        self.post_file_selection()

    def validate_path(self, filepath):
        if not filepath:
            show_info("Invalid selection.")
            return None
        self.filepath = filepath

    def select_filepath(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select file")
        self.validate_path(filepath)
        if self.filepath is None:
            return
        self.single = True

        self.initialize_single_file()
        filename = os.path.basename(self.filepath)
        show_info(f"Selected file: {filename}")
        self.filepath_text.setText(f"Selected file: {filename}")

        self.post_file_selection()

    def reset(self):
        # switch to nellie's file_select tab
        self.nellie.reset()


if __name__ == "__main__":
    import napari
    viewer = napari.Viewer()
    napari.run()
