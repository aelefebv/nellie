import os
import napari
from PyQt5.QtWidgets import QSpinBox
from qtpy.QtWidgets import QWidget, QGridLayout, QLabel, QPushButton, QLineEdit, QFileDialog, QVBoxLayout, QHBoxLayout, QGroupBox
from napari.utils.notifications import show_info
from tifffile import tifffile

from nellie.im_info.verifier import FileInfo, ImInfo
from nellie.utils.general import get_reshaped_image


class NellieFileSelect(QWidget):
    def __init__(self, napari_viewer: 'napari.viewer.Viewer', nellie, parent=None):
        super().__init__(parent)
        self.nellie = nellie
        self.filepath = None
        self.file_info: FileInfo = None
        self.im_info = None

        self.viewer = napari_viewer
        self.viewer.title = 'Nellie Napari'
        # Add text showing what filepath is selected
        self.filepath_text = QLabel(text="No file selected.")
        self.filepath_text.setWordWrap(True)

        # Filepath selector button
        self.filepath_button = QPushButton(text="Select File")
        self.filepath_button.clicked.connect(self.select_filepath)
        self.filepath_button.setEnabled(True)

        self.reset_button = QPushButton(text="Reset")
        self.reset_button.clicked.connect(self.nellie.reset)
        self.reset_button.setEnabled(True)

        self.file_shape_text = QLabel(text="None")
        self.file_shape_text.setWordWrap(True)

        self.current_order_text = QLabel(text="None")
        self.current_order_text.setWordWrap(True)

        self.dim_order_button = QLineEdit(self)
        self.dim_order_button.setText("None")
        self.dim_order_button.setToolTip("Accepted axes: None")
        self.dim_order_button.setEnabled(False)
        self.dim_order_button.textChanged.connect(self.handle_dim_order_changed)

        self.dim_t_text = 'None'
        self.dim_z_text = 'None'
        self.dim_xy_text = 'None'

        self.label_t = QLabel("T resolution (s):")
        self.dim_t_button = QLineEdit(self)
        self.dim_t_button.setText("None")
        self.dim_t_button.setEnabled(False)
        self.dim_t_button.textChanged.connect(self.handle_t_changed)

        self.label_z = QLabel("Z resolution (um):")
        self.dim_z_button = QLineEdit(self)
        self.dim_z_button.setText("None")
        self.dim_z_button.setEnabled(False)
        self.dim_z_button.textChanged.connect(self.handle_z_changed)

        self.label_xy = QLabel("X & Y resolution (um):")
        self.dim_xy_button = QLineEdit(self)
        self.dim_xy_button.setText("None")
        self.dim_xy_button.setEnabled(False)
        self.dim_xy_button.textChanged.connect(self.handle_xy_changed)

        self.label_channel = QLabel("Channel:")
        self.channel_button = QSpinBox(self)
        self.channel_button.setRange(0, 0)
        self.channel_button.setValue(0)
        self.channel_button.setEnabled(False)
        self.channel_button.valueChanged.connect(self.change_channel)

        self.label_time = QLabel("Start frame:")
        self.label_time_2 = QLabel("End frame:")
        self.start_frame_button = QSpinBox(self)
        self.start_frame_button.setRange(0, 0)
        self.start_frame_button.setValue(0)
        self.start_frame_button.setEnabled(False)
        self.start_frame_button.valueChanged.connect(self.change_time)
        self.end_frame_button = QSpinBox(self)
        self.end_frame_button.setRange(0, 0)
        self.end_frame_button.setValue(0)
        self.end_frame_button.setEnabled(False)
        self.end_frame_button.valueChanged.connect(self.change_time)
        self.end_frame_init = False

        self.confirm_button = QPushButton(text="Confirm")
        self.confirm_button.clicked.connect(self.on_confirm)
        self.confirm_button.setEnabled(False)

        self.preview_button = QPushButton(text="Preview image")
        self.preview_button.clicked.connect(self.on_preview)
        self.preview_button.setEnabled(False)

        # self.delete_button = QPushButton(text="Delete image")

        self.process_button = QPushButton(text="Process image")
        self.process_button.clicked.connect(self.on_process)
        self.process_button.setEnabled(False)

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        # File Selection Group
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout()
        file_button_sublayout = QHBoxLayout()
        file_button_sublayout.addWidget(self.filepath_button)
        file_button_sublayout.addWidget(self.reset_button)
        file_layout.addLayout(file_button_sublayout)
        # file_layout.addWidget(self.filepath_button)
        file_sub_layout = QHBoxLayout()
        file_sub_layout.addWidget(QLabel("Selected file:"))
        file_sub_layout.addWidget(self.filepath_text)
        file_layout.addLayout(file_sub_layout)
        file_group.setLayout(file_layout)

        # Axes Info Group
        axes_group = QGroupBox("Axes Information")
        axes_layout = QVBoxLayout()
        for label, button in [
            (QLabel("Dimension order:"), self.dim_order_button),
            (QLabel("File shape:"), self.file_shape_text),
            (QLabel("Current order:"), self.current_order_text)
        ]:
            sub_layout = QHBoxLayout()
            sub_layout.addWidget(label)
            sub_layout.addWidget(button)
            axes_layout.addLayout(sub_layout)
        axes_group.setLayout(axes_layout)

        # Dimensions Group
        dim_group = QGroupBox("Dimension Resolutions")
        dim_layout = QVBoxLayout()
        for label, button in [
            (self.label_t, self.dim_t_button),
            (self.label_z, self.dim_z_button),
            (self.label_xy, self.dim_xy_button)
        ]:
            sub_layout = QHBoxLayout()
            sub_layout.addWidget(label)
            sub_layout.addWidget(button)
            dim_layout.addLayout(sub_layout)
        dim_group.setLayout(dim_layout)

        # Slice Settings Group
        slice_group = QGroupBox("Slice Settings")
        slice_layout = QVBoxLayout()
        for label, button in [
            (self.label_time, self.start_frame_button),
            (self.label_time_2, self.end_frame_button),
            (self.label_channel, self.channel_button)
        ]:
            sub_layout = QHBoxLayout()
            sub_layout.addWidget(label)
            sub_layout.addWidget(button)
            slice_layout.addLayout(sub_layout)
        slice_group.setLayout(slice_layout)

        # Action Buttons Group
        action_group = QGroupBox("Actions")
        action_layout = QHBoxLayout()
        action_layout.addWidget(self.confirm_button)
        action_layout.addWidget(self.preview_button)
        action_layout.addWidget(self.process_button)
        action_group.setLayout(action_layout)

        # Add all groups to main layout
        main_layout.addWidget(file_group)
        main_layout.addWidget(axes_group)
        main_layout.addWidget(dim_group)
        main_layout.addWidget(slice_group)
        main_layout.addWidget(action_group)

        self.setLayout(main_layout)

    def select_filepath(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select file")
        self.validate_path(filepath)
        if self.filepath is None:
            return
        # self.single = True

        self.initialize_single_file()
        filename = os.path.basename(self.filepath)
        self.filepath_text.setText(f"{filename}")

        # self.post_file_selection()

    def validate_path(self, filepath):
        if not filepath:
            show_info("Invalid selection.")
            return None
        self.filepath = filepath

    def initialize_single_file(self):
        self.file_info = FileInfo(self.filepath)
        self.file_info.find_metadata()
        self.file_info.load_metadata()
        self.file_shape_text.setText(f"{self.file_info.shape}")

        self.dim_order_button.setText(f"{self.file_info.axes}")
        self.dim_order_button.setEnabled(True)
        self.on_change()

    def on_change(self):
        self.confirm_button.setEnabled(False)
        self.check_available_dims()
        if len(self.file_info.shape) == 2:
            self.dim_order_button.setToolTip("Accepted axes: 'Y', 'X' (e.g. 'YX')")
        elif len(self.file_info.shape) == 3:
            self.dim_order_button.setToolTip("Accepted axes: ['T' or 'C' or 'Z'], 'Y', 'X' (e.g. 'ZYX')")
        elif len(self.file_info.shape) == 4:
            self.dim_order_button.setToolTip("Accepted axes: ['T' or 'C' or 'Z']x2, 'Y', 'X' (e.g. 'TZYX')")
        elif len(self.file_info.shape) == 5:
            self.dim_order_button.setToolTip("Accepted axes: ['T' or 'C' or 'Z']x3, 'Y', 'X' (e.g. 'TZCYX')")
        elif len(self.file_info.shape) > 5:
            self.dim_order_button.setStyleSheet("background-color: red")
            show_info(f"Error: Too many dimensions found ({self.file_info.shape}).")

        if self.file_info.good_axes:
            current_order_text = f"({', '.join(self.file_info.axes)})"
            self.current_order_text.setText(current_order_text)
            self.dim_order_button.setStyleSheet("background-color: green")
        else:
            self.current_order_text.setText("Invalid")
            self.dim_order_button.setStyleSheet("background-color: red")

        if self.file_info.good_dims and self.file_info.good_axes:
            self.confirm_button.setEnabled(True)

        # self.delete_button.setEnabled(False)
        self.preview_button.setEnabled(False)
        self.process_button.setEnabled(False)
        # check file_info output path. if it exists, enable the delete and preview button
        if os.path.exists(self.file_info.output_path) and self.file_info.good_dims and self.file_info.good_axes:
            # self.delete_button.setEnabled(True)
            self.preview_button.setEnabled(True)
            self.process_button.setEnabled(True)

    def check_available_dims(self):
        def check_dim(dim, dim_button, dim_text):
            dim_button.setStyleSheet("background-color: green")
            if dim in self.file_info.axes:
                dim_button.setEnabled(True)
                if dim_text is None or dim_text == 'None':
                    dim_button.setText(str(self.file_info.dim_res[dim]))
                else:
                    dim_button.setText(dim_text)
                    if self.file_info.dim_res[dim] is None:
                        dim_button.setStyleSheet("background-color: red")
                if dim_button.text() == 'None' or dim_button.text() is None:
                    dim_button.setStyleSheet("background-color: red")
            else:
                dim_button.setEnabled(False)
                if dim in self.file_info.dim_res:
                    dim_button.setText(str(self.file_info.dim_res[dim]))
                else:
                    dim_button.setText("None")

        check_dim('T', self.dim_t_button, self.dim_t_text)
        check_dim('Z', self.dim_z_button, self.dim_z_text)
        check_dim('X', self.dim_xy_button, self.dim_xy_text)
        check_dim('Y', self.dim_xy_button, self.dim_xy_text)

        self.channel_button.setEnabled(False)
        if 'C' in self.file_info.axes:
            self.channel_button.setEnabled(True)
            self.channel_button.setRange(0, self.file_info.shape[self.file_info.axes.index('C')]-1)

        self.start_frame_button.setEnabled(False)
        self.end_frame_button.setEnabled(False)
        if 'T' in self.file_info.axes:
            self.start_frame_button.setEnabled(True)
            self.end_frame_button.setEnabled(True)
            current_start_frame = self.start_frame_button.value()
            current_end_frame = self.end_frame_button.value()
            max_t = self.file_info.shape[self.file_info.axes.index('T')] - 1
            self.start_frame_button.setRange(0, current_end_frame)
            self.end_frame_button.setRange(current_start_frame, max_t)
            if not self.end_frame_init:
                self.start_frame_button.setValue(0)
                self.end_frame_button.setValue(max_t)
                self.end_frame_init = True

    def handle_dim_order_changed(self, text):
        self.file_info.change_axes(text)
        self.end_frame_init = False
        self.on_change()

    def handle_t_changed(self, text):
        self.dim_t_text = text
        try:
            value = float(self.dim_t_text)
            self.file_info.change_dim_res('T', value)
        except ValueError:
            self.file_info.change_dim_res('T', None)
        self.on_change()

    def handle_z_changed(self, text):
        self.dim_z_text = text
        try:
            value = float(self.dim_z_text)
            self.file_info.change_dim_res('Z', value)
        except ValueError:
            self.file_info.change_dim_res('Z', None)
        self.on_change()

    def handle_xy_changed(self, text):
        self.dim_xy_text = text
        try:
            value = float(self.dim_xy_text)
            self.file_info.change_dim_res('X', value)
            self.file_info.change_dim_res('Y', value)
        except ValueError:
            self.file_info.change_dim_res('X', None)
            self.file_info.change_dim_res('Y', None)
        self.on_change()

    def change_channel(self):
        self.file_info.change_selected_channel(self.channel_button.value())
        self.on_change()

    def change_time(self):
        self.file_info.select_temporal_range(self.start_frame_button.value(), self.end_frame_button.value())
        self.on_change()

    def on_confirm(self):
        show_info("Saving OME TIFF file.")
        self.im_info = ImInfo(self.file_info)
        # self.nellie.im_info = self.im_info
        self.on_change()

    def on_delete(self):
        # delete the ome tif saved file from disk
        os.remove(self.file_info.output_path)
        self.on_change()

    def on_process(self):
        # switch to process tab
        self.im_info = ImInfo(self.file_info)
        self.on_change()
        self.nellie.go_process()

    def on_preview(self):
        im_memmap = tifffile.memmap(self.file_info.output_path)
        # num_t = min(2, self.im_info.shape[self.im_info.axes.index('T')])
        if 'Z' in self.file_info.axes:
            scale = (self.file_info.dim_res['Z'], self.file_info.dim_res['Y'], self.file_info.dim_res['X'])
            self.viewer.dims.ndisplay = 3
        else:
            scale = (self.file_info.dim_res['Y'], self.file_info.dim_res['X'])
            self.viewer.dims.ndisplay = 2
        self.viewer.add_image(im_memmap, name=self.file_info.filename_no_ext, scale=scale, blending='additive',
                              interpolation3d='nearest', interpolation2d='nearest')
        self.viewer.scale_bar.visible = True
        self.viewer.scale_bar.unit = 'um'


class NellieFileSelect_old(QWidget):
    def __init__(self, napari_viewer: 'napari.viewer.Viewer', nellie, parent=None):
        super().__init__(parent)
        self.nellie = nellie
        self.filepath = None
        self.im_info = None
        self.im_info_valid = False
        self.file_info = None

        self.num_frames_max = None
        self.num_channels = 1
        self.channel_selected = 0

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
        self.dim_order_button.setToolTip("Accepted axes: ['TZYX', 'TYX', 'TZCYX', 'TCYX', 'TCZYX', 'ZYX', 'ZTYX', 'YX', 'CYX', 'CZYX', 'ZCYX']")
        self.dim_order = None
        self.dim_order_button.setEnabled(False)

        # self.dim_order_button.textChanged.connect(self.handleDimOrderChanged)
        self.channel_input = QPushButton(text="Change channel")
        self.channel_input.setRange(0, 0)
        self.channel_input.setValue(0)
        self.channel_input.setEnabled(False)
        self.channel_input.valueChanged.connect(self.change_channel)
        # add a change channel button
        # add a min and max t button

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

    def change_channel(self):
        self.channel_selected = self.channel_input.value()
        # set max as the size of the 'C' axes in self.file_info
        self.channel_input.setRange(0, self.file_info.shape[self.file_info.axes.index('C')])



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
        self.dim_order_button.setText(self.file_info.axes)

        if not self.file_info.good_axes:
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

        if 'T' in self.file_info.axes:
            self.dim_t = self.file_info.dim_res['T']
            self.dim_t_button.setText(str(self.dim_t))
            self.dim_t_button.setStyleSheet("background-color: green")
        else:
            self.dim_t_button.setEnabled(False)
            self.dim_t_button.setStyleSheet("background-color: red")
            self.dims_valid = False

        if 'Z' in self.file_info.axes:
            self.dim_z = self.file_info.dim_res['Z']
            self.dim_z_button.setText(str(self.dim_z))
            self.dim_z_button.setStyleSheet("background-color: green")
        else:
            self.dim_z_button.setEnabled(False)
            self.dim_z_button.setStyleSheet("background-color: red")
            self.dims_valid = False

        if self.file_info.dim_res['X'] is not None:
            self.dim_xy = self.file_info.dim_res['X']
            self.dim_xy_button.setText(str(self.dim_xy))
            self.dim_xy_button.setStyleSheet("background-color: green")
        elif self.file_info.dim_sizes['Y'] is not None:
            self.dim_xy = self.file_info.dim_sizes['Y']
            self.dim_xy_button.setText(str(self.dim_xy))
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
        if not self.file_info.good_axes:
            show_info("Axes resolutions not set.")
            return False
        if not self.file_info.good_dims:
            show_info("Dimension order is invalid.")
            return False
        self.open_preview_button.setEnabled(True)
        return True

    def change_channel_init(self):
        self.dim_sizes = {'T': None, 'Z': None, 'X': None, 'Y': None}
        self.dim_sizes_changed = False
        self.dim_order = None

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
        self.file_info = FileInfo(self.filepath)
        self.file_info.find_metadata()
        self.file_info.load_metadata()
        # open the file and load its info
        # self.im_info = ImInfo(self.filepath, ch=self.nellie.processor.channel_input.value(),
        #                       dimension_order=dim_order, dim_sizes=dim_sizes)
        # self.nellie.im_info = self.im_info
        # self.dim_order = self.im_info.axes

        # check if the dimension order and axes resolutions are valid
        self.im_info_valid = self.check_im_info_valid()

        self.post_file_selection()

    def initialize_folder(self, filepath):
        dim_sizes, dim_order = self.check_manual_params()
        # open the folder and load all the files it can
        filenames = os.listdir(filepath)
        # only get filenames that end if .tif, .tiff, .ndi, .nd2, .czi, .lif
        valid_filetypes = ['.tif', '.tiff', '.ndi', '.nd2', '.czi', '.lif']
        filenames = [filename for filename in filenames if os.path.splitext(filename)[1] in valid_filetypes]

        self.nellie.valid_files = []
        for filename in filenames:
            show_info(f"Checking file {filename}.")
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
                show_info(f"Invalid dimension order or axes resolutions for file {file.basename_no_ext}.")
                break

        self.im_info = self.nellie.valid_files[0]
        self.post_file_selection()

    def check_analysis_valid(self):
        self.nellie.setTabEnabled(self.nellie.analysis_tab, False)
        if self.single and os.path.exists(self.im_info.pipeline_paths['features_image']):
            self.nellie.setTabEnabled(self.nellie.analysis_tab, True)
            # self.nellie.analyzer.post_init()

    def disable_file_selection(self):
        self.filepath_button.setEnabled(False)
        self.folder_button.setEnabled(False)

    def enable_time_options(self):
        if self.single:
            tab = self.nellie.processor
        else:
            tab = self.nellie.batch_mode

        tab.time_input.setEnabled(False)

        if not self.im_info.no_t and self.num_frames_max is not None:
            tab.time_input.setEnabled(True)
            tab.time_input.setRange(1, self.num_frames_max)
            tab.time_input.setValue(self.num_frames_max)
        else:
            tab.time_input.setRange(self.num_frames_max, self.num_frames_max)
            tab.time_input.setValue(self.num_frames_max)

    def enable_channel_options(self):
        if self.single:
            tab = self.nellie.processor
        else:
            tab = self.nellie.batch_mode

        # if not self.im_info.no_c and self.num_channels > 1:
        # if self.im_info.num_channels_original > 1:
        tab.channel_input.setEnabled(True)
        tab.channel_input.setRange(0, self.im_info.num_channels_original - 1)
        tab.channel_input.setValue(self.nellie.processor.channel_input.value())
        tab.channel_input.setValue(self.channel_selected)
        # else:
        #     tab.channel_input.setEnabled(False)
        #     tab.channel_input.setRange(0, 0)
        #     tab.channel_input.setValue(0)

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

        self.nellie.settings.remove_edges_checkbox.setEnabled(True)
        self.nellie.processor.post_init()

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