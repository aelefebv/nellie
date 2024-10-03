import os
import napari
from PyQt5.QtWidgets import QSpinBox
from qtpy.QtWidgets import QWidget, QLabel, QPushButton, QLineEdit, QFileDialog, QVBoxLayout, QHBoxLayout, QGroupBox
from napari.utils.notifications import show_info
from tifffile import tifffile

from nellie.im_info.verifier import FileInfo, ImInfo


class NellieFileSelect(QWidget):
    """
    A class for selecting and configuring image files for processing in the Nellie pipeline within the napari viewer.

    Attributes
    ----------
    viewer : napari.viewer.Viewer
        The napari viewer instance.
    nellie : object
        Reference to the main Nellie object containing image processing pipelines and functions.
    filepath : str or None
        The selected file path.
    file_info : FileInfo
        Stores metadata and shape information about the selected image file.
    im_info : ImInfo or None
        Contains information about the image for processing after confirmation.
    batch_fileinfo_list : list or None
        List of FileInfo objects when a folder is selected for batch processing.
    filepath_text : QLabel
        Text widget displaying the selected file or folder path.
    filepath_button : QPushButton
        Button to open the file dialog for selecting an image file.
    folder_button : QPushButton
        Button to open the folder dialog for batch processing.
    reset_button : QPushButton
        Button to reset the file selection and clear all settings.
    file_shape_text : QLabel
        Displays the shape of the selected image file.
    current_order_text : QLabel
        Displays the current dimension order (axes) of the image file.
    dim_order_button : QLineEdit
        Input field for entering the dimension order of the image.
    dim_t_button, dim_z_button, dim_xy_button : QLineEdit
        Input fields for entering the resolution of the time (T), Z, and XY dimensions, respectively.
    channel_button : QSpinBox
        Spin box for selecting the channel in the multi-channel image.
    start_frame_button, end_frame_button : QSpinBox
        Spin boxes for selecting the start and end frame for temporal range selection.
    confirm_button : QPushButton
        Button to confirm the file selection and save the OME TIFF file.
    preview_button : QPushButton
        Button to preview the image in the napari viewer.
    process_button : QPushButton
        Button to process the selected image file(s) through the Nellie pipeline.

    Methods
    -------
    init_ui()
        Sets up the user interface layout with file selection, axes information, dimension resolutions, and actions.
    select_filepath()
        Opens a file dialog for selecting an image file and validates the selected file.
    select_folder()
        Opens a folder dialog for batch processing and validates the selected folder.
    validate_path(filepath)
        Validates the selected file or folder path and updates the file path attribute.
    initialize_single_file()
        Initializes the FileInfo for the selected image file, loads metadata, and updates UI elements.
    initialize_folder()
        Initializes the FileInfo objects for all image files in the selected folder.
    on_change()
        Updates UI elements based on the selected file's metadata, checking available dimensions and enabling buttons.
    check_available_dims()
        Checks the availability of specific dimensions (T, Z, XY) in the selected file and enables corresponding UI elements.
    handle_dim_order_changed(text)
        Handles changes in the dimension order (axes) input field and updates the metadata accordingly.
    handle_t_changed(text)
        Handles changes in the time (T) resolution input field and updates the metadata.
    handle_z_changed(text)
        Handles changes in the Z resolution input field and updates the metadata.
    handle_xy_changed(text)
        Handles changes in the XY resolution input field and updates the metadata.
    change_channel()
        Updates the selected channel for the image when the channel spin box is changed.
    change_time()
        Updates the temporal range when the start and end frame spin boxes are changed.
    on_confirm()
        Confirms the file selection, creates an ImInfo object, and prepares the file for processing.
    on_process()
        Prepares the selected file(s) for processing through the Nellie pipeline.
    on_preview()
        Opens a preview of the selected image in the napari viewer, adjusting the display settings based on the file's dimensionality.
    """
    def __init__(self, napari_viewer: 'napari.viewer.Viewer', nellie, parent=None):
        """
        Initializes the NellieFileSelect class.

        Parameters
        ----------
        napari_viewer : napari.viewer.Viewer
            Reference to the napari viewer instance.
        nellie : object
            Reference to the main Nellie object containing image processing pipelines and functions.
        parent : QWidget, optional
            Optional parent widget (default is None).
        """
        super().__init__(parent)
        self.nellie = nellie
        self.filepath = None
        self.file_info: FileInfo = None
        self.im_info = None

        self.batch_fileinfo_list = None

        self.viewer = napari_viewer
        self.viewer.title = 'Nellie Napari'
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

        self.selection_text = QLabel("Selected file:")
        self.selection_text.setWordWrap(True)

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
        """
        Sets up the user interface layout with sections for file selection, axes information, dimension resolutions, and action buttons.
        """
        main_layout = QVBoxLayout()

        # File Selection Group
        file_group = QGroupBox("File Selection")
        file_layout = QVBoxLayout()
        file_button_sublayout = QHBoxLayout()
        file_button_sublayout.addWidget(self.filepath_button)
        file_button_sublayout.addWidget(self.folder_button)
        file_layout.addLayout(file_button_sublayout)
        file_sub_layout = QHBoxLayout()
        file_sub_layout.addWidget(self.selection_text)
        file_sub_layout.addWidget(self.filepath_text)
        file_layout.addLayout(file_sub_layout)
        file_layout.addWidget(self.reset_button)
        file_group.setLayout(file_layout)

        # Axes Info Group
        axes_group = QGroupBox("Axes Information")
        axes_layout = QVBoxLayout()
        sub_layout = QHBoxLayout()
        sub_layout.addWidget(QLabel("Dimension order:"))
        sub_layout.addWidget(self.dim_order_button)
        sub_layout.addWidget(QLabel("Only T, C, Z, Y, and X allowed."))
        axes_layout.addLayout(sub_layout)
        for label, button in [
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
        """
        Opens a file dialog for selecting an image file, validates the selected file, and updates the UI with metadata.
        """
        self.batch_fileinfo_list = None
        filepath, _ = QFileDialog.getOpenFileName(self, "Select file")
        self.validate_path(filepath)
        if self.filepath is None:
            return
        self.selection_text.setText("Selected file:")

        self.file_info = FileInfo(self.filepath)
        self.initialize_single_file()
        filename = os.path.basename(self.filepath)
        self.filepath_text.setText(f"{filename}")

    def select_folder(self):
        """
        Opens a folder dialog for selecting a folder for batch processing and initializes FileInfo objects for all files in the folder.
        """
        folderpath = QFileDialog.getExistingDirectory(self, "Select folder")
        self.validate_path(folderpath)
        if self.filepath is None:
            return
        self.selection_text.setText("Selected folder:")

        self.initialize_folder()
        self.filepath_text.setText(f"{folderpath}")


    def validate_path(self, filepath):
        """
        Validates the selected file or folder path and updates the file path attribute.

        Parameters
        ----------
        filepath : str
            The file or folder path selected by the user.
        """
        if not filepath:
            show_info("Invalid selection.")
            return None
        self.filepath = filepath

    def initialize_single_file(self):
        """
        Initializes the FileInfo object for the selected image file, loads metadata, and updates the dimension resolution fields.
        """
        self.file_info.find_metadata()
        self.file_info.load_metadata()
        self.file_shape_text.setText(f"{self.file_info.shape}")

        self.dim_order_button.setText(f"{self.file_info.axes}")
        self.dim_order_button.setEnabled(True)
        self.on_change()

    def initialize_folder(self):
        """
        Initializes FileInfo objects for all .tif, .tiff, and .nd2 files in the selected folder and loads their metadata.
        """
        # get all .tif, .tiff, and .nd2 files in the folder
        files = [f for f in os.listdir(self.filepath) if f.endswith('.tif') or f.endswith('.tiff') or f.endswith('.nd2')]
        # for each file, create a FileInfo object
        self.batch_fileinfo_list = [FileInfo(os.path.join(self.filepath, f)) for f in files]
        for file_info in self.batch_fileinfo_list:
            file_info.find_metadata()
            file_info.load_metadata()
        self.file_info = self.batch_fileinfo_list[0]
        self.initialize_single_file()
        # This assumes all files in the folder have the same metadata (dim order, resolutions, temporal range, channels)

    def on_change(self):
        """
        Updates the user interface elements, including enabling or disabling buttons based on the file metadata and resolution settings.
        """
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
        if os.path.exists(self.file_info.ome_output_path) and self.file_info.good_dims and self.file_info.good_axes:
            # self.delete_button.setEnabled(True)
            self.preview_button.setEnabled(True)
            self.process_button.setEnabled(True)

    def check_available_dims(self):
        """
        Checks the availability of specific dimensions (T, Z, XY) in the selected file and enables the corresponding input fields for resolutions.
        """
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
        """
        Handles changes in the dimension order input field and updates the FileInfo object accordingly.

        Parameters
        ----------
        text : str
            The new dimension order string entered by the user.
        """
        if self.batch_fileinfo_list is None:
            self.file_info.change_axes(text)
        else:
            for file_info in self.batch_fileinfo_list:
                file_info.change_axes(text)

        self.end_frame_init = False
        self.on_change()

    def handle_t_changed(self, text):
        """
        Handles changes in the time (T) resolution input field and updates the FileInfo object accordingly.

        Parameters
        ----------
        text : str
            The new T resolution entered by the user.
        """
        self.dim_t_text = text
        try:
            value = float(self.dim_t_text)
            if self.batch_fileinfo_list is None:
                self.file_info.change_dim_res('T', value)
            else:
                for file_info in self.batch_fileinfo_list:
                    file_info.change_dim_res('T', value)
        except ValueError:
            if self.batch_fileinfo_list is None:
                self.file_info.change_dim_res('T', None)
            else:
                for file_info in self.batch_fileinfo_list:
                    file_info.change_dim_res('T', None)
        self.on_change()

    def handle_z_changed(self, text):
        """
        Handles changes in the Z resolution input field and updates the FileInfo object accordingly.

        Parameters
        ----------
        text : str
            The new Z resolution entered by the user.
        """
        self.dim_z_text = text
        try:
            value = float(self.dim_z_text)
            if self.batch_fileinfo_list is None:
                self.file_info.change_dim_res('Z', value)
            else:
                for file_info in self.batch_fileinfo_list:
                    file_info.change_dim_res('Z', value)
        except ValueError:
            if self.batch_fileinfo_list is None:
                self.file_info.change_dim_res('Z', None)
            else:
                for file_info in self.batch_fileinfo_list:
                    file_info.change_dim_res('Z', None)
        self.on_change()

    def handle_xy_changed(self, text):
        """
        Handles changes in the XY resolution input field and updates the FileInfo object accordingly.

        Parameters
        ----------
        text : str
            The new XY resolution entered by the user.
        """
        self.dim_xy_text = text
        try:
            value = float(self.dim_xy_text)
            if self.batch_fileinfo_list is None:
                self.file_info.change_dim_res('X', value)
                self.file_info.change_dim_res('Y', value)
            else:
                for file_info in self.batch_fileinfo_list:
                    file_info.change_dim_res('X', value)
                    file_info.change_dim_res('Y', value)
        except ValueError:
            if self.batch_fileinfo_list is None:
                self.file_info.change_dim_res('X', None)
                self.file_info.change_dim_res('Y', None)
            else:
                for file_info in self.batch_fileinfo_list:
                    file_info.change_dim_res('X', None)
                    file_info.change_dim_res('Y', None)
        self.on_change()

    def change_channel(self):
        """
        Updates the selected channel in the FileInfo object when the channel spin box value is changed.
        """
        if self.batch_fileinfo_list is None:
            self.file_info.change_selected_channel(self.channel_button.value())
        else:
            for file_info in self.batch_fileinfo_list:
                file_info.change_selected_channel(self.channel_button.value())
        self.on_change()

    def change_time(self):
        """
        Updates the temporal range in the FileInfo object when the start or end frame spin box values are changed.
        """
        if self.batch_fileinfo_list is None:
            self.file_info.select_temporal_range(self.start_frame_button.value(), self.end_frame_button.value())
        else:
            for file_info in self.batch_fileinfo_list:
                file_info.select_temporal_range(self.start_frame_button.value(), self.end_frame_button.value())
        self.on_change()

    def on_confirm(self):
        """
        Confirms the file selection, creates an ImInfo object for the file, and prepares it for processing.
        """
        show_info("Saving OME TIFF file.")
        if self.batch_fileinfo_list is None:
            self.im_info = ImInfo(self.file_info)
        else:
            self.im_info = [ImInfo(file_info) for file_info in self.batch_fileinfo_list]
        self.on_change()

    def on_process(self):
        """
        Prepares the selected file(s) for processing through the Nellie pipeline by creating ImInfo objects and switching to the processing tab.
        """
        # switch to process tab
        if self.batch_fileinfo_list is None:
            self.im_info = ImInfo(self.file_info)
        else:
            self.im_info = [ImInfo(file_info) for file_info in self.batch_fileinfo_list]
        self.on_change()
        self.nellie.go_process()

    def on_preview(self):
        """
        Opens a preview of the selected image in the napari viewer, adjusting display settings (e.g., 2D or 3D view) based on the file's dimensionality.
        """
        im_memmap = tifffile.memmap(self.file_info.ome_output_path)
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


if __name__ == "__main__":
    import napari
    viewer = napari.Viewer()
    napari.run()
