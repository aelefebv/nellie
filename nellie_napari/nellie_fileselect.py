from __future__ import annotations

import os
from typing import List, Optional, Union

import tifffile
from qtpy.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QLineEdit,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QSpinBox,
    QComboBox,
)
from qtpy.QtGui import QDoubleValidator
from napari.utils.notifications import show_info

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
        The selected file or folder path.
    file_info : FileInfo or None
        Stores metadata and shape information about the currently active image file.
    im_info : ImInfo or list[ImInfo] or None
        Information object(s) for the image file(s) after confirmation / OME-TIFF creation.
    batch_fileinfo_list : list[FileInfo] or None
        List of FileInfo objects when a folder is selected for batch processing.
    filepath_text : QLabel
        Text widget displaying the selected file or folder path.
    filepath_button : QPushButton
        Button to open the file dialog for selecting an image file.
    folder_button : QPushButton
        Button to open the folder dialog for batch processing.
    reset_button : QPushButton
        Button to reset the file selection and clear all settings (via nellie.reset()).
    file_shape_text : QLabel
        Displays the shape of the selected image file.
    current_order_text : QLabel
        Displays the current dimension order (axes) of the image file.
    dim_order_button : QLineEdit
        Input field for entering the dimension order of the image.
    dim_t_button, dim_z_button, dim_xy_button : QLineEdit
        Input fields for entering the resolution of the time (T), Z, and XY dimensions, respectively.
    channel_button : QSpinBox
        Spin box for selecting the channel in the multi-channel image (0-based index).
    start_frame_button, end_frame_button : QSpinBox
        Spin boxes for selecting the start and end frame (0-based indices) for temporal range selection.
    confirm_button : QPushButton
        Button to confirm the file selection and save the OME-TIFF file(s).
    preview_button : QPushButton
        Button to preview the image in the napari viewer.
    process_button : QPushButton
        Button to process the selected image file(s) through the Nellie pipeline.
    """

    def __init__(self, napari_viewer: napari.Viewer, nellie, parent=None):
        """
        Initialize the NellieFileSelect class.

        Parameters
        ----------
        napari_viewer : napari.Viewer
            Reference to the napari viewer instance.
        nellie : object
            Reference to the main Nellie object containing image processing pipelines and functions.
        parent : QWidget, optional
            Optional parent widget (default is None).
        """
        super().__init__(parent)
        self.nellie = nellie
        self.filepath: Optional[str] = None
        self.file_info: Optional[FileInfo] = None
        self.im_info: Optional[Union[ImInfo, List[ImInfo]]] = None

        self.batch_fileinfo_list: Optional[List[FileInfo]] = None

        self.viewer = napari_viewer
        self.viewer.title = "Nellie Napari"

        # --- File selection widgets ---
        self.filepath_text = QLabel(text="No file selected.")
        self.filepath_text.setWordWrap(True)

        self.filepath_button = QPushButton(text="Select File")
        self.filepath_button.clicked.connect(self.select_filepath)
        self.filepath_button.setEnabled(True)

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
        self.dim_order_button.setToolTip("No file selected.")
        self.dim_order_button.setEnabled(False)
        self.dim_order_button.setReadOnly(True)
        self.dim_order_button.setVisible(False)

        self.axes_combo_boxes: List[QComboBox] = []
        self.axes_combo_container = QWidget(self)
        self.axes_combo_layout = QHBoxLayout()
        self.axes_combo_layout.setContentsMargins(0, 0, 0, 0)
        self.axes_combo_container.setLayout(self.axes_combo_layout)
        self.axes_combo_container.setEnabled(False)
        self._updating_axes_combos = False

        # Resolution text caches (as entered by user)
        self.dim_t_text: Optional[str] = "None"
        self.dim_z_text: Optional[str] = "None"
        self.dim_xy_text: Optional[str] = "None"

        # Labels and resolution fields
        self.label_t = QLabel("T resolution (s):")
        self.dim_t_button = QLineEdit(self)
        self.dim_t_button.setText("None")
        self.dim_t_button.setEnabled(False)
        self.dim_t_button.editingFinished.connect(self.handle_t_changed)

        self.label_z = QLabel("Z resolution (µm):")
        self.dim_z_button = QLineEdit(self)
        self.dim_z_button.setText("None")
        self.dim_z_button.setEnabled(False)
        self.dim_z_button.editingFinished.connect(self.handle_z_changed)

        self.label_xy = QLabel("X & Y resolution (µm):")
        self.dim_xy_button = QLineEdit(self)
        self.dim_xy_button.setText("None")
        self.dim_xy_button.setEnabled(False)
        self.dim_xy_button.editingFinished.connect(self.handle_xy_changed)

        # Numeric validators for resolution fields (user input only)
        self._setup_numeric_validators()

        # Channel and time selection
        self.label_channel = QLabel("Channel (0-based):")
        self.channel_button = QSpinBox(self)
        self.channel_button.setRange(0, 0)
        self.channel_button.setValue(0)
        self.channel_button.setEnabled(False)
        self.channel_button.valueChanged.connect(self.change_channel)

        self.label_time = QLabel("Start frame (0-based):")
        self.label_time_2 = QLabel("End frame (0-based):")
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

        # Action buttons
        self.confirm_button = QPushButton(text="Confirm metadata")
        self.confirm_button.setToolTip("This will save the image as an OME-TIFF for use in processing")
        self.confirm_button.clicked.connect(self.on_confirm)
        self.confirm_button.setEnabled(False)

        self.preview_button = QPushButton(text="Preview image")
        self.preview_button.clicked.connect(self.on_preview)
        self.preview_button.setEnabled(False)

        self.process_button = QPushButton(text="Process image(s)")
        self.process_button.clicked.connect(self.on_process)
        self.process_button.setEnabled(False)

        self.validation_label = QLabel("")
        self.validation_label.setWordWrap(True)
        self.validation_label.setStyleSheet("color: red")

        self.init_ui()

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _setup_numeric_validators(self) -> None:
        """
        Attach QDoubleValidator to resolution fields for numeric input.
        """
        validator = QDoubleValidator(0.0, 1e9, 6, self)
        # Allow empty text as "intermediate" during editing
        validator.setNotation(QDoubleValidator.StandardNotation)

        for line_edit in (self.dim_t_button, self.dim_z_button, self.dim_xy_button):
            line_edit.setValidator(validator)

    def _set_line_edit_text(self, line_edit: QLineEdit, text: str) -> None:
        """
        Update a QLineEdit value without triggering signal handlers.
        """
        line_edit.blockSignals(True)
        line_edit.setText(text)
        line_edit.blockSignals(False)

    def _set_validation_messages(self, messages: List[str]) -> None:
        """
        Update the validation label with current errors or a success message.
        """
        if messages:
            self.validation_label.setText("\n".join(messages))
            self.validation_label.setStyleSheet("color: red")
        else:
            self.validation_label.setText("All checks passed.")
            self.validation_label.setStyleSheet("color: green")

    def _clear_axes_combo_boxes(self) -> None:
        """
        Remove all axis combo boxes from the layout.
        """
        for combo in self.axes_combo_boxes:
            self.axes_combo_layout.removeWidget(combo)
            combo.deleteLater()
        self.axes_combo_boxes = []

    def _enforce_axes_combo_uniqueness(self) -> None:
        """
        Ensure axes selections are unique across combo boxes.
        """
        selected = [combo.currentText() for combo in self.axes_combo_boxes]
        for combo in self.axes_combo_boxes:
            model = combo.model()
            for idx in range(combo.count()):
                item = model.item(idx)
                if item is None:
                    continue
                text = combo.itemText(idx)
                is_selected_elsewhere = text in selected and text != combo.currentText()
                item.setEnabled(not is_selected_elsewhere)

    def _sync_axes_combo_boxes(self) -> None:
        """
        Sync axis combo boxes with current axes state.
        """
        if self.file_info is None or self.file_info.axes is None:
            self._clear_axes_combo_boxes()
            self.axes_combo_container.setEnabled(False)
            return

        axes = self.file_info.axes
        if len(self.axes_combo_boxes) != len(axes):
            self._clear_axes_combo_boxes()
            self._updating_axes_combos = True
            for axis in axes:
                combo = QComboBox(self)
                combo.addItems(["T", "Z", "C", "Y", "X"])
                combo.setCurrentText(axis)
                combo.currentTextChanged.connect(self.handle_axes_combo_changed)
                self.axes_combo_layout.addWidget(combo)
                self.axes_combo_boxes.append(combo)
            self._updating_axes_combos = False
        else:
            self._updating_axes_combos = True
            for combo, axis in zip(self.axes_combo_boxes, axes):
                if combo.currentText() != axis:
                    combo.setCurrentText(axis)
            self._updating_axes_combos = False

        self.axes_combo_container.setEnabled(True)
        self._enforce_axes_combo_uniqueness()
        self._set_line_edit_text(self.dim_order_button, "".join(axes))

    def _validate_axes_text(self, text: str, expected_len: int) -> Optional[str]:
        """
        Validate axes text against expected length and allowed axis labels.

        Returns
        -------
        Optional[str]
            Error message if invalid, otherwise None.
        """
        if len(text) != expected_len:
            return f"Axes must be {expected_len} characters for this file."
        allowed_axes = {"T", "Z", "C", "Y", "X"}
        if any(axis not in allowed_axes for axis in text):
            return "Axes must only use T, Z, C, Y, X."
        if len(set(text)) != len(text):
            return "Axes must not contain duplicates."
        if "X" not in text or "Y" not in text:
            return "Axes must include both X and Y."
        return None

    def _each_file_info(self) -> List[FileInfo]:
        """
        Return a list of all FileInfo objects (single or batch).

        Returns
        -------
        List[FileInfo]
            List of FileInfo objects.
        """
        if self.batch_fileinfo_list:
            return self.batch_fileinfo_list
        return [self.file_info] if self.file_info is not None else []

    def _apply_to_each_file_info(self, action: str, func) -> bool:
        """
        Apply a function to each FileInfo and surface the first error to the user.

        Returns
        -------
        bool
            True if all updates succeeded, False if any failed.
        """
        for file_info in self._each_file_info():
            try:
                func(file_info)
            except Exception as exc:
                show_info(f"{action}: {exc}")
                self._set_validation_messages([str(exc)])
                return False
        return True

    @property
    def _has_batch(self) -> bool:
        """
        Return True if in batch mode.

        Returns
        -------
        bool
            True if in batch mode.
        """
        return bool(self.batch_fileinfo_list)

    def _create_im_info(self) -> Union[ImInfo, List[ImInfo]]:
        """
        Create ImInfo instance(s) from the current FileInfo object(s).

        Returns
        -------
        Union[ImInfo, List[ImInfo]]
            ImInfo object or list of ImInfo objects.
        """
        file_infos = self._each_file_info()
        if not file_infos:
            raise RuntimeError("No file(s) selected.")
        im_infos = [ImInfo(fi) for fi in file_infos]
        return im_infos[0] if len(im_infos) == 1 else im_infos

    # -------------------------------------------------------------------------
    # UI setup
    # -------------------------------------------------------------------------

    def init_ui(self) -> None:
        """
        Set up the user interface layout with sections for file selection,
        axes information, dimension resolutions, slice settings, and action buttons.
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
        sub_layout.addWidget(self.axes_combo_container)
        axes_layout.addLayout(sub_layout)

        for label, widget in [
            (QLabel("File shape:"), self.file_shape_text),
            (QLabel("Current order:"), self.current_order_text),
        ]:
            sub_layout = QHBoxLayout()
            sub_layout.addWidget(label)
            sub_layout.addWidget(widget)
            axes_layout.addLayout(sub_layout)

        axes_group.setLayout(axes_layout)

        # Dimensions Group
        dim_group = QGroupBox("Dimension Resolutions")
        dim_layout = QVBoxLayout()
        for label, widget in [
            (self.label_t, self.dim_t_button),
            (self.label_z, self.dim_z_button),
            (self.label_xy, self.dim_xy_button),
        ]:
            sub_layout = QHBoxLayout()
            sub_layout.addWidget(label)
            sub_layout.addWidget(widget)
            dim_layout.addLayout(sub_layout)
        dim_group.setLayout(dim_layout)

        # Slice Settings Group
        slice_group = QGroupBox("Slice Settings")
        slice_layout = QVBoxLayout()
        for label, widget in [
            (self.label_time, self.start_frame_button),
            (self.label_time_2, self.end_frame_button),
            (self.label_channel, self.channel_button),
        ]:
            sub_layout = QHBoxLayout()
            sub_layout.addWidget(label)
            sub_layout.addWidget(widget)
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
        main_layout.addWidget(self.validation_label)
        main_layout.addWidget(action_group)

        self.setLayout(main_layout)

    # -------------------------------------------------------------------------
    # File / folder selection
    # -------------------------------------------------------------------------

    def select_filepath(self) -> None:
        """
        Open a file dialog for selecting an image file, validates the selected
        file, and updates the UI with metadata.
        """
        self.batch_fileinfo_list = None
        filepath, _ = QFileDialog.getOpenFileName(self, "Select file")
        if not self.validate_path(filepath):
            return

        self.selection_text.setText("Selected file:")

        self.file_info = FileInfo(self.filepath, output_naming="detailed")
        self.initialize_single_file()
        filename = os.path.basename(self.filepath)
        self.filepath_text.setText(filename)

    def select_folder(self) -> None:
        """
        Open a folder dialog for selecting a folder for batch processing and
        initializes FileInfo objects for all files in the folder.
        """
        folderpath = QFileDialog.getExistingDirectory(self, "Select folder")
        if not self.validate_path(folderpath):
            return

        if not self.initialize_folder():
            return

        self.selection_text.setText("Selected folder:")
        self.filepath_text.setText(folderpath)

    def validate_path(self, filepath: str) -> bool:
        """
        Validate the selected file or folder path and updates the file path
        attribute. Canceling the dialog is treated as a no-op.

        Parameters
        ----------
        filepath : str
            The file or folder path selected by the user.

        Returns
        -------
        bool
            True if a valid path was selected, False otherwise.
        """
        if not filepath:
            # User cancelled; do not modify existing selection or show an error.
            return False

        self.filepath = filepath
        return True

    # -------------------------------------------------------------------------
    # Initialization of FileInfo objects
    # -------------------------------------------------------------------------

    def initialize_single_file(self, metadata_already_loaded: bool = False) -> None:
        """
        Initialize the FileInfo object for the selected image file, loads metadata,
        and updates the dimension resolution fields.

        Parameters
        ----------
        metadata_already_loaded : bool
            If True, skip calling find_metadata/load_metadata (used for batch
            initialization where metadata was already loaded).
        """
        if self.file_info is None:
            return

        if not metadata_already_loaded:
            self.file_info.find_metadata()
            self.file_info.load_metadata()

        self.file_shape_text.setText(f"{self.file_info.shape}")
        self._set_line_edit_text(self.dim_order_button, "".join(self.file_info.axes))
        self.dim_order_button.setEnabled(True)
        self.on_change()

    def initialize_folder(self) -> bool:
        """
        Initialize FileInfo objects for all .tif, .tiff, and .nd2 files in the
        selected folder and loads their metadata.

        Returns
        -------
        bool
            True if at least one compatible file was found and initialized,
            False otherwise.
        """
        if self.filepath is None:
            return False

        files = [
            f
            for f in os.listdir(self.filepath)
            if f.lower().endswith((".tif", ".tiff", ".nd2"))
        ]

        if not files:
            show_info("No .tif, .tiff, or .nd2 files found in the selected folder.")
            self.batch_fileinfo_list = None
            return False

        files.sort()
        self.batch_fileinfo_list = [
            FileInfo(os.path.join(self.filepath, f), output_naming="detailed") for f in files
        ]
        for file_info in self.batch_fileinfo_list:
            file_info.find_metadata()
            file_info.load_metadata()

        # Ensure batch metadata is compatible across files
        primary_axes = self.batch_fileinfo_list[0].axes
        primary_shape = self.batch_fileinfo_list[0].shape
        mismatched = [
            file_info
            for file_info in self.batch_fileinfo_list[1:]
            if file_info.axes != primary_axes or file_info.shape != primary_shape
        ]
        if mismatched:
            show_info(
                "Batch files have mismatched axes or shapes. "
                "Please select a folder with compatible files."
            )
            self.batch_fileinfo_list = None
            return False

        # Use the first file as the "primary" file for UI state
        self.file_info = self.batch_fileinfo_list[0]
        self.initialize_single_file(metadata_already_loaded=True)
        # Assumes all files share the same metadata (axes, resolutions, etc.)
        return True

    # -------------------------------------------------------------------------
    # UI state updates
    # -------------------------------------------------------------------------

    def on_change(self) -> None:
        """
        Update the user interface elements, including enabling or disabling buttons
        based on the file metadata and resolution settings.
        """
        # Default: disable actions until we prove they are valid
        self.confirm_button.setEnabled(False)
        self.preview_button.setEnabled(False)
        self.process_button.setEnabled(False)

        if self.file_info is None:
            # No file selected; reset relevant UI state
            self.file_shape_text.setText("None")
            self.current_order_text.setText("None")
            self._set_line_edit_text(self.dim_order_button, "None")
            self.dim_order_button.setEnabled(False)
            self.dim_order_button.setToolTip("No file selected.")
            self.axes_combo_container.setToolTip("No file selected.")
            self._clear_axes_combo_boxes()
            self.axes_combo_container.setEnabled(False)
            self.axes_combo_container.setStyleSheet("")
            self.validation_label.setText("")

            # Reset resolution fields
            for btn in (self.dim_t_button, self.dim_z_button, self.dim_xy_button):
                btn.setEnabled(False)
                self._set_line_edit_text(btn, "None")
                btn.setStyleSheet("")

            # Reset slice controls
            for spin in (self.channel_button, self.start_frame_button, self.end_frame_button):
                spin.setEnabled(False)
                spin.setRange(0, 0)
                spin.setValue(0)

            return

        # Update available dimensions and resolution fields
        self.check_available_dims()
        self._sync_axes_combo_boxes()

        # Update tooltip for dimension order based on number of dimensions
        ndim = len(self.file_info.shape)
        if ndim == 2:
            tooltip = "Accepted axes: 'Y', 'X' (e.g. 'YX')"
            self.dim_order_button.setToolTip(tooltip)
            self.axes_combo_container.setToolTip(tooltip)
        elif ndim == 3:
            tooltip = "Accepted axes: ['T' or 'C' or 'Z'], 'Y', 'X' (e.g. 'ZYX')"
            self.dim_order_button.setToolTip(tooltip)
            self.axes_combo_container.setToolTip(tooltip)
        elif ndim == 4:
            tooltip = "Accepted axes: ['T' or 'C' or 'Z']x2, 'Y', 'X' (e.g. 'TZYX')"
            self.dim_order_button.setToolTip(tooltip)
            self.axes_combo_container.setToolTip(tooltip)
        elif ndim == 5:
            tooltip = "Accepted axes: ['T' or 'C' or 'Z']x3, 'Y', 'X' (e.g. 'TZCYX')"
            self.dim_order_button.setToolTip(tooltip)
            self.axes_combo_container.setToolTip(tooltip)
        elif ndim > 5:
            self.dim_order_button.setStyleSheet("background-color: red")
            show_info(f"Error: Too many dimensions found ({self.file_info.shape}).")

        # Axes validity feedback
        if getattr(self.file_info, "good_axes", False):
            current_order_text = f"({', '.join(self.file_info.axes)})"
            self.current_order_text.setText(current_order_text)
            self.dim_order_button.setStyleSheet("background-color: green")
            self.axes_combo_container.setStyleSheet("background-color: green")
        else:
            self.current_order_text.setText("Invalid")
            self.dim_order_button.setStyleSheet("background-color: red")
            self.axes_combo_container.setStyleSheet("background-color: red")

        # Enable confirm if both dims and axes are valid
        if getattr(self.file_info, "good_dims", False) and getattr(
            self.file_info, "good_axes", False
        ):
            self.confirm_button.setEnabled(True)

        # Enable preview/process if OME-TIFF exists and metadata is valid
        if (
            hasattr(self.file_info, "ome_output_path")
            and os.path.exists(self.file_info.ome_output_path)
            and getattr(self.file_info, "good_dims", False)
            and getattr(self.file_info, "good_axes", False)
        ):
            self.preview_button.setEnabled(True)
            self.process_button.setEnabled(True)

        errors = self.file_info.get_validation_errors()
        self._set_validation_messages(errors)

    def check_available_dims(self) -> None:
        """
        Check the availability of specific dimensions (T, Z, XY) in the selected
        file and enables the corresponding input fields for resolutions.
        """
        if self.file_info is None:
            return

        axes = self.file_info.axes
        dim_res = getattr(self.file_info, "dim_res", {})

        def check_dim(dim: str, dim_button: QLineEdit, attr_name: str) -> None:
            dim_text = getattr(self, attr_name)
            dim_button.setStyleSheet("")
            if dim in axes:
                dim_button.setEnabled(True)
                if dim_text is None or dim_text == "None":
                    value = dim_res.get(dim)
                    if value is None:
                        self._set_line_edit_text(dim_button, "None")
                        dim_button.setStyleSheet("background-color: red")
                    else:
                        self._set_line_edit_text(dim_button, str(value))
                        dim_button.setStyleSheet("background-color: green")
                else:
                    self._set_line_edit_text(dim_button, dim_text)
                    if dim_res.get(dim) is None:
                        dim_button.setStyleSheet("background-color: red")
                    else:
                        dim_button.setStyleSheet("background-color: green")
            else:
                dim_button.setEnabled(False)
                value = dim_res.get(dim)
                self._set_line_edit_text(dim_button, str(value) if value is not None else "None")
                dim_button.setStyleSheet("")

        # T and Z use separate fields
        check_dim("T", self.dim_t_button, "dim_t_text")
        check_dim("Z", self.dim_z_button, "dim_z_text")

        # XY share a single field
        self.dim_xy_button.setStyleSheet("")
        has_xy = ("X" in axes) or ("Y" in axes)
        if has_xy:
            self.dim_xy_button.setEnabled(True)
            if self.dim_xy_text is None or self.dim_xy_text == "None":
                x_val = dim_res.get("X")
                y_val = dim_res.get("Y")
                value = x_val if x_val is not None else y_val
                if value is None:
                    self._set_line_edit_text(self.dim_xy_button, "None")
                    self.dim_xy_button.setStyleSheet("background-color: red")
                else:
                    self._set_line_edit_text(self.dim_xy_button, str(value))
                    if x_val is not None and y_val is not None:
                        self.dim_xy_button.setStyleSheet("background-color: green")
            else:
                self._set_line_edit_text(self.dim_xy_button, self.dim_xy_text)
                x_val = dim_res.get("X")
                y_val = dim_res.get("Y")
                if x_val is None or y_val is None:
                    self.dim_xy_button.setStyleSheet("background-color: red")
                else:
                    self.dim_xy_button.setStyleSheet("background-color: green")
        else:
            self.dim_xy_button.setEnabled(False)
            self._set_line_edit_text(self.dim_xy_button, "None")
            self.dim_xy_button.setStyleSheet("")

        # Channel controls
        self.channel_button.setEnabled(False)
        if "C" in axes:
            ax_idx = axes.index("C")
            max_c = self.file_info.shape[ax_idx]
            self.channel_button.setEnabled(True)
            self.channel_button.setRange(0, max_c - 1)
            desired_channel = getattr(self.file_info, "ch", 0)
            if desired_channel < 0 or desired_channel >= max_c:
                desired_channel = 0
            if self.channel_button.value() != desired_channel:
                self.channel_button.blockSignals(True)
                self.channel_button.setValue(desired_channel)
                self.channel_button.blockSignals(False)
        else:
            if self.channel_button.value() != 0:
                self.channel_button.blockSignals(True)
                self.channel_button.setValue(0)
                self.channel_button.blockSignals(False)

        # Temporal range controls
        self.start_frame_button.setEnabled(False)
        self.end_frame_button.setEnabled(False)
        if "T" in axes:
            ax_idx = axes.index("T")
            max_t = self.file_info.shape[ax_idx] - 1

            self.start_frame_button.setEnabled(True)
            self.end_frame_button.setEnabled(True)

            current_start = self.start_frame_button.value()
            current_end = self.end_frame_button.value()

            # Keep ranges consistent with current values
            self.start_frame_button.setRange(0, current_end if current_end <= max_t else max_t)
            self.end_frame_button.setRange(
                current_start if current_start <= max_t else 0, max_t
            )

            desired_start = getattr(self.file_info, "t_start", 0) or 0
            desired_end = getattr(self.file_info, "t_end", None)
            if desired_end is None:
                desired_end = max_t
            if desired_start < 0 or desired_start > max_t:
                desired_start = 0
            if desired_end < desired_start or desired_end > max_t:
                desired_end = max_t

            if (
                self.start_frame_button.value() != desired_start
                or self.end_frame_button.value() != desired_end
            ):
                self.start_frame_button.blockSignals(True)
                self.end_frame_button.blockSignals(True)
                self.start_frame_button.setValue(desired_start)
                self.end_frame_button.setValue(desired_end)
                self.start_frame_button.blockSignals(False)
                self.end_frame_button.blockSignals(False)
            self.end_frame_init = True
        else:
            if self.start_frame_button.value() != 0 or self.end_frame_button.value() != 0:
                self.start_frame_button.blockSignals(True)
                self.end_frame_button.blockSignals(True)
                self.start_frame_button.setValue(0)
                self.end_frame_button.setValue(0)
                self.start_frame_button.blockSignals(False)
                self.end_frame_button.blockSignals(False)
            self.end_frame_init = False

    # -------------------------------------------------------------------------
    # Handlers for UI changes
    # -------------------------------------------------------------------------

    def handle_dim_order_changed(self) -> None:
        """
        Handle changes in the dimension order input field and updates the
        FileInfo object(s) accordingly.

        Parameters
        ----------
        """
        if self.file_info is None or self.file_info.shape is None:
            return

        text = self.dim_order_button.text().strip().upper()
        self._set_line_edit_text(self.dim_order_button, text)

        error = self._validate_axes_text(text, len(self.file_info.shape))
        if error:
            self.dim_order_button.setStyleSheet("background-color: red")
            self.dim_order_button.setToolTip(error)
            self.axes_combo_container.setStyleSheet("background-color: red")
            self._set_validation_messages([error])
            return

        if not self._apply_to_each_file_info(
            "Error updating axes",
            lambda file_info: file_info.change_axes(text),
        ):
            return

        # Axes changes may affect temporal dimension position
        self.end_frame_init = False
        self.on_change()

    def handle_axes_combo_changed(self, _text: str) -> None:
        """
        Handle axes updates from the combo boxes.
        """
        if self._updating_axes_combos:
            return
        if not self.axes_combo_boxes:
            return
        self._enforce_axes_combo_uniqueness()
        axes_text = "".join(combo.currentText() for combo in self.axes_combo_boxes)
        self._set_line_edit_text(self.dim_order_button, axes_text)
        self.handle_dim_order_changed()

    def handle_t_changed(self) -> None:
        """
        Handle changes in the time (T) resolution input field and updates the
        FileInfo object(s) accordingly.

        Parameters
        ----------
        """
        text = self.dim_t_button.text()
        self.dim_t_text = text
        try:
            value = float(text)
        except ValueError:
            value = None

        if not self._apply_to_each_file_info(
            "Error updating T resolution",
            lambda file_info: file_info.change_dim_res("T", value),
        ):
            return

        self.on_change()

    def handle_z_changed(self) -> None:
        """
        Handle changes in the Z resolution input field and updates the
        FileInfo object(s) accordingly.

        Parameters
        ----------
        """
        text = self.dim_z_button.text()
        self.dim_z_text = text
        try:
            value = float(text)
        except ValueError:
            value = None

        if not self._apply_to_each_file_info(
            "Error updating Z resolution",
            lambda file_info: file_info.change_dim_res("Z", value),
        ):
            return

        self.on_change()

    def handle_xy_changed(self) -> None:
        """
        Handle changes in the XY resolution input field and updates the
        FileInfo object(s) accordingly.

        Parameters
        ----------
        """
        text = self.dim_xy_button.text()
        self.dim_xy_text = text
        try:
            value = float(text)
        except ValueError:
            value = None
        def update_xy(file_info: FileInfo) -> None:
            file_info.change_dim_res("X", value)
            file_info.change_dim_res("Y", value)

        if not self._apply_to_each_file_info("Error updating XY resolution", update_xy):
            return

        self.on_change()

    def change_channel(self) -> None:
        """
        Update the selected channel in the FileInfo object(s) when the channel
        spin box value is changed.
        """
        channel = self.channel_button.value()
        if not self._apply_to_each_file_info(
            "Error updating channel selection",
            lambda file_info: file_info.change_selected_channel(channel),
        ):
            return
        self.on_change()

    def change_time(self) -> None:
        """
        Update the temporal range in the FileInfo object(s) when the start or
        end frame spin box values are changed.
        """
        start = self.start_frame_button.value()
        end = self.end_frame_button.value()
        if not self._apply_to_each_file_info(
            "Error updating temporal range",
            lambda file_info: file_info.select_temporal_range(start, end),
        ):
            return
        self.on_change()

    # -------------------------------------------------------------------------
    # Actions: confirm, process, preview
    # -------------------------------------------------------------------------

    def on_confirm(self) -> None:
        """
        Confirm the file selection, creates ImInfo object(s) for the file(s),
        and prepares them for processing (creating OME-TIFF files).
        """
        if self._has_batch:
            show_info("Saving OME-TIFF files.")
        else:
            show_info("Saving OME-TIFF file.")

        try:
            self.im_info = self._create_im_info()
        except Exception as e:
            show_info(f"Error saving OME-TIFF file(s): {e}")
            return

        self.on_change()

    def on_process(self) -> None:
        """
        Prepare the selected file(s) for processing through the Nellie pipeline
        by creating ImInfo object(s) and switching to the processing tab.
        """
        try:
            self.im_info = self._create_im_info()
        except Exception as e:
            show_info(f"Error preparing image(s) for processing: {e}")
            return

        self.on_change()
        self.nellie.go_process()

    def on_preview(self) -> None:
        """
        Open a preview of the selected image in the napari viewer, adjusting
        display settings (e.g., 2D or 3D view) based on the file's dimensionality.
        """
        if self.file_info is None or not hasattr(self.file_info, "ome_output_path"):
            show_info("No OME-TIFF file available for preview.")
            return

        try:
            im_memmap = tifffile.memmap(self.file_info.ome_output_path)
        except Exception as e:
            show_info(f"Error opening OME-TIFF for preview: {e}")
            return

        scale = None
        dim_res = getattr(self.file_info, "dim_res", {})
        axes = self.file_info.axes

        try:
            if "Z" in axes and self.file_info.shape[self.file_info.axes.index("Z")] > 1:
                scale = (
                    float(dim_res["Z"]),
                    float(dim_res["Y"]),
                    float(dim_res["X"]),
                )
                self.viewer.dims.ndisplay = 3
            else:
                scale = (
                    float(dim_res["Y"]),
                    float(dim_res["X"]),
                )
                self.viewer.dims.ndisplay = 2
        except Exception:
            # Fall back to default scale if resolution metadata is missing or invalid
            scale = None

        add_kwargs = dict(
            name=self.file_info.filename_no_ext,
            blending="translucent_no_depth",
            interpolation3d="nearest",
            interpolation2d="nearest",
        )
        if scale is not None:
            add_kwargs["scale"] = scale

        self.viewer.add_image(im_memmap, **add_kwargs)
        self.viewer.scale_bar.visible = True
        self.viewer.scale_bar.unit = "µm"


if __name__ == "__main__":
    import napari

    viewer = napari.Viewer()
    napari.run()
