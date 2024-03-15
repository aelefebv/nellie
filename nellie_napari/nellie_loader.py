import datetime
import os

from qtpy.QtWidgets import QWidget, QGridLayout, QLabel, QPushButton, QSpinBox, QCheckBox, QFileDialog, QMessageBox
from napari.utils.notifications import show_info
from nellie.im_info.im_info import ImInfo
from tifffile import tifffile

from nellie_napari import NellieViewer
from nellie_napari.batch_mode import BatchMode
from nellie_napari.nellie_analysis import NellieAnalysis


class NellieLoader(QWidget):
    def __init__(self, napari_viewer: 'napari.viewer.Viewer', parent=None):
        super().__init__(parent)
        self.filepath = None
        self.im_info = None

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

        # Label above the spinner box
        self.channel_label = QLabel("Channel to analyze:")

        self.channel_input = QSpinBox()
        self.channel_input.setRange(0, 0)
        self.channel_input.setValue(0)
        self.channel_input.setEnabled(False)
        self.channel_input.valueChanged.connect(self.change_channel)

        # Label above the spinner box
        self.time_label = QLabel("Number of temporal frames:")

        self.time_input = QSpinBox()
        self.time_input.setRange(1, 1)
        self.time_input.setValue(1)
        self.time_input.setEnabled(False)

        # process button
        self.process_button = QPushButton(text="Process")
        self.process_button.setEnabled(False)

        # analysis button
        self.analysis_button = QPushButton(text="Analyze")
        self.analysis_button.clicked.connect(self.open_nellie_analyzer)
        self.analysis_button.setEnabled(False)

        # reset button
        self.reset_button = QPushButton(text="Reset")
        self.reset_button.clicked.connect(self.reset)
        self.reset_button.setEnabled(False)

        # Checkbox for 'Remove edges'
        self.remove_edges_checkbox = QCheckBox("Remove image edges")

        # screenshot button
        self.screenshot_button = QPushButton(text="Ctrl/Cmd-Shift-E")
        self.screenshot_button.clicked.connect(self.screenshot)
        self.screenshot_button.setEnabled(False)
        self.viewer.bind_key('Ctrl-Shift-E', self.screenshot, overwrite=True)

        self.layout().addWidget(self.filepath_button, 1, 0, 1, 1)
        self.layout().addWidget(self.folder_button, 1, 1, 1, 1)
        self.layout().addWidget(self.filepath_text, 45, 0, 1, 2)
        self.layout().addWidget(self.channel_label, 46, 0)
        self.layout().addWidget(self.channel_input, 46, 1)
        self.layout().addWidget(self.time_label, 47, 0)
        self.layout().addWidget(self.time_input, 47, 1)
        self.layout().addWidget(self.process_button, 48, 0)
        self.layout().addWidget(self.remove_edges_checkbox, 48, 1)
        self.layout().addWidget(self.analysis_button, 49, 0)
        self.layout().addWidget(self.reset_button, 49, 1)
        self.layout().addWidget(QLabel("\nEasy screenshot"), 50, 0, 1, 2)
        self.layout().addWidget(self.screenshot_button, 51, 0, 1, 2)

        self.nellie_viewer = None
        self.batch_mode = None
        self.nellie_analyzer = None

        self.single = None

        self.valid_files = []
        self.valid_analysis_files = []

    def screenshot(self, event=None):
        # # if there's no layer, return
        # if self.viewer.layers is None:
        #     return

        # easy no prompt screenshot
        dt = datetime.datetime.now()  # year, month, day, hour, minute, second, millisecond up to 3 digits
        dt = dt.strftime("%Y%m%d_%H%M%S%f")[:-3]

        screenshot_folder = self.im_info.screenshot_dir
        if not os.path.exists(screenshot_folder):
            os.makedirs(screenshot_folder)

        im_name = f'{dt}-{self.im_info.basename_no_ext}.png'
        file_path = os.path.join(screenshot_folder, im_name)

        # Take screenshot
        screenshot = self.viewer.screenshot(canvas_only=True)

        # Save the screenshot
        try:
            # save as png to file_path using tifffile
            tifffile.imwrite(file_path, screenshot)
            print(f"Screenshot saved to {file_path}")
        except Exception as e:
            QMessageBox.warning(None, "Error", f"Failed to save screenshot: {str(e)}")
            raise e

    def initialize_single_file(self):
        self.im_info = ImInfo(self.filepath, ch=self.channel_input.value())

        if self.im_info.no_t:
            self.num_frames_max = 1
        else:  # the index of self.im_info.shape that corresponds to the index of 'T' in self.im_info.axes
            self.num_frames_max = self.im_info.shape[self.im_info.axes.index('T')]

        if self.im_info.no_c:
            self.num_channels = 1
        else:  # the index of self.im_info.shape that corresponds to the index of 'C' in self.im_info.axes
            self.num_channels = self.im_info.shape[self.im_info.axes.index('C')]

        self.screenshot_button.setEnabled(True)
        self.process_button.setText("Open Nellie Processor")
        self.analysis_button.setText("Open Nellie Analyzer")

    def initialize_folder(self, filepath):
        filenames = os.listdir(filepath)
        self.valid_files = []
        for filename in filenames:
            try:
                im_info = ImInfo(os.path.join(filepath, filename), ch=self.channel_input.value())
                self.valid_files.append(im_info)
            except:
                pass

        self.im_info = self.valid_files[0]

        if self.im_info.no_t:
            self.num_frames_max = 1
        else:  # the index of self.im_info.shape that corresponds to the index of 'T' in self.im_info.axes
            self.num_frames_max = self.im_info.shape[self.im_info.axes.index('T')]

        if self.im_info.no_c:
            self.num_channels = 1
        else:  # the index of self.im_info.shape that corresponds to the index of 'C' in self.im_info.axes
            self.num_channels = self.im_info.shape[self.im_info.axes.index('C')]

        # change next button text to 'Batch Run'
        self.process_button.setText("Run Batch Process Folder")
        self.analysis_button.setEnabled(False)
        # self.analysis_button.setText("Batch Analyze Folder")

    def check_analysis_valid(self):
        if self.single:
            if os.path.exists(self.im_info.pipeline_paths['adjacency_maps']):
                return True
            else:
                return False

        else:
            return False
            # self.valid_analysis_files = []
            # for valid_file in self.valid_files:
            #     if os.path.exists(valid_file.pipeline_paths['adjacency_maps']):
            #         self.valid_analysis_files.append(valid_file)
            # if self.valid_analysis_files:
            #     return True
            # else:
            #     return False

    def post_file_selection(self):
        self.time_input.setEnabled(True)
        if not self.im_info.no_c:
            self.channel_input.setEnabled(True)
        self.process_button.setEnabled(True)
        if self.check_analysis_valid():
            self.analysis_button.setEnabled(True)
        else:
            self.analysis_button.setEnabled(False)
        self.reset_button.setEnabled(True)
        self.remove_edges_checkbox.setEnabled(True)

        self.filepath_button.setEnabled(False)
        self.folder_button.setEnabled(False)
        self.time_input.setRange(1, self.num_frames_max)
        self.time_input.setValue(self.num_frames_max)
        self.channel_input.setRange(0, self.num_channels - 1)
        self.channel_input.setValue(0)

    def select_folder(self):
        filepath = QFileDialog.getExistingDirectory(self, "Select folder")

        if not filepath:
            show_info("No folder selected.")
            return None

        self.filepath = filepath

        self.single = False
        self.initialize_folder(filepath)
        filename = os.path.basename(self.filepath)
        show_info(f"Selected folder: {filename}")
        self.filepath_text.setText(f"Selected folder: {filename}")

        self.post_file_selection()
        self.process_button.clicked.connect(self.run_batch_processing)

    def select_filepath(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select file")
        if not filepath:
            show_info("No file selected.")
            return None

        self.filepath = filepath

        self.single = True
        self.initialize_single_file()
        filename = os.path.basename(self.filepath)
        show_info(f"Selected file: {filename}")
        self.filepath_text.setText(f"Selected file: {filename}")

        self.post_file_selection()
        self.process_button.clicked.connect(self.open_nellie_viewer)

    def disable_buttons(self):
        self.process_button.setEnabled(False)
        self.analysis_button.setEnabled(False)
        self.time_input.setEnabled(False)
        self.channel_input.setEnabled(False)
        self.remove_edges_checkbox.setEnabled(False)

    def open_nellie_viewer(self):
        self.disable_buttons()
        self.nellie_viewer = NellieViewer()
        self.viewer.window.add_dock_widget(self.nellie_viewer, name='Nellie Viewer', area='right')
        try:
            self.viewer.window.remove_dock_widget(self.batch_mode)
        except:
            pass
        self.nellie_viewer.im_info = self.im_info
        self.nellie_viewer.num_t = self.time_input.value()
        self.nellie_viewer.remove_edges = self.remove_edges_checkbox.isChecked()
        self.nellie_viewer.viewer = self.viewer
        self.nellie_viewer.post_init()

    def open_nellie_analyzer(self):
        self.disable_buttons()
        self.nellie_analyzer = NellieAnalysis()
        self.viewer.window.add_dock_widget(self.nellie_analyzer, name='Nellie Analyzer', area='right')
        self.nellie_analyzer.im_info = self.im_info
        self.nellie_analyzer.viewer = self.viewer
        self.nellie_analyzer.num_t = self.time_input.value()
        self.nellie_analyzer.post_init()

    def run_batch_processing(self):
        # todo maybe open up a batch viewer, to only run certain parts of the pipeline
        self.disable_buttons()
        self.batch_mode = BatchMode()
        self.viewer.window.add_dock_widget(self.batch_mode, name='Nelle Batch Mode', area='right')
        try:
            self.viewer.window.remove_dock_widget(self.nellie_viewer)
        except:
            pass
        self.batch_mode.valid_files = self.valid_files
        self.batch_mode.num_t = self.time_input.value()
        self.batch_mode.remove_edges = self.remove_edges_checkbox.isChecked()
        self.batch_mode.viewer = self.viewer
        self.batch_mode.enable_buttons()

    def reset(self):
        self.filepath = None
        self.im_info = None
        self.num_frames_max = None
        self.num_channels = 1
        self.time_input.setValue(1)
        self.time_input.setEnabled(False)
        self.channel_input.setValue(0)
        self.channel_input.setEnabled(False)
        self.filepath_button.setEnabled(True)
        self.filepath_text.setText("No file selected.")
        self.folder_button.setEnabled(True)
        self.process_button.setEnabled(False)
        self.analysis_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.remove_edges_checkbox.setEnabled(False)
        self.screenshot_button.setEnabled(False)

        self.valid_files = []
        self.valid_analysis_files = []
        self.single = None
        self.remove_edges_checkbox.setChecked(False)

        # remove all dock widgets that are open
        try:
            self.viewer.window.remove_dock_widget(self.nellie_viewer)
            del self.nellie_viewer
            self.nellie_viewer = None
        except:
            pass
        try:
            self.viewer.window.remove_dock_widget(self.batch_mode)
            del self.batch_mode
            self.batch_mode = None
        except:
            pass
        try:
            self.viewer.window.remove_dock_widget(self.nellie_analyzer)
            del self.nellie_analyzer
            self.nellie_analyzer = None
        except:
            pass

    def change_channel(self):
        if self.single:
            self.initialize_single_file()
        else:
            self.initialize_folder(self.filepath)


if __name__ == "__main__":
    import napari
    viewer = napari.Viewer()
    napari.run()
