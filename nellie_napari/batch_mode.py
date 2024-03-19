import os
import napari
from napari.utils.notifications import show_info

from qtpy.QtWidgets import QWidget, QPushButton, QLabel, QGridLayout, QSpinBox, QCheckBox
from nellie.feature_extraction.hierarchical import Hierarchy
from nellie.segmentation.labelling import Label
from nellie.segmentation.mocap_marking import Markers
from nellie.segmentation.networking import Network
from nellie.tracking.hu_tracking import HuMomentTracking
from nellie.tracking.voxel_reassignment import VoxelReassigner
from nellie.segmentation.filtering import Filter


class BatchMode(QWidget):
    def __init__(self, napari_viewer: 'napari.viewer.Viewer', nellie, parent=None):
        super().__init__(parent)
        self.nellie = nellie
        self.viewer = napari_viewer
        self.nellie.valid_files = None
        self.num_t = None

        self.im_info = None

        self.layout = QGridLayout()
        self.setLayout(self.layout)

        self.all_preprocessed = False
        self.all_segmented = False
        self.all_marked = False
        self.all_tracked = False
        self.all_reassigned = False

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
        self.time_input.valueChanged.connect(self.change_t)
        self.num_t = 1

        # Checkbox for 'Remove edges'
        self.remove_edges_checkbox = QCheckBox("Remove image edges")
        self.remove_edges_checkbox.setEnabled(False)
        self.remove_edges_checkbox.setToolTip(
            "Originally for Snouty deskewed images. If you see weird image edge artifacts, enable this.")

        self.run_all_button = QPushButton(text="Run all")
        self.run_all_button.clicked.connect(self.run_all)
        self.run_all_button.setEnabled(True)

        self.preprocess_button = QPushButton(text="Run preprocessing")
        self.preprocess_button.clicked.connect(self.run_preprocessing)
        self.preprocess_button.setEnabled(False)

        self.segment_button = QPushButton(text="Run segmentation")
        self.segment_button.clicked.connect(self.run_segmentation)
        self.segment_button.setEnabled(False)

        self.mocap_button = QPushButton(text="Run mocap marking")
        self.mocap_button.clicked.connect(self.run_mocap)
        self.mocap_button.setEnabled(False)

        self.track_button = QPushButton(text="Run tracking")
        self.track_button.clicked.connect(self.run_tracking)
        self.track_button.setEnabled(False)

        self.reassign_button = QPushButton(text="Run voxel reassignment")
        self.reassign_button.clicked.connect(self.run_reassignment)
        self.reassign_button.setEnabled(False)

        self.feature_export_button = QPushButton(text="Run feature export")
        self.feature_export_button.clicked.connect(self.run_feature_export)
        self.feature_export_button.setEnabled(False)

        # Add buttons
        self.layout.addWidget(self.channel_label, 0, 0)
        self.layout.addWidget(self.channel_input, 0, 1)
        self.layout.addWidget(self.time_label, 1, 0)
        self.layout.addWidget(self.time_input, 1, 1)
        self.layout.addWidget(self.remove_edges_checkbox, 2, 1)

        self.layout.addWidget(QLabel("Run full pipeline"), 42, 0, 1, 2)
        self.layout.addWidget(self.run_all_button, 43, 0, 1, 2)

        self.layout.addWidget(QLabel("Run individual steps / Visualize"), 44, 0, 1, 2)
        self.layout.addWidget(self.preprocess_button, 45, 0, 1, 2)
        self.layout.addWidget(self.segment_button, 46, 0, 1, 2)
        self.layout.addWidget(self.mocap_button, 47, 0, 1, 2)
        self.layout.addWidget(self.track_button, 48, 0, 1, 2)
        self.layout.addWidget(self.feature_export_button, 49, 0, 1, 2)
        self.layout.addWidget(self.reassign_button, 50, 0, 1, 2)

    def change_channel(self):
        if self.nellie.file_select.single:
            self.nellie.file_select.initialize_single_file()
        else:
            self.nellie.file_select.initialize_folder(self.nellie.file_select.filepath)

    def change_t(self):
        self.num_t = self.time_input.value()

    def run_all(self):
        self.run_preprocessing()
        self.run_segmentation()
        self.run_mocap()
        self.run_tracking()
        self.run_reassignment()
        self.run_feature_export()

    def run_feature_export(self):
        for file in self.nellie.valid_files:
            feature_export = Hierarchy(file, self.num_t)
            feature_export.run()
        self.enable_buttons()

    def run_preprocessing(self):
        for file in self.nellie.valid_files:
            try:
                preprocessing = Filter(file, self.num_t, remove_edges=self.remove_edges_checkbox.isChecked())
                preprocessing.run()
            except Exception as e:
                show_info(f"Error in preprocessing {file.filename}: {e}")
        self.enable_buttons()

    def run_segmentation(self):
        for file in self.nellie.valid_files:
            try:
                segmentation = Label(file, self.num_t)
                segmentation.run()
                networking = Network(file, self.num_t)
                networking.run()
            except Exception as e:
                show_info(f"Error in segmentation {file.filename}: {e}")
        self.enable_buttons()

    def run_mocap(self):
        for file in self.nellie.valid_files:
            try:
                mocap_marking = Markers(file, self.num_t)
                mocap_marking.run()
            except Exception as e:
                show_info(f"Error in mocap marking {file.filename}: {e}")
        self.enable_buttons()

    def run_tracking(self):
        for file in self.nellie.valid_files:
            try:
                hu_tracking = HuMomentTracking(file, self.num_t)
                hu_tracking.run()
            except Exception as e:
                show_info(f"Error in tracking {file.filename}: {e}")
        self.enable_buttons()

    def run_reassignment(self):
        for file in self.nellie.valid_files:
            try:
                vox_reassign = VoxelReassigner(file, self.num_t)
                vox_reassign.run()
            except Exception as e:
                show_info(f"Error in voxel reassignment {file.filename}: {e}")
        self.enable_buttons()

    def check_sections(self):
        if self.check_all_exists('im_frangi'):
            self.all_preprocessed = True
        if self.check_all_exists('im_instance_label'):
            self.all_segmented = True
        if self.check_all_exists('im_marker'):
            self.all_marked = True
        if self.check_all_exists('flow_vector_array'):
            self.all_tracked = True
        if self.check_all_exists('im_branch_label_reassigned'):
            self.all_reassigned = True

    def check_all_exists(self, filepath_name):
        for file in self.nellie.valid_files:
            filepath = file.pipeline_paths[filepath_name]
            if not os.path.exists(filepath):
                return False
        return True

    def enable_buttons(self):
        self.check_sections()
        self.preprocess_button.setEnabled(True)

        if self.all_preprocessed:
            self.segment_button.setEnabled(True)
        else:
            self.segment_button.setEnabled(False)

        if self.all_segmented:
            self.mocap_button.setEnabled(True)
        else:
            self.mocap_button.setEnabled(False)

        if self.all_marked:
            self.track_button.setEnabled(True)
        else:
            self.track_button.setEnabled(False)

        if self.all_tracked:
            self.reassign_button.setEnabled(True)
            self.feature_export_button.setEnabled(True)
        else:
            self.reassign_button.setEnabled(False)
            self.feature_export_button.setEnabled(False)


if __name__ == "__main__":
    import napari
    viewer = napari.Viewer()
    napari.run()
