import os

from qtpy.QtWidgets import QWidget, QPushButton, QLabel, QVBoxLayout
from nellie.feature_extraction.hierarchical import Hierarchy
from nellie.segmentation.labelling import Label
from nellie.segmentation.mocap_marking import Markers
from nellie.segmentation.networking import Network
from nellie.tracking.hu_tracking import HuMomentTracking
from nellie.tracking.voxel_reassignment import VoxelReassigner
from nellie.segmentation.filtering import Filter


class BatchMode(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.valid_files = None
        self.num_t = None
        self.remove_edges = None
        self.viewer = None

        self.im_info = None

        self.setLayout(QVBoxLayout())

        self.all_preprocessed = False
        self.all_segmented = False
        self.all_networked = False
        self.all_marked = False
        self.all_tracked = False
        self.all_reassigned = False

        self.run_all_button = QPushButton(text="Run all")
        self.run_all_button.clicked.connect(self.run_all)
        self.run_all_button.setEnabled(True)

        self.preprocessed_button = QPushButton(text="Run preprocessing")
        self.preprocessed_button.clicked.connect(self.run_preprocessing)
        self.preprocessed_button.setEnabled(False)

        self.segmentation_button = QPushButton(text="Run segmentation")
        self.segmentation_button.clicked.connect(self.run_segmentation)
        self.segmentation_button.setEnabled(False)

        self.network_button = QPushButton(text="Run networking")
        self.network_button.clicked.connect(self.run_networking)
        self.network_button.setEnabled(False)

        self.mocap_button = QPushButton(text="Run mocap marking")
        self.mocap_button.clicked.connect(self.run_mocap)
        self.mocap_button.setEnabled(False)

        self.tracking_button = QPushButton(text="Run tracking")
        self.tracking_button.clicked.connect(self.run_tracking)
        self.tracking_button.setEnabled(False)

        self.reassign_button = QPushButton(text="Run voxel reassignment")
        self.reassign_button.clicked.connect(self.run_reassignment)
        self.reassign_button.setEnabled(False)

        self.feature_export_button = QPushButton(text="Run feature export")
        self.feature_export_button.clicked.connect(self.run_feature_export)
        self.feature_export_button.setEnabled(False)

        self.layout().addWidget(QLabel("\n Run full pipeline"))
        self.layout().addWidget(self.run_all_button)
        self.layout().addWidget(QLabel("\n Run individual sections"))
        self.layout().addWidget(self.preprocessed_button)
        self.layout().addWidget(self.segmentation_button)
        self.layout().addWidget(self.network_button)
        self.layout().addWidget(self.mocap_button)
        self.layout().addWidget(self.tracking_button)
        self.layout().addWidget(self.reassign_button)
        self.layout().addWidget(self.feature_export_button)

    def run_all(self):
        self.run_preprocessing()
        self.run_segmentation()
        self.run_networking()
        self.run_mocap()
        self.run_tracking()
        self.run_reassignment()
        self.run_feature_export()

    def run_feature_export(self):
        for file in self.valid_files:
            feature_export = Hierarchy(file, self.num_t)
            feature_export.run()
        self.enable_buttons()

    def run_preprocessing(self):
        for file in self.valid_files:
            preprocessing = Filter(file, self.num_t, remove_edges=self.remove_edges)
            preprocessing.run()
        self.enable_buttons()

    def run_segmentation(self):
        for file in self.valid_files:
            segmentation = Label(file, self.num_t)
            segmentation.run()
        self.enable_buttons()

    def run_networking(self):
        for file in self.valid_files:
            networking = Network(file, self.num_t)
            networking.run()
        self.enable_buttons()

    def run_mocap(self):
        for file in self.valid_files:
            mocap_marking = Markers(file, self.num_t)
            mocap_marking.run()
        self.enable_buttons()

    def run_tracking(self):
        for file in self.valid_files:
            hu_tracking = HuMomentTracking(file, self.num_t)
            hu_tracking.run()
        self.enable_buttons()

    def run_reassignment(self):
        for file in self.valid_files:
            vox_reassign = VoxelReassigner(file, self.num_t)
            vox_reassign.run()
        self.enable_buttons()

    def check_sections(self):
        if self.check_all_exists('im_frangi'):
            self.all_preprocessed = True
        if self.check_all_exists('im_instance_label'):
            self.all_segmented = True
        if self.check_all_exists('im_skel_relabelled'):
            self.all_networked = True
        if self.check_all_exists('im_marker'):
            self.all_marked = True
        if self.check_all_exists('flow_vector_array'):
            self.all_tracked = True
        if self.check_all_exists('im_instance_label_reassigned'):
            self.all_reassigned = True

    def check_all_exists(self, filepath_name):
        for file in self.valid_files:
            filepath = file.pipeline_paths[filepath_name]
            if not os.path.exists(filepath):
                return False
        return True

    def enable_buttons(self):
        self.check_sections()
        self.preprocessed_button.setEnabled(True)

        if self.all_preprocessed:
            self.segmentation_button.setEnabled(True)
        else:
            self.segmentation_button.setEnabled(False)

        if self.all_segmented:
            self.network_button.setEnabled(True)
        else:
            self.network_button.setEnabled(False)

        if self.all_networked:
            self.mocap_button.setEnabled(True)
        else:
            self.mocap_button.setEnabled(False)

        if self.all_marked:
            self.tracking_button.setEnabled(True)
        else:
            self.tracking_button.setEnabled(False)

        if self.all_tracked:
            self.reassign_button.setEnabled(True)
            self.feature_export_button.setEnabled(True)
        else:
            self.reassign_button.setEnabled(False)
            self.feature_export_button.setEnabled(False)
