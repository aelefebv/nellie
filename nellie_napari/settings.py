from qtpy.QtWidgets import QWidget, QGridLayout, QCheckBox, QSpinBox, QLabel
import napari


class Settings(QWidget):
    def __init__(self, napari_viewer: 'napari.viewer.Viewer', nellie, parent=None):
        super().__init__(parent)
        self.nellie = nellie
        self.viewer = napari_viewer

        self.layout = QGridLayout()
        self.setLayout(self.layout)

        # Checkbox for 'Remove edges'
        self.remove_edges_checkbox = QCheckBox("Remove image edges")
        self.remove_edges_checkbox.setEnabled(False)
        self.remove_edges_checkbox.setToolTip(
            "Originally for Snouty deskewed images. If you see weird image edge artifacts, enable this.")

        self.voxel_reassign = QCheckBox("Auto-run voxel reassignment")
        self.voxel_reassign.setChecked(False)
        self.voxel_reassign.setEnabled(True)

        # Analyze node level
        self.analyze_node_level = QCheckBox("Analyze node level (slow)")
        self.analyze_node_level.setChecked(False)
        self.analyze_node_level.setEnabled(True)

        # Track all frames
        self.track_all_frames = QCheckBox("Track all frames' voxels")
        self.track_all_frames.setChecked(True)
        self.track_all_frames.setEnabled(True)

        # Label above the spinner box
        self.skip_vox_label = QLabel("Track every N voxel. N=")

        self.skip_vox = QSpinBox()
        self.skip_vox.setRange(1, 10000)
        self.skip_vox.setValue(5)
        self.skip_vox.setEnabled(False)

        self.layout.addWidget(self.remove_edges_checkbox, 0, 0)
        self.layout.addWidget(self.analyze_node_level, 1, 0)
        self.layout.addWidget(self.voxel_reassign, 1, 1)

        self.layout.addWidget(self.skip_vox_label, 6, 0, 1, 1)
        self.layout.addWidget(self.skip_vox, 6, 1, 1, 1)
        self.layout.addWidget(self.track_all_frames, 7, 0, 1, 1)


        self.initialized = False

    def post_init(self):
        self.initialized = True


if __name__ == "__main__":
    import napari
    viewer = napari.Viewer()
    napari.run()
