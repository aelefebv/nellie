from qtpy.QtWidgets import QTabWidget

from nellie_napari import NellieProcessor
from nellie_napari.nellie_analysis import NellieAnalysis
from nellie_napari.nellie_fileselect import NellieFileSelect


class NellieLoader(QTabWidget):
    def __init__(self, napari_viewer: 'napari.viewer.Viewer', parent=None):
        super().__init__(parent)
        self.file_select = NellieFileSelect(napari_viewer, self)
        self.addTab(self.file_select, "File select")

        self.processor = NellieProcessor(napari_viewer, self)
        self.addTab(self.processor, "Processing")

        self.analyzer = NellieAnalysis(napari_viewer, self)
        self.addTab(self.analyzer, "Analysis")


if __name__ == "__main__":
    import napari
    viewer = napari.Viewer()
    napari.run()
