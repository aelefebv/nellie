from qtpy.QtWidgets import QTabWidget

from nellie_napari import NellieProcessor
from nellie_napari.batch_mode import BatchMode
from nellie_napari.nellie_analysis import NellieAnalysis
from nellie_napari.nellie_fileselect import NellieFileSelect


class NellieLoader(QTabWidget):
    def __init__(self, napari_viewer: 'napari.viewer.Viewer', parent=None):
        super().__init__(parent)
        self.file_select = NellieFileSelect(napari_viewer, self)
        self.processor = NellieProcessor(napari_viewer, self)
        self.analyzer = NellieAnalysis(napari_viewer, self)
        self.batch_mode = BatchMode(napari_viewer, self)

        self.file_select_tab = None
        self.processor_tab = None
        self.analysis_tab = None

        self.add_tabs()

    def add_tabs(self):
        self.file_select_tab = self.addTab(self.file_select, "File select")
        self.processor_tab = self.addTab(self.processor, "Processing")
        self.analysis_tab = self.addTab(self.analyzer, "Analysis")

        self.setTabEnabled(self.processor_tab, False)
        self.setTabEnabled(self.analysis_tab, False)

    def reset(self):
        # needs to be in reverse order
        self.removeTab(self.analysis_tab)
        self.removeTab(self.processor_tab)
        self.removeTab(self.file_select_tab)

        self.file_select = NellieFileSelect(self.file_select.viewer, self)
        self.processor = NellieProcessor(self.processor.viewer, self)
        self.analyzer = NellieAnalysis(self.analyzer.viewer, self)
        self.batch_mode = BatchMode(self.batch_mode.viewer, self)

        self.add_tabs()


if __name__ == "__main__":
    import napari
    viewer = napari.Viewer()
    napari.run()
