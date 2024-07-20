from napari.utils.notifications import show_info
from qtpy.QtWidgets import QTabWidget

from nellie_napari import NellieProcessor
from nellie_napari.batch_mode import BatchMode
from nellie_napari.home import Home
from nellie_napari.nellie_analysis import NellieAnalysis
from nellie_napari.nellie_fileselect import NellieFileSelect
from nellie_napari.settings import Settings
from nellie_napari.visualizer import NellieVisualizer


class NellieLoader(QTabWidget):
    def __init__(self, napari_viewer: 'napari.viewer.Viewer', parent=None):
        super().__init__(parent)
        self.home = Home(napari_viewer, self)
        self.file_select = NellieFileSelect(napari_viewer, self)
        self.processor = NellieProcessor(napari_viewer, self)
        self.visualizer = NellieVisualizer(napari_viewer, self)
        self.analyzer = NellieAnalysis(napari_viewer, self)
        self.batch_mode = BatchMode(napari_viewer, self)
        self.settings = Settings(napari_viewer, self)

        self.home_tab = None
        self.file_select_tab = None
        self.processor_tab = None
        self.visualizer_tab = None
        self.analysis_tab = None
        self.batch_tab = None
        self.settings_tab = None

        self.add_tabs()
        self.currentChanged.connect(self.on_tab_change)  # Connect the signal to the slot

        self.im_info = None
        self.valid_files = []

    def add_tabs(self):
        self.home_tab = self.addTab(self.home, "Home")
        self.file_select_tab = self.addTab(self.file_select, "File select")
        self.processor_tab = self.addTab(self.processor, "Process")
        self.visualizer_tab = self.addTab(self.visualizer, "Visualize")
        self.analysis_tab = self.addTab(self.analyzer, "Analyze")
        self.batch_tab = self.addTab(self.batch_mode, "Batch")
        self.settings_tab = self.addTab(self.settings, "Settings")

        self.setTabEnabled(self.processor_tab, False)
        self.setTabEnabled(self.visualizer_tab, False)
        self.setTabEnabled(self.analysis_tab, False)
        self.setTabEnabled(self.batch_tab, False)

    def reset(self):
        self.setCurrentIndex(self.home_tab)

        # needs to be in reverse order
        self.removeTab(self.settings_tab)
        self.removeTab(self.batch_tab)
        self.removeTab(self.analysis_tab)
        self.removeTab(self.visualizer_tab)
        self.removeTab(self.processor_tab)
        self.removeTab(self.file_select_tab)

        self.file_select = NellieFileSelect(self.file_select.viewer, self)
        self.processor = NellieProcessor(self.processor.viewer, self)
        self.visualizer = NellieVisualizer(self.visualizer.viewer, self)
        self.analyzer = NellieAnalysis(self.analyzer.viewer, self)
        self.batch_mode = BatchMode(self.batch_mode.viewer, self)
        self.settings = Settings(self.settings.viewer, self)

        self.add_tabs()

        self.im_info = None

    # def file_ready(self):
    #     # self.im_info = self.file_select.im_info
    #     self.processor.post_init()
    #     self.visualizer.post_init()

    def on_tab_change(self, index):
        if index == self.analysis_tab:  # Check if the Analyze tab is selected
            if not self.analyzer.initialized:
                show_info("Initializing analysis tab")
                self.analyzer.post_init()
        elif index == self.visualizer_tab:
            if not self.visualizer.initialized:
                show_info("Initializing visualizer tab")
                self.visualizer.post_init()
        # elif index == self.processor_tab:
            # if not self.processor.initialized:
            #     show_info("Initializing processor tab")
        # elif index == self.settings_tab:
        #     if not self.settings.initialized:
        #         show_info("Initializing settings tab")
        self.settings.post_init()
        # else:
        #     return


if __name__ == "__main__":
    import napari
    viewer = napari.Viewer()
    napari.run()
