from napari.utils.notifications import show_info
from qtpy.QtWidgets import QTabWidget

from nellie_napari import NellieProcessor
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
        self.settings = Settings(napari_viewer, self)

        self.home_tab = None
        self.file_select_tab = None
        self.processor_tab = None
        self.visualizer_tab = None
        self.analysis_tab = None
        self.settings_tab = None

        self.add_tabs()
        self.currentChanged.connect(self.on_tab_change)  # Connect the signal to the slot

        self.im_info = None
        self.im_info_list = None

    def add_tabs(self):
        self.home_tab = self.addTab(self.home, "Home")
        self.file_select_tab = self.addTab(self.file_select, "File validation")
        self.processor_tab = self.addTab(self.processor, "Process")
        self.visualizer_tab = self.addTab(self.visualizer, "Visualize")
        self.analysis_tab = self.addTab(self.analyzer, "Analyze")
        self.settings_tab = self.addTab(self.settings, "Settings")

        self.setTabEnabled(self.processor_tab, False)
        self.setTabEnabled(self.visualizer_tab, False)
        self.setTabEnabled(self.analysis_tab, False)

    def reset(self):
        self.setCurrentIndex(self.home_tab)

        # needs to be in reverse order
        self.removeTab(self.settings_tab)
        self.removeTab(self.analysis_tab)
        self.removeTab(self.visualizer_tab)
        self.removeTab(self.processor_tab)
        self.removeTab(self.file_select_tab)

        self.file_select = NellieFileSelect(self.file_select.viewer, self)
        self.processor = NellieProcessor(self.processor.viewer, self)
        self.visualizer = NellieVisualizer(self.visualizer.viewer, self)
        self.analyzer = NellieAnalysis(self.analyzer.viewer, self)
        self.settings = Settings(self.settings.viewer, self)

        self.add_tabs()

        self.im_info = None
        self.im_info_list = None

    def on_tab_change(self, index):
        if index == self.analysis_tab:  # Check if the Analyze tab is selected
            if not self.analyzer.initialized:
                show_info("Initializing analysis tab")
                self.analyzer.post_init()
        elif index == self.visualizer_tab:
            if not self.visualizer.initialized:
                show_info("Initializing visualizer tab")
                self.visualizer.post_init()
        self.settings.post_init()

    def go_process(self):
        if self.file_select.batch_fileinfo_list is None:
            self.im_info = self.file_select.im_info
        else:
            self.im_info = self.file_select.im_info[0]
            self.im_info_list = self.file_select.im_info
            print(self.im_info_list)
        self.setTabEnabled(self.processor_tab, True)
        self.setTabEnabled(self.visualizer_tab, True)
        self.processor.post_init()
        self.visualizer.post_init()
        self.on_tab_change(self.processor_tab)
        self.setCurrentIndex(self.processor_tab)


if __name__ == "__main__":
    import napari
    viewer = napari.Viewer()
    napari.run()
