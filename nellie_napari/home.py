from qtpy.QtWidgets import QLabel, QGridLayout, QWidget, QPushButton, QMessageBox
import napari
import os
import datetime
import tifffile
from napari.utils.notifications import show_info


class Home(QWidget):
    def __init__(self, napari_viewer: 'napari.viewer.Viewer', nellie, parent=None):
        super().__init__(parent)
        self.nellie = nellie
        self.viewer = napari_viewer

        self.layout = QGridLayout()
        self.setLayout(self.layout)

        # screenshot button
        self.screenshot_button = QPushButton(text="Ctrl/Cmd-Shift-E")
        self.screenshot_button.clicked.connect(self.screenshot)
        self.screenshot_button.setEnabled(True)
        self.viewer.bind_key('Ctrl-Shift-E', self.screenshot, overwrite=True)

        self.layout.addWidget(QLabel("\nEasy screenshot"), 50, 0, 1, 2)
        self.layout.addWidget(self.screenshot_button, 51, 0, 1, 2)

    def screenshot(self, event=None):
        if self.nellie.im_info is None:
            show_info("No file selected, cannot take screenshot")
            return

        # easy no prompt screenshot
        dt = datetime.datetime.now()  # year, month, day, hour, minute, second, millisecond up to 3 digits
        dt = dt.strftime("%Y%m%d_%H%M%S%f")[:-3]

        screenshot_folder = self.nellie.im_info.screenshot_dir
        if not os.path.exists(screenshot_folder):
            os.makedirs(screenshot_folder)

        im_name = f'{dt}-{self.nellie.im_info.basename_no_ext}.png'
        file_path = os.path.join(screenshot_folder, im_name)

        # Take screenshot
        screenshot = self.viewer.screenshot(canvas_only=True)

        # Save the screenshot
        try:
            # save as png to file_path using tifffile
            tifffile.imwrite(file_path, screenshot)
            show_info(f"Screenshot saved to {file_path}")
        except Exception as e:
            QMessageBox.warning(None, "Error", f"Failed to save screenshot: {str(e)}")
            raise e


if __name__ == "__main__":
    import napari
    viewer = napari.Viewer()
    napari.run()
