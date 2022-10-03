import sys

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from image_pixelation.imageprocessing import ImageProcessing
from ui.design_file_pixelation import Ui_MainWindow


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.image = None
        self.pixelation_dimens = 0
        self.pixel_height = 0
        self.pixel_width = 0
        self.file_path = ""
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.select_file)
        self.ui.push_button_pixelate.clicked.connect(self.pixelate_image)

    def select_file(self):
        self.file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        self.image = ImageProcessing(self.file_path)
        self.image.show_image(self.ui.graphics_view_image)

    def pixelate_image(self):
        try:
            self.pixel_width = int(self.ui.line_edit_width.text())
            self.pixel_height = int(self.ui.line_edit_height.text())
            self.pixelation_dimens = (self.pixel_width, self.pixel_height)
            self.image.pixelate_image(self.pixelation_dimens)
            self.image.show_image(self.ui.graphics_view_image)
        except ValueError:
            QMessageBox.warning(self, "Warning", "Please enter a number")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())