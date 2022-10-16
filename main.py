# use ui file 'C:\Users\orhan\PycharmProjects\imageProcessingPixelation\ui_desing_file\design_file_pixelation.ui'
# 1) The user is asked to write the file path of the image he wants to process.
# 2) The size information of the image is saved in the predetermined "row_image" (row) and "column_image" variables.
# 3) a function checks if the image is grayscale or RGB and if it is grayscale the "image_gray_scale"
# variable is false if not true. print image is grayscale or RGB is printed.
# 4) The values that the user wants to divide the photo into parcels according to the "block_vertical"
# "block_horizontal" values are taken from the user. The vertical length value to the "block_vertical"
# variable is assigned the horizontal length value to the "block_horizontal" variable. These values
# are assigned to the "block_dim_tuple" variable.
# 5) The image is zoned according to the "block_dim_tuple" values. Each region is zoned so that the
# average of the colors there is in its color.
# 6) the new image output is saved to the specified file path.
# Subject of this project is pixelation of image. The image is divided into small parts and the average of the colors
# in each part is taken and the color of the part is changed to the average color. The image is pixelated. We use
# the cv2 library to read the image and the numpy library to process the image. The image is divided into small parts
# and the average of the colors in each part is taken and the color of the part is changed to the average color.


import sys

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from image_processing.imageprocessing import ImageProcessing
from ui_desing_file.design_file_pixelation import Ui_MainWindow


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.image = None
        self.file_path = ""
        self.block_dim_tuple_user = 0
        self.block_horizontal_user = 0
        self.block_vertical_user = 0
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.label_value_white.setText("0")
        self.ui.label_value_black.setText("0")
        self.ui.pushButton.clicked.connect(self.open_file)
        self.ui.push_button_update.clicked.connect(self.update_image)
        self.ui.push_button_reflect.clicked.connect(self.reflect_image)
        self.ui.push_button_crop.clicked.connect(self.crop_image)
        self.ui.push_button_resize.clicked.connect(self.resize_image)
        self.ui.push_button_shift.clicked.connect(self.shift_image)
        self.ui.push_button_rgb_hsi.clicked.connect(self.rgb_hsi)
        # sliders
        self.ui.horizontal_slider_white.setMinimum(0)
        self.ui.horizontal_slider_white.setMaximum(255)
        self.ui.horizontal_slider_black.setMinimum(0)
        self.ui.horizontal_slider_black.setMaximum(255)
        self.ui.horizontal_slider_white.valueChanged.connect(self.ui.label_value_white.setNum)
        self.ui.horizontal_slider_black.valueChanged.connect(self.ui.label_value_black.setNum)
        self.ui.push_button_add_1_white.clicked.connect(lambda: self.ui.horizontal_slider_white.setValue(
            self.ui.horizontal_slider_white.value() + 1))
        self.ui.push_button_add_1_black.clicked.connect(lambda: self.ui.horizontal_slider_black.setValue(
            self.ui.horizontal_slider_black.value() + 1))
        self.ui.push_button_subtract_1_white.clicked.connect(lambda: self.ui.horizontal_slider_white.setValue(
            self.ui.horizontal_slider_white.value() - 1))
        self.ui.push_button_subtract_1_black.clicked.connect(lambda: self.ui.horizontal_slider_black.setValue(
            self.ui.horizontal_slider_black.value() - 1))
        # listen push_button_set_transfer_function button click if clicked call transfer_function_process function
        self.ui.push_button_set_transfer_function.clicked.connect(self.set_transfer_function)
        # listen push_button_stretch button click if clicked call stretch_process function
        self.ui.push_button_stretching.clicked.connect(self.stretch_image)
        # listen push_button_equalization button click if clicked call equalization_process function
        self.ui.push_button_equalization.clicked.connect(self.equalization_image)

    def equalization_image(self):
        self.image.histogram_equalization_process()
        self.image.show_image(self.ui.graphics_view_output)

    def stretch_image(self):
        self.image.histogram_stretching_process()
        self.image.show_image(self.ui.graphics_view_output)

    def set_transfer_function(self):
        # take the values from the user and send them to the function
        try:
            white_value = int(self.ui.label_value_white.text())
            black_value = int(self.ui.label_value_black.text())
            self.image.transfer_function_process(white_value, black_value)
            # show returned image from self.image.transfer_function_process(white_value, black_value)
            self.image.show_image(self.ui.graphics_view_output)
        except ValueError:
            QMessageBox.warning(self, "Warning", "Please enter a number")

    def rgb_hsi(self):
        while True:
            if self.ui.radio_button_vertical.isChecked():
                self.image.rgb_to_hsi_process()
            elif self.ui.radio_button_horizontal.isChecked():
                self.image.hsi_to_rgb_process()
            self.image.show_image(self.ui.graphics_view_output)
            break

    def shift_image(self):
        try:
            # if line_edit_shift is not empty take the value from the user and send it to the function
            if self.ui.line_edit_shift.text() == "":
                return

            shift_value = int(self.ui.line_edit_shift.text())
            # right or left or up or down
            if self.ui.radio_button_up.isChecked():
                self.image.shifting_image_process("up", shift_value)
            elif self.ui.radio_button_down.isChecked():
                self.image.shifting_image_process("down", shift_value)
            elif self.ui.radio_button_left.isChecked():
                self.image.shifting_image_process("left", shift_value)
            elif self.ui.radio_button_right.isChecked():
                self.image.shifting_image_process("right", shift_value)
            # convert brg to rgb
            if self.image.image_gray_scale is False:
                self.image.convert_image_bgr_to_rgb()
            self.image.show_image(self.ui.graphics_view_output)
        except ValueError:
            QMessageBox.warning(self, "Warning", "Please enter a number")

    def crop_image(self):
        # take the values from the user and send them to the function
        try:
            x1 = int(self.ui.line_edit_horizontal.text())
            y1 = int(self.ui.line_edit_vertical.text())
            x2 = int(self.ui.line_edit_width.text())
            y2 = int(self.ui.line_edit_height.text())
            self.image.crop_image_process(x1, y1, x2, y2)
            # bgr to rgb
            if self.image.image_gray_scale is False:
                self.image.convert_image_bgr_to_rgb()
            self.image.show_image(self.ui.graphics_view_output)
        except ValueError:
            QMessageBox.warning(self, "Warning", "Please enter a number")

    def resize_image(self):
        # take the values from the user and send them to the function
        try:
            user_specified_value_row = int(self.ui.line_edit_height.text())
            user_specified_value_column = int(self.ui.line_edit_width.text())
            self.image.resize_image_process(user_specified_value_row, user_specified_value_column)
            # convert image bgr to rgb
            if self.image.image_gray_scale is False:
                self.image.convert_image_bgr_to_rgb()
            self.image.show_image_new_window()
        except ValueError:
            QMessageBox.warning(self, "Warning", "Please enter a number")

    def open_file(self):
        # *.png *.jpg *.bmp *.tif *.tiff
        self.file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "",
                                                        "Image Files (*.png *.jpg *.bmp *.tif *.tiff)")
        self.image = ImageProcessing(self.file_path)
        self.image.show_image(self.ui.graphics_view_input)

    def update_image(self):
        try:
            self.block_vertical_user = int(self.ui.line_edit_vertical.text())
            self.block_horizontal_user = int(self.ui.line_edit_horizontal.text())
            self.block_dim_tuple_user = (self.block_vertical_user, self.block_horizontal_user)
            # take pixelate image function result and show image
            self.image.pixelate_image(self.block_dim_tuple_user)
            self.image.show_image(self.ui.graphics_view_output)
        except ValueError:
            QMessageBox.warning(self, "Warning", "Please enter a number")

    def reflect_image(self):
        # check radio_button_horizontal and radio_button_vertical state assign the state of these buttons to the
        # variable and send the variable to the function
        if self.ui.radio_button_horizontal.isChecked() and self.ui.radio_button_vertical.isChecked():
            self.image.reflect_image_process(reflect_type="both")
        elif self.ui.radio_button_horizontal.isChecked():
            self.image.reflect_image_process(reflect_type="horizontal")
        elif self.ui.radio_button_vertical.isChecked():
            self.image.reflect_image_process(reflect_type="vertical")
        else:
            QMessageBox.warning(self, "Warning", "Please select a reflection type")

            # convert image bgr to rgb
        if self.image.image_gray_scale is False:
            self.image.convert_image_bgr_to_rgb()
        self.image.show_image(self.ui.graphics_view_output)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
