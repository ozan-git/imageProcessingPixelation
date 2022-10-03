import cv2
from PyQt5 import QtWidgets, QtGui, QtCore

class ImageProcessing:
    def __init__(self, file_path):
        self.image_path = file_path
        self.image = cv2.imread(self.image_path)
        self.image_width, self.image_height, self.image_channel = self.image.shape
        self.image_gray_scale = False
        if self.image_channel == 1:
            self.image_gray_scale = True
            print("Image is grayscale")
        else:
            print("Image is RGB")

    def pixelate_image(self, pixelation_dimens):
        # Check if scaled_dimens are bigger then zero
        if pixelation_dimens[0] > 0 and pixelation_dimens[1] > 0:
            # Scale image to a lower ratio base on pixelation area.
            # Then scale it back to original size to get pixelated image.
            scaled_dimens = (self.image_width // pixelation_dimens[0], self.image_height // pixelation_dimens[1])

            self.image = cv2.resize(self.image, scaled_dimens, interpolation=cv2.INTER_NEAREST)
            self.image = cv2.resize(self.image, (self.image_height, self.image_width), interpolation=cv2.INTER_NEAREST)

    def show_image(self, graphics_view):
        if self.image_gray_scale:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # Format_Grayscale8 for grayscale images

        image = QtGui.QImage(self.image, self.image_height, self.image_width, QtGui.QImage.Format_BGR888)
        pixmap = QtGui.QPixmap.fromImage(image)
        graphics_view.scene = QtWidgets.QGraphicsScene()
        graphics_view.scene.addPixmap(pixmap)
        graphics_view.setScene(graphics_view.scene)
        graphics_view.fitInView(graphics_view.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
