import cv2
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QImage, QPixmap


class ImageProcessing:
    def __init__(self, file_path):
        self.file_path = file_path
        self.image = cv2.imread(self.file_path)
        self.row_image, self.column_image, self.channel_image = self.image.shape
        self.image_gray_scale = False
        if self.channel_image == 1:
            self.image_gray_scale = True
            print("Image is grayscale")
        else:
            print("Image is RGB")

    def pixelate_image(self, block_dim_tuple):
        self.image = cv2.resize(self.image, block_dim_tuple, interpolation=cv2.INTER_NEAREST)
        self.image = cv2.resize(self.image, (self.column_image, self.row_image), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(self.file_path, self.image)

    def show_image(self, graphics_view):
        if self.image_gray_scale:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # Format_Grayscale8 for grayscale images

        image = QtGui.QImage(self.image, self.column_image, self.row_image, QtGui.QImage.Format_BGR888)
        pixmap = QtGui.QPixmap.fromImage(image)
        graphics_view.scene = QtWidgets.QGraphicsScene()
        graphics_view.scene.addPixmap(pixmap)
        graphics_view.setScene(graphics_view.scene)
        graphics_view.show()
