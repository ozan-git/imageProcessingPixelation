import cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QImage


class ImageProcessing:
    def __init__(self, file_path):
        self.file_path = file_path
        # read image from file path and save to image variable as numpy array
        self.image = cv2.imread(self.file_path)
        self.row_image, self.column_image, self.channel_image = self.image.shape
        self.image_gray_scale = False
        print(self.row_image, self.column_image, self.channel_image)
        # if image is grayscale
        if self.image.shape[2] == 1:
            self.image_gray_scale = True
            print("Image is grayscale")
        # if image is RGB
        else:
            self.image_gray_scale = False
            print("Image is RGB")

    def pixelate_image(self, block_dim_tuple_user):
        # The image is divided into blocks according to the block_dim_tuple values. Each region is zoned so that the
        # average of the colors there is in its color.
        # tuple row, column is block_dim_tuple
        # block_dim_tuple = (row, column)
        # we have two situation grayscale and RGB image
        # if image is grayscale
        if self.image_gray_scale:
            # create new image with same size
            new_image = np.zeros((self.row_image, self.column_image, 1), np.uint8)
            # create block
            for i in range(0, self.row_image, block_dim_tuple_user[0]):
                for j in range(0, self.column_image, block_dim_tuple_user[1]):
                    # get average of block
                    average = np.average(self.image[i:i + block_dim_tuple_user[0], j:j + block_dim_tuple_user[1]])
                    # set average to block
                    new_image[i:i + block_dim_tuple_user[0], j:j + block_dim_tuple_user[1]] = average
            self.image = new_image
        # if image is RGB
        else:
            # create new image with same size
            new_image = np.zeros((self.row_image, self.column_image, 3), np.uint8)
            # create block
            for i in range(0, self.row_image, block_dim_tuple_user[0]):
                for j in range(0, self.column_image, block_dim_tuple_user[1]):
                    # get average of block
                    average = np.average(self.image[i:i + block_dim_tuple_user[0], j:j + block_dim_tuple_user[1]],
                                         axis=(0, 1))
                    # set average to block
                    new_image[i:i + block_dim_tuple_user[0], j:j + block_dim_tuple_user[1]] = average
            self.image = new_image
        return self.image
        # block_row_size, block_column_size = block_dim_tuple
        #
        # for i in range(0, self.row_image, block_row_size):
        #     for j in range(0, self.column_image, block_column_size):
        #         self.image[i:i + block_row_size, j:j + block_column_size] = \
        #             cv2.mean(self.image[i:i + block_row_size, j:j + block_column_size])[:int(self.channel_image)]
        # return self.image

    def show_image(self, graphics_view):
        if self.image_gray_scale:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # Format_Grayscale8 for grayscale images

        image = QtGui.QImage(self.image, self.column_image, self.row_image, QImage.Format_BGR888)
        pixmap = QtGui.QPixmap.fromImage(image)
        graphics_view.scene = QtWidgets.QGraphicsScene()
        graphics_view.scene.addPixmap(pixmap)
        graphics_view.setScene(graphics_view.scene)
        graphics_view.show()
