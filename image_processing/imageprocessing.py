import cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QImage, QPixmap


def check_image_is_gray_scale(image):
    if len(image) == 2:
        return True
    else:
        return False


class ImageProcessing:
    def __init__(self, file_path):
        self.file_path = file_path
        # read image from file path and save to image variable as numpy array
        self.image = cv2.imread(self.file_path)
        self.row_image, self.column_image, self.channel_image = self.image.shape
        self.image_gray_scale = False
        print(self.row_image, self.column_image, self.channel_image)

        self.image_gray_scale = check_image_is_gray_scale(self.image.shape)
        if self.image_gray_scale:
            print("Grayscale")
        else:
            print("RGB")

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
            for i in range(start=0, stop=self.row_image, step=block_dim_tuple_user[0]):
                for j in range(start=0, stop=self.column_image, step=block_dim_tuple_user[1]):
                    # get average of block
                    average = np.average(self.image[i:i + block_dim_tuple_user[0], j:j + block_dim_tuple_user[1]],
                                         axis=(0, 1))
                    # set average to block
                    new_image[i:i + block_dim_tuple_user[0], j:j + block_dim_tuple_user[1]] = average
            self.image = new_image
        return self.image

    def reflect_image_process(self, reflect_type):
        if reflect_type == "horizontal":
            self.image = self.image[:, ::-1]
        elif reflect_type == "vertical":
            self.image = self.image[::-1, :]
        elif reflect_type == "both":
            self.image = self.image[::-1, ::-1]
        return self.image

    def resize_image_process(self, row, column):
        # resize without cv2.resize
        # get image size
        row_image, column_image, channel_image = self.image.shape
        # create new image with same size
        new_image = np.zeros((row, column, channel_image), np.uint8)
        # get resize factor
        resize_factor_row = row_image / row
        resize_factor_column = column_image / column
        # resize image
        for i in range(row):
            for j in range(column):
                new_image[i, j] = self.image[int(i * resize_factor_row), int(j * resize_factor_column)]
        self.image = new_image
        return self.image

    def crop_image_process(self, x1, y1, x2, y2):
        # create new image with same size
        new_image = np.zeros((self.row_image, self.column_image, self.channel_image), np.uint8)
        # crop image
        for i in range(x1, x2):
            for j in range(y1, y2):
                new_image[i, j] = self.image[i, j]
        self.image = new_image
        return self.image

    def shifting_image_process(self, shifting_type, shift):
        # create new image with same size
        new_image = np.zeros((self.row_image, self.column_image, self.channel_image), np.uint8)

        # if shifting type is right
        if shifting_type == "right":
            # shifting image
            for i in range(self.row_image):
                for j in range(self.column_image):
                    if j + shift < self.column_image:
                        new_image[i, j + shift] = self.image[i, j]
        # if shifting type is left
        elif shifting_type == "left":
            # shifting image
            for i in range(self.row_image):
                for j in range(self.column_image):
                    if j - shift >= 0:
                        new_image[i, j - shift] = self.image[i, j]
        # if shifting type is up
        elif shifting_type == "up":
            # shifting image
            for i in range(self.row_image):
                for j in range(self.column_image):
                    if i - shift >= 0:
                        new_image[i - shift, j] = self.image[i, j]
        # if shifting type is down
        elif shifting_type == "down":
            # shifting image
            for i in range(self.row_image):
                for j in range(self.column_image):
                    if i + shift < self.row_image:
                        new_image[i + shift, j] = self.image[i, j]

        self.image = new_image
        return self.image

    def rgb_to_hsi_process(self):
        if self.image_gray_scale:
            print("Image is not RGB")
            return None
        # convert GBR to HSI
        # https://stackoverflow.com/questions/52834537/rgb-to-hsi-conversion-hue-always-calculated-as-0
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                # get value of each channel
                b, g, r = cv2.split(self.image)
                # convert to float
                b = b.astype(float)
                g = g.astype(float)
                r = r.astype(float)
                # calculate I
                i = (b + g + r) / 3
                # calculate S
                s = 1 - (3 / (b + g + r)) * np.minimum(b, np.minimum(g, r))
                # calculate H
                h = np.arccos((0.5 * ((r - g) + (r - b))) / np.sqrt((r - g) ** 2 + (r - b) * (g - b)))
                # set H
                h[b > g] = 360 - h[b > g]
                h = h / 2
                # create new image
                new_image = np.zeros((self.row_image, self.column_image, 3), np.uint8)
                # set value to new image
                new_image[:, :, 0] = h
                new_image[:, :, 1] = s
                new_image[:, :, 2] = i
                # convert to uint8
                new_image = new_image.astype(np.uint8)
                self.image = new_image
                return self.image

    def show_image(self, graphics_view):
        if self.image_gray_scale:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        # convert to QImage
        q_image = QImage(self.image, self.column_image, self.row_image, self.column_image * 3, QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(q_image)
        graphics_view.scene = QtWidgets.QGraphicsScene()
        graphics_view.scene.addPixmap(pixmap)
        graphics_view.setScene(graphics_view.scene)
        graphics_view.show()

    def show_image_new_window(self):
        cv2.imshow("image", self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def convert_image_bgr_to_rgb(self):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
