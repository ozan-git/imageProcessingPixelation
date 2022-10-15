import cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QImage


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

    def resize_image_process(self, user_specified_value_row, user_specified_value_column):
        # resize image using bi-linear interpolation method without using cv2.resize function
        # create new image with same size
        new_image = np.zeros((user_specified_value_row, user_specified_value_column, self.channel_image), np.uint8)
        # resize image
        for i in range(user_specified_value_row):
            for j in range(user_specified_value_column):
                # calculate new pixel
                x = i * self.row_image / user_specified_value_row
                y = j * self.column_image / user_specified_value_column
                new_image[i, j] = self.bi_linear_interpolation(x, y)

        self.image = new_image
        return self.image

    def bi_linear_interpolation(self, x, y):
        # calculate new pixel
        x1 = int(x)
        x2 = x1 + 1
        y1 = int(y)
        y2 = y1 + 1
        # calculate new pixel
        # if  index 512 is out of bounds for axis 1 with size 512
        if x2 >= self.row_image:
            x2 = self.row_image - 1
        if y2 >= self.column_image:
            y2 = self.column_image - 1
        # calculate new pixel
        new_pixel = (x2 - x) * (y2 - y) * self.image[x1, y1] +\
                    (x2 - x) * (y - y1) * self.image[x1, y2] +\
                    (x - x1) * (y2 - y) * self.image[x2, y1] +\
                    (x - x1) * (y - y1) * self.image[x2, y2]
        return new_pixel
        # main idea of this method is from https://stackoverflow.com/questions/12729228/simple-efficient-bilinear-interpolation-of-images-in-numpy-and-python

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
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                # get value of each channel
                r = self.image[:, :, 0]
                g = self.image[:, :, 1]
                b = self.image[:, :, 2]
                # get image size
                row_image, column_image, channel_image = self.image.shape
                # create new image with same size
                new_image = np.zeros((row_image, column_image, channel_image), np.uint8)
                # convert rgb to hsi
                for k in range(row_image):
                    for j in range(column_image):
                        # get value of each channel
                        r = self.image[k, j, 0]
                        g = self.image[k, j, 1]
                        b = self.image[k, j, 2]
                        # calculate h
                        h = np.arccos(((0.5 * ((r - g) + (r - b))) / np.sqrt((r - g) ** 2 + (r - b) * (g - b)))) * \
                            180 / np.pi
                        if b <= g:
                            h = h
                        else:
                            h = 360 - h
                        # calculate s
                        s = 1 - (3 / (r + g + b)) * np.min([r, g, b])
                        # calculate i
                        i = (r + g + b) / 3
                        # set value of each channel
                        new_image[k, j, 0] = h
                        new_image[k, j, 1] = s
                        new_image[k, j, 2] = i
                self.image = new_image
                return self.image

    def hsi_to_rgb_process(self):
        if self.image_gray_scale:
            print("Image is not HSI")
            return None
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                # get value of each channel
                h = self.image[:, :, 0]
                s = self.image[:, :, 1]
                i = self.image[:, :, 2]
                # get image size
                row_image, column_image, channel_image = self.image.shape
                # create new image with same size
                new_image = np.zeros((row_image, column_image, channel_image), np.uint8)
                # convert hsi to rgb
                for k in range(row_image):
                    for j in range(column_image):
                        # get value of each channel
                        h = self.image[k, j, 0]
                        s = self.image[k, j, 1]
                        i = self.image[k, j, 2]
                        # calculate r
                        r = i + (i * s * np.cos(h * np.pi / 180)) / np.cos((60 - h) * np.pi / 180)
                        # calculate g
                        g = i + (i * s * np.cos((h - 120) * np.pi / 180)) / np.cos((180 - h) * np.pi / 180)
                        # calculate b
                        b = 3 * i - r - g
                        # set value of each channel
                        new_image[k, j, 0] = b
                        new_image[k, j, 1] = g
                        new_image[k, j, 2] = r
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
