import cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QImage


def check_image_is_gray_scale(image):
    if len(image) == 2:
        return True
    else:
        return False


def apply_desired_transfer_function(pixel, white_level, black_level):
    if pixel > white_level:
        return 255
    elif pixel < black_level:
        return 0
    else:
        return np.round(((pixel - black_level) / (white_level - black_level)) * 255)


class ImageProcessing:
    def __init__(self, file_path):
        self.filter_type = None
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

    # main idea of this method is from
    # https://stackoverflow.com/questions/12729228/simple-efficient-bilinear-interpolation-of-images-in-numpy-and-python
    # https://en.wikipedia.org/wiki/Bilinear_interpolation
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
        new_pixel = (x2 - x) * (y2 - y) * self.image[x1, y1] + \
                    (x2 - x) * (y - y1) * self.image[x1, y2] + \
                    (x - x1) * (y2 - y) * self.image[x2, y1] + \
                    (x - x1) * (y - y1) * self.image[x2, y2]
        return new_pixel

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

    # show specified image with specified name in new window
    def show_image_new_window_specified(self, image, name):
        cv2.imshow(name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def convert_image_bgr_to_rgb(self):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

    def histogram_stretching_process(self):
        # get image size
        row_image, column_image, channel_image = self.image.shape
        # create new image with same size
        new_image = np.zeros((row_image, column_image, channel_image), np.uint8)
        # get value of each channel
        b = self.image[:, :, 0]
        g = self.image[:, :, 1]
        r = self.image[:, :, 2]
        # calculate min and max value of each channel
        r_min = np.min(r)
        r_max = np.max(r)
        g_min = np.min(g)
        g_max = np.max(g)
        b_min = np.min(b)
        b_max = np.max(b)
        # calculate new value of each channel
        for k in range(row_image):
            for j in range(column_image):
                new_image[k, j, 2] = ((b[k, j] - b_min) / (b_max - b_min)) * 250
                new_image[k, j, 1] = ((g[k, j] - g_min) / (g_max - g_min)) * 250
                new_image[k, j, 0] = ((r[k, j] - r_min) / (r_max - r_min)) * 250
        self.image = new_image
        return self.image

    # cdf(v) = round(((cdf(v) - cdf_min)/(M*N - cdf_min))*(L-1))
    def histogram_equalization_process(self):
        # get image size
        row_image, column_image, channel_image = self.image.shape
        # create new image with same size
        new_image = np.zeros((row_image, column_image, channel_image), np.uint8)
        # get value of each channel
        b = self.image[:, :, 0]
        g = self.image[:, :, 1]
        r = self.image[:, :, 2]
        # calculate cdf of each channel
        b_cdf = np.zeros(256)
        g_cdf = np.zeros(256)
        r_cdf = np.zeros(256)
        for k in range(256):
            b_cdf[k] = np.sum(b <= k)
            g_cdf[k] = np.sum(g <= k)
            r_cdf[k] = np.sum(r <= k)
        # calculate min value of cdf
        b_cdf_min = np.min(b_cdf)
        g_cdf_min = np.min(g_cdf)
        r_cdf_min = np.min(r_cdf)
        # calculate new value of each channel
        for k in range(row_image):
            for j in range(column_image):
                new_image[k, j, 2] = round(
                    ((b_cdf[b[k, j]] - b_cdf_min) / ((row_image * column_image) - b_cdf_min)) * 255)
                new_image[k, j, 1] = round(
                    ((g_cdf[g[k, j]] - g_cdf_min) / ((row_image * column_image) - g_cdf_min)) * 255)
                new_image[k, j, 0] = round(
                    ((r_cdf[r[k, j]] - r_cdf_min) / ((row_image * column_image) - r_cdf_min)) * 255)
        self.image = new_image
        return self.image

    def histogram_equalization_process_2(self):
        row, column, channel = self.image.shape
        # unique = value of each pixel counts = count of each pixel
        unique, counts = np.unique(self.image, return_counts=True)
        # calculate probability of each pixel
        total = 0
        new_value_list = []
        new_pixel_list = []
        new_cdf_list = []
        new_equalization_list = []
        # trip each counts length and calculate probability of each pixel
        for i in range(len(counts)):
            # value of pixel (0-255)
            value_of_pixel = unique[i]
            # because of 3 channel (RGB) so we need to calculate probability of each channel
            count_of_pixel = counts[i] = counts[i] / 3
            new_value_list.append(value_of_pixel)
            new_pixel_list.append(count_of_pixel)

        for j in range(len(new_pixel_list)):
            total += new_pixel_list[j]
            new_cdf_list.append(total)
            # calculate new value of each pixel
            min_cdf = np.min(new_cdf_list)
            value_of_pixel = round(((new_cdf_list[j] - min_cdf) / (row * column - min_cdf)) * 255)
            new_equalization_list.append(value_of_pixel)

        equalization = dict(zip(new_value_list, new_equalization_list))
        for k in range(row):
            for j in range(column):
                self.image[k, j, 2] = equalization[self.image[k, j, 0]]
                self.image[k, j, 1] = equalization[self.image[k, j, 1]]
                self.image[k, j, 0] = equalization[self.image[k, j, 2]]
        return self.image

    # This function applies the desired transfer function to an RGB image. An image will be selected from the list and
    # the transfer function will be applied to the image. For example, since 184 is displayed, the slide bar for White
    # level is displayed, so anything above 184 in the image is assigned a value of 255. Since the black scroll bar is
    # at 51, pixels with a value less than 51 are assigned 0 pixels. There can be a minimum of 3 pixels between the
    # white and black scrollbar. That is, if the white level is 130, the black level can be a maximum of 127.
    def transfer_function_process(self, white_level, black_level):
        # get image size
        row_image, column_image, channel_image = self.image.shape
        # create new image with same size
        new_image = np.zeros((row_image, column_image, channel_image), np.uint8)
        # get value of each channel
        r = self.image[:, :, 0]
        g = self.image[:, :, 1]
        b = self.image[:, :, 2]
        tres_hold_1 = black_level
        tres_hold_2 = white_level
        if tres_hold_1 > tres_hold_2:
            tres_hold_1 = white_level
            tres_hold_2 = black_level
        # check difference between white level and black level is greater than 3
        if abs(tres_hold_2 - tres_hold_1) >= 3:
            # calculate new value of each channel
            for k in range(row_image):
                for j in range(column_image):
                    new_image[k, j, 0] = apply_desired_transfer_function(r[k, j], tres_hold_2, tres_hold_1)
                    new_image[k, j, 1] = apply_desired_transfer_function(g[k, j], tres_hold_2, tres_hold_1)
                    new_image[k, j, 2] = apply_desired_transfer_function(b[k, j], tres_hold_2, tres_hold_1)
            self.image = new_image
            return self.image
        else:
            return self.image

    def min_filter_process(self, row, column):
        # row and column of image
        row_image, column_image, channel_image = self.image.shape

        # create new image with same size
        new_image = np.zeros((row_image, column_image, channel_image), np.uint8)

        for i in range(row_image - row - 1):
            for j in range(column_image - column - 1):
                new_image[i + 1][j + 1] = np.amin(self.image[i:i + row, j:j + column], axis=(0, 1))

        result_array_image = np.require(new_image, np.uint8, 'C')
        self.image = result_array_image
        return result_array_image

    def max_filter_process(self, row, column):
        # row and column of image
        row_image, column_image, channel_image = self.image.shape

        # create new image with same size
        new_image = np.zeros((row_image, column_image, channel_image), np.uint8)

        for i in range(row_image - row - 1):
            for j in range(column_image - column - 1):
                new_image[i + 1][j + 1] = np.amax(self.image[i:i + row, j:j + column], axis=(0, 1))

        result_array_image = np.require(new_image, np.uint8, 'C')
        self.image = result_array_image
        return result_array_image

    def average_filter_process(self, row, column, image):
        # get image size and create new image with same size
        row_image, column_image, channel_image = self.image.shape
        new_image = np.zeros((row_image, column_image, channel_image), np.uint8)
        mask = np.ones((row, column)) / 9
        img = self.image
        for k in range(1, row_image - 1):
            for j in range(1, column_image - 1):
                temp = img[k - 1, j - 1] * mask[0, 0] + img[k - 1, j] * mask[0, 1] + img[k - 1, j + 1] * mask[0, 2] + \
                       img[k, j - 1] * mask[1, 0] + img[k, j] * mask[1, 1] + img[k, j + 1] * mask[1, 2] + \
                       img[k + 1, j - 1] * mask[2, 0] + img[k + 1, j] * mask[2, 1] + img[k + 1, j + 1] * mask[2, 2]
                new_image[k, j] = temp
        self.image = new_image
        return self.image

    def median_filter_process(self):
        # get image size and create new image with same size
        row_image, column_image, channel_image = self.image.shape
        new_image = np.zeros((row_image, column_image, channel_image), np.uint8)
        img = self.image[:, :, 0]
        for k in range(1, row_image - 1):
            for j in range(1, column_image - 1):
                temp = [img[k - 1, j - 1], img[k - 1, j], img[k - 1, j + 1],
                        img[k, j - 1], img[k, j], img[k, j + 1],
                        img[k + 1, j - 1], img[k + 1, j], img[k + 1, j + 1]]
                temp = sorted(temp)
                new_image[k, j] = temp[4]
        self.image = new_image
        return new_image.astype(np.uint8)