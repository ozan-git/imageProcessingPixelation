import numpy as np


# using highpass filtering and threshold for image enhancement
# take original image as parameter highpass filter is used to enhance the edges of the image and remove the noise from
# the image after applying highpass filter, threshold is applied to the image
def highpass_filter_process(image):
    # get image size and create new image with same size
    row_image, column_image, channel_image = image.shape
    new_image = np.zeros((row_image, column_image, channel_image), np.uint8)
    # create filter
    highpass_filter = np.array([[-1, -1, -1],
                                [-1, 8, -1],
                                [-1, -1, -1]])

    # apply filter to image
    for k in range(1, row_image - 1):
        for j in range(1, column_image - 1):
            temp = image[k - 1, j - 1] * highpass_filter[0, 0] + image[k - 1, j] * highpass_filter[0, 1] + \
                   image[k - 1, j + 1] * highpass_filter[0, 2] + \
                   image[k, j - 1] * highpass_filter[1, 0] + image[k, j] * highpass_filter[1, 1] + \
                   image[k, j + 1] * highpass_filter[1, 2] + \
                   image[k + 1, j - 1] * highpass_filter[2, 0] + image[k + 1, j] * highpass_filter[2, 1] + \
                   image[k + 1, j + 1] * highpass_filter[2, 2]

            new_image[k, j] = temp
    image = new_image
    return image


# threshold is used to remove the noise from the image BHPF (order 4 with a cutoff frequency 50)
def thresh_hold_to_image(image):
    # get image size and create new image with same size
    row_image, column_image, channel_image = image.shape
    new_image = np.zeros((row_image, column_image, channel_image), np.uint8)
    # apply threshold to image
    for k in range(1, row_image - 1):
        for j in range(1, column_image - 1):
            if new_image[k, j] < 50:
                new_image[k, j] = 0
            else:
                new_image[k, j] = 255
    image = new_image
    return image


# we have moire pattern noise. take spectrum. than butterworth notch reject filter multiplied by the fourier transform of the image.
# then take inverse fourier transform of the result. this will remove the moire pattern noise.
def butterworth_notch_reject_filter(image):
    # get image size and create new image with same size
    row_image, column_image, channel_image = image.shape
    new_image = np.zeros((row_image, column_image, channel_image), np.uint8)
    # create filter
    butterworth_notch_reject = np.array([[0, 0, 0],
                                         [0, 1, 0],
                                         [0, 0, 0]])

    # apply filter to image
    for k in range(1, row_image - 1):
        for j in range(1, column_image - 1):
            temp = image[k - 1, j - 1] * butterworth_notch_reject[0, 0] + image[k - 1, j] * butterworth_notch_reject[
                0, 1] + \
                   image[k - 1, j + 1] * butterworth_notch_reject[0, 2] + \
                   image[k, j - 1] * butterworth_notch_reject[1, 0] + image[k, j] * butterworth_notch_reject[1, 1] + \
                   image[k, j + 1] * butterworth_notch_reject[1, 2] + \
                   image[k + 1, j - 1] * butterworth_notch_reject[2, 0] + image[k + 1, j] * butterworth_notch_reject[
                       2, 1] + \
                   image[k + 1, j + 1] * butterworth_notch_reject[2, 2]

            new_image[k, j] = temp
    image = new_image
    return image
