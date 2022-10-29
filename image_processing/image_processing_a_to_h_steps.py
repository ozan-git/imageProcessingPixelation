import cv2
import numpy as np


def sobel_mask(final_image_array):
    mask = cv2.blur(final_image_array, (5, 5))
    return mask


# b) Laplacian of original image
def laplace(image):
    laplacian_filter = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]

    # setting an array element with a sequence.
    [rows, columns] = np.shape(image)

    filtered_image = np.zeros((rows, columns))
    for i in range(rows - 2):
        for j in range(columns - 2):

            dot_product = np.dot(laplacian_filter, image[i:i + 3, j:j + 3])
            result_of_sum = sum(map(sum, dot_product))
            filtered_image[i + 1][j + 1] = result_of_sum
    filtered_image = cv2.Laplacian(image, cv2.CV_64F)
    return filtered_image


def sobels(image):
    kernel_x = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
    kernel_y = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
    [rows, columns] = np.shape(image)
    sobel_filters = np.zeros(shape=(rows, columns))
    for i in range(rows - 2):
        for j in range(columns - 2):
            gx = np.sum(np.multiply(kernel_x, image[i:i + 3, j:j + 3]))
            gy = np.sum(np.multiply(kernel_y, image[i:i + 3, j:j + 3]))
            sobel_filters[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2)
    sobel_filters = np.require(sobel_filters, np.uint8, 'C')
    return sobel_filters


def sobel_filter(img):
    final_image_array = sobels(img)
    return final_image_array


def apply_sobel_mask(image):
    # average filter 5 by 5
    mask = cv2.blur(image, (5, 5))
    return mask


# sobel_image_smoothed  sobel image smoothed with a 5x5 averaging filter
def sobel_image_smoothed(image):
    final_image_array = sobels(image)
    final_image_array = apply_sobel_mask(final_image_array)
    return final_image_array


# d) Sobel gradient of original image
def sobel_gradient_filter(img):
    final_image_array = sobels(img)
    # final_image_array = self.apply_sobel_mask(final_image_array)
    return final_image_array


def sobel_gradient_smoothed_filter(img):
    final_image_array = apply_sobel_mask(img)
    return final_image_array


# b) Laplacian of original image
def laplacian_filter(image):
    laplacian_array = laplace(image)
    laplacian_array = (((laplacian_array - laplacian_array.min()) / (
            laplacian_array.max() - laplacian_array.min())) * 255.9).astype(np.uint8)
    return laplacian_array


# c) Sharpened image obtained by adding original image and laplacian of original image
def sharpened_filter(image):
    laplacian_array = laplace(image)
    laplacian_array = np.uint8(np.absolute(laplacian_array))
    sharpen_filter = cv2.add(image, laplacian_array)
    return sharpen_filter

# f) Mask image formed by the product of c) sharpened image and e) sobel image smoothed.
# def apply_mask_filter_sharpened_product_with_smoothed(image):
#     final_image_array = sobels(image)
#     blur_filter = apply_sobel_mask(final_image_array)
#     laplacian_array = laplace(image)
#     laplacian_array = np.uint8(np.absolute(laplacian_array))
#     shape_filter = cv2.add(image, laplacian_array)
#     masked_image = cv2.bitwise_and(blur_filter, shape_filter)
#     return masked_image

# f) Mask image formed by the product of c) sharpened image and e) sobel image smoothed.
def apply_filter_sharpened_product_with_smoothed(image):
    sharpened_image = sharpened_filter(image)
    smoothed_sobel_image = sobel_image_smoothed(image)

    # product of sharpened image and smoothed sobel image
    product_mask_image = cv2.bitwise_and(sharpened_image, smoothed_sobel_image)
    return product_mask_image


# g) Sharpened image obtained by the sum of (a) and (f).
def apply_mask_filter_sharpened_sum_with_smoothed(image):
    final_image_array = sobels(image)
    blur_filter = apply_sobel_mask(final_image_array)
    laplacian_array = laplace(image)
    laplacian_array = np.uint8(np.absolute(laplacian_array))
    shape_filter = cv2.add(image, laplacian_array)
    masked_image = cv2.bitwise_and(blur_filter, shape_filter)
    g_filters = cv2.add(image, masked_image)
    return g_filters


# def apply_mask_filter_sharpened_sum_with_smoothed(image):
#     laplacian_array = laplace(image)
#     laplacian_array = np.uint8(np.absolute(laplacian_array))
#     sharpened_filter = cv2.add(image, laplacian_array)
#     blur_filter = apply_sobel_mask(sharpened_filter)
#     return blur_filter


def power_law_transformation(image):
    final_image_array = sobels(image)
    blur_filter = apply_sobel_mask(final_image_array)

    laplacian_array = laplace(image)
    laplacian_array = np.uint8(np.absolute(laplacian_array))
    shape_filter = cv2.add(image, laplacian_array)

    masked_image = cv2.bitwise_and(blur_filter, shape_filter)
    g_filters = cv2.add(image, masked_image)
    power_image = np.array(255 * (g_filters / 255) ** 0.5, dtype='uint8')
    return power_image
