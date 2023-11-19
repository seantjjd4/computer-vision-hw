import cv2
import numpy as np

taipei101_img = cv2.imread("./test_img/taipei101.png", cv2.IMREAD_COLOR)
plane_img = cv2.imread("./test_img/aeroplane.png", cv2.IMREAD_COLOR)

# Q1: turn rgb image to grayscale image
def rgbToGray(img):
    height, width, _ = img.shape
    gray = np.zeros((height, width, 1), np.uint8)
    for i in range(height):
        for j in range(width):
            r, g, b = img[i, j]
            gray_value = int(0.2989 * r + 0.5870 * g + 0.1140 * b)
            gray[i, j] = gray_value

    return gray

# Q2: doing convolution
# assume the kernel shape is a square and the length is odd.
def convolution(img, kernel, stride=1):
    pad_img = zeroPadding(img)
    img_height, img_width, _ = pad_img.shape
    kernel_height, kernel_width = kernel.shape

    h_start_index = kernel_height // 2
    h_end_index = img_height - (kernel_height // 2)
    w_start_index = kernel_width // 2
    w_end_index = img_width - (kernel_width // 2)

    result_img = np.zeros((img_height, img_width, 1), np.uint8)

    for i in range(h_start_index, h_end_index, stride):
        for j in range(w_start_index, w_end_index, stride):
            result_img[i, j] = arrayMutipleAndAddAll(img[i-1:i+2, j-1:j+2, 0], kernel)

    return result_img

# function for Q2, doing image zero padding
def zeroPadding(img):
    image_height, image_width, _ = img.shape
    result = np.zeros((image_height + 2, image_width + 2, 1), np.uint8)
    result[1:-1, 1:-1] = img[:,:]
    return result

# function for Q2, doing the calculation of convolution
# assume the arrays shape are the same.
def arrayMutipleAndAddAll(array1, array2):
    array_height, array_width = array1.shape
    result = 0

    for i in range(array_height):
        for j in range(array_width):
            result += int(array1[i,j] * array2[i,j])
    if result >= 255:
        return 255
    elif result <= 0:
        return 0
    else:
        return result

# Q3: max pooling with given matrix size and stride
def maxPooling(img, size=(2,2), stride=2):
    img_height, img_width, _ = img.shape
    result_img = np.zeros((img_height // size[0], img_width // size[1], 1), np.uint8)

    x = 0
    for i in range(0, img_height-1, stride):
        y = 0
        for j in range(0, img_width-1, stride):
            try:
                result_img[x, y, 0] = chooseMaxFromArray(img[i:i+size[0], j:j+size[1], 0])
            except:
                print(x, y)
                exit(0)
            y += 1
        x += 1

    return result_img

# fucntion of Q3: picking the max value
def chooseMaxFromArray(array):
    height, width = array.shape
    max = 0
    for i in range(height):
        for j in range(width):
            if max < array[i,j]:
                max = array[i,j]

    return max

# Q4: binarization with customize threshold
# assume the image is in grayscale
def binarization(img, threshold=128):
    img_height, img_width, _ = img.shape
    result_img = np.zeros((img_height, img_width, 1), np.uint8)
    for i in range(img_height):
        for j in range(img_width):
            if img[i,j,0] >= threshold:
                result_img[i,j,0] = 255
            else:
                result_img[i,j,0] = 0

    return result_img

if __name__ == "__main__":
    g_taipei101_img = rgbToGray(taipei101_img)
    g_plane_img = rgbToGray(plane_img)

    cv2.imwrite('./result_img/taipei101_Q1.png', g_taipei101_img)
    cv2.imwrite('./result_img/aeroplane_Q1.png', g_plane_img)

    kernel = np.array([[-1, -1, -1],
                        [-1, 8, -1],
                        [-1, -1, -1]])

    g_edge_taipei101_img = convolution(g_taipei101_img, kernel, stride=1)
    g_edge_plane_img = convolution(g_plane_img, kernel, stride=1)

    cv2.imwrite('./result_img/taipei101_Q2.png', g_edge_taipei101_img)
    cv2.imwrite('./result_img/aeroplane_Q2.png', g_edge_plane_img)

    pooling_size = (2,2)

    g_pool_taipei101_img = maxPooling(g_taipei101_img, pooling_size, stride=2)
    g_pool_plane_img = maxPooling(g_plane_img, pooling_size, stride=2)

    cv2.imwrite('./result_img/taipei101_Q3.png', g_pool_taipei101_img)
    cv2.imwrite('./result_img/aeroplane_Q3.png', g_pool_plane_img)

    threshold = 128

    g_binary_taipei101_img = binarization(g_taipei101_img, threshold)
    g_binary_plane_img = binarization(g_plane_img, threshold)

    cv2.imwrite('./result_img/taipei101_Q4.png', g_binary_taipei101_img)
    cv2.imwrite('./result_img/aeroplane_Q4.png', g_binary_plane_img)