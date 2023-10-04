import cv2
import numpy as np

taipei101_img = cv2.imread("./test_img/taipei101.png", cv2.IMREAD_COLOR)
plane_img = cv2.imread("./test_img/aeroplane.png", cv2.IMREAD_COLOR)

def rgbToGray(img):
    height, width, _ = img.shape
    gray = np.zeros((height, width, 1), np.uint8)
    for i in range(height):
        for j in range(width):
            r, g, b = img[i, j]
            gray_value = int(0.2989 * r + 0.5870 * g + 0.1140 * b)
            gray[i, j] = gray_value

    return gray

def edgeDetect(img):
    kernel = np.array([[-1, -1, -1],
                        [-1, 8, -1],
                        [-1, -1, -1]])
    
    image_height, image_width, _ = img.shape
    kernel_height, kernel_width = kernel.shape

g_taipei101_img = rgbToGray(taipei101_img)
g_plane_img = rgbToGray(plane_img)

cv2.imwrite('./result_img/taipei101_Q1.png', g_taipei101_img)
cv2.imwrite('./result_img/aeroplane_Q1.png', g_plane_img)