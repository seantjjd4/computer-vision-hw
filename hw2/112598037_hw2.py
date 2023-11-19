import cv2
import numpy as np
import matplotlib.pyplot as plt

# Q1: Mean Filter ( stride = 1, kernel size = 3x3, zero padding = true )
def convolution(img, kernel, stride=1):

    img_height, img_width = img.shape
    p_img = zeroPadding(img)

    p_img_height, p_img_width = p_img.shape
    kernel_height, kernel_width = kernel.shape

    h_start_index = kernel_height // 2
    h_end_index = p_img_height - (kernel_height // 2)
    w_start_index = kernel_width // 2
    w_end_index = p_img_width - (kernel_width // 2)

    result_img = np.zeros((img_height, img_width), np.uint8)

    for i in range(h_start_index, h_end_index, stride):
        for j in range(w_start_index, w_end_index, stride):
            result_img[i-h_start_index, j-h_start_index] = convolutionCalculation(img[i-h_start_index:i+h_start_index+1, j-h_start_index:j+h_start_index+1], kernel)

    return result_img

def convolutionCalculation(array1, array2):
    array_height, array_width = array1.shape
    result = 0

    for i in range(array_height):
        for j in range(array_width):
            result += int(array1[i,j] * array2[i,j] // 1)

    if result >= 255:
        return 255
    elif result <= 0:
        return 0
    else:
        return result


def zeroPadding(img):
    image_height, image_width = img.shape
    result = np.zeros((image_height + 2, image_width + 2), np.uint8)
    result[1:-1, 1:-1] = img[:,:]
    return result


# Q2: Median Filter ( stride = 1, kernel size = 3x3, zero padding = true )
def medianFilter(img, kernel_size = np.array([[0,0,0],[0,0,0],[0,0,0]]), stride=1):
    img_height, img_width = img.shape
    p_img = zeroPadding(img)

    p_img_height, p_img_width = p_img.shape
    kernel_height, kernel_width = kernel_size.shape

    h_start_index = kernel_height // 2
    h_end_index = p_img_height - (kernel_height // 2)
    w_start_index = kernel_width // 2
    w_end_index = p_img_width - (kernel_width // 2)

    result_img = np.zeros((img_height, img_width), np.uint8)

    for i in range(h_start_index, h_end_index, stride):
        for j in range(w_start_index, w_end_index, stride):
            result_img[i-h_start_index, j-h_start_index] = medianCalculation(img[i-h_start_index:i+h_start_index+1, j-h_start_index:j+h_start_index+1])

    return result_img

def medianCalculation(array):
    array_height, array_width = array.shape
    elements = []
    k = 0
    for i in range(array_height):
        for j in range(array_width):
            elements.append(array[i][j])
            k += 1

    sorted = bubbleSort(elements)
    return sorted[len(sorted) // 2]

def bubbleSort(array):
    length = len(array)
    for i in range(length):
        sorted = True
        for j in range(length - i - 1):
            if array[j] > array[j + 1]:
                array[j], array[j + 1] = array[j + 1], array[j]
                sorted = False
        if sorted:
            break
    return array

# Q3: Histogram
def getHistogramData(img):
    img_height, img_width = img.shape
    result = [0]*256
    for i in range(img_height):
        for j in range(img_width):
            result[img[i][j]] += 1
    return result

def getFlat(img):
    img_height, img_width = img.shape
    result = []
    for i in range(img_height):
        for j in range(img_width):
            result.append(img[i][j])
    return result



if __name__ == "__main__":
    noise1 = cv2.imread("./test_img/noise1.png", cv2.IMREAD_GRAYSCALE)
    noise2 = cv2.imread("./test_img/noise2.png", cv2.IMREAD_GRAYSCALE)

    mean_kernel = np.array([[1/9, 1/9, 1/9],
                    [1/9, 1/9, 1/9],
                    [1/9, 1/9, 1/9]])

    mean_noise1 = convolution(noise1, mean_kernel)
    mean_noise2 = convolution(noise2, mean_kernel)

    cv2.imwrite('./result_img/noise1_q1.png', mean_noise1)
    cv2.imwrite('./result_img/noise2_q1.png', mean_noise2)

    median_noise1 = medianFilter(noise1)
    median_noise2 = medianFilter(noise2)

    cv2.imwrite('./result_img/noise1_q2.png', median_noise1)
    cv2.imwrite('./result_img/noise2_q2.png', median_noise2)

    plt.figure()
    plt.hist(getFlat(noise1), 256, [0, 256])
    plt.savefig('./result_img/noise1_his.png')

    plt.figure()
    plt.hist(getFlat(noise2), 256, [0, 256])
    plt.savefig('./result_img/noise2_his.png')

    plt.figure()
    plt.hist(getFlat(mean_noise1), 256, [0, 256])
    plt.savefig('./result_img/noise1_q1_his.png')

    plt.figure()
    plt.hist(getFlat(mean_noise2), 256, [0, 256])
    plt.savefig('./result_img/noise2_q1_his.png')

    plt.figure()
    plt.hist(getFlat(median_noise1), 256, [0, 256])
    plt.savefig('./result_img/noise1_q2_his.png')

    plt.figure()
    plt.hist(getFlat(median_noise2), 256, [0, 256])
    plt.savefig('./result_img/noise2_q2_his.png')