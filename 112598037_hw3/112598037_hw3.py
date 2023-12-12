import cv2
import numpy as np

# Q1: Guassian Filter ( stride = 1, kernel size = 3x3, zero padding = true )
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

def zeroPadding(img):
    image_height, image_width = img.shape
    result = np.zeros((image_height + 2, image_width + 2), np.uint8)
    result[1:-1, 1:-1] = img[:,:]
    return result


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
    

# Q2: Canny Edge Detection
def getCanny(img, isBlured=False):
    if isBlured == False:
        guassian_kernel = np.array([[0.0585, 0.0965, 0.0585],
                            [0.0965, 0.1591, 0.0965],
                            [0.0585, 0.0965, 0.0585]])
        img = convolution(img, guassian_kernel)

    (magnitude, angle) = gradientCalculation(img)

    nms_img = nonMaximumSuppression(magnitude, angle)

    (thres_img, weak, strong) = doubleThreshold(nms_img)

    result = edgeTrackingByHysteresis(thres_img, weak, strong)

    return result

def gradientCalculation(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    img_x = convolution(img, Kx)
    img_y = convolution(img, Ky)

    magnitude = np.hypot(img_x, img_y)
    magnitude = magnitude / magnitude.max() * 255
    angle = np.arctan2(img_y, img_x)

    return (magnitude, angle)

def nonMaximumSuppression(magnitude, angle):
    height, width = magnitude.shape
    result = np.zeros((height,width), dtype=np.int32)
    angle = angle * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, height-1):
        for j in range(1, width-1):
            try:
                q = 255
                r = 255
                
                #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = magnitude[i, j+1]
                    r = magnitude[i, j-1]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = magnitude[i+1, j-1]
                    r = magnitude[i-1, j+1]
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = magnitude[i+1, j]
                    r = magnitude[i-1, j]
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = magnitude[i-1, j-1]
                    r = magnitude[i+1, j+1]

                if (magnitude[i,j] >= q) and (magnitude[i,j] >= r):
                    result[i,j] = magnitude[i,j]
                else:
                    result[i,j] = 0

            except IndexError as e:
                pass

    return result

def doubleThreshold(img, low=10, high=25):
    strong = 255
    weak = 50
    height, width = img.shape
    result = np.zeros((height,width), dtype=np.int32)

    for i in range(height):
        for j in range(width):
            if img[i,j] > high:
                result[i,j] = strong
            elif high > img[i,j] and img[i,j] > low:
                result[i,j] = weak
    return (result, weak, strong)

def edgeTrackingByHysteresis(img, weak=50, strong=255):
    height, width = img.shape
    result = np.zeros((height,width), dtype=np.int32)

    for i in range(height):
        for j in range(width):
            if img[i,j] == weak:
                if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                    result[i, j] = strong
                else:
                    result[i, j] = 0
            elif img[i,j] == strong:
                result[i,j] = img[i,j]

    return result


def houghTransform(edges, origin_img, threshold = 100):
    theta_res = 1
    rho_res = 1

    theta_vals = np.deg2rad(np.arange(-90, 90, theta_res))
    width, height = edges.shape
    diag_len = int(np.ceil(np.sqrt(width * width + height * height)))
    rho_vals = np.arange(-diag_len, diag_len, rho_res)

    hough_space = np.zeros((len(rho_vals), len(theta_vals)), dtype=np.uint64)

    edge_points = np.argwhere(edges != 0)
    for y, x in edge_points:
        for t_idx, theta in enumerate(theta_vals):
            rho = int(x * np.cos(theta) + y * np.sin(theta))
            rho_idx = np.argmin(np.abs(rho_vals - rho))
            hough_space[rho_idx, t_idx] += 1

    rho_idxs, theta_idxs = np.where(hough_space > threshold)

    for rho_idx, theta_idx in zip(rho_idxs, theta_idxs):
        rho = rho_vals[rho_idx]
        theta = theta_vals[theta_idx]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(origin_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return origin_img


    

if __name__ == "__main__":
    img1 = cv2.imread("./test_img/img1.png", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("./test_img/img2.png", cv2.IMREAD_GRAYSCALE)
    img3 = cv2.imread("./test_img/img3.png", cv2.IMREAD_GRAYSCALE)

    guassian_kernel = np.array([[0.0585, 0.0965, 0.0585],
                                [0.0965, 0.1591, 0.0965],
                                [0.0585, 0.0965, 0.0585]])
    
    g_img1 = convolution(img1, guassian_kernel)
    g_img2 = convolution(img2, guassian_kernel)
    g_img3 = convolution(img3, guassian_kernel)

    cv2.imwrite('./result_img/img1_q1.png', g_img1)
    cv2.imwrite('./result_img/img2_q1.png', g_img2)
    cv2.imwrite('./result_img/img3_q1.png', g_img3)

    c_img1 = getCanny(g_img1, True)
    c_img2 = getCanny(g_img2, True)
    c_img3 = getCanny(g_img3, True)
    cv2.imwrite('./result_img/img1_q2.png', c_img1)
    cv2.imwrite('./result_img/img2_q2.png', c_img2)
    cv2.imwrite('./result_img/img3_q2.png', c_img3)


    h_img1 = houghTransform(c_img1, img1, 120)
    h_img2 = houghTransform(c_img2, img2, 120)
    h_img3 = houghTransform(c_img3, img3, 120)

    cv2.imwrite('./result_img/img1_q3.png', h_img1)
    cv2.imwrite('./result_img/img2_q3.png', h_img2)
    cv2.imwrite('./result_img/img3_q3.png', h_img3)