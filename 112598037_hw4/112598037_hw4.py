import cv2
import numpy as np

def active_contour(img_gradient, points, min_energy_table, region, alpha, beta, gamma, num_points):
    img_height, img_width = img_gradient.shape

    # calculate the continous energy (internal)
    def cal_cont_energy(x1,y1,x2,y2):
        result = (np.hypot((x1 - x2), (y1 - y2)))**2
        return result
    
    # calculate the curves energy (internal)
    def cal_curv_energy(x1,y1,x2,y2,x3,y3):
        result = (np.hypot((x1 - 2*x2 + x3), (y1 - 2*y2 + y3)))**2
        return result
    
    # calculate the image energy (external)
    def cal_image_energy(x, y):
        result = -(img_gradient[x, y])
        return result
    
    shift_index = [i - region//2 for i in range(region)]
    
    for i in range(num_points):
        j = i-1
        k = i+1
        if i == 0:
            j = num_points-1
        if i == num_points-1:
            k = 0
        e_temp = min_energy_table[i]
        new_x = points[i,0]
        new_y = points[i,1]
        for _x in shift_index:
            for _y in shift_index:
                x = points[i,0] + _x
                y = points[i,1] + _y
                if x > img_width or y > img_height or x < 0 or y < 0:
                    continue
                e_cont = cal_cont_energy(x, y, points[j,0], points[j,1])
                e_curv = cal_curv_energy(points[j,0], points[j,1], x, y, points[k,0], points[k,1])
                e_image = cal_image_energy(x, y)
                e_total = alpha*e_cont + beta*e_curv + gamma*e_image

                if e_total < e_temp:
                    e_temp = e_total
                    new_x, new_y = x, y

        points[i,0] = new_x
        points[i,1] = new_y
    return points

def guassian_blur(img):
    guassian_kernel = np.array([[0.0585, 0.0965, 0.0585],
                            [0.0965, 0.1591, 0.0965],
                            [0.0585, 0.0965, 0.0585]])
    
    result = cv2.filter2D(src=img, ddepth=-1, kernel=guassian_kernel)
    return result

def get_image_gradient(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    sobel_x = cv2.filter2D(src=img, ddepth=-1, kernel=Kx)
    sobel_y = cv2.filter2D(src=img, ddepth=-1, kernel=Ky)

    magnitude = np.hypot(sobel_x, sobel_y)
    result = magnitude / magnitude.max() * 255
    return result

def get_fixed_initial_point(img, num_points):

    img_height, img_width = img.shape

    center = [img_height/2, img_width/2]
    radius = img_height*0.3 if img_height < img_width else img_width*0.3

    theta = np.linspace(0, 2*np.pi, num_points)

    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    x = [int(i) for i in x]
    y = [int(i) for i in y]
    return np.stack((x, y), axis=1)

def get_initial_energy_table(img_gradient, points, alpha, beta, gamma, num_points):
    energy_table = np.zeros(num_points)
    for i in range(num_points):
        j = i-1
        k = i+1
        if i == 0:
            j = num_points-1
        if i == num_points-1:
            k = 0
        e_cont = ((points[i,0] - points[j,0])**2 + (points[i,1] - points[j,1])**2)
        
        e_curv = ((points[j,0] - 2*points[i,0] + points[k,0])**2 + (points[j,1] - 2*points[i,1] + points[k,1])**2)
        e_image = -(img_gradient[points[i,0], points[i,1]])
        energy_table[i] = alpha*e_cont + beta*e_curv + gamma*e_image
    return energy_table

def draw_points(img, points, num_points):
    new_image = img.copy()
    for i in range(num_points):
        j = i+1
        if i == num_points-1:
            j = 0
        new_image = cv2.circle(new_image, (int(points[i,0]),int(points[i,1])), radius=5, color=(0,0,255), thickness=-1)
        new_image = cv2.line(new_image, (int(points[i][0]), int(points[i][1])), (int(points[j][0]), int(points[j][1])), color=(255, 0, 0), thickness=2)

    return new_image


def find_contour(img, alpha=0.1, beta=0.4, gamma=0.5, max_iteration=2000, region=7, num_points=50, video_name=""):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g_img = guassian_blur(gray)
    img_gradient = get_image_gradient(g_img)

    points = get_fixed_initial_point(g_img, num_points)

    # draw initial circle
    for i in range(num_points):
        j = i+1
        if i == num_points-1:
            j = 0
        img = cv2.circle(img, (int(points[i,0]),int(points[i,1])), radius=5, color=(0,0,0), thickness=-1)
        img = cv2.line(img, (int(points[i][0]), int(points[i][1])), (int(points[j][0]), int(points[j][1])), color=(0, 0, 0), thickness=2)

    # record video
    if (video_name):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('./result_img/'+video_name+'.avi', fourcc, 5.0, (1000, 1000))
        out.write(img)

    min_energy_table = get_initial_energy_table(img_gradient, points, alpha, beta, gamma, num_points)

    for i in range(0, max_iteration):
        pre_points = points.copy()
        points = active_contour(img_gradient, points, min_energy_table, region, alpha, beta, gamma, num_points)
        if (video_name):
            frame = draw_points(img, points, num_points)
            out.write(frame)
        if (pre_points == points).all():
            break
    
    img = draw_points(img, points, num_points)
    if (video_name):
        out.write(frame)
        out.release()

    return img
    





if __name__ == "__main__":
    img1 = cv2.imread("./test_img/img1.jpg")
    img2 = cv2.imread("./test_img/img2.jpg")
    img3 = cv2.imread("./test_img/img3.jpg")

    result1 = find_contour(img1, video_name="vid_img1")
    result2 = find_contour(img2, video_name="vid_img2")
    result3 = find_contour(img3, video_name="vid_img3")

    cv2.imwrite('./result_img/result_img1.png', result1)
    cv2.imwrite('./result_img/result_img2.png', result2)
    cv2.imwrite('./result_img/result_img3.png', result3)