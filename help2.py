import cv2
import numpy as np


def crop_black_borders(img):
    img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)
    return img[y:y + h, x:x + w]


def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    cv2.imshow("White balans", result)
    return result


def draw_boxes(img, mask):
    canny_output = cv2.Canny(mask, 50, 100)

    contours, h = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)
    centers = [None] * len(contours)
    radius = [None] * len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
    cv2.imshow("canny", canny_output)
    drawing = img.copy()

    for i in range(len(contours)):
        color = (0, 0, 255)
        cv2.rectangle(drawing,
                      (int(boundRect[i][0] - 0.5 * boundRect[i][2]), int(boundRect[i][1] - boundRect[i][3] * 0.5)), \
                      (int(boundRect[i][0] + boundRect[i][2] * 1.5), int(boundRect[i][1] + boundRect[i][3] * 1.5)),
                      color, 2)

    return drawing


image = cv2.imread('./gg/g2.jpg')
image = crop_black_borders(image)

img_balanced = white_balance(image)
img_balanced = cv2.blur(img_balanced, (30, 30))
cv2.imshow("blur", img_balanced)

hsv_img = cv2.cvtColor(img_balanced, cv2.COLOR_BGR2HSV).astype(np.float64)

min, max = ([35.0, 40.0, 31.0], [60.0, 163.0, 54.0])

green_low = np.array(min)
green_high = np.array(max)
curr_mask = cv2.inRange(hsv_img, green_low, green_high)
cv2.imshow("curr", curr_mask)
result_with_boxes = draw_boxes(image, curr_mask)
cv2.imshow("infection", result_with_boxes)
cv2.waitKey(0)

img=cv2.imread("./1.jpg",cv2.IMREAD_ANYCOLOR)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(img_hsv)
min = [
np.quantile(h, 0.1),
np.quantile(s, 0.1),
np.quantile(v, 0.1)
]
max = [
np.quantile(h, 0.9),
np.quantile(s, 0.9),
np.quantile(v, 0.9)
]
print(max, min)