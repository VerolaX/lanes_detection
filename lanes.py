import cv2
import numpy as np

# truncate out a portion of lines detected by Hough Transform
def make_coordinate(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * 3/5)
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

# smooth the detected lines by taking their averages respectively
def average_slope_intercept(image, lines):
    left_fit = []
    rigth_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        # fit a linear function to the points
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            rigth_fit.append((slope, intercept))

    # take the average of the parameters
    left_fit_average = np.average(left_fit, axis=0)
    rigth_fit_average = np.average(rigth_fit, axis=0)

    # save the truncated line coordinates
    left_line = make_coordinate(image, left_fit_average)
    right_line = make_coordinate(image, rigth_fit_average)

    return np.array([left_line, right_line])

# perform canny edge detector
def Canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    canny = cv2.Canny(gray, 50, 150)
    return canny

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            # add lines to the line image
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

# crop out the region of interest
def region_of_interest(image):
    height = image.shape[0]
    # draw a triangle
    polygons = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(image)
    # combine the mask with the triangle
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image
#
# image = cv2.imread('./test_image.jpg')
# lane_image = np.copy(image)
# canny_image = Canny(lane_image)
# cropped_image = region_of_interest(canny_image)
# lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
# averaged_lines = average_slope_intercept(lane_image, lines)
# line_image = display_lines(lane_image, averaged_lines)
# combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
#
# cv2.imshow('result', combo_image)
# cv2.waitKey(0)

cap = cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
    _, frame = cap.read() # read the frames from video
    canny_image = Canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)

    # combine the images
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    cv2.imshow('result', combo_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
