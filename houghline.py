import cv2
import numpy as np

def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i ][: , 1:]
        XYs = []
        for j in np.unique(npXYs[: , 0]):
            XY = npXYs[npXYs[: , 0] == j ][: , 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs

# Load and preprocess an Image
default_file = 'problems/isolated.csv'

# Load the polyline data from the .csv file
polyline_data = read_csv(default_file)

# Create an empty image
image = np.zeros((263, 263, 3), dtype=np.uint8)

# Draw the polylines onto the image
for path in polyline_data:
    for polyline in path:
        polyline = np.int32(polyline)
        cv2.polylines(image, [polyline], isClosed=False, color=(255,255,255))

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Perform edge detection
edges = cv2.Canny(blur, 50, 150)

# Perform Hough Line Transform
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)

# Draw the lines
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Find contours
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Classify shapes
for contour in contours:
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    vertices = len(approx)
    
    if vertices == 3:
        shape = "Triangle"
    elif vertices == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        shape = "Square" if ar >= 0.95 and ar <= 1.05 else "Rectangle"
    elif vertices == 5:
        shape = "Pentagon"
    elif vertices < 10:
        shape = "Star"
    else:
        area = cv2.contourArea(contour)
        if area > 1000:
            shape = "Circle"
        else:
            shape = "Polygon"

    cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)
    cv2.putText(image, shape, (approx[0][0][0], approx[0][0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Save Results
cv2.imwrite('solutions/isolated_sol.png', image)
