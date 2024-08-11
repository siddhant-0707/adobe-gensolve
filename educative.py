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
# image = cv2.imread('problems/isolated.png')
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
edges = cv2.Canny(blur, 30, 150)

# Find contours
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Classifying shapes
for contour in contours:
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    vertices = len(approx)
    if vertices == 3:
        shape = "Triangle"
    elif vertices == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
    elif vertices == 5:
        shape = "Pentagon"
    else:
        shape = "Circle"
    
    cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)
    cv2.putText(image, shape, (approx[0][0][0], approx[0][0][1]+5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

# Save Results
cv2.imwrite('DetectedShapes_educative.png',image)