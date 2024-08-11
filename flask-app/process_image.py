import cv2
import numpy as np
import svgwrite
import cairosvg

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


def polylines2svg(paths_XYs, svg_path):
    W, H = 0, 0
    for path_XYs in paths_XYs:
        for XY in path_XYs:
            W, H = max(W, np.max(XY[:, 0])), max(H, np.max(XY[:, 1]))
    padding = 0.1
    W, H = int(W + padding * W), int(H + padding * H)

    # Create a new SVG drawing
    dwg = svgwrite.Drawing(svg_path, profile='tiny', shape_rendering='crispEdges')
    group = dwg.g()
    colours = ["red", "blue", "green"]

    for i, path in enumerate(paths_XYs):
        path_data = []
        c = colours[i % len(colours)]
        for XY in path:
            path_data.append(("M", (XY[0, 0], XY[0, 1])))
            for j in range(1, len(XY)):
                path_data.append(("L", (XY[j, 0], XY[j, 1])))
            if not np.allclose(XY[0], XY[-1]):
                path_data.append(("Z", None))
        group.add(dwg.path(d=path_data, fill=c, stroke='none', stroke_width=2))

    dwg.add(group)
    dwg.save()

    png_path = svg_path.replace('.svg', '.png')
    fact = max(1, 1024 / min(H, W))
    cairosvg.svg2png(url=svg_path, write_to=png_path, 
                     parent_width=W, parent_height=H, 
                     output_width=fact*W, output_height=fact*H, 
                     background_color='white')

    return


def check_symmetry(mask, tolerance=0.25):
    h, w = mask.shape
    mid_w = w // 2
    mid_h = h // 2
    
    symmetries = {
        "vertical": False,
        "horizontal": False,
        "diagonal_1": False,  # Top-left to bottom-right
        "diagonal_2": False,  # Top-right to bottom-left
    }
    
    # Vertical symmetry check
    left_half = mask[:, :mid_w]
    right_half = mask[:, mid_w:]
    if w % 2 != 0:
        right_half = right_half[:, :-1]
    flipped_right_half = cv2.flip(right_half, 1)
    diff_vertical = cv2.absdiff(left_half, flipped_right_half)
    non_zero_vertical = cv2.countNonZero(diff_vertical)
    total_pixels_vertical = h * mid_w
    if non_zero_vertical / total_pixels_vertical <= tolerance:
        symmetries["vertical"] = True
    
    # Horizontal symmetry check
    top_half = mask[:mid_h, :]
    bottom_half = mask[mid_h:, :]
    if h % 2 != 0:
        bottom_half = bottom_half[:-1, :]
    flipped_bottom_half = cv2.flip(bottom_half, 0)
    diff_horizontal = cv2.absdiff(top_half, flipped_bottom_half)
    non_zero_horizontal = cv2.countNonZero(diff_horizontal)
    total_pixels_horizontal = w * mid_h
    if non_zero_horizontal / total_pixels_horizontal <= tolerance:
        symmetries["horizontal"] = True
    
    # Diagonal symmetry check
    mask_diag1 = mask.diagonal()
    flipped_diag1 = np.flip(mask.diagonal())
    diff_diag1 = cv2.absdiff(mask_diag1, flipped_diag1)
    non_zero_diag1 = np.count_nonzero(diff_diag1)
    if non_zero_diag1 / len(mask_diag1) <= tolerance:
        symmetries["diagonal_1"] = True
    
    # Diagonal symmetry check
    mask_diag2 = np.fliplr(mask).diagonal()
    flipped_diag2 = np.flip(mask_diag2)
    diff_diag2 = cv2.absdiff(mask_diag2, flipped_diag2)
    non_zero_diag2 = np.count_nonzero(diff_diag2)
    if non_zero_diag2 / len(mask_diag2) <= tolerance:
        symmetries["diagonal_2"] = True
    
    return symmetries

def draw_symmetry_lines(image, bounding_box, symmetries):
    x, y, w, h = bounding_box
    
    # vertical symmetry line
    if symmetries["vertical"]:
        x_center = x + w // 2
        cv2.line(image, (x_center, y), (x_center, y + h), (0, 0, 255), 1)
    
    # horizontal symmetry line
    if symmetries["horizontal"]:
        y_center = y + h // 2
        cv2.line(image, (x, y_center), (x + w, y_center), (255, 0, 0), 1)
    
    # diagonal symmetry lines
    if symmetries["diagonal_1"]:
        cv2.line(image, (x, y), (x + w, y + h), (0, 255, 0), 1)  # Top-left to bottom-right
    
    if symmetries["diagonal_2"]:
        cv2.line(image, (x + w, y), (x, y + h), (255, 255, 0), 1)  # Top-right to bottom-left

def process_image(input_path, output_path):
    default_file = input_path

    polyline_data = read_csv(default_file)

    image = np.zeros((263, 263, 3), dtype=np.uint8)

    for path in polyline_data:
        for polyline in path:
            polyline = np.int32(polyline)
            cv2.polylines(image, [polyline], isClosed=False, color=(255,255,255))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        vertices = len(approx)
        
        if vertices == 3:
            shape = "Triangle"
        elif vertices == 4:
            shape = "Square" if cv2.boundingRect(approx)[2] / float(cv2.boundingRect(approx)[3]) >= 0.95 else "Rectangle"
        elif vertices == 5:
            shape = "Pentagon"
        elif vertices == 6:
            shape = "Hexagon"
        elif vertices < 10:
            shape = "Star"
        else:
            shape = "Circle" if cv2.contourArea(contour) > 1000 else "Polygon"

        bounding_box = cv2.boundingRect(approx)

        mask = np.zeros((bounding_box[3], bounding_box[2]), dtype=np.uint8)
        cv2.drawContours(mask, [contour - [bounding_box[0], bounding_box[1]]], -1, 255, -1)

        symmetries = check_symmetry(mask)
        draw_symmetry_lines(image, bounding_box, symmetries)

        symmetry_status = ", ".join([k for k, v in symmetries.items() if v]) or "Not Symmetric"
        cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)
        cv2.putText(image, f"{shape} ({symmetry_status})", (bounding_box[0], bounding_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imwrite(output_path, image)
