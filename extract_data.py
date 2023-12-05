import cv2
import numpy as np

def draw_lines(frame, line_arr, B, G, R):
    """
    Draw lines on an image.

    Args:
        frame (numpy.ndarray): Input image.
        line_arr (numpy.ndarray): Array containing lines.
        B (int): Blue color intensity.
        G (int): Green color intensity.
        R (int): Red color intensity.
    """
    if line_arr.size > 4:
        for line in line_arr:
            line = np.squeeze(line)  # If the shape is (1, 1, 4), use squeeze
            cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (B, G, R), 2)
    else:
        cv2.line(frame, (line_arr[0], line_arr[1]), (line_arr[2], line_arr[3]), (B, G, R), 2)

def hough_line_raw(edges, val):
    """
    Apply Hough line transform to detect lines in an image.

    Args:
        edges (numpy.ndarray): Image with edges detected.
        val (int): Threshold value for line detection.

    Returns:
        numpy.ndarray: Detected lines.
    """
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=val)
    return lines

def line_arr_slope_degrees(lines):
    line_arr = []
    slope = []
    if lines is not None and len(lines) > 0:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            line_arr.append([x1, y1, x2, y2])
            slope.append(np.arctan2(np.sin(theta), np.cos(theta)) * 180 / np.pi - 90)
    return np.array(line_arr), np.array(slope)

def find_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
    m1, b1 = (None, x1) if x1 == x2 else ((y2 - y1) / (x2 - x1), y1 - ((y2 - y1) / (x2 - x1)) * x1)
    m2, b2 = (None, x3) if x3 == x4 else ((y4 - y3) / (x4 - x3), y3 - ((y4 - y3) / (x4 - x3)) * x3)

    if m1 == m2:
        return None

    x, y = (b1, m2 * b1 + b2) if m1 is None else (b2, m1 * b2 + b1) if m2 is None else ((b2 - b1) / (m1 - m2), m1 * ((b2 - b1) / (m1 - m2)) + b1)
    
    return int(x), int(y)


def slope_filter(slopesL, linesL_arr, slopesR, linesR_arr):
    # 수평 기울기 제한
    linesL_arr = linesL_arr[np.abs(slopesL)>15]
    slopesL = slopesL[np.abs(slopesL)>15]
    linesR_arr = linesR_arr[np.abs(slopesR)>15]
    slopesR = slopesR[np.abs(slopesR)>15]
    # 수직 기울기 제한
    linesL_arr, linesR_arr = linesL_arr[np.abs(slopesL)<75], linesR_arr[np.abs(slopesR)<75]
    slopesL, slopesR = slopesL[np.abs(slopesL)<75], slopesR[np.abs(slopesR)<75]
    
    # 왜인지 모르겠으나 각도 기준이 거꾸로임
    linesL, linesR = linesL_arr[slopesL < 0], linesR_arr[slopesR > 0]
    slopesL, slopesR = slopesL[slopesL < 0], slopesR[slopesR > 0]
    
    return linesL, linesR, slopesL, slopesR
