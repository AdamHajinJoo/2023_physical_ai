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

def find_intersection(a1, b1, c1, d1, a2, b2, c2, d2):
    # roi 잡아놓은 것 때문에 평행하거나 뭐 요소가 없거나 등의 예외가 발생할 수가 없음
    
    m1 = (d1-b1)/(c1-a1)
    m2 = (d2-b2)/(c2-a2)
    
    cx = ( (m1 * a1) - b1 - (m2 * a2) + b2 )/(m1-m2)
    cy = m2 * (cx - a2) + b2

    return int(cx), int(cy)

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
