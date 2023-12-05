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
    if line_arr.shape[0] > 0:
        for line in line_arr:
            line = np.squeeze(line)  # If the shape is (1, 1, 4), use squeeze
            cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (B, G, R), 2)

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
    # 첫 번째 직선의 기울기와 y 절편 계산
    m1 = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else None
    b1 = y1 - m1 * x1 if m1 is not None else x1
    
    # 두 번째 직선의 기울기와 y 절편 계산
    m2 = (y4 - y3) / (x4 - x3) if (x4 - x3) != 0 else None
    b2 = y3 - m2 * x3 if m2 is not None else x3
    
    # 두 직선이 평행인 경우
    if m1 == m2:
        return 0, 0  # 수정된 부분: 기본값으로 (0, 0) 반환
    
    # 교점의 x, y 좌표 계산
    if m1 is not None and m2 is not None:
        x_intersect = (b2 - b1) / (m1 - m2)
        y_intersect = m1 * x_intersect + b1
    elif m1 is None:
        x_intersect = b1
        y_intersect = m2 * x_intersect + b2
    else:  # m2 is None
        x_intersect = b2
        y_intersect = m1 * x_intersect + b1
    
    return int(x_intersect), int(y_intersect)

def slope_filter(slopesL, linesL_arr, slopesR, linesR_arr, roi_w, roi_y, roi_h, side):
    # 수평 기울기 제한
    linesL_arr, linesR_arr = linesL_arr[np.abs(slopesL)>15], linesR_arr[np.abs(slopesR)>15]
    slopesL, slopesR = slopesL[np.abs(slopesL)>15], slopesR[np.abs(slopesR)>15]
    # 수직 기울기 제한
    linesL_arr, linesR_arr = linesL_arr[np.abs(slopesL)<75], linesR_arr[np.abs(slopesR)<75]
    slopesL, slopesR = slopesL[np.abs(slopesL)<75], slopesR[np.abs(slopesR)<75]
    # 필터링된 직선 버리기
    L_lines, R_lines = linesL_arr[(slopesL>0)], linesR_arr[(slopesR<0)]
    L_lines, R_lines = L_lines[:,None], R_lines[:,None]
    
    # NaN 값이 있는지 확인 후 정수로 변환
    if L_lines.size > 0:
        L_mean_line = np.expand_dims(np.nanmean(L_lines, axis=0), axis=0).astype(int)
    else:
        L_mean_line = np.zeros((1, 1, 4), dtype=int)
    
    if R_lines.size > 0:
        R_mean_line = np.expand_dims(np.nanmean(R_lines, axis=0), axis=0).astype(int)
    else:
        R_mean_line = np.zeros((1, 1, 4), dtype=int)

    # L_lines와 R_lines를 합치기
    L_lines[:, :, [0, 2]] += 0
    L_lines[:, :, [1, 3]] += roi_y
    R_lines[:, :, [0, 2]] += (side - roi_w)
    R_lines[:, :, [1, 3]] += roi_y

    all_lines = np.concatenate((L_lines, R_lines), axis=0)
    mean_line = np.concatenate((L_mean_line, R_mean_line), axis=0)

    # all_lines[:, :, [1, 3]] += roi_y
    # mean_line[:, :, [1, 3]] += roi_y

    # print(mean_line[0, 0, 0], mean_line[0, 0, 1],
            # mean_line[0, 0, 2], mean_line[0, 0, 3],
            # mean_line[1, 0, 0], mean_line[1, 0, 1],
            # mean_line[1, 0, 2], mean_line[1, 0, 3])

    x_van, y_van = find_intersection(
                    mean_line[0, 0, 0], mean_line[0, 0, 1],
                    mean_line[0, 0, 2], mean_line[0, 0, 3],
                    mean_line[1, 0, 0], mean_line[1, 0, 1],
                    mean_line[1, 0, 2], mean_line[1, 0, 3]
                )

    return all_lines, mean_line, x_van, y_van
