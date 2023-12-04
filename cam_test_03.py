import cv2
import numpy as np
import hough_line_test as hough

# test3: 화면상 좌표 없이 직선의 기울기와 소실점 구하기.

def find_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
    # 첫 번째 직선의 기울기와 y 절편 계산
    m1 = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else None
    b1 = y1 - m1 * x1 if m1 is not None else x1
    
    # 두 번째 직선의 기울기와 y 절편 계산
    m2 = (y4 - y3) / (x4 - x3) if (x4 - x3) != 0 else None
    b2 = y3 - m2 * x3 if m2 is not None else x3
    
    # 두 직선이 평행인 경우
    if m1 == m2:
        return None
    
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
    
    return x_intersect, y_intersect

def gen_slope_degree(d_lines):
    slope = []
    if d_lines is not None and len(d_lines) > 0:
        for line in d_lines:
            rho, theta = line[0]
            slope.append(np.arctan2(np.sin(theta), np.cos(theta)) * 180 / np.pi - 90)
    return np.array(slope)

# 기울기에 따른 lines 필터링
def lines_filtered_lines(d_slope_degree, d_lines):
    # 수평 기울기 제한
    d_line_arr = d_line_arr[np.abs(d_slope_degree)>20]
    d_slope_degree = d_slope_degree[np.abs(d_slope_degree)>20]
    # 수직 기울기 제한
    d_line_arr = d_line_arr[np.abs(d_slope_degree)<75]
    d_slope_degree = d_slope_degree[np.abs(d_slope_degree)<75]
    # 필터링된 직선 버리기
    L_lines, R_lines = d_line_arr[(d_slope_degree>0),:], d_line_arr[(d_slope_degree<0),:]
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
    all_lines = np.concatenate((L_lines, R_lines), axis=0)
    mean_line = np.concatenate((L_mean_line, R_mean_line), axis=0)
    
    return L_lines, R_lines, all_lines, L_mean_line, R_mean_line, mean_line

roi_y = 213

x_vanish = 0
y_vanish = 0

def main():
    # 카메라 초기화
    camera_index = 1 #0
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Could not open webcam")
        exit()
    
    while True:
        ret, frame = cap.read()
        
        if ret:
            n_frame = cv2.resize(frame, (416, 416))
            roi_fr = n_frame[roi_y : ,  : ]       # [y:y+h, x:x+w]
            
            # 허프 변환으로 선 감지
            _, blur, edges = hough.preprocessing_for_hough(roi_fr, 5, 0.2)
            lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=110)
            line_arr = np.squeeze(hough.create_hough_lines(lines))
            
            # 선 솎아내기
            if line_arr.size == 0:
                pass
                # print("None")
            elif line_arr.ndim == 1:
                pass
                # print(1)
                # x1, y1, x2, y2 = line_arr[0], line_arr[1]+roi_y, line_arr[2], line_arr[3]+roi_y
            else:
                # print(line_arr.size)
                slope_degree = gen_slope_degree(lines)
                # _, _, all_lines, _, _, mean_line = hough.lines_filtered(slope_degree, line_arr)

                # 선 그리기
                # all_lines[:, :, [1, 3]] += roi_y
                # mean_line[:, :, [1, 3]] += roi_y

                # 소실점 생성
                # x_vanish, y_vanish = find_intersection(
                #     mean_line[0, 0, 0], mean_line[0, 0, 1],
                #     mean_line[0, 0, 2], mean_line[0, 0, 3],
                #     mean_line[1, 0, 0], mean_line[1, 0, 1],
                #     mean_line[1, 0, 2], mean_line[1, 0, 3]
                # )

                # hough.draw_lines(n_frame, all_lines, 128, 0, 0)
                # hough.draw_lines(n_frame, mean_line, 0, 0, 255)
 
            cv2.imshow("n_frame", n_frame)
            
        if cv2.waitKey(30) & 0xFF == ord('q'):
            print(slope_degree)
            print(slope_degree_2)
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()