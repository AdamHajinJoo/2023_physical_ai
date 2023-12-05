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

def calculate_vanishing_point(lines):
    rho1, theta1 = lines[0][0]
    rho2, theta2 = lines[1][0]

    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    
    b = np.array([rho1, rho2])
    
    try:
        x, y = np.linalg.solve(A, b)
        return int(x), int(y)
    except np.linalg.LinAlgError:
        return None

def gen_slope_degree(d_lines):
    slope = []
    if d_lines is not None and len(d_lines) > 0:
        for line in d_lines:
            rho, theta = line[0]
            slope.append(np.arctan2(np.sin(theta), np.cos(theta)) * 180 / np.pi - 90)
    return np.array(slope)

roi_y = 213

x_van = 0
y_van = 0

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
            _, blur, edges = hough.preprocessing_for_hough(roi_fr, 5, 0.1)
            lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=110)
            line_arr = np.squeeze(hough.create_hough_lines(lines))
            
            # 선 솎아내기
            if line_arr.size == 0:
                pass
            elif line_arr.ndim == 1:
                pass
            else:
                # print(line_arr.size)
                x1, y1, x2, y2 = line_arr[:, 0], line_arr[:, 1]+roi_y, line_arr[:, 2], line_arr[:, 3]+roi_y
                slope_degree = gen_slope_degree(lines)

                _, _, all_lines, _, _, mean_line = hough.lines_filtered(slope_degree, line_arr)
                all_lines[:, :, [1, 3]] += roi_y
                mean_line[:, :, [1, 3]] += roi_y

                # 소실점 생성
                x_van, y_van = find_intersection(
                    mean_line[0, 0, 0], mean_line[0, 0, 1],
                    mean_line[0, 0, 2], mean_line[0, 0, 3],
                    mean_line[1, 0, 0], mean_line[1, 0, 1],
                    mean_line[1, 0, 2], mean_line[1, 0, 3]
                )
                
                hough.draw_lines(n_frame, all_lines, 128, 0, 0)
                hough.draw_lines(n_frame, mean_line, 0, 0, 255)
                cv2.circle(n_frame, (x_van, y_van), 5, (0, 255, 0), 2)
 
            cv2.imshow("n_frame", n_frame)
            
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()