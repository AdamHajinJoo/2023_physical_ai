import cv2
import numpy as np

# 영상 전처리 (허프 변환을 적용하기 전에 필요한 전처리 수행)
def preprocessing_for_hough(frame, blurVal):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blurVal, blurVal), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)
    return edges
# 허프 변환 수행
def create_hough_lines(lines):
    line_arr = []
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
    return np.array(line_arr)
# frame에 직선 그리기
def draw_lines(frame, line_arr):
    if line_arr.shape[0] > 0:
        for line in line_arr:
            line = np.squeeze(line)  # 만약 shape이 (1, 1, 4)인 경우 squeeze 사용
            cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 2)
# ROI 설정
def region_of_interest_frame(frame):
    cv2.rectangle(frame, (xi, yi, xf, yf), 255, 2)
    return None
# 직선 솎아내기
def lines_filtered(d_line_arr, d_slope_degree):
    # 수평 기울기 제한
    d_line_arr = d_line_arr[np.abs(d_slope_degree)>20]
    d_slope_degree = d_slope_degree[np.abs(d_slope_degree)>20]
    # 수직 기울기 제한
    d_line_arr = d_line_arr[np.abs(d_slope_degree)<85]
    d_slope_degree = d_slope_degree[np.abs(d_slope_degree)<85]
    # 필터링된 직선 버리기
    L_lines, R_lines = d_line_arr[(d_slope_degree>0),:], d_line_arr[(d_slope_degree<0),:]
    L_lines, R_lines = L_lines[:,None], R_lines[:,None]
    # L_lines와 R_lines를 합치기
    all_lines = np.concatenate((L_lines, R_lines), axis=0)
    
    return L_lines, R_lines, all_lines


def main():
    # 비디오 캡처 객체 초기화
    frame = cv2.imread('img_test3.jpg')
    n_frame = cv2.resize(frame, (416, 416))
    
    # 허프 변환으로 선 감지
    edges = preprocessing_for_hough(n_frame)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=120)
    line_arr = np.squeeze(create_hough_lines(lines))
    
    # 선 솎아내기
    x1, y1, x2, y2 = line_arr[:, 0], line_arr[:, 1], line_arr[:, 2], line_arr[:, 3]
    slope_degree = (np.arctan2(y2 - y1, x2 - x1) * 180) / np.pi
    L_lines, R_lines, all_lines = lines_filtered(line_arr, slope_degree)

    # 선 그리기
    draw_lines(n_frame, all_lines)
    region_of_interest(n_frame)
    cv2.imshow("Hough transform test", n_frame)
    cv2.waitKey()
    # 비디오 캡처 객체 해제
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()