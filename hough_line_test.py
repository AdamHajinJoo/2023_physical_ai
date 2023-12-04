import cv2
import numpy as np

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

def draw_lines(frame, line_arr):
    for line in line_arr:
        cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 2)

def main():
    # 비디오 캡처 객체 초기화
    frame = cv2.imread('img_test.jpg')
    dframe = cv2.resize(frame, (416, 416))
    
    # 영상 전처리 (허프 변환을 적용하기 전에 필요한 전처리 수행)
    gray = cv2.cvtColor(dframe, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)

    # 허프 변환을 사용하여 선 감지
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=130)
    line_arr = create_hough_lines(lines)
    # 기울기 구하기
    line_arr = np.squeeze(line_arr)   # line_arr 배열을 변형
    
    # 직선의 시작점과 끝
    x1, y1, x2, y2 = line_arr[:, 0], line_arr[:, 1], line_arr[:, 2], line_arr[:, 3]
    slope_degree = (np.arctan2(y2 - y1, x2 - x1) * 180) / np.pi
    print(slope_degree)
    
    # 수평 기울기 제한
    line_arr = line_arr[np.abs(slope_degree)>20]
    slope_degree = slope_degree[np.abs(slope_degree)>20]
    # 수직 기울기 제한
    line_arr = line_arr[np.abs(slope_degree)<85]
    slope_degree = slope_degree[np.abs(slope_degree)<85]
    # 필터링된 직선 버리기
    L_lines, R_lines = line_arr[(slope_degree>0),:], line_arr[(slope_degree<0),:]
    L_lines, R_lines = L_lines[:,None], R_lines[:,None]

    # 허프 변환 선 그리기
    draw_lines(dframe, line_arr)

    # 화면에 결과 표시
    cv2.imshow("Hough Transform", dframe)
    cv2.waitKey()
    #비디오 캡처 객체 해제
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()