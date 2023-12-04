import cv2
import numpy as np

# 영상 전처리 (허프 변환을 적용하기 전에 필요한 전처리 수행)
def preprocessing_for_hough(frame, blurVal, darkening_factor):
    # 이미지의 어두운 부분을 조금 낮추되, 밝은 부분은 그대로 둔다.
    def adjust_brightness(image, darkening_factor):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        equ = cv2.equalizeHist(gray)
        adjusted_image = cv2.addWeighted(equ, 1 - darkening_factor, gray, darkening_factor, 0)
        return cv2.cvtColor(adjusted_image, cv2.COLOR_GRAY2BGR)

    # 전처리
    ad_frame = adjust_brightness(frame, darkening_factor)
    gray = cv2.cvtColor(ad_frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blurVal, blurVal), 0)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)
    return gray, blur, edges
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
def draw_lines(frame, line_arr, B, G, R):
    if line_arr.shape[0] > 0:
        for line in line_arr:
            line = np.squeeze(line)  # 만약 shape이 (1, 1, 4)인 경우 squeeze 사용
            cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (B, G, R), 2)

# 기울기에 따른 직선 필터링
def lines_filtered(d_slope_degree, d_line_arr=None, d_lines=None):
    # 수평 기울기 제한
    mask_horizontal = np.abs(d_slope_degree) > 20
    d_line_arr = apply_mask(d_line_arr, mask_horizontal)
    d_lines = apply_mask(d_lines, mask_horizontal)
    d_slope_degree = d_slope_degree[mask_horizontal]

    # 수직 기울기 제한
    mask_vertical = np.abs(d_slope_degree) < 75
    d_line_arr = apply_mask(d_line_arr, mask_vertical)
    d_lines = apply_mask(d_lines, mask_vertical)
    d_slope_degree = d_slope_degree[mask_vertical]

    # 필터링된 직선 버리기
    L_line_arr, R_line_arr = apply_mask(d_line_arr, d_slope_degree > 0), apply_mask(d_line_arr, d_slope_degree < 0)
    L_lines, R_lines = apply_mask(d_lines, d_slope_degree > 0), apply_mask(d_lines, d_slope_degree < 0)

    # NaN 값이 있는지 확인 후 정수로 변환
    L_mean_line, R_mean_line = nan_mean_and_convert(L_line_arr), nan_mean_and_convert(R_line_arr)
    L_mean_line_r_th, R_mean_line_r_th = nan_mean_and_convert(L_lines), nan_mean_and_convert(R_lines)

    # L_line_arr와 R_line_arr를 합치기
    all_lines = concatenate_arrays(L_line_arr, R_line_arr)
    mean_line = concatenate_arrays(L_mean_line, R_mean_line)
    mean_line_r_th = concatenate_arrays(L_mean_line_r_th, R_mean_line_r_th)

    return L_line_arr, R_line_arr, all_lines, L_mean_line, R_mean_line, mean_line, L_mean_line_r_th, R_mean_line_r_th, mean_line_r_th
# 배열이 주어지 않았거나 None이면 그대로 반환
def apply_mask(data, mask):
    return data[mask] if data is not None else None
# 배열에 대한 평균 계산, NaN이 있는지 확인, 정수 배열로 변환
def nan_mean_and_convert(data):
    if data is not None and data.size > 0:
        mean_line = np.nanmean(data, axis=0).astype(int)
        return np.zeros((1, 1, 4), dtype=int) if np.isnan(mean_line).any() else mean_line
    return None
# 두 배열 연결. 어느 하나가 None이면 다른 배열 반환, 둘다 None이면 None 반환
def concatenate_arrays(arr1, arr2):
    if arr1 is not None and arr2 is not None:
        return np.concatenate((arr1, arr2), axis=0)
    elif arr1 is not None:
        return arr1
    elif arr2 is not None:
        return arr2
    else:
        return None


def main():
    # 비디오 캡처 객체 초기화
    frame = cv2.imread('img_test2.jpg')
    n_frame = cv2.resize(frame, (416, 416))
    
    # 허프 변환으로 선 감지
    gray, blur, edges = preprocessing_for_hough(n_frame, 7, 0.1)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=110)
    line_arr = np.squeeze(create_hough_lines(lines))
    
    # 선 솎아내기
    x1, y1, x2, y2 = line_arr[:, 0], line_arr[:, 1], line_arr[:, 2], line_arr[:, 3]
    slope_degree = (np.arctan2(y2 - y1, x2 - x1) * 180) / np.pi
    _, _, all_lines, _, _, mean_line = lines_filtered(slope_degree, line_arr)

    # print(all_lines)
    # print(mean_line)

    # 선 그리기
    draw_lines(n_frame, mean_line, 0, 0, 255)
    cv2.imshow("Hough transform test", n_frame)
    # for img in [blur, edges]:
    #     cv2.imshow(str(img), img)
    cv2.waitKey()
    # 비디오 캡처 객체 해제
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()