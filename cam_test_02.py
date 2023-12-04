import cv2
import numpy as np
import hough_line_test as hough

def main():
    # 카메라 초기화
    camera_index = 0
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Could not open webcam")
        exit()
    
    while True:
        ret, frame = cap.read()
        
        if ret:
            n_frame = cv2.resize(frame, (416, 416))
            roi_fr = n_frame[213 : ,  : ]       # [y:y+h, x:x+w]
            
            # 허프 변환으로 선 감지
            edges = hough.preprocessing_for_hough(roi_fr, 5)
            lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=120)
            line_arr = np.squeeze(hough.create_hough_lines(lines))
            
            # 선 솎아내기
            if line_arr.size == 0:
                print("None")
            elif line_arr.ndim == 1:
                print(1)
                x1, y1, x2, y2 = line_arr[0], line_arr[1]+213, line_arr[2], line_arr[3]+213
            else:
                print(line_arr.size)
                x1, y1, x2, y2 = line_arr[:, 0], line_arr[:, 1]+213, line_arr[:, 2], line_arr[:, 3]+213
                slope_degree = (np.arctan2(y2 - y1, x2 - x1) * 180) / np.pi
                L_lines, R_lines, all_lines = hough.lines_filtered(line_arr, slope_degree)

                # 선 그리기
                all_lines[:, :, 1] += 213
                all_lines[:, :, 3] += 213
                hough.draw_lines(n_frame, all_lines)
            
            cv2.imshow("Hough transform test", n_frame)
            
        if cv2.waitKey(15) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()