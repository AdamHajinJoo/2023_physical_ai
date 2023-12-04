import cv2
import numpy as np
import hough_line_test as hough

roi_y = 213

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
                # print(1)
                x1, y1, x2, y2 = line_arr[0], line_arr[1]+roi_y, line_arr[2], line_arr[3]+roi_y
            else:
                # print(line_arr.size)
                x1, y1, x2, y2 = line_arr[:, 0], line_arr[:, 1]+roi_y, line_arr[:, 2], line_arr[:, 3]+roi_y
                slope_degree = (np.arctan2(y2 - y1, x2 - x1) * 180) / np.pi
                _, _, all_lines, _, _, mean_line = hough.lines_filtered(slope_degree, line_arr)

                # 선 그리기
                all_lines[:, :, [1, 3]] += roi_y
                mean_line[:, :, [1, 3]] += roi_y

                hough.draw_lines(n_frame, all_lines, 128, 0, 0)
                hough.draw_lines(n_frame, mean_line, 0, 0, 255)
             
            cv2.imshow("n_frame", n_frame)
            cv2.imshow("blur", blur)
            cv2.imshow("edges", edges)
            
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()