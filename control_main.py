import cv2
import numpy as np
from dijkstra import *
from preprocessing import *
from extract_data import *

# roi_frame = [y1:y2, x1:x2]
side = 416
roi_w = 140
roi_h = 200
roi_y = 200

def s(frame, blurVal, darkenin, threshold):
    # edges = img_preprocess(frame, blurVal, darkenin)
    # lines_rth = hough_line_rth(edges, threshold)
    # line_arr, slope_degrees = line_arr_slope_degrees(lines_rth)
    pass

def r():
    pass

def l():
    pass

def main():
    # 카메라 초기화
    camera_index = 1
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Could not open webcam")
        exit()

    
    while True:
        if len(movement_plan) <= 0:
            break

        ret, o_frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        frame = cv2.resize(o_frame, (side, side))
        roi_L_frame = frame[roi_y : (roi_y + roi_h), 0 : roi_w]
        roi_R_frame = frame[roi_y : (roi_y + roi_h), (side - roi_w) : side ]

        roiL_edges = img_preprocess(roi_L_frame, 5, 0.2)
        roiR_edges = img_preprocess(roi_R_frame, 5, 0.2)

        linesL_arr, slopesL = line_arr_slope_degrees(hough_line_raw(roiL_edges, 120))
        linesR_arr, slopesR = line_arr_slope_degrees(hough_line_raw(roiR_edges, 120))
        
        raw_lines = np.concatenate((linesL_arr, linesR_arr), axis=0)

        if raw_lines.size == 0:
            pass
        elif raw_lines.ndim == 1:
            pass
        else:
            all_lines, mean_line, x_van, y_van = slope_filter(slopesL, linesL_arr, slopesR, linesR_arr, roi_w, roi_y, roi_h, side)
            # print(line_arr.size)
            
            cv2.rectangle(frame, (0, roi_y), (roi_w, (roi_y + roi_h)), (255, 128, 128), 2)
            cv2.rectangle(frame, ((side - roi_w), roi_y), (side, (roi_y + roi_h)), (255, 128, 128), 2)
            
            draw_lines(frame, all_lines, 128, 0, 0)
            draw_lines(frame, mean_line, 0, 0, 255)
            cv2.circle(frame, (x_van, y_van), 5, (0, 255, 0), 2)
            
        # order = movement_plan.pop(0) 
        
        # if order == 's':
        #     s()
        
        # if order == 'r':
        #     r()

        # if order == 'l':
        #     l()
                    
        cv2.imshow('ON_AIR', frame)
        
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()