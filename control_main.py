import cv2
import numpy as np
from dijkstra import *
from preprocessing import *
from extract_data import *

cursor_position = (0, 0)
INT_MIN = np.iinfo(np.int32).min
# roi_frame = [y1:y2, x1:x2]
side = 416

roiL_coord = [(0, 200), (160, 400)]
roiR_coord = [(256, 200), (416, 400)]

def mouse_callback(event, x, y, flags, param):
    global cursor_position
    cursor_position = (x, y)

def main():
    # 카메라 초기화
    camera_index = 0
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Could not open webcam")
        exit()
    
    while True:
        ret, o_frame = cap.read()
        
        if not ret:
        # if not ret or len(movement_plan) <= 0:
            print("종료")
            break

        frame = cv2.resize(o_frame, (side, side))
        
         # 창에 마우스 이벤트 콜백 함수 등록
        cv2.namedWindow('ON_AIR')
        cv2.setMouseCallback('ON_AIR', mouse_callback)
        
        roiL = frame[roiL_coord[0][1] : roiL_coord[1][1], roiL_coord[0][0] : roiL_coord[1][0]]
        roiR = frame[roiR_coord[0][1] : roiR_coord[1][1], roiR_coord[0][0] : roiR_coord[1][0]]

        roiL_edges = img_preprocess(roiL, 5, 0.2)
        roiR_edges = img_preprocess(roiR, 5, 0.2)

        linesL_arr, slopesL = line_arr_slope_degrees(hough_line_raw(roiL_edges, 120))
        linesR_arr, slopesR = line_arr_slope_degrees(hough_line_raw(roiR_edges, 120))
                
        if linesL_arr.size == 0 or linesR_arr.size == 0:
            pass
        elif linesL_arr.ndim == 1 or linesR_arr.ndim == 1:
            pass
            # print("1dim_L: {0} (count: {1}) \n1dim_R: {2} (count: {3})\n".format(linesL_arr, int(linesL_arr.size / 4), linesR_arr, int(linesR_arr.size / 4)))
        else:
            # print("L: {0} (count: {1}) \nR: {2} (count: {3})\n".format(linesL_arr, int(linesL_arr.size / 4), linesR_arr, int(linesR_arr.size / 4)))
            # print("slopeL: {0} (count: {1}) \nslopeR: {2} (count: {3})\n".format(slopesL, int(slopesL.size), slopesR, int(slopesR.size)))
            cv2.rectangle(frame, (roiL_coord[0][0], roiL_coord[0][1]), (roiL_coord[1][0], roiL_coord[1][1]), (255, 128, 128), 2)              # roi_l
            cv2.rectangle(frame, (roiR_coord[0][0], roiR_coord[0][1]), (roiR_coord[1][0], roiR_coord[1][1]), (255, 128, 128), 2)  # roi_R

            linesL, linesR, _, _ = slope_filter(slopesL, linesL_arr, slopesR, linesR_arr)
            L_mean = np.nanmean(linesL, axis=0).astype(int)
            R_mean = np.nanmean(linesR, axis=0).astype(int)
            
            # L_mean이나 R_mean에 INT_MIN 값만 있는 경우에 continue
            if np.all(L_mean == INT_MIN) or np.all(R_mean == INT_MIN):
                continue
            
            try:
                if np.isnan(L_mean).any() or np.isnan(R_mean).any():
                    raise ValueError("빈 배열이 포함되어 있습니다.")
            except ValueError as e:
                print("Warning:", e)
                continue
            
            L_coor = np.array([roiL_coord[0][0], roiL_coord[0][1], roiL_coord[0][0], roiL_coord[0][1]])
            R_coor = np.array([roiR_coord[0][0], roiR_coord[0][1], roiR_coord[0][0], roiR_coord[0][1]])
            
            L_mean_shifted = L_mean + L_coor
            R_mean_shifted = R_mean + R_coor
            
            draw_lines(frame, L_mean_shifted, 0, 0, 255)
            draw_lines(frame, R_mean_shifted, 0, 0, 255)
            
            x_van, y_van = find_intersection(
                L_mean_shifted[0], L_mean_shifted[1], L_mean_shifted[2], L_mean_shifted[3],
                R_mean_shifted[0], R_mean_shifted[1], R_mean_shifted[2], L_mean_shifted[3],
            )
            
            print("L_mean: ({0},{1})~({2},{3})".format(L_mean_shifted[0], L_mean_shifted[1], L_mean_shifted[2], L_mean_shifted[3]))
            print("R_mean: ({0},{1})~({2},{3})".format(R_mean_shifted[0], R_mean_shifted[1], R_mean_shifted[2], R_mean_shifted[3]))
            print(x_van, y_van)
            print()
            
            # 현재 마우스 좌표 표시
        cv2.putText(frame, f"마우스 좌표: {cursor_position}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # cv2.circle(frame, (x_van, y_van), 5, (0, 255, 0), 2)
            
        # order = movement_plan.pop(0) 
        
        # if order == 's':
        #     s()
        
        # if order == 'r':
        #     r()

        # if order == 'l':
        #     l()
                    
        cv2.imshow('ON_AIR', frame)
        # cv2.imshow('L', roiL)
        # cv2.imshow('R', roiR)
        
        if cv2.waitKey(40) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()