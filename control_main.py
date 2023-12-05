import cv2
import numpy as np
import modi
from dijkstra import *
from preprocessing import *
from extract_data import *

global x_van
global y_van

cursor_position = (0, 0)
INT_MIN = np.iinfo(np.int32).min

side = 416

roiL_coord = [(0, 200), (160, 400)]
roiR_coord = [(256, 200), (416, 400)]

# P 제어 상수
kp = 0.01

# 목표 vanish_point (실제로는 여러 방법을 사용하여 이 값을 얻어야 함)
target_vanish_point = (213, 250)

# 모터 관련 변수 (실제로는 하드웨어 및 제어 라이브러리에 따라 다름)
left_motor = 0
right_motor = 0

base_rpm = 1

def control_motors(current_x, target_x):
    # P 제어 수행
    error = target_x - current_x
    control_input = kp * error

    # 좌우 모터 RPM 조절
    left_motor_rpm = base_rpm + control_input
    right_motor_rpm = base_rpm - control_input

    # 모터 RPM을 제어하는 코드 (하드웨어 및 라이브러리에 따라 다름)
    set_left_motor_rpm(left_motor_rpm)
    set_right_motor_rpm(right_motor_rpm)

def straight_P_control():
    # 현재 vanish point와 목표 vanish point 간의 수평 오차 계산
    error_x = x_van - target_vanish_point[0]

    # P 제어를 사용하여 모터 RPM 조절
    control_motors(x_van, target_vanish_point[0])

def turn_right():
    pass

def turn_left():
    pass

def main():
    # 카메라 초기화
    camera_index = 1
    cap = cv2.VideoCapture(camera_index)

    # modi 초기화
    bundle = modi.MODI(1)
    motor1 = bundle.motors[0]
    motor2 = bundle.motors[1]

    def set_left_motor_rpm(motor1, motor2, left_motor_rpm):
        motor1.speed[0] = left_motor_rpm
        motor2.speed[0] = left_motor_rpm
        # motor1의 왼쪽 바퀴와 motor2의 왼쪽 바퀴 모두 rpm 설정

    def set_right_motor_rpm(motor1, motor2, right_motor_rpm):
        motor1.speed[1] = -right_motor_rpm
        motor2.speed[1] = -right_motor_rpm
        # motor1의 왼쪽 바퀴와 motor2의 오른쪽 바퀴 모두 rpm 설정

    if not cap.isOpened():
        print("Could not open webcam")
        exit()
    
    x_van = 213
    y_van = 250

    while True:
        ret, o_frame = cap.read()
        
        if not ret:
        # if not ret or len(movement_plan) <= 0:
            print("종료")
            break

        frame = cv2.resize(o_frame, (side, side))
                
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
        else:
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
            
            cx, cy = find_intersection(
                L_mean_shifted[0], L_mean_shifted[1], L_mean_shifted[2], L_mean_shifted[3],
                R_mean_shifted[0], R_mean_shifted[1], R_mean_shifted[2], R_mean_shifted[3],
            )

            x_van = cx
            y_van = cy


        cv2.circle(frame, (x_van, y_van), 5, (0, 255, 0), 2)
        
        cv2.rectangle(frame, (roiL_coord[0][0], roiL_coord[0][1]), (roiL_coord[1][0], roiL_coord[1][1]), (255, 128, 128), 2)  # roi_l
        cv2.rectangle(frame, (roiR_coord[0][0], roiR_coord[0][1]), (roiR_coord[1][0], roiR_coord[1][1]), (255, 128, 128), 2)  # roi_R
        
        order = movement_plan.pop(0) 
        
        if order == 's':
            pass
        if order == 'r':
            turn_right()
        if order == 'l':
            turn_left()
                    
        cv2.imshow('ON_AIR', frame)
        # cv2.imshow('L', roiL)
        # cv2.imshow('R', roiR)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            print(x_van, y_van)
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()