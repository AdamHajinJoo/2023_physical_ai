import cv2
import numpy as np
import modi
import time
from dijkstra import *
from preprocessing import *
from extract_data import *

'''
알고리즘 개요:
응애응애
'''

global x_curr
global y_curr

INT_MIN = np.iinfo(np.int32).min

roiL_coord = [(0, 200), (160, 400)]
roiR_coord = [(256, 200), (416, 400)]

# P 제어 상수
kp = 0.01

# 목표 vanish_point (실제로는 여러 방법을 사용하여 이 값을 얻어야 함)
target_vanish_point = (213, 250)

base_rpm = 1

# 두 평균직선을 도출한다.
def process_shifted_lines(linesL_arr, slopesL, linesR_arr, slopesR):
    if linesL_arr.size == 0 or linesR_arr.size == 0:
        return None

    if linesL_arr.ndim == 1 or linesR_arr.ndim == 1:
        return None

    linesL, linesR, _, _ = slope_filter(slopesL, linesL_arr, slopesR, linesR_arr)
    L_mean = np.nanmean(linesL, axis=0).astype(int)
    R_mean = np.nanmean(linesR, axis=0).astype(int)

    # L_mean이나 R_mean에 INT_MIN 값만 있는 경우에는 처리 중단
    if np.all(L_mean == INT_MIN) or np.all(R_mean == INT_MIN):
        return None

    try:
        if np.isnan(L_mean).any() or np.isnan(R_mean).any():
            raise ValueError("빈 배열이 포함되어 있습니다.")
    except ValueError as e:
        print("Warning:", e)
        return None

    L_coor = np.array([roiL_coord[0][0], roiL_coord[0][1], roiL_coord[0][0], roiL_coord[0][1]])
    R_coor = np.array([roiR_coord[0][0], roiR_coord[0][1], roiR_coord[0][0], roiR_coord[0][1]])

    L_mean_shifted = L_mean + L_coor
    R_mean_shifted = R_mean + R_coor

    return L_mean_shifted, R_mean_shifted

def main():
    # 카메라 초기화
    camera_index = 1
    cap = cv2.VideoCapture(camera_index)

    # modi 초기화
    bundle = modi.MODI(1)
    motor1 = bundle.motors[0]
    motor2 = bundle.motors[1]
    
    # 소실점 초기화
    x_curr, y_curr = 213, 250
    
    # 왼쪽 바퀴들의 속도를 조절한다.
    def set_left_motor_rpm(motor1, motor2, left_motor_rpm):
        motor1.speed[0] = left_motor_rpm
        motor2.speed[0] = left_motor_rpm
        # motor1의 왼쪽 바퀴와 motor2의 왼쪽 바퀴 모두 rpm 설정
    # 오른쪽 바퀴들의 속도를 조절한다.
    def set_right_motor_rpm(motor1, motor2, right_motor_rpm):
        motor1.speed[1] = -right_motor_rpm
        motor2.speed[1] = -right_motor_rpm
        # motor1의 왼쪽 바퀴와 motor2의 오른쪽 바퀴 모두 rpm 설정
    
    # x_curr와 target_x에 따른 P 제어를 실시한다. 속도는 base_rpm에 기반한다.
    def control_motors(x_curr, target_x):
        # P 제어 수행
        error = target_x - x_curr
        control_input = kp * error

        # 좌우 모터 RPM 조절
        left_motor_rpm = base_rpm + control_input
        right_motor_rpm = base_rpm - control_input

        # 모터 RPM 제어 (하드웨어 및 라이브러리에 따라 다름)
        set_left_motor_rpm(left_motor_rpm)
        set_right_motor_rpm(right_motor_rpm)

    # 직진
    def straight_P_control():
        # P 제어를 사용하여 모터 RPM 조절
        control_motors(x_curr, target_vanish_point[0])

    # 우회전
    def turn_right():
        pass

    # 좌회전
    def turn_left():
        pass

    # movement_plan으로부터 움직일 매뉴얼을 뽑아낸다.
    order = movement_plan.pop(0)

    if not cap.isOpened():
        print("Could not open webcam")
        exit()

    while True:
        ret, o_frame = cap.read()
        
        if not ret:
        # if not ret or len(movement_plan) <= 0:
            print("종료")
            break
        
        frame = cv2.resize(o_frame, (416, 416))
        
        # ROI를 전처리한다.
        roiL = frame[roiL_coord[0][1] : roiL_coord[1][1], roiL_coord[0][0] : roiL_coord[1][0]]
        roiR = frame[roiR_coord[0][1] : roiR_coord[1][1], roiR_coord[0][0] : roiR_coord[1][0]]

        roiL_edges = img_preprocess(roiL, 5, 0.2)
        roiR_edges = img_preprocess(roiR, 5, 0.2)

        # ROI를 허프 변환하여 소실점을 구하기 위한 직선 두 개를 구한다.
        linesL_arr, slopesL = line_arr_slope_degrees(hough_line_raw(roiL_edges, 120))
        linesR_arr, slopesR = line_arr_slope_degrees(hough_line_raw(roiR_edges, 120))
        
        result = process_shifted_lines(linesL_arr, slopesL, linesR_arr, slopesR)

        # 현재 소실점 좌표 (x_curr, y_curr)를 구한다.
        if result is not None:
            L_mean_shifted, R_mean_shifted = result

            draw_lines(frame, L_mean_shifted, 0, 0, 255)
            draw_lines(frame, R_mean_shifted, 0, 0, 255)

            cx, cy = find_intersection(
                L_mean_shifted[0], L_mean_shifted[1], L_mean_shifted[2], L_mean_shifted[3],
                R_mean_shifted[0], R_mean_shifted[1], R_mean_shifted[2], R_mean_shifted[3],
            )

            x_curr = cx
            y_curr = cy

        cv2.circle(frame, (x_curr, y_curr), 5, (0, 255, 0), 2)
        
        cv2.rectangle(frame, (roiL_coord[0][0], roiL_coord[0][1]), (roiL_coord[1][0], roiL_coord[1][1]), (255, 128, 128), 2)  # roi_l
        cv2.rectangle(frame, (roiR_coord[0][0], roiR_coord[0][1]), (roiR_coord[1][0], roiR_coord[1][1]), (255, 128, 128), 2)  # roi_R
        
        # 매뉴얼에 따른 움직임 제어
        if order == 's':
            straight_P_control()
        if order == 'r':
            turn_right()
        if order == 'l':
            turn_left()
                    
        cv2.imshow('ON_AIR', frame)

        control_motors(x_curr, target_vanish_point[0])
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            print(x_curr, y_curr)
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()