import cv2
import numpy as np
# import modi
import time
from dijkstra import *
from preprocessing import *
from extract_data import *

'''
알고리즘 개요:
'''

global x_curr
global y_curr

global maybe_time_to_turn
turn_time_len = 3000     # 시간(ms) 지나면 다음 order 수행
frequency = 40

movement_plan = generate_movement_plan()

INT_MIN = np.iinfo(np.int32).min

roiL_coord = [(0, 240), (150, 320)]
roiR_coord = [(416 - 150, 240), (416, 320)]

# P 제어 상수
kp = 0.1

# 목표 vanish_point (실제로는 여러 방법을 사용하여 이 값을 얻어야 함)
target_vanish_point = (213, 160)

# 차체 바퀴 속도 관련
basic_rpm = 100
turn_time = 4.000        # 좌회전 / 우회전하는 시간
turn_speed_reduced = 20

def pop(li):
    comp = li.pop(0)
    # print(''.join(li))
    return comp

# 두 평균직선을 도출한다.
def process_shifted_lines(linesL_arr, slopesL, linesR_arr, slopesR):
    linesL, linesR, _, _ = slope_filter(slopesL, linesL_arr, slopesR, linesR_arr)
    
    # NaN이 아닌 값만 추출
    valid_linesL = linesL[~np.isnan(linesL).any(axis=1)]
    valid_linesR = linesR[~np.isnan(linesR).any(axis=1)]

    # NaN이 없는 경우에만 평균 계산 및 형변환 수행
    if valid_linesL.size > 0:
        L_mean = np.nanmean(valid_linesL, axis=0).astype(int)
    else:
        # NaN이 없는 값이 없는 경우에 대한 처리
        if linesL.size > 0:
            L_mean = np.full_like(linesL[0], np.nan, dtype=int)
        else:
            # linesL이 비어있는 경우에 대한 처리
            L_mean = np.full(4, np.nan, dtype=int)  # 여기서 4는 linesL의 예상된 열 수입니다.

    if valid_linesR.size > 0:
        R_mean = np.nanmean(valid_linesR, axis=0).astype(int)
    else:
        # NaN이 없는 값이 없는 경우에 대한 처리
        if linesR.size > 0:
            R_mean = np.full_like(linesR[0], np.nan, dtype=int)
        else:
            # linesR이 비어있는 경우에 대한 처리
            R_mean = np.full(4, np.nan, dtype=int)  # 여기서 4는 linesR의 예상된 열 수입니다.

    if valid_linesL.size == 0 or valid_linesR.size == 0 or np.all(L_mean == INT_MIN) or np.all(R_mean == INT_MIN):
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
    camera_index = 0
    cap = cv2.VideoCapture(camera_index)

    # # modi 초기화
    # bundle = modi.MODI(1)
    # motor1 = bundle.motors[0]
    # motor2 = bundle.motors[1]
    
    # 전역변수들 초기화
    x_curr, y_curr = 213, 200
    maybe_time_to_turn = 0
    
    # # 왼쪽 앞바퀴의 속도를 조절한다.
    # def set_left_motor_rpm(motor1, motor2, left_motor_rpm):
    #     motor1.speed[0] = left_motor_rpm
    #     motor2.speed[0] = basic_rpm
    #     # motor1의 왼쪽 바퀴와 motor2의 왼쪽 바퀴 모두 rpm 설정
    
    # # 오른쪽 앞바퀴의 속도를 조절한다.
    # def set_right_motor_rpm(motor1, motor2, right_motor_rpm):
    #     motor1.speed[1] = -right_motor_rpm
    #     motor2.speed[1] = -basic_rpm
    #     # motor1의 오른쪽 바퀴와 motor2의 오른쪽 바퀴 모두 rpm 설정
    
    # x_curr와 target_x에 따른 P 제어를 실시한다. 속도는 basic_rpm에 기반한다.
    def control_motors(x_curr, target_x):
        # P 제어 수행
        error = target_x - x_curr
        control_input = kp * error

        # 좌우 모터 RPM 조절
        left_motor_rpm = basic_rpm + control_input
        right_motor_rpm = basic_rpm - control_input

        # 모터 RPM 제어
        # set_left_motor_rpm(left_motor_rpm)
        # set_right_motor_rpm(right_motor_rpm)
        return left_motor_rpm, right_motor_rpm

    # 우회전: 오른쪽 바퀴를 느리게 회전한다.
    def turn_right(rpm):
        # set_left_motor_rpm(motor1, motor2, basic_rpm)
        # set_right_motor_rpm(motor1, motor2, rpm)
        time.sleep(turn_time)
        
    # 좌회전: 왼쪽 바퀴를 느리게 회전한다.
    def turn_left(rpm):
        # set_left_motor_rpm(motor1, motor2, rpm)
        # set_right_motor_rpm(motor1, motor2, basic_rpm)
        time.sleep(turn_time)
        
    # movement_plan으로부터 움직일 매뉴얼을 뽑아낸다.
    order = pop(movement_plan)

    if not cap.isOpened():
        print("Could not open webcam")
        exit()

    while True:
        ret, o_frame = cap.read()
        
        if not ret or len(movement_plan) <= 0:
            print("종료")
            break
        
        frame = cv2.resize(o_frame, (416, 416))
        
        # ROI를 전처리한다.
        roiL = frame[roiL_coord[0][1] : roiL_coord[1][1], roiL_coord[0][0] : roiL_coord[1][0]]
        roiR = frame[roiR_coord[0][1] : roiR_coord[1][1], roiR_coord[0][0] : roiR_coord[1][0]]

        roiL_edges = img_preprocess(roiL, 7, 0.2)
        roiR_edges = img_preprocess(roiR, 7, 0.2)

        # ROI를 허프 변환하여 소실점을 구하기 위한 직선 두 개를 구한다.
        linesL_arr, slopesL = line_arr_slope_degrees(hough_line_raw(roiL_edges, 70))
        linesR_arr, slopesR = line_arr_slope_degrees(hough_line_raw(roiR_edges, 70))
        
        # 직선차로가 인식되지 않으면 maybe_time_to_turn을 증가시키도록 함
        if linesL_arr.size == 0 or linesR_arr.size == 0 or linesL_arr.ndim == 1 or linesR_arr.ndim == 1:
            result = None
            maybe_time_to_turn += int(turn_time_len / frequency)
        else:
            maybe_time_to_turn = 0
            result = process_shifted_lines(linesL_arr, slopesL, linesR_arr, slopesR)

        # 현재 소실점 좌표 (x_curr, y_curr)를 구한다.
        if result is not None:
            L_mean_shifted, R_mean_shifted = result

            draw_lines(frame, L_mean_shifted, 0, 0, 255)
            draw_lines(frame, R_mean_shifted, 0, 0, 255)

            cx, cy = find_intersection(
                L_mean_shifted[0], L_mean_shifted[1], L_mean_shifted[2], L_mean_shifted[3],
                R_mean_shifted[0], R_mean_shifted[1], R_mean_shifted[2], R_mean_shifted[3]
            )

            x_curr = cx
            y_curr = cy
        
        # 화면에 정보를 표시.
        cv2.circle(frame, (x_curr, y_curr), 5, (0, 255, 0), 2)  # 소실점
        # movement_plan 현황
        cv2.putText(frame, f"order: {order}   plan_left: {''.join(movement_plan)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)
        if maybe_time_to_turn < turn_time_len:
            cv2.putText(frame, f"maybe_time_to_turn: {maybe_time_to_turn}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)
        else:
            cv2.putText(frame, "Turn!", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 128, 0), 2)
        # ROI 영역 표시
        cv2.rectangle(frame, (roiL_coord[0][0], roiL_coord[0][1]), (roiL_coord[1][0], roiL_coord[1][1]), (255, 128, 128), 2)  # roi_l
        cv2.rectangle(frame, (roiR_coord[0][0], roiR_coord[0][1]), (roiR_coord[1][0], roiR_coord[1][1]), (255, 128, 128), 2)  # roi_R
        
        # movement_plan에 따라 움직임을 제어한다.
        if order == 's':
            front_left, front_right = control_motors(x_curr, target_vanish_point[0])
            cv2.putText(frame, "Turn!", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 128, 0), 2)
            
            cv2.putText(frame, f"L: {front_left}, R: {front_right}", (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
            cv2.putText(frame, f"L: {basic_rpm}, R: {basic_rpm}", (10, 375), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)

            # maybe_time_to_turn이 turn_time_len을 넘어가는 순간 다음 단계로 넘어간다.
            if maybe_time_to_turn - frequency < turn_time_len and maybe_time_to_turn >= turn_time_len:
                order = pop(movement_plan)
            
        if order == 'r':
            rpm_turn = basic_rpm - turn_speed_reduced
            cv2.putText(frame, f"L: {basic_rpm}, R: {rpm_turn}", (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
            cv2.putText(frame, f"L: {basic_rpm}, R: {basic_rpm}", (10, 375), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
            turn_right(rpm_turn)
            order = pop(movement_plan)

        if order == 'l':
            rpm_turn = basic_rpm - turn_speed_reduced
            cv2.putText(frame, f"L: {rpm_turn}, R: {basic_rpm}", (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
            cv2.putText(frame, f"L: {basic_rpm}, R: {basic_rpm}", (10, 375), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
            turn_left(rpm_turn)
            order = pop(movement_plan)
                    
        cv2.imshow('ON_AIR', frame)
        # cv2.imshow('L', roiL_edges)
        # cv2.imshow('R', roiR_edges)

        control_motors(x_curr, target_vanish_point[0])
        
        if cv2.waitKey(frequency) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()