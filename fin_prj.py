import cv2
import numpy as np
import dijkstra as di
import matplotlib.pyplot as plt
import sys
# sys.path.append('/home/pi/Desktop/modules/pymodi_master/pymodi-master/modi')
# from modi import modi
# from modules.pymodi_master.pymodi-master.modi import modi
# sys.path.append('/home/pi/Desktop/modules/yolov5_master')
# from yolov5_master import detect as yolo_det


def calculate_vanishing_point(lines):
    if lines is not None:
        # 모든 직선의 두 교차점을 최소 자승 문제로 계산
        points = []
        for line1 in lines:
            for line2 in lines:
                rho1, theta1 = line1[0]
                rho2, theta2 = line2[0]
                A = np.array([[np.cos(theta1), np.sin(theta1)],
                              [np.cos(theta2), np.sin(theta2)]])
                b = np.array([rho1, rho2])
                point, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                points.append(point)

        # 교차점의 평균 계산
        if points:
            vanishing_point = np.mean(points, axis=0)
            return vanishing_point.astype(int)
    return None
def draw_lines(frame, lines):
    # pass
    if lines is not None and len(lines) > 0:
        for line in lines:
            rho, theta = line
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

def region_of_interest(img, vertices, color3=(255,255,255), color1=255):
    pass
    # mask = np.zeros_like(img) # mask = img와 같은 크기의 빈 이미지
    
    # if len(img.shape) > 2: # Color 이미지(3채널)라면 :
    #     color = color3
    # else: # 흑백 이미지(1채널)라면 :
    #     color = color1
        
    # # vertices에 정한 점들로 이뤄진 다각형부분(ROI 설정부분)을 color로 채움 
    # cv2.fillPoly(mask, vertices, color)
    
    # # 이미지와 color로 채워진 ROI를 합침
    # ROI_image = cv2.bitwise_and(img, mask)
    return ROI_image


def yolo():
    pass
    # YOLO 코드 추가

# modi 제어를 위한 함수들
def turn_right():
    pass
    # print("Turning right")
    # 우회전에 대한 모터 동작 등을 추가
def turn_left():
    pass
    # print("Turning left")
    # 좌회전에 대한 모터 동작 등을 추가
def stop():
    pass
    # print("Stopping")
    # 정지에 대한 모터 동작 등을 추가

def main():
    # modi 초기화
    bundle = modi.MODI()
    motor = bundle.motors[0]
    # 카메라 초기화
    camera_index = 0
    cap = cv2.VideoCapture(camera_index)
    # OpenCV 및 YOLO 초기화
    net = cv2.dnn.readNet("yolov5.weights", "yolov5.cfg")
    layer_names = net.getUnconnectedOutLayersNames()
    
    # dijkstra.py 문서로부터 map과 manual 가져오기
    map_res = di.dijkstra(di.graph_map, di.destinations, di.first_dir)
    manual = di.path_to_movement_plan(map_res[1], di.graph_map, di.first_dir)

    if not cap.isOpened():
        print("Could not open webcam")
        exit()

    for order in manual:
        while True:
            ret, frame = cap.read()

            if not cap.isOpened():
                print("Could not open webcam")
                exit()

            if ret:
                dframe = cv2.resize(frame, (416, 416))
                height, width, channels = dframe.shape
                
                if order == 's':
                    yolo()  # YOLO를 사용하여 물체 감지

                    # 동작 조건
                    # 영상 전처리 (허프 변환을 적용하기 전에 필요한 전처리 수행)
                    gray = cv2.cvtColor(dframe, cv2.COLOR_BGR2GRAY)
                    blur = cv2.GaussianBlur(gray, (3, 3), 0)
                    edges = cv2.Canny(blur, 50, 150, apertureSize=3)

                    # 허프 변환을 사용하여 선 감지
                    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=125)
                    # lines = np.squeeze(lines)

                    # 기울기 구하기
                    slope_degree = (np.arctan2(lines[:,1] - lines[:,3], lines[:,0] - lines[:,2]) * 180) / np.pi

                    # 수평 기울기 제한
                    lines = lines[np.abs(slope_degree)<160]
                    slope_degree = slope_degree[np.abs(slope_degree)<160]
                    # 수직 기울기 제한
                    lines = lines[np.abs(slope_degree)>95]
                    slope_degree = slope_degree[np.abs(slope_degree)>95]
                    # 필터링된 직선 버리기
                    L_lines, R_lines = lines[(slope_degree>0),:], lines[(slope_degree<0),:]
                    L_lines, R_lines = L_lines[:,None], R_lines[:,None]

                    # 허프 변환 선 그리기
                    draw_lines(dframe, lines)

                    # 선이 감지된 경우에만 소실점 계산
                    # vanishing_point = calculate_vanishing_point(lines)
                    # if vanishing_point is not None:
                    #     # 소실점 출력
                    #     print("Vanishing Point:", vanishing_point)
                    #     # 소실점에 원 그리기
                    #     cv2.circle(dframe, tuple(vanishing_point), 5, (0, 255, 0), -1)

                    # 화면에 결과 표시
                    cv2.imshow("Hough Transform", dframe)

                    # 자세한 동작 조건에 따라 modi 제어 함수 호출
                    # ... (자세한 동작 조건을 여기에 작성)

                    # modi로부터 동작이 완료됐는지의 신호를 받음
                    # ... (modi에서 동작 완료 시 신호를 받는 코드)

                    # 조건이 충족되면 break
                    break

                elif order == 'r':
                    turn_right()
                    yolo()

                elif order == 'l':
                    turn_left()
                    yolo()

    # 모든 작업이 끝나면 종료
    stop()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()