import cv2

# 카메라 초기화
camera_index = 0
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print("Could not open webcam")
    exit()
    
while True:
    ret, frame = cap.read()
    
    if ret:
        d_frame = cv2.resize(frame, (416, 416))
        cv2.imshow("RESIZED", d_frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()