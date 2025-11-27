from ultralytics import YOLO
import cv2

# Load model
model = YOLO('models/yolo8_license_plate/weights/best.pt')

# DroidCam thường là camera 1 hoặc 2
cap = cv2.VideoCapture("http://192.168.2.5:4747/video")  # nếu không được thì thử 0, 2, 3

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không lấy được khung hình!")
        break

    # Detect
    results = model(frame)
    annotated = results[0].plot()

    cv2.imshow("DroidCam YOLO", annotated)

    # Bấm q để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
