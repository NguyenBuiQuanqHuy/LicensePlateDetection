from ultralytics import YOLO 

# Load model YOLOv8n (nhẹ)
model = YOLO('yolov8n.pt')  

# Train
model.train(
    data='dataset/data.yaml',  # file config dataset
    epochs=50,
    imgsz=640,
    batch=16,
    project='models',
    name='yolo8_license_plate',
    exist_ok=True
)
