import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Workaround for OpenMP conflict

from ultralytics import YOLO

# Load a fresh YOLOv8 model (or use 'yolov8n.pt' for pretrained weights)
model = YOLO("yolov8n.yaml")  # or "yolov8n.pt" for transfer learning

# Train the model (update paths as needed)
results = model.train(
    data=r"C:\Users\User\Downloads\TPR_Car_Logo_Detector.v2i.yolov8\data.yaml",
    epochs=50,  # Increase if needed (~50-100 is typical)
    imgsz=640,  # Image size
    batch=8,    # Reduce if you get CUDA out-of-memory errors
    name="car_logo_detector"  # Saves results under 'runs/detect/car_logo_detector'
)
