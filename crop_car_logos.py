import cv2
import os
from ultralytics import YOLO

# Load your trained model
model = YOLO(r"runs/detect/car_logo_detector/weights/best.pt")

# Configure paths
input_folder = r"C:\path\to\your\car_images"
output_folder = r"C:\path\to\cropped_logos"
os.makedirs(output_folder, exist_ok=True)

# Process images
for filename in os.listdir(input_folder):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        
        # Detect logos
        results = model.predict(img, conf=0.2)  # Adjust confidence as needed
        
        # Crop and save
        for i, box in enumerate(results[0].boxes.xyxy):
            x1, y1, x2, y2 = map(int, box.tolist())
            cropped = img[y1:y2, x1:x2]
            cv2.imwrite(os.path.join(output_folder, f"{filename.split('.')[0]}_logo_{i}.jpg"), cropped)
        
        print(f"Processed {filename} | Logos found: {len(results[0].boxes)}")
