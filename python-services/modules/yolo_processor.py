from ultralytics import YOLO
import cv2

def detect_objects(image_input):
    model = YOLO("yolov8n.pt")
    
    if isinstance(image_input, str):
        results = model(image_input)
    else:
        results = model(image_input)
    
    boxes = []
    class_names = []
    for box in results[0].boxes:
        boxes.append(box.xyxy.cpu().numpy().tolist()[0])
        class_names.append(results[0].names[int(box.cls)])
    
    return boxes, class_names