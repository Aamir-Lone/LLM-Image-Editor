# modules/yolo_processor.py
from ultralytics import YOLO

def detect_objects(image_path):
    """
    Detects objects in the given image using YOLOv8.

    Args:
        image_path (str): Path to the input image.

    Returns:
        list: Bounding boxes in the format [x_min, y_min, x_max, y_max].
    """
    # Load YOLO model
    model = YOLO("yolov8n.pt")  # Automatically downloads YOLOv8n on first run

    # Detect objects
    results = model(image_path)

    # Extract bounding boxes
    boxes = []
    for box in results[0].boxes.xyxy:
        boxes.append(box.cpu().numpy().tolist())

    return boxes
