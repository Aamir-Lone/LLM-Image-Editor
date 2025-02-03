# # modules/yolo_processor.py
# from ultralytics import YOLO

# def detect_objects(image_path):
#     """
#     Detects objects in the given image using YOLOv8.

#     Args:
#         image_path (str): Path to the input image.

#     Returns:
#         list: Bounding boxes in the format [x_min, y_min, x_max, y_max].
#     """
#     # Load YOLO model
#     model = YOLO("yolov8n.pt")  # Automatically downloads YOLOv8n on first run

#     # Detect objects
#     results = model(image_path)

#     # Extract bounding boxes
#     boxes = []
#     for box in results[0].boxes.xyxy:
#         boxes.append(box.cpu().numpy().tolist())

#     return boxes
# ******************************************************************
from ultralytics import YOLO

def detect_objects(image_path):
    """
    Detects objects in the given image using YOLOv8.

    Args:
        image_path (str): Path to the input image.

    Returns:
        list: Bounding boxes in the format [x_min, y_min, x_max, y_max].
        list: Class names of the detected objects.
    """
    # Load YOLO model
    # model = YOLO("yolov8n.pt")  # Automatically downloads YOLOv8n on first run
    model = YOLO("yolov8m.pt")

    # Detect objects
    results = model(image_path)

    # Extract bounding boxes and class names
    boxes = []
    class_names = []
    for box in results[0].boxes:
        boxes.append(box.xyxy.cpu().numpy().tolist()[0])  # Bounding box
        class_names.append(results[0].names[int(box.cls)])  # Class name

    return boxes, class_names