
# ******************************************************************
# from ultralytics import YOLO

# def detect_objects(image_path):
#     """
#     Detects objects in the given image using YOLOv8.

#     Args:
#         image_path (str): Path to the input image.

#     Returns:
#         list: Bounding boxes in the format [x_min, y_min, x_max, y_max].
#         list: Class names of the detected objects.
#     """
#     # Load YOLO model
#     # model = YOLO("yolov8n.pt")  # Automatically downloads YOLOv8n on first run
#     model = YOLO("yolov8m.pt")

#     # Detect objects
#     results = model(image_path)

#     # Extract bounding boxes and class names
#     boxes = []
#     class_names = []
#     for box in results[0].boxes:
#         boxes.append(box.xyxy.cpu().numpy().tolist()[0])  # Bounding box
#         class_names.append(results[0].names[int(box.cls)])  # Class name

#     return boxes, class_names
#********************************************************************************

# modules/yolo_processor.py
from ultralytics import YOLO

def detect_objects(image_input):
    """
    Detects objects in the given image using YOLOv8.

    Args:
        image_input (str or numpy.ndarray): Path to the input image or the image array.

    Returns:
        list: Bounding boxes in the format [x_min, y_min, x_max, y_max].
        list: Class names of the detected objects.
    """
    model = YOLO("yolov8m.pt")
    
    # Detect objects
    if isinstance(image_input, str):
        results = model(image_input)
    else:
        results = model(image_input)

    # Extract bounding boxes and class names
    boxes = []
    class_names = []
    for box in results[0].boxes:
        boxes.append(box.xyxy.cpu().numpy().tolist()[0])
        class_names.append(results[0].names[int(box.cls)])

    return boxes, class_names