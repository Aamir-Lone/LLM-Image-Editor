# python-services/main.py
from modules.yolo_processor import detect_objects
from modules.sam_processor import segment_objects
from modules.overlay_masks import overlay_masks
import cv2
import os

def main(image_path):
    # Step 1: Detect objects using YOLO
    print("Detecting objects...")
    boxes = detect_objects(image_path)
    print(f"Detected bounding boxes: {boxes}")

    # Step 2: Segment objects using SAM
    print("Segmenting objects...")
    masks, image = segment_objects(image_path, boxes)
    print("Segmentation complete.")

    # Step 3: Overlay masks and save result
    print("Overlaying masks...")
    result = overlay_masks(image, masks)

    cv2.imwrite("segmented_image.jpg", result)
    print("Result saved as 'segmented_image.jpg'.")

if __name__ == "__main__":
    main("image.jpg")  # Replace with your input image
