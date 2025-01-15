from modules.yolo_processor import detect_objects
from modules.sam_processor import segment_objects
from modules.overlay_masks import overlay_masks
from modules.background_removal import remove_background
import cv2

def main(image_path):
    # Step 1: Detect objects using YOLO
    print("Detecting objects...")
    boxes = detect_objects(image_path)
    print(f"Detected bounding boxes: {boxes}")

    # Step 2: Segment objects using SAM
    print("Segmenting objects...")
    masks, image = segment_objects(image_path, boxes)
    print("Segmentation complete.")

    # Step 3: Remove background using masks

    print("Removing background...")
    
    print(f"Image shape: {image.shape if image is not None else 'None'}")
    print(f"Number of masks: {len(masks)}")
    if masks:
        print(f"First mask shape: {masks[0].shape if masks[0] is not None else 'None'}")


    result = remove_background(image, masks)
    cv2.imwrite("background_removed_image.jpg", result)
    print("Background removed image saved as 'background_removed_image.jpg'.")

    # Optional Step 4: Overlay masks for visualization
    print("Overlaying masks for visualization...")
    overlayed_image = overlay_masks(image, masks)
    cv2.imwrite("segmented_image.jpg", overlayed_image)
    print("Segmented image saved as 'segmented_image.jpg'.")
if __name__ == "__main__":
    main("image.jpg")  # Replace with the path to your image
