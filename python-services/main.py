# from modules.yolo_processor import detect_objects
# from modules.sam_processor import segment_objects
# from modules.overlay_masks import overlay_masks
# from modules.background_removal import remove_background
# from modules.background_blur import blur_background
# import cv2

# def main(image_path):
#     # Step 1: Detect objects using YOLO
#     print("Detecting objects...")
#     boxes = detect_objects(image_path)
#     print(f"Detected bounding boxes: {boxes}")

#     # Step 2: Segment objects using SAM
#     print("Segmenting objects...")
#     masks, image = segment_objects(image_path, boxes)
#     print("Segmentation complete.")

#     # Step 3: Remove background using masks

#     print("Removing background...")
    
#     print(f"Image shape: {image.shape if image is not None else 'None'}")
#     print(f"Number of masks: {len(masks)}")
#     if masks:
#         print(f"First mask shape: {masks[0].shape if masks[0] is not None else 'None'}")


#     result = remove_background(image, masks)
#     cv2.imwrite("background_removed_image.jpg", result)
#     print("Background removed image saved as 'background_removed_image.jpg'.")

#     # Step 4: Blur background
#     print("Blurring background...")
#     background_blurred_image = blur_background(image, masks,blur_strength=(35, 35))
#     cv2.imwrite("background_blurred_image.jpg", background_blurred_image)
#     print("Background blurred image saved as 'background_blurred_image.jpg'.")

#     # Optional Step 5: Overlay masks for visualization
#     print("Overlaying masks for visualization...")
#     overlayed_image = overlay_masks(image, masks)
#     cv2.imwrite("segmented_image.jpg", overlayed_image)
#     print("Segmented image saved as 'segmented_image.jpg'.")
# if __name__ == "__main__":
#     main("image.jpg")  # Replace with the path to your image


#*************************************************************************************************************

from modules.yolo_processor import detect_objects
from modules.sam_processor import segment_objects
from modules.overlay_masks import overlay_masks
from modules.background_removal import remove_background
from modules.background_blur import blur_background
import cv2
from transformers import pipeline

from transformers import pipeline

# def interpret_prompt(prompt):
#     # Load a zero-shot classification pipeline
#     classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

#     # Define the possible actions
#     candidate_labels = ["blur the background", "remove the background", "overlay masks","other editing"]

#     # Classify the prompt
#     result = classifier(prompt, candidate_labels)

#     print("Classification Result:", result)

#     # Return the label with the highest score
#     return result['labels'][0]

def interpret_prompt(prompt):
    print(f"Prompt: {prompt}")  # Debug: Print the prompt
    try:
        # Try the ML model first
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        candidate_labels = ["blur the background", "remove the background", "highlight objects with masks", "other editing"]
        result = classifier(prompt, candidate_labels)
        print(f"Classification Result: {result}")  # Debug: Print the classification result

        label_to_action = {
            "blur the background": "blur_background",
            "remove the background": "remove_background",
            "highlight objects with masks": "overlay_masks",
            "other editing": "other"
        }
        action = label_to_action[result['labels'][0]]
        print(f"Action: {action}")  # Debug: Print the action
    except Exception as e:
        print(f"Error in ML model: {e}")  # Debug: Print any errors
        # Fallback to keyword matching
        prompt_lower = prompt.lower()
        if "blur" in prompt_lower:
            action = "blur_background"
        elif "remove" in prompt_lower or "background" in prompt_lower:
            action = "remove_background"
        elif "highlight" in prompt_lower or "mask" in prompt_lower:
            action = "overlay_masks"
        else:
            action = "other"
        print(f"Fallback Action: {action}")  # Debug: Print the fallback action
    return action
def process_image(image_path, prompt):
    # Step 1: Detect objects using YOLO
    boxes = detect_objects(image_path)
    
    # Step 2: Segment objects using SAM
    masks, image = segment_objects(image_path, boxes)
    
    # Step 3: Interpret the prompt using the LLM
    action = interpret_prompt(prompt)
    print(f"Final Action: {action}")  # Debug
    
    # Step 4: Perform the action
    if action == "blur_background":  # Use mapped action name
        result = blur_background(image, masks)
    elif action == "remove_background":  # Use mapped action name
        result = remove_background(image, masks)
    elif action == "overlay_masks":  # Use mapped action name
        result = overlay_masks(image, masks)
    else:
        raise ValueError("Unsupported action.")
    
    return result
if __name__ == "__main__":
    image_path = "image.jpg"  # Replace with the path to your image
    prompt = "remove the background of this image."  # Replace with the user's prompt
    result = process_image(image_path, prompt)
    cv2.imwrite("output_image.jpg", result)
    print("Processed image saved as 'output_image.jpg'.")