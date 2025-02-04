

#*************************************************************************************************************

from modules.yolo_processor import detect_objects
from modules.sam_processor import segment_objects
from modules.overlay_masks import overlay_masks
from modules.background_removal import remove_background
from modules.background_blur import blur_background
from modules.remove_object import remove_object
import cv2
from transformers import pipeline

from transformers import pipeline


def interpret_prompt(prompt):
    print(f"Prompt: {prompt}")  # Debug: Print the prompt
    try:
        # Try the ML model first
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        candidate_labels = [
            "blur the background", 
            "remove the background", 
            "highlight objects with masks", 
            "remove an object", 
            "other editing"
        ]
        result = classifier(prompt, candidate_labels)
        print(f"Classification Result: {result}")  # Debug: Print the classification result

        label_to_action = {
            "blur the background": "blur_background",
            "remove the background": "remove_background",
            "highlight objects with masks": "overlay_masks",
            "remove an object": "remove_object",
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
        elif "remove" in prompt_lower and "background" in prompt_lower:
            action = "remove_background"
        elif "remove" in prompt_lower:
            action = "remove_object"
        elif "highlight" in prompt_lower or "mask" in prompt_lower:
            action = "overlay_masks"
        else:
            action = "other"
        print(f"Fallback Action: {action}")  # Debug: Print the fallback action
    return action
def extract_object_name(prompt):
    """
    Extracts the object name from the prompt using zero-shot classification.

    Args:
        prompt (str): The user's prompt (e.g., "remove the car from this image").

    Returns:
        str: The name of the object to remove (e.g., "car").
    """
    # Define a list of common objects (you can expand this list)
    candidate_labels = [
        "person", "car", "dog", "cat", "bicycle", "motorcycle", 
        "bus", "truck", "chair", "table", "other"
    ]

    # Use zero-shot classification to identify the object
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    result = classifier(prompt, candidate_labels)

    # Return the most likely object
    return result['labels'][0]
def process_image(image_path, prompt):
    # Step 1: Detect objects using YOLO
    boxes, class_names = detect_objects(image_path)
    print(f"Detected objects: {class_names}")  # Debug: Print detected objects
    
    # Step 2: Segment objects using SAM
    masks, image = segment_objects(image_path, boxes)
    
    # Step 3: Interpret the prompt using the LLM
    action = interpret_prompt(prompt)
    print(f"Final Action: {action}")  # Debug
    
    # Step 4: Perform the action
    if action == "blur_background":
        result = blur_background(image, masks)
    elif action == "remove_background":
        result = remove_background(image, masks)
    elif action == "overlay_masks":
        result = overlay_masks(image, masks)
    elif action == "remove_object":
        # Step 4.1: Extract the object name from the prompt
        object_name = extract_object_name(prompt)
        print(f"Object to remove: {object_name}")  # Debug

        # Step 4.2: Find all instances of the object in the list of detected objects
        object_indices = [i for i, name in enumerate(class_names) if name == object_name]

        if not object_indices:
            print(f"Object '{object_name}' not found in the image. Detected objects: {class_names}")
            return image  # Return the original image if the object is not found

        # Step 4.3: Remove all instances of the object
        result = image.copy()  # Start with the original image
        for i in object_indices:
            print(f"Removing object {i + 1}: {object_name}")  # Debug: Print which object is being removed
            result = remove_object(result, masks, i)  # Remove the object

        print(f"All instances of '{object_name}' have been removed.")  # Debug
    else:
        raise ValueError("Unsupported action.")
    
    return result
if __name__ == "__main__":
    image_path = "image.jpg"  # Replace with the path to your image
    prompt = "remove the truck from this image."  # Replace with the user's prompt
    result = process_image(image_path, prompt)
    cv2.imwrite("output_image.jpg", result)
    print("Processed image saved as 'output_image.jpg'.")