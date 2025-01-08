import cv2
import numpy as np

def process_with_cv2(image_path: str, task: str):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        return None, "Image could not be loaded."

    # Process based on the task
    if task == "blur background":
        # Create a blurred version of the image
        blurred = cv2.GaussianBlur(image, (51, 51), 0)
        return blurred, None

    elif task == "remove background":
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Threshold to create a mask
        _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        # Invert the mask to keep the subject
        mask_inv = cv2.bitwise_not(mask)
        # Apply the mask to the image
        result = cv2.bitwise_and(image, image, mask=mask_inv)
        return result, None

    else:
        return None, f"Task '{task}' not recognized."
