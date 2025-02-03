import cv2
import numpy as np

def remove_object(image, masks, object_index):
    """
    Removes a specific object from the image using its mask.

    Args:
        image (numpy.ndarray): The original image.
        masks (list): List of masks for each detected object.
        object_index (int): Index of the object to remove.

    Returns:
        numpy.ndarray: Image with the object removed.
    """
    if object_index < 0 or object_index >= len(masks):
        raise ValueError("Invalid object index.")

    # Get the mask for the specified object
    mask = masks[object_index]

    # Handle multi-channel masks
    if len(mask.shape) == 3:
        mask = np.max(mask, axis=0)

    # Invert the mask to keep everything except the object
    inverted_mask = cv2.bitwise_not(mask)

    # Apply the inverted mask to the image
    result = cv2.bitwise_and(image, image, mask=inverted_mask)

    return result