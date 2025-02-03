# import cv2
# import numpy as np

# def remove_object(image, masks, object_index):
#     """
#     Removes a specific object from the image using its mask.

#     Args:
#         image (numpy.ndarray): The original image.
#         masks (list): List of masks for each detected object.
#         object_index (int): Index of the object to remove.

#     Returns:
#         numpy.ndarray: Image with the object removed.
#     """
#     if object_index < 0 or object_index >= len(masks):
#         raise ValueError("Invalid object index.")

#     # Get the mask for the specified object
#     mask = masks[object_index]

#     # Handle multi-channel masks
#     if len(mask.shape) == 3:
#         mask = np.max(mask, axis=0)

#     # Invert the mask to keep everything except the object
#     inverted_mask = cv2.bitwise_not(mask)

#     # Apply the inverted mask to the image
#     result = cv2.bitwise_and(image, image, mask=inverted_mask)

#     return result
# ****************************************************************************

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
    # Get the mask for the specified object
    mask = masks[object_index]

    # Handle multi-channel masks
    if len(mask.shape) == 3:
        mask = np.max(mask, axis=0)

    # Ensure the mask is binary (0 or 255)
    mask = (mask > 0).astype(np.uint8) * 255

    # Debug: Save the mask for inspection
    cv2.imwrite(f"debug_mask_{object_index}.png", mask)

    # Invert the mask to keep everything except the object
    inverted_mask = cv2.bitwise_not(mask)

    # Debug: Save the inverted mask for inspection
    cv2.imwrite(f"debug_inverted_mask_{object_index}.png", inverted_mask)

    # Apply the inverted mask to the image
    result = cv2.bitwise_and(image, image, mask=inverted_mask)

    # Debug: Save the result after applying the mask
    cv2.imwrite(f"debug_result_{object_index}.png", result)

    return result