import numpy as np
import cv2

def overlay_masks(image, masks):
    """
    Overlays masks on the input image.

    Args:
        image (numpy.ndarray): The original image.
        masks (list): List of masks for each detected object.

    Returns:
        numpy.ndarray: Image with masks overlaid.
    """
    overlay = image.copy()

    for mask in masks:
        # Handle multi-channel masks
        if len(mask.shape) == 3:
            print(f"Converting multi-channel mask of shape {mask.shape} to single-channel.")
            # Convert to single-channel by taking the maximum value across channels
            mask = np.max(mask, axis=0)

        # Convert the mask to uint8
        mask_uint8 = mask.astype(np.uint8)

        # Validate dimensions
        if len(mask_uint8.shape) != 2:
            raise ValueError(f"Invalid mask shape after processing: {mask_uint8.shape}")
        if len(image.shape) != 3:
            raise ValueError(f"Invalid image shape: {image.shape}")

        print("Processed mask shape:", mask_uint8.shape)
        print("Image shape:", image.shape)

        # Resize the mask to match the image size
        resized_mask = cv2.resize(mask_uint8, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Ensure the mask is binary (0 or 1)
        binary_mask = (resized_mask > 0).astype(np.uint8)

        # Overlay the mask on the image
        overlay = cv2.addWeighted(overlay, 0.7, np.stack([binary_mask * 255] * 3, axis=-1), 0.3, 0)

    return overlay
