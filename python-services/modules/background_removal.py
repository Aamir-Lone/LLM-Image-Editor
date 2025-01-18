import cv2
import numpy as np

def remove_background(image, masks, background_color=(255, 255, 255)):
    """
    Removes the background from the input image using masks.

    Args:
        image (numpy.ndarray): The original image.
        masks (list): List of masks for each detected object.
        background_color (tuple): RGB color for the background. Default is white.

    Returns:
        numpy.ndarray: Image with the background removed (transparent or colored).
    """
    if image is None or len(image.shape) != 3:
        raise ValueError(f"Invalid image: {image}")

    # Initialize combined mask to zero (black)
    combined_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # Process each mask
    for i, mask in enumerate(masks):
        if mask is None:
            raise ValueError(f"Invalid mask: {mask}")

        # Handle multi-channel masks by converting to single-channel
        if len(mask.shape) == 3:
            mask = np.max(mask, axis=0)  # Collapse multi-channel mask to single-channel

        if len(mask.shape) != 2:
            raise ValueError(f"Processed mask is not 2D: {mask.shape}")

        # Resize mask to match the image dimensions
        resized_mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Combine all masks into a single mask
        combined_mask = cv2.bitwise_or(combined_mask, resized_mask)

    # Debugging: Save and inspect the combined mask
    cv2.imwrite("combined_mask_debug.png", combined_mask * 255)

    # Create an inverted mask (background is 1, objects are 0)
    inverted_mask = cv2.bitwise_not(combined_mask)

    # Debugging: Save the inverted mask
    cv2.imwrite("inverted_mask_debug.png", inverted_mask * 255)

    # Extract the foreground using the combined mask
    foreground = cv2.bitwise_and(image, image, mask=combined_mask)

    # Debugging: Save the foreground
    cv2.imwrite("foreground_debug.png", foreground)

    # Create a background image with the specified color
    background = np.full_like(image, background_color, dtype=np.uint8)
    background = cv2.bitwise_and(background, background, mask=inverted_mask)

    # Debugging: Save the background
    cv2.imwrite("background_debug.png", background)

    # Combine the foreground and background to get the final result
    result = cv2.add(foreground, background)

    # Debugging: Save the final result
    cv2.imwrite("final_debug_result.png", result)

    return result
