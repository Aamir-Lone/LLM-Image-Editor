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

    # Initialize a blank combined mask
    combined_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # Process and combine each mask
    for i, mask in enumerate(masks):
        if mask is None:
            raise ValueError(f"Invalid mask: {mask}")

        # Convert multi-channel mask to single-channel
        if len(mask.shape) == 3:
            mask = np.max(mask, axis=0)

        # Normalize mask to binary (0 or 1)
        binary_mask = (mask > 0).astype(np.uint8)

        # Save individual binary mask for debugging
        cv2.imwrite(f"debug_mask_{i}.png", binary_mask * 255)

        # Combine masks using logical OR
        combined_mask = np.logical_or(combined_mask, binary_mask).astype(np.uint8)

    # Debugging: Save the combined binary mask
    cv2.imwrite("debug_combined_mask.png", combined_mask * 255)

    # Extract the foreground using the combined mask
    foreground = cv2.bitwise_and(image, image, mask=combined_mask)

    # Debugging: Save the foreground
    cv2.imwrite("debug_foreground.png", foreground)

    # Create a solid background with the specified color
    background = np.full_like(image, background_color, dtype=np.uint8)

    # Create an inverted mask for the background
    inverted_mask = cv2.bitwise_not((combined_mask * 255).astype(np.uint8))

    # Debugging: Save the inverted mask
    cv2.imwrite("debug_inverted_mask.png", inverted_mask)

    # Apply the inverted mask to the background
    background = cv2.bitwise_and(background, background, mask=inverted_mask)

    # Debugging: Save the isolated background
    cv2.imwrite("debug_background.png", background)

    # Combine the foreground and background
    result = cv2.add(foreground, background)

    # Debugging: Save the final result
    cv2.imwrite("debug_final_result.png", result)

    return result
