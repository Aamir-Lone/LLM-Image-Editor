import cv2
import numpy as np

def blur_background(image, masks, blur_strength=(35, 35)):
    """
    Blurs the background of the input image using the masks.

    Args:
        image (numpy.ndarray): The original image.
        masks (list): List of masks for each detected object.
        blur_strength (tuple): Kernel size for Gaussian blur. Default is (25, 25).

    Returns:
        numpy.ndarray: Image with blurred background and focused foreground.
    """
    if image is None or len(image.shape) != 3:
        raise ValueError(f"Invalid image: {image}")

    # Initialize a blank combined mask
    combined_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # Combine all masks into a single binary mask
    for i, mask in enumerate(masks):
        if mask is None:
            raise ValueError(f"Invalid mask: {mask}")

        # Convert multi-channel masks to single-channel
        if len(mask.shape) == 3:
            mask = np.max(mask, axis=0)

        # Normalize mask to binary (0 or 1)
        binary_mask = (mask > 0).astype(np.uint8)

        # Combine masks using logical OR
        combined_mask = np.logical_or(combined_mask, binary_mask).astype(np.uint8)

    # Debugging: Save the combined mask
    cv2.imwrite("debug_combined_mask_blur.png", combined_mask * 255)

    # Invert the combined mask to get the background mask
    inverted_mask = cv2.bitwise_not((combined_mask * 255).astype(np.uint8))

    # Debugging: Save the inverted mask
    cv2.imwrite("debug_inverted_mask_blur.png", inverted_mask)

    # Blur the entire image
    blurred_image = cv2.GaussianBlur(image, blur_strength, 0)

    # Use the combined mask to extract the foreground from the original image
    foreground = cv2.bitwise_and(image, image, mask=combined_mask)

    # Use the inverted mask to extract the blurred background
    background = cv2.bitwise_and(blurred_image, blurred_image, mask=inverted_mask)

    # Combine the foreground and blurred background
    result = cv2.add(foreground, background)

    # Debugging: Save the final blurred background result
    cv2.imwrite("debug_blurred_background_result.png", result)

    return result
