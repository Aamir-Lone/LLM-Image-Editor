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

    combined_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # Process each mask
    for i, mask in enumerate(masks):
        if mask is None:
            raise ValueError(f"Invalid mask: {mask}")

        # Handle multi-channel masks by converting to single-channel
        if len(mask.shape) == 3:
            mask = np.max(mask, axis=0)  # Collapse multi-channel mask to single-channel

        # Visualize mask for debugging
        print(f"Mask {i} sum: {np.sum(mask)}")  # Print sum of mask to verify non-zero values
        cv2.imwrite(f"mask_{i}.png", mask * 255)  # Save mask image for visual inspection

        if len(mask.shape) != 2:
            raise ValueError(f"Processed mask is not 2D: {mask.shape}")

        # Resize mask to match the image dimensions
        resized_mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        combined_mask = cv2.bitwise_or(combined_mask, resized_mask)

    # Visualize the combined mask
    print(f"Combined mask sum: {np.sum(combined_mask)}")
    cv2.imwrite("combined_mask.png", combined_mask * 255)  # Save combined mask for inspection

    # Create an inverted mask (background is 1, objects are 0)
    inverted_mask = cv2.bitwise_not(combined_mask)

    # Create a background image
    background = np.full_like(image, background_color, dtype=np.uint8)

    # Mask the original image and the background
    foreground = cv2.bitwise_and(image, image, mask=combined_mask)
    background = cv2.bitwise_and(background, background, mask=inverted_mask)

    # Combine the foreground and background
    result = cv2.add(foreground, background)

    # If you want transparency instead of a solid background color, you could add an alpha channel.
    # result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    # result[:, :, 3] = combined_mask  # Add the mask as alpha channel for transparency

    # Visualize the final result
    print(f"Foreground sum: {np.sum(foreground)}")
    print(f"Background sum: {np.sum(background)}")

    return result
