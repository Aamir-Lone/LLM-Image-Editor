import cv2

def process_image_logic(input_path: str, output_path: str, task: str):
    image = cv2.imread(input_path)

    if task == "blur background":
        # Example: Apply Gaussian blur
        blurred = cv2.GaussianBlur(image, (51, 51), 0)
        cv2.imwrite(output_path, blurred)
    elif task == "remove background":
        # Add logic for background removal
        pass
    else:
        raise ValueError(f"Unsupported task: {task}")
