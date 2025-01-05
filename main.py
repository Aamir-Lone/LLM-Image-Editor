import cv2
import matplotlib.pyplot as plt # for displaying in notebooks
from transformers import pipeline

def display_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # OpenCV uses BGR, matplotlib uses RGB
    plt.imshow(img)
    plt.axis('off') # Hide axis ticks and numbers
    plt.show()

image_path = "C:/LLM image editor/what-would-tony-stark-think-of-john-walker-v0-28mug8fgrexc1.webp"  # Replace with your image path
# display_image(image_path)

# import cv2

def adjust_brightness(image, factor):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = cv2.multiply(hsv[:,:,2], factor) # Adjust V channel (brightness)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def adjust_contrast(image, factor):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    adjusted_image = cv2.convertScaleAbs(image, alpha=factor, beta=0)
    return adjusted_image

def apply_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Example usage:
img = cv2.imread(image_path)
brighter_image = adjust_brightness(img, 1.5) # Increase brightness by 50%
contrast_image = adjust_contrast(img, 1.2)
gray_image = apply_grayscale(img)
cv2.imwrite("brighter.jpg", brighter_image)
cv2.imwrite("contrast.jpg", contrast_image)
cv2.imwrite("gray.jpg", gray_image)



# Choose a suitable model. This is a smaller, faster one.
text_generation = pipeline("text-generation", model="distilgpt2")

command_map = {
    "make brighter": adjust_brightness,
    "increase brightness": adjust_brightness,
    "make darker": adjust_brightness, # We'll need to handle the factor differently
    "make grayscale": apply_grayscale,
    "convert to grayscale": apply_grayscale,
    "increase contrast": adjust_contrast,
}

def process_command(image, command):
    for keyword, func in command_map.items():
        if keyword in command.lower():
            if func == adjust_brightness:
                factor = 1.5 if "brighter" in command.lower() else 0.5 # Example logic
                return func(image, factor)
            elif func == adjust_contrast:
                factor = 1.2
                return func(image, factor)
            else:
                return func(image)
    return image # Return original if no command matches

user_command = "make brighter"
img = cv2.imread(image_path)
edited_image = process_command(img, user_command)
cv2.imwrite("edited_image.jpg", edited_image)