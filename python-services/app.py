import streamlit as st
import cv2
import numpy as np
from PIL import Image
from modules.yolo_processor import detect_objects
from modules.sam_processor import segment_objects
from modules.background_removal import remove_background
from modules.background_blur import blur_background
from modules.overlay_masks import overlay_masks
from modules.remove_object import remove_object
from transformers import pipeline

# Title of the app
st.title("NLP-Based Image Editor")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Read the image
    image = Image.open(uploaded_file)
    image = np.array(image)  # Convert to NumPy array for processing
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Input prompt
    prompt = st.text_input("Enter your prompt (e.g., 'blur the background'):")

    if prompt:
        # Step 1: Detect objects using YOLO
        boxes, class_names = detect_objects(image)
        st.write("Detected objects:", class_names)

        # Step 2: Segment objects using SAM
        masks, segmented_image = segment_objects(image, boxes)
        st.image(segmented_image, caption="Segmented Image", use_column_width=True)

        # Step 3: Interpret the prompt using the LLM
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        candidate_labels = [
            "remove an object", "blur the background", "remove the background", 
            "highlight objects with masks", "change object color", "other editing"
        ]
        result = classifier(prompt, candidate_labels)
        action = result['labels'][0]
        st.write(f"Action: {action}")

        # Step 4: Perform the action
        if action == "remove an object":
            # Extract object name from the prompt
            object_name = classifier(prompt, candidate_labels=["person", "car", "dog", "cat", "other"])['labels'][0]
            st.write(f"Object to remove: {object_name}")

            # Find the object index
            object_index = -1
            for i, name in enumerate(class_names):
                if name == object_name:
                    object_index = i
                    break

            if object_index == -1:
                st.error(f"Object '{object_name}' not found in the image.")
            else:
                # Remove the object
                result_image = remove_object(image, masks, object_index)
                st.image(result_image, caption="Result Image", use_column_width=True)

        elif action == "blur the background":
            result_image = blur_background(image, masks)
            st.image(result_image, caption="Blurred Background", use_column_width=True)

        elif action == "remove the background":
            result_image = remove_background(image, masks)
            st.image(result_image, caption="Background Removed", use_column_width=True)

        elif action == "highlight objects with masks":
            result_image = overlay_masks(image, masks)
            st.image(result_image, caption="Highlighted Objects", use_column_width=True)

        else:
            st.error("Unsupported action.")