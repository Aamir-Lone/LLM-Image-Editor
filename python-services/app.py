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
import os

# Configuration
MODEL_PATH = "models/sam_vit_h_4b8939.pth"
YOLO_MODEL = "yolov8n.pt"

def main():
    st.title("NLP-Powered Image Editor")
    st.markdown("Upload an image and describe your edits using natural language")
    
    # File upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read and display image
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        st.image(image_np, caption="Original Image", use_column_width=True)
        
        # Get user prompt
        prompt = st.text_input("Describe your edits (e.g., 'Remove the car on the left'):")
        
        if prompt:
            with st.spinner("Processing your request..."):
                try:
                    # Step 1: Object detection
                    boxes, class_names = detect_objects(image_np)
                    
                    # Step 2: Object segmentation
                    masks, segmented_image = segment_objects(image_np, boxes)
                    
                    # Step 3: Prompt interpretation
                    action_interpreter = pipeline(
                        "zero-shot-classification",
                        model="typeform/distilbert-base-uncased-mnli"
                    )
                    
                    # Define possible actions
                    actions = [
                        "remove object", "blur background", 
                        "remove background", "highlight objects",
                        "color adjustment", "other"
                    ]
                    
                    # Get action prediction
                    action_result = action_interpreter(prompt, actions)
                    selected_action = action_result['labels'][0]
                    
                    # Display processing steps
                    with st.expander("Processing Details"):
                        st.write("Detected Objects:", class_names)
                        st.write("Interpreted Action:", selected_action)
                        st.image(segmented_image, caption="Segmented Objects")
                    
                    # Perform the action
                    if "remove object" in selected_action:
                        object_interpreter = pipeline(
                            "zero-shot-classification",
                            model="facebook/bart-large-mnli"
                        )
                        object_labels = list(set(class_names))  # Use detected classes
                        object_result = object_interpreter(prompt, object_labels)
                        target_object = object_result['labels'][0]
                        
                        if target_object in class_names:
                            object_index = class_names.index(target_object)
                            result = remove_object(image_np, masks, object_index)
                        else:
                            st.error(f"Could not find {target_object} in the image")
                            return
                            
                    elif "blur background" in selected_action:
                        result = blur_background(image_np, masks)
                        
                    elif "remove background" in selected_action:
                        result = remove_background(image_np, masks)
                        
                    elif "highlight objects" in selected_action:
                        result = overlay_masks(image_np, masks)
                        
                    else:
                        st.error("This action is not yet supported")
                        return
                    
                    # Display result
                    st.subheader("Edited Image")
                    st.image(result, use_column_width=True)
                    st.success("Processing complete!")
                    
                    # Add download button
                    result_pil = Image.fromarray(result)
                    st.download_button(
                        label="Download Result",
                        data=cv2.imencode('.png', result)[1].tobytes(),
                        file_name="edited_image.png",
                        mime="image/png"
                    )
                    
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()