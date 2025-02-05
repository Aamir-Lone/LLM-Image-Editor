# # modules/sam_processor.py
# from segment_anything import SamPredictor, sam_model_registry
# import cv2
# import os
# import numpy as np
# # Path to the SAM checkpoint in the models folder
# # SAM_CHECKPOINT = os.path.join(os.path.dirname(__file__), "../models/sam_vit_h_4b8939.pth")
# SAM_CHECKPOINT = r"C:\LLM_image_editor\python-services\models\sam_vit_h_4b8939.pth"
# # Load the SAM model from the checkpoint

# def load_sam_model():
#     """
#     Loads the SAM model.

#     Returns:
#         SamPredictor: The loaded SAM predictor.
#     """
#     sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
#     return SamPredictor(sam)

# def segment_objects(image_path, boxes):
#     """
#     Segments objects in the given image using SAM and bounding boxes from YOLO.

#     Args:
#         image_path (str): Path to the input image.
#         boxes (list): Bounding boxes in the format [x_min, y_min, x_max, y_max].

#     Returns:
#         list: Masks for each bounding box.
#         numpy.ndarray: Original image.
#     """
#     # Load image
#     image = cv2.imread(image_path)

#     # Load SAM model
#     predictor = load_sam_model()
#     predictor.set_image(image)

#     # Segment objects
#     masks = []
#     for box in boxes:
#         input_box = np.array(box).reshape(1, -1)  # Convert to [1, 4] format
#         mask, _, _ = predictor.predict(box=input_box)
#         masks.append(mask.astype(np.uint8))  # Ensure masks are uint8

#     return masks, image

#***************************************************************************************************************************
# modules/sam_processor.py
from segment_anything import SamPredictor, sam_model_registry
import cv2
import numpy as np
SAM_CHECKPOINT = r"C:\LLM_image_editor\python-services\models\sam_vit_h_4b8939.pth"
# # Load the SAM model from the checkpoint

def load_sam_model():
#     """
#     Loads the SAM model.

#     Returns:
#         SamPredictor: The loaded SAM predictor.
#     """
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
    return SamPredictor(sam)


def segment_objects(image_input, boxes):
    """
    Segments objects in the image using SAM and bounding boxes from YOLO.

    Args:
        image_input (str or numpy.ndarray): Path to the input image or the image array.
        boxes (list): Bounding boxes from YOLO.

    Returns:
        list: Masks for each bounding box.
        numpy.ndarray: Original image.
    """
    # Load image (works for both file paths and NumPy arrays)
    if isinstance(image_input, str):
        image = cv2.imread(image_input)
    else:
        image = image_input.copy()

    # Load SAM model
    predictor = load_sam_model()
    predictor.set_image(image)

    # Segment objects
    masks = []
    for box in boxes:
        input_box = np.array(box).reshape(1, -1)
        mask, _, _ = predictor.predict(box=input_box)
        masks.append(mask.astype(np.uint8))

    return masks, image