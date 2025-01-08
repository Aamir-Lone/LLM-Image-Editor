
# from fastapi import FastAPI, File, Form, UploadFile
# import logging

# app = FastAPI()

# # Configure logging
# logging.basicConfig(level=logging.INFO)

# @app.post("/process-image")
# async def process_image(file: UploadFile = File(...), task: str = Form(...)):
#     try:
#         # Log the received file and task
#         logging.info(f"File Name: {file.filename}")
#         logging.info(f"Content Type: {file.content_type}")
#         logging.info(f"Task: {task}")

#         # Check if the file or task is missing
#         if not file or file.filename == "":
#             return {"error": "No file uploaded"}
#         if not task:
#             return {"error": "No task provided"}

#         # Read the file's content (optional for debugging)
#         file_content = await file.read()
#         logging.info(f"File Size: {len(file_content)} bytes")

#         # Return a success response
#         return {
#             "message": "File received successfully",
#             "filename": file.filename,
#             "content_type": file.content_type,
#             "task": task,
#         }

#     except Exception as e:
#         logging.error(f"Error while processing: {e}")
#         return {"error": str(e)}




from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
import shutil
from image_processor import process_image_logic

app = FastAPI()

@app.post("/process-image")
async def process_image(file: UploadFile = File(...), task: str = Form(...)):
    temp_file_path = f"/tmp/{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Process the image
    processed_image_path = f"/tmp/processed_{file.filename}"
    process_image_logic(temp_file_path, processed_image_path, task)

    # Return processed image
    return FileResponse(processed_image_path, media_type="image/jpeg")

from fastapi import FastAPI
from image_processor import process_image_logic

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Image Processing API"}

@app.post("/process-image")
async def process_image(file: bytes, task: str):
    return process_image_logic(file, task)

