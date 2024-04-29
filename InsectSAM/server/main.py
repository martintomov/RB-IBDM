from torch.nn.functional import threshold, normalize
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI(title="InsectSAM API",
              version="0.0.1",
              description="A simple api server using FastAPI interface and Pydantic data validation")

# To all CORS enabled origins
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
#     expose_headers=["*"],
# )


@app.post("/get_cuda_availability")
def check_cuda_availability():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {"device": device}
@app.post("/insectsam_dino")
def insectsam_grounding_dino_inference(prompt: str, insect_photo: UploadFile):
    # Get the file extension
    file_extension = os.path.splitext(insect_photo.filename)[1]

    # Generate a unique file name
    file_name = f"temp{file_extension}"

    # Save the file locally
    with open(file_name, "wb") as file:
        file.write(insect_photo.file.read())
    #load filename"

    segmented_array = []
    segmented_picture = []

    return {"file_name": file_name, "segments_array": segmented_array, "segmented_picture": segmented_picture}

@app.post("/insectsam")
def insectsam_inference(prompt: str, insect_photo: UploadFile):
    # Get the file extension
    file_extension = os.path.splitext(insect_photo.filename)[1]

    # Generate a unique file name
    file_name = f"temp{file_extension}"

    # Save the file locally
    with open(file_name, "wb") as file:
        file.write(insect_photo.file.read())
    #load filename"

    segmented_array = []
    segmented_picture = []

    return {"file_name": file_name, "segments_array": segmented_array, "segmented_picture": segmented_picture}
# Redirect root to docs lets users see the documentation when they visit the root of the server
@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


## Run the server
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
