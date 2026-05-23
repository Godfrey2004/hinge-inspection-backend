import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# Import detector logic
from detector import detector

# Initialize FastAPI
app = FastAPI(
    title="AI Hinge Inspection API",
    description="Backend for the AI-powered industrial hinge inspection system.",
    version="1.0.0"
)

# CORS Middleware Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://hinge-inspection-frontend.vercel.app",
        "https://hinge-inspection-frontend-grryc3foq-godfrey-j-projects8.vercel.app",
        "http://localhost:5173",  # Local dev
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories for managing videos
UPLOAD_DIR = "uploads"
OUTPUTS_DIR = "outputs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


@app.get("/")
async def health_check():
    """
    Health check endpoint to verify backend status.
    """
    return {"message": "AI Hinge Inspection Backend Running"}


@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """
    Accepts video uploads, validates the format, saves it, and runs YOLO AI processing.
    Returns the analytics JSON response.
    """
    # Validate extension
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Only .mp4, .avi, and .mov are allowed."
        )

    try:
        # Generate a unique filename to prevent collisions
        file_ext = os.path.splitext(file.filename)[1]
        unique_id = uuid.uuid4().hex
        input_filename = f"{unique_id}{file_ext}"
        output_filename = f"processed_{unique_id}.mp4"
        
        input_path = os.path.join(UPLOAD_DIR, input_filename)
        output_path = os.path.join(OUTPUTS_DIR, output_filename)

        # Save the uploaded file
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run AI processing
        analytics_result = await detector.process_video(input_path, output_path)

        # The analytics_result already matches the requested JSON schema:
        # { "left_hinge": "OK", "right_hinge": "MISSING", "inspection": "FAIL", "total_hinges": 1, "confidence": 98, "output_video": "..." }
        return analytics_result

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing the video: {str(e)}"
        )


@app.get("/processed-video/{filename}")
async def get_processed_video(filename: str):
    """
    Serves the processed video output to the frontend.
    """
    file_path = os.path.join(OUTPUTS_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Processed video not found"
        )
        
    return FileResponse(
        path=file_path,
        media_type="video/mp4",
        filename=filename
    )
