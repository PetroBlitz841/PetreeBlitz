from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict
import pandas as pd
import os
import uuid
from datetime import datetime

from model.inference import TreeIdentifier
from model.model_loader import load_resnet18, get_transform
from model.learning import rebuild_identifier_with_learning
from model.utils.loader import populate_albums_from_df

# ===== Global State =====
identifier = None
model = None
transform = None

# ===== In-memory storage =====
SAMPLES: Dict = {}
ALBUMS: Dict = {}

# ===== Pydantic Models =====
class PredictionModel(BaseModel):
    label: str
    confidence: float


class FeedbackModel(BaseModel):
    was_correct: Optional[bool] = None
    correct_label: Optional[str] = None


class FeedbackRequestModel(BaseModel):
    sample_id: str
    was_correct: bool
    correct_label: Optional[str] = None


class FeedbackResponseModel(BaseModel):
    status: str


class ImageResponseModel(BaseModel):
    sample_id: str
    image_url: str
    predictions: List[PredictionModel]
    feedback: Optional[FeedbackModel] = None
    timestamp: str


class AlbumModel(BaseModel):
    album_id: str
    name: str
    num_images: int


# ===== FastAPI App =====
app = FastAPI(title="PetreeBlitz API")

# ===== Static files =====
if os.path.exists('data/patches'):
    app.mount("/patches", StaticFiles(directory="./data/patches"), name="patches")
if os.path.exists('data/Trees'):
    app.mount("/trees", StaticFiles(directory="./data/Trees"), name="trees")


# ===== API Endpoints =====
@app.post("/identify", response_model=dict)
async def identify_image(file: UploadFile = File(...)):
    """Identify a tree image and return predictions."""
    if not identifier:
        return JSONResponse(status_code=503, content={"detail": "Model not ready"})
    
    image_bytes = await file.read()
    predictions = identifier.identify(image_bytes)
    
    sample_id = str(uuid.uuid4())
    # Store the image bytes so we can recompute embeddings when learning from feedback
    SAMPLES[sample_id] = {
        "image_bytes": image_bytes,
        "predictions": predictions,
        "feedback": None,
        "timestamp": __import__('time').time(),
        "album_id": None  # Will be set when feedback is received
    }
    
    return {
        "predictions": predictions,
        "sample_id": sample_id
    }


@app.post("/feedback", response_model=FeedbackResponseModel)
async def submit_feedback(feedback: FeedbackRequestModel):
    """Submit feedback for a sample to help the model learn."""
    global identifier
    
    sample_id = feedback.sample_id
    
    # Check if sample exists
    if sample_id not in SAMPLES:
        return JSONResponse(
            status_code=404,
            content={"status": "not found"}
        )
    
    sample = SAMPLES[sample_id]
    
    # Store feedback
    sample["feedback"] = {
        "was_correct": feedback.was_correct,
        "correct_label": feedback.correct_label
    }
    
    should_rebuild = False
    correct_label = None
    
    # Learn from corrected predictions
    if not feedback.was_correct and feedback.correct_label and sample.get("image_bytes"):
        should_rebuild = True
        correct_label = feedback.correct_label.replace(" ", "_")
        sample["album_id"] = correct_label
        
        # Create or update album
        if correct_label not in ALBUMS:
            ALBUMS[correct_label] = {"name": feedback.correct_label, "sample_ids": []}
        
        if sample_id not in ALBUMS[correct_label]["sample_ids"]:
            ALBUMS[correct_label]["sample_ids"].append(sample_id)
    
    # Also learn from correct predictions to reinforce them
    elif feedback.was_correct and sample.get("predictions") and sample.get("image_bytes"):
        # Get the top predicted label
        if sample["predictions"]:
            top_prediction = sample["predictions"][0]
            correct_label = top_prediction["label"].replace(" ", "_")
            should_rebuild = True
            
            # Ensure album exists
            if correct_label not in ALBUMS:
                ALBUMS[correct_label] = {"name": top_prediction["label"], "sample_ids": []}
            
            if sample_id not in ALBUMS[correct_label]["sample_ids"]:
                ALBUMS[correct_label]["sample_ids"].append(sample_id)
    
    # Rebuild identifier if we learned something
    if should_rebuild and correct_label:
        new_identifier = rebuild_identifier_with_learning(
            sample_id, sample["image_bytes"], correct_label, SAMPLES, ALBUMS, model, transform
        )
        
        if new_identifier:
            identifier = new_identifier
            print(f"Identifier rebuilt with feedback. Now using {len(identifier.known_embeddings)} embeddings")
            return FeedbackResponseModel(status="success")
        else:
            print(f"Failed to rebuild identifier")
            return JSONResponse(status_code=500, content={"status": "failed"})
    
    return FeedbackResponseModel(status="success")

@app.get("/albums", response_model=List[AlbumModel])
def get_albums():
    """Get all albums with image counts."""
    result = []
    for album_id in ALBUMS:
        # Count unique samples from CSV (those with image_path)
        csv_samples = set()
        for sample_id in ALBUMS[album_id]["sample_ids"]:
            if "image_path" in SAMPLES[sample_id]:
                csv_samples.add(SAMPLES[sample_id]["image_path"])
        result.append(AlbumModel(
            album_id=album_id,
            name=ALBUMS[album_id]["name"],
            num_images=len(csv_samples)
        ))
    return result


@app.get("/albums/{album_id}/images", response_model=List[ImageResponseModel])
def get_album_images(album_id: str):
    """Get all images in an album."""
    if album_id not in ALBUMS:
        return JSONResponse(status_code=404, content={"detail": "Album not found"})
    
    result = []
    for sample_id in ALBUMS[album_id]["sample_ids"]:
        s = SAMPLES[sample_id]
        # Skip samples without image_path (these are learned embeddings)
        if "image_path" not in s:
            continue
        
        predictions = [
            PredictionModel(label=p["label"], confidence=p["confidence"])
            for p in s.get("predictions", [])
        ]
        feedback = None
        if s.get("feedback"):
            feedback = FeedbackModel(
                was_correct=s["feedback"].get("was_correct"),
                correct_label=s["feedback"].get("correct_label")
            )
        
        timestamp = datetime.fromtimestamp(s["timestamp"]).isoformat() + "Z"
        result.append(ImageResponseModel(
            sample_id=sample_id,
            image_url=s["image_path"],
            predictions=predictions,
            feedback=feedback,
            timestamp=timestamp
        ))
    
    return result


# ===== Startup Event =====
@app.on_event("startup")
async def startup_event():
    """Initialize model and data on application startup."""
    global identifier, model, transform
    
    print("path: ", os.getcwd())
    csv_path = 'data/tree_patches_with_clusters.csv'
    
    # Load model and transform once for use in learning
    model = load_resnet18()
    transform = get_transform()
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        populate_albums_from_df(df, ALBUMS, SAMPLES)
        
        # Initialize the identifier only if data is loaded
        identifier = TreeIdentifier(model, transform, SAMPLES)
        
        print(f"Loaded {len(SAMPLES)} samples into {len(ALBUMS)} albums")
    else:
        print(f"WARNING: {csv_path} not found. Identification will be disabled.")
