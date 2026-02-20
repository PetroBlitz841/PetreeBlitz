from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from typing import List, Optional
import pandas as pd
import os
import uuid
from datetime import datetime
import numpy as np
from sqlalchemy.orm import Session

from db import init_db, get_db, Album, Sample, Feedback, Embedding, SessionLocal
from model.inference import TreeIdentifier
from model.model_loader import load_resnet18, get_transform
from model.learning import update_identifier_with_learning
from model.utils.loader import populate_albums_from_db

# ===== Global State =====
identifier = None
model = None
transform = None

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
# Ensure folders exist so StaticFiles can be mounted reliably
os.makedirs(os.path.join('data', 'patches'), exist_ok=True)
os.makedirs(os.path.join('data', 'Trees'), exist_ok=True)

# Mount static directories for serving images
app.mount("/patches", StaticFiles(directory="./data/patches"), name="patches")
app.mount("/trees", StaticFiles(directory="./data/Trees"), name="trees")

def get_image_url(path: str) -> str:
    """
    Convert a local path like 'data/patches/tree1.jpg' to a URL like '/patches/tree1.jpg'
    """
    if path.startswith("data/patches"):
        filename = os.path.basename(path)
        return f"/patches/{filename}"
    elif path.startswith("data/Trees"):
        filename = os.path.basename(path)
        return f"/trees/{filename}"
    else:
        return path  # fallback


# ===== API Endpoints =====
@app.post("/identify", response_model=dict)
async def identify_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Identify a tree image and return predictions."""
    if not identifier:
        return JSONResponse(status_code=503, content={"detail": "Model not ready"})
    
    image_bytes = await file.read()
    predictions = identifier.identify(image_bytes)
    
    sample_id = str(uuid.uuid4())

    # Ensure patches directory exists and save the uploaded file so it can be shown in albums
    try:
        patches_dir = os.path.join('data', 'patches')
        os.makedirs(patches_dir, exist_ok=True)

        # Determine filename extension from uploaded filename
        _, ext = os.path.splitext(file.filename or '')
        if not ext:
            ext = '.jpg'

        filename = f"{sample_id}{ext}"
        file_fs_path = os.path.join(patches_dir, filename)
        with open(file_fs_path, 'wb') as f:
            f.write(image_bytes)

        # Store a web-accessible image path in the DB (consistent with CSV loader '/trees/...')
        image_path_db = f"/patches/{filename}"
    except Exception as e:
        # If saving fails, still store the raw bytes but leave image_path empty
        print(f"Warning: failed to save uploaded patch file: {e}")
        image_path_db = None

    # Store sample in database
    db_sample = Sample(
        sample_id=sample_id,
        image_bytes=image_bytes,
        image_path=image_path_db,
        predictions=predictions,
        timestamp=datetime.utcnow()
    )
    db.add(db_sample)
    db.commit()

    return {
        "predictions": predictions,
        "sample_id": sample_id
    }


@app.post("/feedback", response_model=FeedbackResponseModel)
async def submit_feedback(feedback: FeedbackRequestModel, db: Session = Depends(get_db)):
    """Submit feedback for a sample to help the model learn."""
    global identifier
    
    sample_id = feedback.sample_id
    
    # Check if sample exists
    sample = db.query(Sample).filter(Sample.sample_id == sample_id).first()
    if not sample:
        return JSONResponse(
            status_code=404,
            content={"status": "not found"}
        )
    
    # Store feedback in database
    feedback_id = str(uuid.uuid4())
    db_feedback = Feedback(
        feedback_id=feedback_id,
        sample_id=sample_id,
        was_correct=feedback.was_correct,
        correct_label=feedback.correct_label
    )
    db.add(db_feedback)
    
    should_rebuild = False
    correct_label = None
    
    # Learn from corrected predictions
    if not feedback.was_correct and feedback.correct_label and sample.image_bytes:
        should_rebuild = True
        correct_label = feedback.correct_label.replace(" ", "_")
        
        # Create or get album
        album = db.query(Album).filter(Album.album_id == correct_label).first()
        if not album:
            album = Album(album_id=correct_label, name=feedback.correct_label)
            db.add(album)
            db.flush()
        
        sample.album_id = correct_label

        # If the sample has image bytes but no stored image_path (e.g., recently identified),
        # save the image to the patches directory so it appears in album image lists.
        try:
            if sample.image_bytes and not sample.image_path:
                patches_dir = os.path.join('data', 'patches')
                os.makedirs(patches_dir, exist_ok=True)
                filename = f"{sample.sample_id}.jpg"
                file_fs_path = os.path.join(patches_dir, filename)
                with open(file_fs_path, 'wb') as f:
                    f.write(sample.image_bytes)
                sample.image_path = f"/patches/{filename}"
        except Exception as e:
            print(f"Warning: failed to persist sample image on feedback: {e}")
    
    # Also learn from correct predictions to reinforce them
    elif feedback.was_correct and sample.predictions:
        if sample.predictions:
            top_prediction = sample.predictions[0]
            correct_label = top_prediction["label"].replace(" ", "_")
            should_rebuild = True
            
            # Create or get album
            album = db.query(Album).filter(Album.album_id == correct_label).first()
            if not album:
                album = Album(album_id=correct_label, name=top_prediction["label"])
                db.add(album)
                db.flush()
            
            sample.album_id = correct_label

            # Ensure image_path exists for display in albums
            try:
                if sample.image_bytes and not sample.image_path:
                    patches_dir = os.path.join('data', 'patches')
                    os.makedirs(patches_dir, exist_ok=True)
                    filename = f"{sample.sample_id}.jpg"
                    file_fs_path = os.path.join(patches_dir, filename)
                    with open(file_fs_path, 'wb') as f:
                        f.write(sample.image_bytes)
                    sample.image_path = f"/patches/{filename}"
            except Exception as e:
                print(f"Warning: failed to persist sample image on feedback: {e}")
    
    db.commit()
    
    # Rebuild identifier if we learned something
    if should_rebuild and correct_label:
        new_identifier = update_identifier_with_learning(
            identifier, sample_id, sample.image_bytes, correct_label, db, model, transform
        )
        
        if new_identifier:
            identifier = new_identifier
            samples_count = db.query(Sample).count()
            print(f"Identifier rebuilt with feedback. Using {samples_count} samples")
            return FeedbackResponseModel(status="success")
        else:
            print(f"Failed to rebuild identifier")
            return JSONResponse(status_code=500, content={"status": "failed"})
    
    return FeedbackResponseModel(status="success")

@app.get("/albums", response_model=List[AlbumModel])
def get_albums(db: Session = Depends(get_db)):
    """Get all albums with image counts."""
    albums = db.query(Album).all()
    result = []
    
    for album in albums:
        # Count samples in this album
        num_images = db.query(Sample).filter(Sample.album_id == album.album_id).count()
        result.append(AlbumModel(
            album_id=album.album_id,
            name=album.name,
            num_images=num_images
        ))
    
    return result


@app.get("/albums/{album_id}/images", response_model=List[ImageResponseModel])
def get_album_images(album_id: str, db: Session = Depends(get_db)):
    """Get all images in an album."""
    album = db.query(Album).filter(Album.album_id == album_id).first()
    if not album:
        return JSONResponse(status_code=404, content={"detail": "Album not found"})
    
    samples = db.query(Sample).filter(Sample.album_id == album_id).all()
    result = []
    
    for sample in samples:
        # Skip samples without image_path (learned embeddings without original images)
        if not sample.image_path:
            continue
        
        predictions = [
            PredictionModel(label=p["label"], confidence=p["confidence"])
            for p in sample.predictions or []
        ]
        
        feedback = None
        if sample.feedback:
            feedback = FeedbackModel(
                was_correct=sample.feedback.was_correct,
                correct_label=sample.feedback.correct_label
            )
        
        timestamp = sample.timestamp.isoformat() + "Z" if sample.timestamp else ""
        result.append(ImageResponseModel(
            sample_id=sample.sample_id,
            image_url=get_image_url(sample.image_path),
            predictions=predictions,
            feedback=feedback,
            timestamp=timestamp
        ))
    
    return result


# ===== Startup Event =====
@app.on_event("startup")
async def startup_event():
    """Initialize database and model on application startup."""
    global identifier, model, transform
    
    print("path: ", os.getcwd())
    
    # Initialize database
    init_db()
    
    csv_path = 'data/tree_patches_with_clusters.csv'
    
    # Load model and transform once for use in learning
    model = load_resnet18()
    transform = get_transform()
    
    # Use SessionLocal for operations
    db = SessionLocal()
    try:
        # Check if database is already populated
        existing_albums = db.query(Album).count()
        
        if existing_albums == 0 and os.path.exists(csv_path):
            # Database is empty, load from CSV
            print(f"Loading data from {csv_path}...")
            populate_albums_from_db(csv_path, db)
            print("CSV data loaded into database")
        elif existing_albums > 0:
            print(f"Database already populated with {existing_albums} albums, skipping CSV load")
        else:
            print(f"WARNING: {csv_path} not found and database is empty. Identification will be disabled.")
        
        # Count loaded data
        samples_count = db.query(Sample).count()
        albums_count = db.query(Album).count()
        
        # Initialize the identifier only if data is loaded
        if albums_count > 0:
            identifier = TreeIdentifier(model, transform, db)
            print(f"Identifier initialized with {samples_count} samples from {albums_count} albums")
        else:
            print("No data available. Identifier will not be initialized.")
    finally:
        db.close()
