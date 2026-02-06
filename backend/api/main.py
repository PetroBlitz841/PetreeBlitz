from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict
import pandas as pd
import os, time, uuid
from datetime import datetime
from model.inference import TreeIdentifier
from model.model_loader import load_resnet18, get_transform

Identifier = None

app = FastAPI(title="PetreeBlitz API")

# ===== Models =====
class PredictionModel(BaseModel):
    label: str
    confidence: float

class FeedbackModel(BaseModel):
    was_correct: Optional[bool] = None
    correct_label: Optional[str] = None

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

# ===== In-memory storage =====
SAMPLES: Dict = {}
ALBUMS: Dict = {}

# ===== Static files =====
if os.path.exists('./patches'):
    app.mount("/patches", StaticFiles(directory="./patches"), name="patches")
if os.path.exists('./Trees'):
    app.mount("/trees", StaticFiles(directory="./Trees"), name="trees")

# ===== API endpoints =====
@app.post("/identify", response_model=dict)
async def identify_image(file: UploadFile = File(...)):
    if not identifier:
        return JSONResponse(status_code=503, content={"detail": "Model not ready"})
    
    image_bytes = await file.read()
    result = identifier.identify(image_bytes)
    return result

@app.get("/albums", response_model=List[AlbumModel])
def get_albums():
    return [AlbumModel(album_id=a, name=ALBUMS[a]["name"], num_images=len(ALBUMS[a]["sample_ids"]))
            for a in ALBUMS]

@app.get("/albums/{album_id}/images", response_model=List[ImageResponseModel])
def get_album_images(album_id: str):
    if album_id not in ALBUMS:
        return JSONResponse(status_code=404, content={"detail":"Album not found"})
    result = []
    for sample_id in ALBUMS[album_id]["sample_ids"]:
        s = SAMPLES[sample_id]
        predictions = [PredictionModel(label=p["label"], confidence=p["confidence"]) for p in s["predictions"]]
        feedback = None
        if s["feedback"]:
            feedback = FeedbackModel(was_correct=s["feedback"].get("was_correct"),
                                     correct_label=s["feedback"].get("correct_label"))
        timestamp = datetime.fromtimestamp(s["timestamp"]).isoformat() + "Z"
        result.append(ImageResponseModel(
            sample_id=sample_id,
            image_url=s["image_path"],
            predictions=predictions,
            feedback=feedback,
            timestamp=timestamp
        ))
    return result

# ===== Populate albums from CSV =====
def add_sample_to_album(album_id, album_name, sample_id, embedding, predictions, image_path):
    if album_id not in ALBUMS:
        ALBUMS[album_id] = {"name": album_name, "sample_ids":[]}
    ALBUMS[album_id]["sample_ids"].append(sample_id)
    SAMPLES[sample_id] = {"embedding":embedding,"predictions":predictions,"feedback":None,
                          "image_path":image_path,"timestamp":time.time(),"album_id":album_id}

def populate_albums_from_df(df):
    import os, uuid
    for cluster_id in df['cluster'].unique():
        cluster_df = df[df['cluster']==cluster_id]
        album_id = f"cluster_{cluster_id}"
        album_name = f"Cluster {cluster_id}"
        for _, row in cluster_df.iterrows():
            sample_id = str(uuid.uuid4())
            embedding = row.get('normalized_embedding') or row.get('embedding')
            patch_path = row['patch_path']
            image_url = f"/{os.path.relpath(patch_path, '.').replace('\\','/')}" if os.path.exists(patch_path) else "/patches/placeholder.png"
            predictions = [{"label":"Species A","confidence":0.7},{"label":"Species B","confidence":0.3}]
            add_sample_to_album(album_id, album_name, sample_id, embedding, predictions, image_url)

@app.on_event("startup")
async def startup_event():
    print("path: ", os.getcwd())
    csv_path = 'data/tree_patches_with_clusters.csv'
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        populate_albums_from_df(df)
        
        # Initialize the helper only if data is loaded
        global identifier
        identifier = TreeIdentifier(load_resnet18(), get_transform(), SAMPLES)
        
        print(f"Loaded {len(SAMPLES)} samples into {len(ALBUMS)} albums")
    else:
        print(f"WARNING: {csv_path} not found. Identification will be disabled.")
