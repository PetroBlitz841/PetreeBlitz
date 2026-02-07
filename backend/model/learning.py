"""Model learning utilities for handling feedback and model improvements."""

import time
import uuid
import numpy as np
import pickle
from PIL import Image
from io import BytesIO
import torch
from datetime import datetime
from sqlalchemy.orm import Session

from model.inference import TreeIdentifier
from db import Sample, Album, Embedding


def compute_patch_embeddings(image_bytes, album_id, model, transform):
    """
    Compute embeddings from image bytes for all 16 patches.
    
    Args:
        image_bytes: Raw image data
        album_id: Label for the patches
        model: PyTorch model for embedding
        transform: Image transform pipeline
        
    Returns:
        List of (embedding, label) tuples
    """
    if not model or not transform:
        raise RuntimeError("Model not initialized")
    
    # Extract 16 patches from the image
    im = Image.open(BytesIO(image_bytes)).convert('RGB')
    im = im.crop((0, 0, 896, 896))
    pw = ph = 896 // 4
    patches = []
    for r in range(4):
        for c in range(4):
            patches.append(im.crop((c*pw, r*ph, (c+1)*pw, (r+1)*ph)))
    
    # Compute embeddings for each patch
    embeddings = []
    for patch in patches:
        tensor = transform(patch).unsqueeze(0)
        with torch.no_grad():
            emb = model(tensor).squeeze().cpu().numpy()
            emb = emb.flatten()
            emb = emb / (np.linalg.norm(emb) + 1e-12)
            embeddings.append((emb, album_id))
    
    return embeddings


def add_learned_embeddings(sample_id: str, image_bytes: bytes, album_id: str, db: Session, model, transform, amplification_factor=5):
    """
    Add new embeddings from corrected feedback to the training set.
    This makes the model learn by having more examples of the correct label.
    Amplification factor multiplies the impact of user corrections.
    
    Args:
        sample_id: Original sample identifier
        image_bytes: Raw image data
        album_id: Correct label
        db: SQLAlchemy database session
        model: PyTorch model for embedding
        transform: Image transform pipeline
        amplification_factor: How many times to add the same patches (default 5)
        
    Returns:
        List of created embedding IDs if successful, None if failed
    """
    try:
        new_embeddings = compute_patch_embeddings(image_bytes, album_id, model, transform)
        
        # Get or create album
        album = db.query(Album).filter(Album.album_id == album_id).first()
        if not album:
            raise ValueError(f"Album {album_id} not found")
        
        # Create embeddings from the learned patches
        # Add them multiple times to amplify the learning effect
        created_embedding_ids = []
        for repeat in range(amplification_factor):
            for idx, (emb, label) in enumerate(new_embeddings):
                embedding_id = str(uuid.uuid4())
                embedding_bytes = pickle.dumps(emb)
                
                embedding_obj = Embedding(
                    embedding_id=embedding_id,
                    album_id=album_id,
                    original_sample_id=sample_id,
                    embedding_vector=embedding_bytes,
                    embedding_dim=len(emb),
                    is_learned=True,
                    amplification_iteration=repeat,
                    patch_index=idx
                )
                db.add(embedding_obj)
                created_embedding_ids.append(embedding_id)
        
        db.commit()
        print(f"Model learned from {len(new_embeddings)} patches × {amplification_factor} amplification = {len(created_embedding_ids)} total embeddings for sample {sample_id} -> {album_id}")
        return created_embedding_ids
    except Exception as e:
        print(f"Error learning from embeddings: {e}")
        db.rollback()
        return None


def rebuild_identifier_with_learning(sample_id: str, image_bytes: bytes, album_id: str, db: Session, model, transform):
    """
    Process feedback correction by learning and rebuilding the model.
    
    Args:
        sample_id: Original sample identifier
        image_bytes: Raw image data
        album_id: Correct label
        db: SQLAlchemy database session
        model: PyTorch model for embedding
        transform: Image transform pipeline
        
    Returns:
        New TreeIdentifier if successful, None if failed
    """
    # Add learned embeddings
    created_embeddings = add_learned_embeddings(sample_id, image_bytes, album_id, db, model, transform)
    
    if created_embeddings:
        # Rebuild identifier with new learned embeddings
        try:
            identifier = TreeIdentifier(model, transform, db)
            return identifier
        except Exception as e:
            print(f"Error rebuilding identifier: {e}")
            return None
    
    return None
