"""Data loading utilities for populating albums from CSV."""

import os
import uuid
import numpy as np
import pickle
from datetime import datetime
from sqlalchemy.orm import Session
from db import Album, Sample, Embedding


def add_sample_to_album_db(db: Session, album_id: str, album_name: str, sample_id: str, embedding: np.ndarray, predictions: list, image_path: str):
    """Add a sample to an album in the database."""
    # Create or get album
    album = db.query(Album).filter(Album.album_id == album_id).first()
    if not album:
        album = Album(album_id=album_id, name=album_name)
        db.add(album)
        db.flush()
    
    # Create sample
    sample = Sample(
        sample_id=sample_id,
        album_id=album_id,
        image_path=image_path,
        predictions=predictions,
        timestamp=datetime.utcnow()
    )
    db.add(sample)
    db.flush()
    
    # Store embedding
    embedding_id = str(uuid.uuid4())
    embedding_bytes = pickle.dumps(embedding)
    
    embedding_obj = Embedding(
        embedding_id=embedding_id,
        album_id=album_id,
        original_sample_id=sample_id,
        embedding_vector=embedding_bytes,
        embedding_dim=len(embedding),
        is_learned=False
    )
    db.add(embedding_obj)


def populate_albums_from_db(csv_path: str, db: Session):
    """Populate albums and samples from a CSV file into the database."""
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    
    # Extract specie name from original_name for each row
    df['specie_name'] = df['original_name'].apply(
        lambda x: x.rsplit(' ', 1)[0] if ' ' in x else x.replace('.jpg', '')
    )
    
    # Track which images we've already added to avoid duplicates within this load
    added_images = set()
    
    # Pre-load existing samples to prevent database duplicates on re-runs
    existing_samples = set()
    existing_sample_records = db.query(Sample).all()
    for record in existing_sample_records:
        if record.image_path:
            existing_samples.add(record.image_path)
    
    samples_added = 0
    
    # Group by specie name to create one album per species
    for specie_name in df['specie_name'].unique():
        specie_df = df[df['specie_name'] == specie_name]
        album_id = specie_name.replace(' ', '_')
        album_name = specie_name
        
        for _, row in specie_df.iterrows():
            original_name = row['original_name']
            
            # Skip if we've already added this image to this album within this load
            image_key = (album_id, original_name)
            if image_key in added_images:
                continue
            
            # Skip if this image is already in the database
            image_url = f"/trees/{original_name}"
            if image_url in existing_samples:
                continue
            
            added_images.add(image_key)
            
            sample_id = str(uuid.uuid4())
            # Parse embedding from string representation in CSV
            embedding_str = row.get('normalized_embedding') or row.get('embedding')
            try:
                # Parse the string representation of the numpy array
                # Clean up whitespace and newlines
                embedding_str = embedding_str.replace('\n', ' ').strip()
                # Remove brackets and split by whitespace
                embedding_str = embedding_str.strip('[]')
                embedding = np.array([float(x) for x in embedding_str.split() if x])
                if len(embedding) == 0:
                    print(f"Warning: Empty embedding for {sample_id}, skipping")
                    continue
            except (ValueError, AttributeError, TypeError) as e:
                print(f"Warning: Could not parse embedding for {sample_id}: {e}, skipping")
                continue
            
            # Find the actual tree image in the Trees directory
            tree_image_path = os.path.join('data/Trees', original_name)
            image_path = f"/trees/{original_name}" if os.path.exists(tree_image_path) else "/trees/placeholder.png"
            
            predictions = [{"label": specie_name, "confidence": 0.9}]
            add_sample_to_album_db(db, album_id, album_name, sample_id, embedding, predictions, image_path)
            samples_added += 1
    
    db.commit()
    print(f"Populated database from {csv_path}: Added {samples_added} samples")
