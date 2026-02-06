"""Data loading utilities for populating albums from CSV."""

import os
import uuid
import numpy as np


def add_sample_to_album(albums, samples, album_id, album_name, sample_id, embedding, predictions, image_path):
    """Add a sample to an album."""
    if album_id not in albums:
        albums[album_id] = {"name": album_name, "sample_ids": []}
    albums[album_id]["sample_ids"].append(sample_id)
    samples[sample_id] = {
        "embedding": embedding,
        "predictions": predictions,
        "feedback": None,
        "image_path": image_path,
        "timestamp": __import__('time').time(),
        "album_id": album_id
    }


def populate_albums_from_df(df, albums, samples):
    """Populate albums and samples from a CSV DataFrame."""
    # Extract specie name from original_name for each row
    df['specie_name'] = df['original_name'].apply(
        lambda x: x.rsplit(' ', 1)[0] if ' ' in x else x.replace('.jpg', '')
    )
    
    # Group by specie name to create one album per species
    for specie_name in df['specie_name'].unique():
        specie_df = df[df['specie_name'] == specie_name]
        album_id = specie_name.replace(' ', '_')
        album_name = specie_name
        
        for _, row in specie_df.iterrows():
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
            
            original_name = row['original_name']
            
            # Find the actual tree image in the Trees directory
            tree_image_path = os.path.join('data/Trees', original_name)
            image_url = f"/trees/{original_name}" if os.path.exists(tree_image_path) else "/trees/placeholder.png"
            
            predictions = [{"label": specie_name, "confidence": 0.9}]
            add_sample_to_album(albums, samples, album_id, album_name, sample_id, embedding, predictions, image_url)
