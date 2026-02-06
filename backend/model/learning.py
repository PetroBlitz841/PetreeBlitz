"""Model learning utilities for handling feedback and model improvements."""

import time
import numpy as np
from PIL import Image
from io import BytesIO
import torch

from model.inference import TreeIdentifier


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


def add_learned_embeddings(sample_id, image_bytes, album_id, samples, model, transform, amplification_factor=5):
    """
    Add new embeddings from corrected feedback to the training set.
    This makes the model learn by having more examples of the correct label.
    Amplification factor multiplies the impact of user corrections.
    
    Args:
        sample_id: Original sample identifier
        image_bytes: Raw image data
        album_id: Correct label
        samples: Sample storage dictionary
        model: PyTorch model for embedding
        transform: Image transform pipeline
        amplification_factor: How many times to add the same patches (default 5)
        
    Returns:
        List of created sample IDs if successful, None if failed
    """
    try:
        new_embeddings = compute_patch_embeddings(image_bytes, album_id, model, transform)
        
        # Create temporary samples from the learned embeddings
        # Add them multiple times to amplify the learning effect
        created_samples = []
        for repeat in range(amplification_factor):
            for idx, (emb, label) in enumerate(new_embeddings):
                learned_sample_id = f"{sample_id}_patch_{idx}_v{repeat}"
                samples[learned_sample_id] = {
                    "embedding": emb,
                    "album_id": label,
                    "feedback": {"was_correct": True, "correct_label": None},
                    "image_bytes": None,  # Don't store all patch image data
                    "timestamp": time.time(),
                    "predictions": []
                }
                created_samples.append(learned_sample_id)
        
        print(f"Model learned from {len(new_embeddings)} patches × {amplification_factor} amplification = {len(created_samples)} total embeddings for sample {sample_id} -> {album_id}")
        return created_samples
    except Exception as e:
        print(f"Error learning from embeddings: {e}")
        return None


def rebuild_identifier_with_learning(sample_id, image_bytes, album_id, samples, albums, model, transform):
    """
    Process feedback correction by learning and rebuilding the model.
    
    Args:
        sample_id: Original sample identifier
        image_bytes: Raw image data
        album_id: Correct label
        samples: Sample storage dictionary
        albums: Album storage dictionary
        model: PyTorch model for embedding
        transform: Image transform pipeline
        
    Returns:
        New TreeIdentifier if successful, None if failed
    """
    # Add learned embeddings
    created_samples = add_learned_embeddings(sample_id, image_bytes, album_id, samples, model, transform)
    
    if created_samples:
        # Rebuild identifier with new learned embeddings
        try:
            identifier = TreeIdentifier(model, transform, samples)
            return identifier
        except Exception as e:
            print(f"Error rebuilding identifier: {e}")
            return None
    
    return None
