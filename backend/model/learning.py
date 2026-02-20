"""Model learning utilities for handling feedback and model improvements."""
import uuid
import numpy as np
import torch
from datetime import datetime
from PIL import Image
from io import BytesIO
from sqlalchemy.orm import Session

from db import Embedding, Album

PATCH_SIZE = 896
GRID = 4

def compute_patch_embeddings(image_bytes, model, transform, device="cpu"):
    """
    Efficiently compute normalized embeddings for 16 patches using batch inference.
    """
    if model is None or transform is None:
        raise RuntimeError("Model not initialized")

    model.eval()
    model.to(device)

    # Load and resize image safely
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    if image.size != (PATCH_SIZE, PATCH_SIZE):
        image = image.resize((PATCH_SIZE, PATCH_SIZE), Image.BILINEAR)

    pw = ph = PATCH_SIZE // GRID

    patches = []
    for r in range(GRID):
        for c in range(GRID):
            patch = image.crop((c * pw, r * ph, (c + 1) * pw, (r + 1) * ph))
            patches.append(transform(patch))

    batch = torch.stack(patches).to(device)

    with torch.no_grad():
        embeddings = model(batch)

    embeddings = embeddings.cpu().numpy()
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)

    return embeddings.astype(np.float32)

def add_learned_embeddings(
    sample_id: str,
    image_bytes: bytes,
    album_id: str,
    db: Session,
    model,
    transform,
    learning_strength: float = 5.0,
    device: str = "cpu",
):
    """
    Learn from corrected feedback efficiently using weighted embeddings.
    No duplication. Scalable. Safe.
    """

    try:
        # Prevent duplicate learning
        existing = db.query(Embedding).filter(
            Embedding.original_sample_id == sample_id,
            Embedding.album_id == album_id,
            Embedding.is_learned == True
        ).first()

        if existing:
            print(f"Sample {sample_id} already learned for {album_id}")
            return []

        embeddings = compute_patch_embeddings(
            image_bytes=image_bytes,
            model=model,
            transform=transform,
            device=device,
        )

        album = db.query(Album).filter(Album.album_id == album_id).first()
        if not album:
            raise ValueError(f"Album {album_id} not found")

        created_ids = []

        for patch_index, emb in enumerate(embeddings):
            embedding_id = str(uuid.uuid4())

            embedding_obj = Embedding(
                embedding_id=embedding_id,
                album_id=album_id,
                original_sample_id=sample_id,
                embedding_vector=emb.tobytes(),  # fast + portable
                embedding_dim=len(emb),
                is_learned=True,
                weight=learning_strength,  # amplified learning
                patch_index=patch_index,
                created_at=datetime.utcnow(),
            )

            db.add(embedding_obj)
            created_ids.append(embedding_id)

        db.commit()

        print(
            f"Learned {len(created_ids)} weighted embeddings "
            f"(strength={learning_strength}) "
            f"for sample {sample_id} -> {album_id}"
        )

        return created_ids

    except Exception as e:
        db.rollback()
        print(f"Learning error: {e}")
        return None

def update_identifier_with_learning(
    identifier,
    sample_id: str,
    image_bytes: bytes,
    album_id: str,
    db: Session,
    model,
    transform,
    learning_strength: float = 5.0,
    device: str = "cpu",
):
    """
    Learn + incrementally update identifier.
    Avoids full index rebuild.
    """

    created_ids = add_learned_embeddings(
        sample_id=sample_id,
        image_bytes=image_bytes,
        album_id=album_id,
        db=db,
        model=model,
        transform=transform,
        learning_strength=learning_strength,
        device=device,
    )

    if not created_ids:
        return identifier

    # Incremental update (must be implemented in TreeIdentifier)
    identifier.add_embeddings_by_ids(created_ids)

    return identifier

def add_embeddings_by_ids(self, embedding_ids):
    """
    Load embeddings by ID and insert into ANN index
    without full rebuild.
    """
    new_embeddings = self._load_embeddings_by_ids(embedding_ids)
    self.index.add(new_embeddings["vectors"])
    self.metadata.extend(new_embeddings["metadata"])