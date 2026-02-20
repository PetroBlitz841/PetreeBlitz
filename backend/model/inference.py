import torch
import numpy as np
from PIL import Image
from io import BytesIO
from collections import defaultdict
from sqlalchemy.orm import Session

PATCH_SIZE = 896
GRID = 4

class TreeIdentifier:
    def __init__(self, model, transform, db: Session, device="cpu"):
        self.model = model.eval().to(device)
        self.transform = transform
        self.device = device

        self.known_embeddings = None
        self.known_labels = []
        self.known_weights = None

        self._load_embeddings(db)

    # ---------------------------
    # Load embeddings efficiently
    # ---------------------------
    def _load_embeddings(self, db):
        from db import Embedding

        embeddings = db.query(Embedding).all()

        vectors = []
        labels = []
        weights = []

        for emb_obj in embeddings:
            vec = np.frombuffer(emb_obj.embedding_vector, dtype=np.float32)
            vec = vec.flatten()

            vectors.append(vec)
            labels.append(emb_obj.album_id)
            weights.append(getattr(emb_obj, "weight", 1.0))

        if vectors:
            self.known_embeddings = np.vstack(vectors)
            self.known_weights = np.array(weights, dtype=np.float32)
        else:
            self.known_embeddings = np.empty((0, 512), dtype=np.float32)
            self.known_weights = np.empty((0,), dtype=np.float32)

        self.known_labels = labels

    # ---------------------------
    # Incremental update
    # ---------------------------
    def add_embeddings(self, vectors, labels, weights):
        """
        Add new embeddings without rebuilding everything.
        """
        if self.known_embeddings.size == 0:
            self.known_embeddings = vectors
            self.known_weights = weights
        else:
            self.known_embeddings = np.vstack([self.known_embeddings, vectors])
            self.known_weights = np.concatenate([self.known_weights, weights])

        self.known_labels.extend(labels)

    # ---------------------------
    # Patch extraction
    # ---------------------------
    def _get_patches(self, image_bytes):
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        if image.size != (PATCH_SIZE, PATCH_SIZE):
            image = image.resize((PATCH_SIZE, PATCH_SIZE), Image.BILINEAR)

        pw = PATCH_SIZE // GRID
        patches = []

        for r in range(GRID):
            for c in range(GRID):
                patches.append(
                    image.crop((c * pw, r * pw, (c + 1) * pw, (r + 1) * pw))
                )

        return patches

    # ---------------------------
    # Batch embedding
    # ---------------------------
    def _embed_patches(self, patches):
        batch = torch.stack([self.transform(p) for p in patches]).to(self.device)

        with torch.no_grad():
            embeddings = self.model(batch)

        embeddings = embeddings.cpu().numpy()
        embeddings = embeddings / (
            np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        )

        return embeddings.astype(np.float32)

    # ---------------------------
    # Fast cosine similarity
    # ---------------------------
    def _cosine_similarity(self, A, B):
        """
        Fast cosine similarity using matrix multiplication.
        Assumes vectors are normalized.
        """
        return np.dot(A, B.T)

    # ---------------------------
    # Identification
    # ---------------------------
    def identify(self, image_bytes, top_k=5):
        if len(self.known_labels) == 0:
            return [{"label": "unknown", "confidence": 0.0}]

        patches = self._get_patches(image_bytes)
        new_embeddings = self._embed_patches(patches)

        # Compute similarity matrix
        similarity = self._cosine_similarity(new_embeddings, self.known_embeddings)

        # Get top-k nearest for each patch
        top_indices = np.argsort(-similarity, axis=1)[:, :top_k]

        vote_scores = defaultdict(float)

        for patch_idx in range(top_indices.shape[0]):
            for idx in top_indices[patch_idx]:
                label = self.known_labels[idx]
                weight = self.known_weights[idx]
                score = similarity[patch_idx, idx]

                # Weighted voting
                vote_scores[label] += score * weight

        # Normalize scores
        total_score = sum(vote_scores.values())
        if total_score == 0:
            return [{"label": "unknown", "confidence": 0.0}]

        predictions = sorted(
            [
                {
                    "label": label.replace("cluster_", ""),
                    "confidence": round(score / total_score, 3),
                }
                for label, score in vote_scores.items()
            ],
            key=lambda x: x["confidence"],
            reverse=True,
        )

        return predictions