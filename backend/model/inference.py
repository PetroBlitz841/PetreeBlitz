import torch
import numpy as np
from PIL import Image
from io import BytesIO
from collections import Counter
from scipy.spatial.distance import cdist

class TreeIdentifier:
    def __init__(self, model, transform, known_samples):
        self.model = model
        self.transform = transform
        # Pre-calculate matrix of known embeddings for speed
        embeddings_list = []
        labels_list = []
        
        for s in known_samples.values():
            # Skip samples that don't have embeddings (e.g., uploaded samples without computed embeddings)
            if "embedding" not in s:
                continue
            
            emb = s["embedding"]
            # Ensure embedding is a numpy array
            if not isinstance(emb, np.ndarray):
                emb = np.array(emb)
            # Flatten to 1D if needed
            emb = emb.flatten()
            embeddings_list.append(emb)
            labels_list.append(s["album_id"])
        
        # Create 2D array (num_samples x embedding_dim)
        self.known_embeddings = np.array(embeddings_list)
        self.known_labels = labels_list
        
        if self.known_embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings array, got shape {self.known_embeddings.shape}")

    def _get_patches(self, image_bytes):
        im = Image.open(BytesIO(image_bytes)).convert('RGB')
        im = im.crop((0, 0, 896, 896))
        pw = ph = 896 // 4
        patches = []
        for r in range(4):
            for c in range(4):
                patches.append(im.crop((c*pw, r*ph, (c+1)*pw, (r+1)*ph)))
        return patches

    def identify(self, image_bytes):
        patches = self._get_patches(image_bytes)
        new_embeddings = []
        
        # Process patches (could be batched for speed)
        for patch in patches:
            tensor = self.transform(patch).unsqueeze(0)
            with torch.no_grad():
                emb = self.model(tensor).squeeze().cpu().numpy()
                # Flatten to 1D
                emb = emb.flatten()
                # Normalize
                emb = emb / (np.linalg.norm(emb) + 1e-12)
                new_embeddings.append(emb)

        # Convert to 2D array (num_patches x embedding_dim)
        new_embeddings = np.array(new_embeddings)
        if new_embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings array, got shape {new_embeddings.shape}")
        
        # Vector comparison
        distances = cdist(new_embeddings, self.known_embeddings, metric='cosine')
        closest_indices = np.argmin(distances, axis=1)
        votes = [self.known_labels[idx] for idx in closest_indices]
        
        # Get vote counts for all species
        vote_counts = Counter(votes)
        total_votes = len(votes)
        
        # Create predictions list sorted by confidence (highest first)
        predictions = []
        for species, count in vote_counts.most_common():
            predictions.append({
                "label": species.replace("cluster_", ""),
                "confidence": round(count / total_votes, 2)
            })
        
        return predictions