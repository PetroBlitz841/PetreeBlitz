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
        self.known_embeddings = np.array([s["embedding"] for s in known_samples.values()])
        self.known_labels = [s["album_id"] for s in known_samples.values()]

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
                emb = emb / (np.linalg.norm(emb) + 1e-12)
                new_embeddings.append(emb)

        # Vector comparison
        distances = cdist(new_embeddings, self.known_embeddings, metric='cosine')
        closest_indices = np.argmin(distances, axis=1)
        votes = [self.known_labels[idx] for idx in closest_indices]
        
        winner, count = Counter(votes).most_common(1)[0]
        return {
            "species": winner.replace("cluster_", ""),
            "confidence": count / 16,
            "agreement": f"{count}/16 patches"
        }