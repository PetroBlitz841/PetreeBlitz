import torch
import torchvision
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np
import pickle
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from umap import UMAP
import pandas as pd
from tqdm import tqdm
import time
import uuid
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime

# Load ResNet18 model (pretrained)
model = models.resnet18(pretrained=True)
model.eval()

# Remove the final classification layer to get feature embeddings (output of avgpool)
model = torch.nn.Sequential(*list(model.children())[:-1])

# Transform: convert to grayscale, resize, normalize
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ============================================================================
# FastAPI Models and Application
# ============================================================================

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


# Global data structures
app = FastAPI(title="PetreeBlitz API")
SAMPLES: Dict = {}  # sample_id -> {embedding, predictions, feedback, image_path, timestamp}
ALBUMS: Dict = {}   # album_id -> {name, sample_ids}


def add_sample_to_album(album_id: str, album_name: str, sample_id: str, 
                       embedding: np.ndarray, predictions: List[Dict], 
                       image_path: str) -> None:
    """Add a sample to an album."""
    if album_id not in ALBUMS:
        ALBUMS[album_id] = {"name": album_name, "sample_ids": []}
    
    ALBUMS[album_id]["sample_ids"].append(sample_id)
    
    SAMPLES[sample_id] = {
        "embedding": embedding,
        "predictions": predictions,
        "feedback": None,
        "image_path": image_path,
        "timestamp": time.time(),
        "album_id": album_id
    }


# ============================================================================
# Serve static images
# ============================================================================
if os.path.exists('./patches'):
    app.mount("/patches", StaticFiles(directory="./patches"), name="patches")
if os.path.exists('./Trees'):
    app.mount("/trees", StaticFiles(directory="./Trees"), name="trees")


@app.get("/albums", response_model=List[AlbumModel])
def get_albums():
    """Get list of all albums."""
    result = []
    for album_id, album_data in ALBUMS.items():
        result.append(AlbumModel(
            album_id=album_id,
            name=album_data["name"],
            num_images=len(album_data["sample_ids"])
        ))
    return result


@app.get("/albums/{album_id}/images", response_model=List[ImageResponseModel])
def get_album_images(album_id: str):
    """Get all images in an album with their predictions and feedback."""
    if album_id not in ALBUMS:
        return JSONResponse(status_code=404, content={"detail": "Album not found"})
    
    sample_ids = ALBUMS[album_id]["sample_ids"]
    result = []
    
    for sample_id in sample_ids:
        if sample_id in SAMPLES:
            sample = SAMPLES[sample_id]
            
            # Convert predictions to response format
            predictions = [
                PredictionModel(label=p["label"], confidence=p["confidence"])
                for p in sample["predictions"]
            ]
            
            # Convert feedback if present
            feedback = None
            if sample["feedback"]:
                feedback = FeedbackModel(
                    was_correct=sample["feedback"].get("was_correct"),
                    correct_label=sample["feedback"].get("correct_label")
                )
            
            # Format timestamp
            timestamp_str = datetime.fromtimestamp(sample["timestamp"]).isoformat() + "Z"
            
            result.append(ImageResponseModel(
                sample_id=sample_id,
                image_url=sample["image_path"],
                predictions=predictions,
                feedback=feedback,
                timestamp=timestamp_str
            ))
    
    return result


def data_preprocess(images_dir='./Trees', patches_dir='./patches'):
    """Read images from `images_dir`, resize each to 896x896, split into 4x4 patches
    (16 patches of 224x224), save patches to `patches_dir` and return a pandas DataFrame
    with columns `patch_path` and `original_name`.

    Returns:
        pandas.DataFrame: rows for each saved patch with keys `patch_path` and `original_name`.
    """
    os.makedirs(patches_dir, exist_ok=True)
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()

    records = []
    for img_file in tqdm(image_files, desc="Processing images"):
        src_path = os.path.join(images_dir, img_file)
        try:
            im = Image.open(src_path).convert('RGB')
        except Exception as e:
            print(f"Skipping {img_file}: cannot open ({e})")
            continue

        # Crop the top-left 896x896 region. Skip images smaller than that.
        w, h = im.size
        if w < 896 or h < 896:
            print(f"Skipping {img_file}: image smaller than 896x896 ({w}x{h})")
            continue
        im_cropped = im.crop((0, 0, 896, 896))

        pw = 896 // 4
        ph = 896 // 4
        base = os.path.splitext(img_file)[0]
        idx = 0

        for r in range(4):
            for c in range(4):
                left = c * pw
                upper = r * ph
                right = left + pw
                lower = upper + ph
                patch = im_cropped.crop((left, upper, right, lower))

                patch_name = f"{base}_patch{idx}.png"
                patch_path = os.path.join(patches_dir, patch_name)
                try:
                    patch.save(patch_path)
                except Exception as e:
                    print(f"Failed to save patch {patch_path}: {e}")
                    continue

                records.append({
                    'patch_path': patch_path,
                    'original_name': img_file,
                })
                idx += 1

    df = pd.DataFrame(records)
    print(f"Created {len(df)} patches in {patches_dir}")
    return df


def patch_to_embedding(patch_path):
    """Load an image patch and return its embedding as a 1D numpy array."""
    try:
        im = Image.open(patch_path).convert('RGB')
    except Exception as e:
        raise RuntimeError(f"Failed to open patch {patch_path}: {e}")

    tensor = transform(im).unsqueeze(0)
    with torch.no_grad():
        emb = model(tensor)
    emb = emb.squeeze().reshape(-1).cpu().numpy()
    return emb


def apply_embeddings_to_df(df, patch_column='patch_path', emb_column='embedding'):
    """Compute embeddings for each patch in `df[patch_column]` and store them in `df[emb_column]`.

    Returns the DataFrame with a new column containing numpy arrays.
    """
    embeddings = []
    for p in tqdm(df[patch_column].tolist(), desc="Computing embeddings"):
        try:
            e = patch_to_embedding(p)
        except Exception as exc:
            print(f"Warning: embedding failed for {p}: {exc}")
            e = None
        embeddings.append(e)
    df[emb_column] = embeddings
    return df


def cluster_embeddings_dbscan(df, emb_column='embedding', eps=1000, min_samples=2, metric='cosine'):
    """Cluster embeddings in dataframe using DBSCAN.
    
    - `emb_column`: column name containing embedding vectors.
    - `eps`: maximum distance between samples in a cluster.
    - `min_samples`: minimum number of samples in a cluster.
    - `metric`: distance metric ('cosine', 'euclidean', etc.).
    
    Returns the dataframe with a new 'cluster' column added.
    """
    # Extract embeddings, filtering out None values
    vecs = []
    valid_idx = []
    for i, v in enumerate(df[emb_column]):
        if v is not None:
            vecs.append(np.asarray(v, dtype=float))
            valid_idx.append(i)
    
    if len(vecs) == 0:
        raise RuntimeError(f'No valid embeddings found in column {emb_column}')
    
    X = np.vstack(vecs)

    # If using cosine, normalize rows to unit length (cosine distance on normalized vectors)
    if metric == 'cosine':
        X = normalize(X, norm='l2')

    # Try DBSCAN first with provided parameters
    clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    labels = clusterer.fit_predict(X)

    # If DBSCAN found no clusters (all noise), fall back to KMeans to ensure grouping
    num_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
    if num_clusters_found == 0:
        # Choose a reasonable k based on number of valid samples
        k = max(2, min(10, max(2, len(valid_idx) // 8)))
        print(f"DBSCAN found 0 clusters; falling back to KMeans with k={k}")
        km = KMeans(n_clusters=k, random_state=0)
        labels = km.fit_predict(X)
    
    # Create cluster column, default to -1 (noise) for rows with None embedding
    clusters = np.full(len(df), -1, dtype=int)
    for i, label in zip(valid_idx, labels):
        clusters[i] = label
    
    df['cluster'] = clusters
    
    # Print summary
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"DBSCAN clustering: {n_clusters} clusters, {n_noise} noise points")
    
    # Compute and print cosine similarities
    if metric == 'cosine':
        print("\nCosine distance matrix (sample):")
        from sklearn.metrics.pairwise import cosine_distances
        cos_dist = cosine_distances(X)
        # Print distance from first 5 points to all others
        for i in range(min(5, len(X))):
            print(f"  Point {i}: {np.round(cos_dist[i][:min(10, len(X))], 3)}")
        # Print cluster composition
        print("\nCluster assignments (cosine DBSCAN):")
        for cluster_id in sorted(set(labels)):
            members = np.where(labels == cluster_id)[0]
            print(f"  Cluster {cluster_id}: {len(members)} points - indices {members[:20]}" + (" ..." if len(members) > 20 else ""))
    
    return df


def apply_majority_vote_clustering(df, tree_col='tree_name', cluster_col='cluster'):
    """Apply majority vote clustering: cluster trees based on the distribution of their patch clusters.
    
    For each tree, compute the count of patches in each cluster (the 'vote' distribution).
    Then cluster the trees using DBSCAN on these normalized count vectors.
    This groups together trees with similar cluster compositions, effectively grouping similar images.
    
    Returns the dataframe with updated cluster assignments.
    """
    # Get all possible clusters
    all_clusters = sorted(df[cluster_col].unique())
    
    # For each tree, compute the count vector
    tree_vectors = {}
    for tree in df[tree_col].unique():
        counts = df[df[tree_col] == tree][cluster_col].value_counts()
        vector = [counts.get(c, 0) for c in all_clusters]
        tree_vectors[tree] = vector
    
    # Create matrix
    tree_list = list(tree_vectors.keys())
    matrix = np.array([tree_vectors[tree] for tree in tree_list], dtype=float)
    
    # Normalize the matrix (to sum to 1, making it a distribution)
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)
    
    # Normalize to unit vectors for cosine distance
    matrix_normalized = normalize(matrix, norm='l2')
    
    # DBSCAN clustering on the count distributions using cosine metric
    # Using higher eps (0.8) to group trees by species rather than individual variations
    db = DBSCAN(eps=0.8, min_samples=1, metric='cosine')
    labels = db.fit_predict(matrix_normalized)
    
    # Map back to df
    tree_cluster_dict = dict(zip(tree_list, labels))
    df[cluster_col] = df[tree_col].map(tree_cluster_dict)
    
    # Print summary
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"\nMajority vote clustering: {n_clusters} clusters, {n_noise} noise trees")
    
    # Compute and print cosine distances between tree vectors
    print("\nCosine distance matrix (tree vectors):")
    from sklearn.metrics.pairwise import cosine_distances
    cos_dist_trees = cosine_distances(matrix_normalized)
    for i, tree in enumerate(tree_list[:min(5, len(tree_list))]):
        distances = cos_dist_trees[i][:min(10, len(tree_list))]
        print(f"  {tree}: {np.round(distances, 3)}")
    
    # Print details of each cluster
    print("\nCluster composition:")
    for cluster_id in sorted(set(labels)):
        trees_in_cluster = [tree for tree, cid in tree_cluster_dict.items() if cid == cluster_id]
        if cluster_id == -1:
            print(f"  Noise cluster: {len(trees_in_cluster)} trees - {trees_in_cluster}")
        else:
            print(f"  Cluster {cluster_id}: {len(trees_in_cluster)} trees - {trees_in_cluster}")
    
    return df


def plot_umap_from_df(df, emb_col='normalized_embedding', name_col='tree_name',
                      n_neighbors=15, min_dist=0.1, savefile=None, figsize=(12, 9)):
    """Plot a 2D UMAP of normalized embeddings from `df`.

    - `emb_col`: column name with normalized vectors (or raw vectors; function will normalize if needed).
    - `name_col`: column name with the tree name (used for labels and legend grouping).
    - `n_neighbors`, `min_dist`: UMAP parameters.
    """
    if UMAP is None:
        raise RuntimeError('UMAP is not available. Install umap-learn to use this feature.')

    vecs = []
    names = []
    for _, row in df.iterrows():
        v = row.get(emb_col)
        vecs.append(v)
        names.append(row.get(name_col))

    if len(vecs) == 0:
        raise RuntimeError('No embeddings available for UMAP plotting')

    X = np.vstack(vecs)

    reducer = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    X2 = reducer.fit_transform(X)

    fig, ax = plt.subplots(figsize=figsize)

    uniq = list(dict.fromkeys(names))
    cmap = plt.get_cmap('tab20')
    colors = [cmap(uniq.index(n) % cmap.N) for n in names]

    ax.scatter(X2[:, 0], X2[:, 1], c=colors, s=30, alpha=0.9)

    for (x0, y0), nm in zip(X2, names):
        ax.text(x0 + 0.002, y0 + 0.002, nm, fontsize=6, alpha=0.9)

    handles = []
    for i, nm in enumerate(uniq):
        handles.append(plt.Line2D([0], [0], marker='o', color='w', label=nm,
                                  markerfacecolor=cmap(i % cmap.N), markersize=6))
    ax.legend(handles=handles, title='Tree name', bbox_to_anchor=(1.05, 1), loc='upper left')

    ax.set_title('UMAP projection of normalized patch embeddings')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    if savefile:
        plt.savefig(savefile, dpi=150, bbox_inches='tight')
        print(f"Saved UMAP plot to {savefile}")
    plt.show()




def populate_albums_from_df(df):
    """Populate ALBUMS and SAMPLES from dataframe with actual patch images."""
    # Group by cluster to create albums
    for cluster_id in df['cluster'].unique():
        cluster_df = df[df['cluster'] == cluster_id]
        album_id = f"cluster_{cluster_id}"
        album_name = f"Cluster {cluster_id}"
        
        for _, row in cluster_df.iterrows():
            sample_id = str(uuid.uuid4())
            embedding = row['normalized_embedding'] if 'normalized_embedding' in row else row.get('embedding')
            
            # Get the actual patch path and convert to relative path for serving
            patch_path = row['patch_path']
            if isinstance(patch_path, str) and os.path.exists(patch_path):
                # Convert absolute path to relative URL
                # e.g., "patches/tree1_patch0.png" -> "/patches/tree1_patch0.png"
                relative_patch = os.path.relpath(patch_path, '.').replace('\\', '/')
                image_url = f"/{relative_patch}"
            else:
                image_url = f"/patches/placeholder.png"
            
            # Create dummy predictions (you can modify this based on your actual predictions)
            predictions = [
                {"label": "Species A", "confidence": 0.7},
                {"label": "Species B", "confidence": 0.3}
            ]
            
            add_sample_to_album(album_id, album_name, sample_id, embedding, predictions, image_url)


@app.on_event("startup")
async def startup_event():
    """Load data on startup if CSV exists."""
    if os.path.exists('tree_patches_with_clusters.csv'):
        df = pd.read_csv('tree_patches_with_clusters.csv')
        populate_albums_from_df(df)
        print(f"Loaded {len(SAMPLES)} samples into {len(ALBUMS)} albums")


if __name__ == '__main__':
    import uvicorn
    
    # Check if we're running the FastAPI app or the clustering script
    import sys
    if '--api' in sys.argv:
        # Run FastAPI server
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        # Run clustering
        df = data_preprocess()
        print(f"Patches dataframe rows: {len(df)}")
        df = apply_embeddings_to_df(df)
        print(df.head())
        df['normalized_embedding'] = df.embedding.apply(
            lambda x: x / (np.linalg.norm(x) + 1e-12)
        )    
        df['tree_name'] = df.original_name.str[:-4]

        plot_umap_from_df(df, emb_col='normalized_embedding', name_col='tree_name',
                          n_neighbors=20, min_dist=0.1, savefile='umap_tree_patches.png')

        # Cluster using normalized embeddings and cosine distance (more stable for ResNet features)
        df = cluster_embeddings_dbscan(df, emb_column='normalized_embedding', eps=0.1, min_samples=2, metric='euclidean')
        
        # Apply majority vote clustering: assign each tree to its majority cluster
        df = apply_majority_vote_clustering(df, tree_col='tree_name', cluster_col='cluster')
        
        print("\n✓ Majority vote clustering completed successfully")
        print(f"Final dataframe shape: {df.shape}")
        print(f"Unique clusters: {sorted(df['cluster'].unique())}")
        print(f"Sample of final results:\n{df[['tree_name', 'patch_path', 'cluster']].head(10)}")
            
        df.to_csv('tree_patches_with_clusters.csv', index=False)


