import torch
import torchvision
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np
import pickle
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize

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

def compute_embeddings(images_dir='./Trees'):
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()

    names = []
    feats = []

    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Skipping {img_file}: failed to open ({e})")
            continue

        tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            emb = model(tensor)

        emb = emb.squeeze()  # likely shape (512, 1, 1) or (512,)
        emb = emb.reshape(-1).cpu().numpy()

        names.append(img_file)
        feats.append(emb)
        print(f"Loaded: {img_file} (embedding {emb.shape})")

    if len(feats) == 0:
        raise RuntimeError('No images found or no embeddings computed.')

    X = np.vstack(feats)
    return names, X

def cluster_embeddings(names, X, eps=0.3, min_samples=2):
    # Normalize embeddings to unit vectors for cosine distance
    Xn = normalize(X, norm='l2')

    db = DBSCAN(metric='cosine', eps=eps, min_samples=min_samples)
    labels = db.fit_predict(Xn)

    clusters = {}
    for name, label in zip(names, labels):
        if label == -1:
            key = 'untitled'
        else:
            key = f'cluster_{label}'
        clusters.setdefault(key, []).append(name)

    return labels, clusters


def save_embeddings(names, X, filepath='embeddings.pkl'):
    """Save embeddings and image names to a pickle file."""
    data = {'names': names, 'embeddings': X}
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"Embeddings saved to {filepath}")


def load_embeddings(filepath='embeddings.pkl'):
    """Load embeddings and image names from a pickle file."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print(f"Embeddings loaded from {filepath}")
    return data['names'], data['embeddings']


if __name__ == '__main__':
    images_dir = './Trees'
    embeddings_file = 'embeddings.pkl'
    
    # Load embeddings from file if it exists, otherwise compute from images
    if os.path.exists(embeddings_file):
        print(f"Loading pre-computed embeddings from {embeddings_file}...")
        names, X = load_embeddings(embeddings_file)
    else:
        print("Computing embeddings from images...")
        names, X = compute_embeddings(images_dir)
        save_embeddings(names, X, embeddings_file)
    
    # Cluster the embeddings
    labels, clusters = cluster_embeddings(names, X, eps=0.1, min_samples=2)

    print('\nClustering results:')
    for k, v in clusters.items():
        print(f"{k}: {len(v)} -> {v}")

    print(f"Total images: {len(names)} | clusters found: {len([k for k in clusters.keys() if k!='untitled'])} | untitled: {len(clusters.get('untitled', []))}")