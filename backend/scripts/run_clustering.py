from model.preprocessing import data_preprocess
from model.model_loader import load_resnet18, get_transform, patch_to_embedding
from model.clustering import cluster_embeddings_dbscan, apply_majority_vote_clustering, plot_umap_from_df
import pandas as pd
import numpy as np

if __name__ == '__main__':
    # 1. Preprocess images into patches
    df = data_preprocess()

    # 2. Load model and transform
    model = load_resnet18()
    transform = get_transform()

    # 3. Compute embeddings
    embeddings = []
    for patch in df['patch_path']:
        embeddings.append(patch_to_embedding(patch, model, transform))
    df['embedding'] = embeddings
    df['normalized_embedding'] = df.embedding.apply(lambda x: x / (np.linalg.norm(x)+1e-12))
    df['tree_name'] = df.original_name.str[:-4]

    # 4. UMAP plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_umap_from_df(df, emb_col='normalized_embedding', name_col='tree_name',
                      n_neighbors=20, min_dist=0.1, savefile=f'../generated/umap/umap_tree_patches_{timestamp}.png')

    # 5. Patch-level clustering
    df = cluster_embeddings_dbscan(df, emb_column='normalized_embedding', eps=0.1, min_samples=2, metric='euclidean')

    # 6. Majority vote to cluster trees
    df = apply_majority_vote_clustering(df, tree_col='tree_name', cluster_col='cluster')

    # 7. Save result
    df.to_csv('data/tree_patches_with_clusters.csv', index=False)
    print("Clustering completed and saved to tree_patches_with_clusters.csv")
