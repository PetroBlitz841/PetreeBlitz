import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_distances
from umap import UMAP
import matplotlib.pyplot as plt

def cluster_embeddings_dbscan(df, emb_column='embedding', eps=1000, min_samples=2, metric='cosine'):
    vecs = []
    valid_idx = []
    for i, v in enumerate(df[emb_column]):
        if v is not None:
            vecs.append(np.asarray(v, dtype=float))
            valid_idx.append(i)
    if len(vecs) == 0:
        raise RuntimeError(f'No valid embeddings in {emb_column}')
    X = np.vstack(vecs)
    if metric == 'cosine':
        X = normalize(X, norm='l2')

    clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    labels = clusterer.fit_predict(X)

    if len(set(labels) - {-1}) == 0:
        k = max(2, min(10, len(valid_idx)//8))
        print(f"DBSCAN found 0 clusters; falling back to KMeans k={k}")
        labels = KMeans(n_clusters=k, random_state=0).fit_predict(X)

    clusters = np.full(len(df), -1, dtype=int)
    for i, label in zip(valid_idx, labels):
        clusters[i] = label
    df['cluster'] = clusters
    return df

def apply_majority_vote_clustering(df, tree_col='tree_name', cluster_col='cluster'):
    all_clusters = sorted(df[cluster_col].unique())
    tree_vectors = {}
    for tree in df[tree_col].unique():
        counts = df[df[tree_col] == tree][cluster_col].value_counts()
        vector = [counts.get(c, 0) for c in all_clusters]
        tree_vectors[tree] = vector

    tree_list = list(tree_vectors.keys())
    matrix = np.array([tree_vectors[tree] for tree in tree_list], dtype=float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)
    matrix_normalized = normalize(matrix, norm='l2')

    db = DBSCAN(eps=0.8, min_samples=1, metric='cosine')
    labels = db.fit_predict(matrix_normalized)

    tree_cluster_dict = dict(zip(tree_list, labels))
    df[cluster_col] = df[tree_col].map(tree_cluster_dict)
    return df

def plot_umap_from_df(df, emb_col='normalized_embedding', name_col='tree_name',
                      n_neighbors=15, min_dist=0.1, savefile=None, figsize=(12,9)):
    X = np.vstack(df[emb_col])
    reducer = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    X2 = reducer.fit_transform(X)

    fig, ax = plt.subplots(figsize=figsize)
    uniq = list(dict.fromkeys(df[name_col]))
    cmap = plt.get_cmap('tab20')
    colors = [cmap(uniq.index(n) % cmap.N) for n in df[name_col]]
    ax.scatter(X2[:,0], X2[:,1], c=colors, s=30, alpha=0.9)

    for (x0, y0), nm in zip(X2, df[name_col]):
        ax.text(x0+0.002, y0+0.002, nm, fontsize=6, alpha=0.9)

    handles = [plt.Line2D([0],[0], marker='o', color='w', label=n,
                          markerfacecolor=cmap(i % cmap.N), markersize=6)
               for i, n in enumerate(uniq)]
    ax.legend(handles=handles, title='Tree name', bbox_to_anchor=(1.05,1), loc='upper left')
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    if savefile: plt.savefig(savefile, dpi=150, bbox_inches='tight')
    plt.show()
