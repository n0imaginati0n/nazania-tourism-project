from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

def kmeans_clustering(X, cluster_range=(2, 10)):
    best_score = -1
    best_model = None
    best_k = None

    print("KMeans Clustering Grid Search Results:\n")
    for k in range(cluster_range[0], cluster_range[1] + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = model.fit_predict(X)

        # Silhouette Score (higher is better)
        score = silhouette_score(X, labels)
        print(f"n_clusters={k} → Silhouette Score: {score:.4f}")

        if score > best_score:
            best_score = score
            best_model = model
            best_k = k

    print("\nBest KMeans Model:")
    print(f"n_clusters={best_k} → Best Silhouette Score: {best_score:.4f}")

    return best_model, best_k, best_score
