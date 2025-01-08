# hierarchical_cluster.py
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
import numpy as np

def run_hierarchical(data_scaled):
    best_n_clusters = None
    best_linkage = None
    best_score = -1
    results = []

    n_clusters_range = range(2, 50)
    linkage_methods = ['ward', 'complete', 'average', 'single']

    for n_clusters in n_clusters_range:
        for linkage_method in linkage_methods:
            clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
            labels = clustering.fit_predict(data_scaled)
            score = silhouette_score(data_scaled, labels)
            results.append((n_clusters, linkage_method, score))

            if score > best_score:
                best_score = score
                best_n_clusters = n_clusters
                best_linkage = linkage_method

    print(f"Best Number of Clusters: {best_n_clusters}")
    print(f"Best Linkage Method: {best_linkage}")
    print(f"Best Silhouette Score: {best_score}")

    n_clusters_values, linkage_values, silhouette_scores = zip(*results)

    linkage_unique = sorted(set(linkage_values))
    n_clusters_unique = sorted(set(n_clusters_values))
    heatmap_data = np.zeros((len(n_clusters_unique), len(linkage_unique)))

    for n_clusters, linkage_method, score in results:
        i = n_clusters_unique.index(n_clusters)
        j = linkage_unique.index(linkage_method)
        heatmap_data[i, j] = score

    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", xticklabels=linkage_unique, yticklabels=n_clusters_unique, cmap='viridis')
    plt.xlabel("Linkage Method")
    plt.ylabel("Number of Clusters")
    plt.title("Silhouette Scores for Hierarchical Clustering")
    plt.show()

    linkage_matrix = linkage(data_scaled, method=best_linkage)
    plt.figure(figsize=(12, 8))
    dendrogram(linkage_matrix)
    plt.title(f"Dendrogram (Linkage: {best_linkage})")
    plt.xlabel("Samples")
    plt.ylabel("Distance")
    plt.show()

    final_clustering = AgglomerativeClustering(n_clusters=best_n_clusters, linkage=best_linkage)
    final_labels = final_clustering.fit_predict(data_scaled)

    plt.figure(figsize=(10, 8))
    plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=final_labels, cmap='viridis', alpha=0.6)
    plt.title(f"Hierarchical Clustering (Clusters={best_n_clusters}, Linkage={best_linkage})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar(label="Cluster Label")
    plt.show()

    return best_score
