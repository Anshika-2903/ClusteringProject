

def hierarchical_clustering(data_scaled):
    best_score = -1
    best_n_clusters = None
    best_linkage = None
    n_clusters_range = range(2, 50)
    linkage_methods = ['ward', 'complete', 'average', 'single']

    for n_clusters in n_clusters_range:
        for link_method in linkage_methods:
            clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage=link_method)
            labels = clusterer.fit_predict(data_scaled)
            score = silhouette_score(data_scaled, labels)
            if score > best_score:
                best_score = score
                best_n_clusters = n_clusters
                best_linkage = link_method

    # Plot dendrogram
    plt.figure(figsize=(12, 8))
    dendro = dendrogram(linkage(data_scaled, method=best_linkage))
    plt.title(f"Dendrogram ({best_linkage})")
   
    plt.show()

    # Plot clusters
    clusterer = AgglomerativeClustering(n_clusters=best_n_clusters, linkage=best_linkage)
    labels = clusterer.fit_predict(data_scaled)

    plt.figure(figsize=(10, 8))
    plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.title(f"Hierarchical Clustering (n_clusters={best_n_clusters}, linkage={best_linkage})")
    # plt.savefig("plots/hierarchical_clusters.png")
    plt.show()

    return best_score, (best_n_clusters, best_linkage)
