
def kmeans_clustering(data_scaled):
    best_score = -1
    best_k = None
    k_range = range(2, 50)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data_scaled)
        score = silhouette_score(data_scaled, labels)
        if score > best_score:
            best_score = score
            best_k = k

    # Plot clusters
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data_scaled)

    plt.figure(figsize=(10, 8))
    plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.title(f"K-Means Clustering with k={best_k}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
  
    plt.show()

    return best_score, best_k
