

def hdbscan_clustering(data_scaled):
    best_score = -1
    best_params = {}
    min_samples_range = range(5, 20, 5)
    min_cluster_size_range = range(15, 300, 5)

    for min_samples in min_samples_range:
        for min_cluster_size in min_cluster_size_range:
            clusterer = hdbscan.HDBSCAN(min_samples=min_samples, min_cluster_size=min_cluster_size)
            labels = clusterer.fit_predict(data_scaled)

            filtered_data = data_scaled[labels != -1]
            filtered_labels = labels[labels != -1]

            if len(np.unique(filtered_labels)) > 1:
                score = silhouette_score(filtered_data, filtered_labels)
                if score > best_score:
                    best_score = score
                    best_params = {'min_samples': min_samples, 'min_cluster_size': min_cluster_size}

    # Plot clusters
    clusterer = hdbscan.HDBSCAN(**best_params)
    labels = clusterer.fit_predict(data_scaled)

    plt.figure(figsize=(12, 8))
    for label in np.unique(labels):
        cluster_points = data_scaled[labels == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {label}")
    plt.title("HDBSCAN Clusters")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    # plt.savefig("plots/hdbscan_clusters.png")
    plt.show()

    return best_score, best_params
