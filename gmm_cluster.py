
def gmm_clustering(data_scaled):
    best_score = -1
    best_n_components = None
    best_covariance_type = None
    n_components_range = range(2, 15)
    covariance_types = ['full', 'tied', 'diag', 'spherical']

    for n_components in n_components_range:
        for cov_type in covariance_types:
            gmm = GaussianMixture(n_components=n_components, covariance_type=cov_type, random_state=42)
            labels = gmm.fit_predict(data_scaled)

            if len(set(labels)) > 1:
                score = silhouette_score(data_scaled, labels)
                if score > best_score:
                    best_score = score
                    best_n_components = n_components
                    best_covariance_type = cov_type

    # Plot clusters
    gmm = GaussianMixture(n_components=best_n_components, covariance_type=best_covariance_type, random_state=42)
    labels = gmm.fit_predict(data_scaled)

    plt.figure(figsize=(10, 8))
    plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.title(f"GMM Clustering (n={best_n_components}, cov={best_covariance_type})")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    # plt.savefig("plots/gmm_clusters.png")
    plt.show()

    return best_score, (best_n_components, best_covariance_type)
