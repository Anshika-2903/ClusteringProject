
if __name__ == "__main__":
    data, data_scaled = load_and_preprocess_data("ML Assignment Dataset.csv")

    print("Running HDBSCAN...")
    hdbscan_score, hdbscan_params = hdbscan_clustering(data_scaled)
    print(f"HDBSCAN Score: {hdbscan_score}, Parameters: {hdbscan_params}")

    print("Running K-Means...")
    kmeans_score, kmeans_k = kmeans_clustering(data_scaled)
    print(f"K-Means Score: {kmeans_score}, Optimal K: {kmeans_k}")

    print("Running Gaussian Mixture Model...")
    gmm_score, gmm_params = gmm_clustering(data_scaled)
    print(f"GMM Score: {gmm_score}, Parameters: {gmm_params}")

    print("Running Hierarchical Clustering...")
    hierarchical_score, hierarchical_params = hierarchical_clustering(data_scaled)
    print(f"Hierarchical Score: {hierarchical_score}, Parameters: {hierarchical_params}")
