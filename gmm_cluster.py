# gmm_cluster.py
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

def run_gmm(data_scaled):
    best_n_components = None
    best_covariance_type = None
    best_score = -1
    results = []

    n_components_range = range(2, 15)
    covariance_types = ['full', 'tied', 'diag', 'spherical']

    for n_components in n_components_range:
        for covariance_type in covariance_types:
            gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=42)
            labels = gmm.fit_predict(data_scaled)

            if len(np.unique(labels)) > 1:
                score = silhouette_score(data_scaled, labels)
                results.append((n_components, covariance_type, score))

    best_result = max(results, key=lambda x: x[2])
    best_n_components, best_covariance_type, best_score = best_result

    print(f"Best Number of Components: {best_n_components}")
    print(f"Best Covariance Type: {best_covariance_type}")
    print(f"Best Silhouette Score: {best_score}")

    components, cov_types, silhouette_scores = zip(*results)

    plt.figure(figsize=(12, 6))
    for cov_type in covariance_types:
        scores = [score for comp, cov, score in results if cov == cov_type]
        plt.plot(n_components_range, scores, marker='o', label=f'Covariance Type: {cov_type}')

    plt.title('GMM Silhouette Scores for Different Parameters')
    plt.xlabel('Number of Components')
    plt.ylabel('Silhouette Score')
    plt.legend()
    plt.grid(True)
    plt.show()

    best_gmm = GaussianMixture(n_components=best_n_components, covariance_type=best_covariance_type, random_state=42)
    final_labels = best_gmm.fit_predict(data_scaled)

    plt.figure(figsize=(10, 8))
    plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=final_labels, cmap='viridis', alpha=0.6)
    plt.title(f"GMM Clustering (n_components={best_n_components}, covariance_type={best_covariance_type})")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.colorbar(label="Cluster Label")
    plt.show()

    return best_score
