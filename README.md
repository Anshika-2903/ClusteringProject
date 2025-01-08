# README for Clustering Assignment

## Overview
This assignment aims to explore various clustering algorithms to analyze and identify meaningful patterns within geospatial data. Several clustering techniques were applied, including HDBSCAN, K-Means, Gaussian Mixture Models (GMM), and Hierarchical Clustering. The project also involves hyperparameter tuning and evaluation using silhouette scores to select the best model for clustering.

## Approach
### Data Preprocessing
1. **Duplicate Removal**: Duplicate entries were removed to ensure data integrity.
2. **Missing Value Handling**:
   - `Latitude` and `Longitude` missing values were replaced with their respective column means.
3. **Feature Scaling**: StandardScaler was used to normalize the features, ensuring uniform contribution of features to the clustering process.

### Visualization
Initial visualization of the data was performed by plotting `Latitude` vs. `Longitude` to understand the data distribution and spatial characteristics.

### Clustering Techniques
1. **HDBSCAN**:
   - Hyperparameters (`min_samples` and `min_cluster_size`) were tuned using grid search.
   - Silhouette score was used to evaluate cluster quality.
   - Noise points were excluded from evaluation for better accuracy.

2. **K-Means**:
   - Grid search was performed over `k` (number of clusters) and `random_state`.
   - The best `k` was identified based on silhouette scores.
   - Clustering results were visualized.

3. **Gaussian Mixture Model (GMM)**:
   - Grid search was conducted over the number of components and covariance types.
   - Clusters were evaluated using silhouette scores.

4. **Hierarchical Clustering**:
   - Grid search was performed over `n_clusters` and linkage methods (`ward`, `complete`, `average`, `single`).
   - A dendrogram was used for visualization of hierarchical relationships.

### Evaluation Metrics
- **Silhouette Score**: This metric measures how similar a data point is to its own cluster compared to other clusters. Higher scores indicate better-defined clusters.

## Assumptions
1. Missing data is assumed to be missing at random and can be imputed using column means.
2. Features are scaled to zero mean and unit variance for clustering algorithms that are sensitive to feature magnitude.
3. For GMM and Hierarchical Clustering, the data is assumed to have a meaningful Euclidean structure.

## Hurdles
1. **Noise Handling**: HDBSCAN produced noise points which required careful handling during silhouette score computation.
2. **Computational Overhead**: Exhaustive grid searches over large hyperparameter spaces were computationally expensive.
3. **Cluster Visualization**: Representing high-dimensional data on 2D plots while preserving interpretability was challenging.

## Solution
- Efficient preprocessing and feature scaling minimized the effect of outliers and improved clustering.
- Grid search was optimized by limiting parameter ranges based on domain knowledge and initial visualizations.
- Visualization tools (scatter plots, dendrograms) were employed to provide insights into clustering results.

## Results
1. **HDBSCAN**:
   - Best Parameters: `min_samples={best_params['min_samples']}`, `min_cluster_size={best_params['min_cluster_size']}`.
   - Silhouette Score (without noise): {silhouette_hdbscan}
2. **K-Means**:
   - Best Parameters: `k={best_k}`, `random_state={best_random_state}`.
   - Silhouette Score: {best_score}
3. **Gaussian Mixture Model**:
   - Best Parameters: `n_components={best_n_components}`, `covariance_type={best_covariance_type}`.
   - Silhouette Score: {final_silhouette_score}
4. **Hierarchical Clustering**:
   - Best Parameters: `n_clusters={best_n_clusters}`, `linkage={best_linkage}`.
   - Silhouette Score: {best_score}

## Conclusions
- HDBSCAN effectively identified noise points, providing high-quality clusters for well-structured data.
- K-Means and GMM offered flexibility with hyperparameter tuning, showing competitive performance with silhouette scores.
- Hierarchical Clustering provided valuable insights into data relationships through dendrograms.
- Each algorithm demonstrated strengths in different aspects, making them suitable for varied clustering requirements.

## Future Improvements
1. Experiment with dimensionality reduction techniques (e.g., PCA, t-SNE) to improve visualization and clustering efficiency.
2. Explore alternative metrics like Davies-Bouldin Index and Calinski-Harabasz Score for a holistic evaluation.
3. Implement parallelization to reduce computation time during hyperparameter tuning.

