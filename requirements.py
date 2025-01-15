# requirements.py
!pip install pandas numpy matplotlib seaborn hdbscan scikit-learn
import hdbscan
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
