# EXPERIMENT 6: K-MEANS CLUSTERING

# Step 1: Import libraries
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Step 2: Generate dataset
X, y = make_blobs(
    n_samples=100,
    centers=3,
    cluster_std=1.0,
    random_state=42
)

# Step 3: Create K-Means model
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

# Step 4: Train model and predict clusters
labels = kmeans.fit_predict(X)

# Step 5: Get cluster centroids
centroids = kmeans.cluster_centers_

# Step 6: Visualize clusters
plt.figure(figsize=(8, 5))

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=labels
)

# Display centroids
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="X",
    s=200,
    label="Centroids"
)

plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

# Step 7: Print centroids
print("Cluster Centroids:")
print(centroids)

# Step 8: Print cluster labels
print("Cluster Labels:")
print(labels)

# Step 9: Calculate inertia
print("Inertia:", kmeans.inertia_)