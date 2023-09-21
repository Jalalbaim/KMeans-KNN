## K means algorithm

import numpy as np

# distance euclidienne

def euclidean_distance(x1, x2):
    ndg1 = x1[0]  
    ndg2 = x2[0]  
    return np.abs(ndg1 - ndg2)

# Algorithm
max_iterations = 100

def kmeans (data, k):
    # initialisation aléatoire des centroids
    centroids = data[:, np.random.choice(range(data.shape[1]), k, replace=False)].T

    for i in range(1000):
      # association de chaque point à un cluster
        clusters = [[] for _ in range(k)]
        for point in data.T:
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(point)

      # associer de nouveau centroids au median des point à chaque cluster
        new_centroids = [np.mean(cluster, axis=0) if cluster else centroid for cluster, centroid in zip(clusters, centroids)]

      # Check pour la convergence
        if np.all(np.array(new_centroids) == np.array(centroids)):
          break
        centroids = new_centroids
    return centroids, clusters
