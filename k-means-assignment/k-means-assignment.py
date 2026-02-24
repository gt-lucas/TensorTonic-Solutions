import numpy as np
def k_means_assignment(points, centroids):
    """
    Assign each point to the nearest centroid.
    """
    # Write code here
    if len(points) == 0 or len(centroids) == 0:
        return []
    P = np.asanyarray(points)
    C = np.asanyarray(centroids)

    diff = P[:, np.newaxis, :] - C
    dist_sq = np.sum(diff**2, axis=2)
    closet_centroids = np.argmin(dist_sq, axis=1)
    return closet_centroids.tolist()