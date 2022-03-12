import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from sfm import multibody_sfm
from synthetic import generate_synthetic_points
from utils import set_axes_equal

def segment(points, K):
    """Compute object segmentations

    M = number of frames
    N = number of points

    Inputs:
        points - M x N x 2 2d pixel positions of N points over M frames
        K - 3x3 intrinsics matrix

    Returns:
        segmentation - N object ids
    """
    M, N, _ = points.shape
    segmentation = np.zeros(N, int)
    pts, _ = multibody_sfm(points[:, segmentation == 0], K)
    km = KMeans(n_clusters=5)
    segmentation = km.fit_predict(pts)
    print(segmentation)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2])
    ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], km.cluster_centers_[:, 2])
    ax.set_box_aspect([1,1,1])
    set_axes_equal(ax)
    plt.show()

    return segmentation

if __name__ == "__main__":
    M = 16
    K = np.array([
        [320, 0, 320],
        [0, 320, 240],
        [0, 0, 1]
    ])
    pts, p = generate_synthetic_points(K, M)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[0, :, 0], pts[0, :, 1], pts[0, :, 2])
    ax.set_box_aspect([1,1,1])
    set_axes_equal(ax)
    plt.show()

    print(segment(p, K))

