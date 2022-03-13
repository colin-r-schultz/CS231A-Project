from re import I
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

    def is_one_object(pts, res):
        return np.mean(np.linalg.norm(res, axis=-1)) < 2.0

    M, N, _ = points.shape
    segmentation = np.zeros(N, int)

    iters = 4
    for i in range(iters):
        new_segmentation = np.zeros_like(segmentation)
        num_objects = 0

        # SPLIT
        for o in range(np.max(segmentation)+1):
            obj_pts = (segmentation == o)
            if np.count_nonzero(obj_pts) <= 4:
                new_segmentation[obj_pts] = num_objects
                num_objects += 1
                continue
            pts, _, res = multibody_sfm(points[:, obj_pts], K)
            if is_one_object(pts, res):
                new_segmentation[obj_pts] = num_objects
                num_objects += 1
            else:
                km = KMeans(n_clusters=2)
                new_segmentation[obj_pts] = km.fit_predict(pts) + num_objects
                num_objects += 2

        segmentation = new_segmentation
        print("Split: ", segmentation)

        new_segmentation = np.zeros_like(segmentation)
        objects = list(range(num_objects))
        num_objects = 0

        # MERGE
        while len(objects) > 0:
            o1 = objects.pop(0)
            merged = False
            for o2 in objects:
                combined_pts = (segmentation == o1) + (segmentation == o2)
                pts, _, res = multibody_sfm(points[:, combined_pts], K)
                if is_one_object(pts, res):
                    objects.remove(o2)
                    new_segmentation[combined_pts] = num_objects
                    merged = True
                    break
            if not merged:
                new_segmentation[segmentation == o1] = num_objects
            num_objects += 1

        segmentation = new_segmentation
        print("Merge: ", segmentation)

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

    S = segment(p, K)
    print()
    print(S[:8])
    print(S[8:12])
    print(S[12:16])
    print(S[16:40])
    print(S[40:54])


