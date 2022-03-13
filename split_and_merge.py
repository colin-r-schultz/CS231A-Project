import numpy as np
from sklearn.cluster import KMeans

from sfm import singlebody_sfm

def split_and_merge(points, K):
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

    def split(pts, res):
        km = KMeans(n_clusters=2)
        s = km.fit_predict(pts)
        return s


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
            pts, _, res = singlebody_sfm(points[:, obj_pts], K)
            if is_one_object(pts, res):
                new_segmentation[obj_pts] = num_objects
                num_objects += 1
            else:
                new_segmentation[obj_pts] = split(pts, res) + num_objects
                print("Split ", o, " into ", num_objects, " and ", num_objects+1)
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
                if np.count_nonzero(combined_pts) < 4:
                    continue
                pts, _, res = singlebody_sfm(points[:, combined_pts], K)
                if is_one_object(pts, res):
                    objects.remove(o2)
                    new_segmentation[combined_pts] = num_objects
                    merged = True
                    print("Merging ", o1, " and ", o2, " into ", num_objects)
                    break
            if not merged:
                new_segmentation[segmentation == o1] = num_objects
            num_objects += 1

        segmentation = new_segmentation
        print("Merge: ", segmentation)

    return segmentation

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from synthetic import default_synthetic_points
    from utils import set_axes_equal

    M = 16
    K = np.array([
        [320, 0, 320],
        [0, 320, 240],
        [0, 0, 1]
    ])
    pts, p = default_synthetic_points(K, M)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[0, :, 0], pts[0, :, 1], pts[0, :, 2])
    ax.set_box_aspect([1,1,1])
    set_axes_equal(ax)
    plt.show()

    S = split_and_merge(p, K)
    print()
    print(S[:8])
    # print(S[8:12])
    # print(S[12:16])
    # print(S[16:40])
    # print(S[40:54])
    print(S[8:32])
    print(S[32:46])


