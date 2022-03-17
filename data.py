import numpy as np

from synthetic import get_random_transforms, project_points, transform_shape, make_cube, make_cylinder, make_tetra
from utils import set_axes_equal


def save_dataset(objs, M, fname):
    points = []
    ids = []
    lines = []
    prev_id = 0
    Rs = []
    Ts = []
    for i, (pts, lns) in enumerate(objs):
        ids += [i] * pts.shape[0]
        points.append(pts)
        lines.append(lns + prev_id)
        prev_id += pts.shape[0]
        R, T = get_random_transforms(M)
        Rs.append(R.as_matrix())
        Ts.append(T)
    points = np.concatenate(points, axis=0)
    ids = np.array(ids)
    lines = np.concatenate(lines, axis=0)
    Rs = np.stack(Rs, axis=0)
    Ts = np.stack(Ts, axis=0)
    np.savez(fname, points=points, ids=ids, lines=lines, Rs=Rs, Ts=Ts)

def load_dataset(fname, K, num_frames=None):
    data = np.load(fname)
    points = data["points"]
    ids = data["ids"]
    Rs = data["Rs"][:num_frames]
    Ts = data["Ts"][:num_frames]

    objs = []

    O = np.max(ids) + 1
    for i in range(O):
        pts = points[ids == i]
        pts = transform_shape(pts, Rs[i], Ts[i])
        objs.append(pts)

    points = np.concatenate(objs, axis=1)
    pixels = project_points(points, K)
    return pixels, ids

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    data = np.load("datasets/8mixed_8.npz")
    points = data["points"]
    ids = data["ids"]

    for i in range(np.max(ids) + 1):
        pts = points[ids==i]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2])
        ax.set_box_aspect([1,1,1])
        set_axes_equal(ax)
        plt.show()