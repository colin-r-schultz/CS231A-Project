import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

def get_random_transforms(M):
    """PRoduce M random tranformations

    Args:
        M (int): number of transformations

    Returns:
        R: set of scipy rotations
        T (M x 3)
    """
    R = Rotation.random(M)
    T = 5 * (np.random.rand(M, 3) - 0.5) + [0, 0, 10]
    return R, T

def transform_shape(shape, R, T):
    """Randomly rotate and translate a shape M times

    Args:
        shape (N x 3): 3d points of shape
        R: (M x 3 x 3) rotation matrices
        T: (M x 3) translations

    Returns:
        points (M x N x 3)
    """
    points = (R @ shape.T).transpose(0, 2, 1)
    points = points + T.reshape(-1, 1, 3)
    return points

def project_points(points, K):
    """Project 3d points to pixels

    Args:
        points (M x N x 3): 3d points
        K (3 x 3): Camera intrinsics

    Returns:
        pixels: (M x N x 2)
    """
    pixels = points @ K.T
    pixels = pixels[..., :2] / pixels[..., 2:]

    return pixels

def make_cylinder(r, h, n):
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xyz = np.array([r * np.cos(theta), r * np.sin(theta), np.full(n, -h/2)]).T
    xyz_2 = xyz.copy()
    xyz_2[:, 2] += h
    points = np.concatenate([xyz, xyz_2], axis=0)
    idx = np.arange(n)
    lines_top = np.c_[idx, (idx + 1) % n]
    lines_bottom = lines_top + n
    lines_side = np.c_[idx, idx + n]
    lines = np.concatenate([lines_top, lines_bottom, lines_side])
    return points, lines

def make_cube(r=1):
    return make_cylinder(r * np.sqrt(2), r * 2, 4)

def make_tetra(r=1):
    tetra = np.array([
        [0, 0, 1],
        [np.sqrt(8/9), 0, -1/3],
        [-np.sqrt(2/9), np.sqrt(2/3), -1/3],
        [-np.sqrt(2/9), -np.sqrt(2/3), -1/3]
    ])
    lines = np.array([
        [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]
    ])
    return tetra * r, lines

def generate_synthetic_points(K, M=4):
    """Generate synthetic points from random rigid bodies
    """
    objs = [
        make_cube(1)[0],
        make_cylinder(1, 2, 12)[0],
        make_cylinder(1.5, 1, 7)[0]
    ]
    points = np.concatenate([
        transform_shape(cube, M),
        # transform_shape(tetra, M),
        # transform_shape(tetra, M),
        transform_shape(make_cylinder(1, 2, 12), M),
        transform_shape(make_cylinder(1.5, 1, 7), M)
    ], axis=1)

    return points, project_points(points, K)


if __name__ == "__main__":
    M = 4
    K = np.array([
        [320, 0, 320],
        [0, 320, 240],
        [0, 0, 1]
    ])
    pts, p = default_synthetic_points(K, M)
    np.savetxt("3d_points.txt", pts.ravel('F'))
    np.savetxt("pixels.txt", p.ravel('F'))

    for i in range(M):
        plt.axis("equal")
        plt.xlim([0, 640])
        plt.ylim([0, 480])
        plt.scatter(p[i, :, 0], p[i, :, 1])
        plt.show()

