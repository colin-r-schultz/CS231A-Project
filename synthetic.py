import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

def transform_shape(shape, M):
    """Randomly rotate and translate a shape M times

    Args:
        shape (N x 3): 3d points of shape
        M (int): number of transformations to return

    Returns:
        points (M x N x 3)
    """
    R = Rotation.random(M).as_matrix()
    T = 3 * (np.random.rand(M, 3) - 0.5) + [0, 0, 6]
    points = (R @ shape.T).transpose(0, 2, 1)
    points = points + T.reshape(-1, 1, 3)

    return points

def make_cylinder(r, h, n):
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xyz = np.array([r * np.cos(theta), r * np.sin(theta), np.full(n, -h/2)]).T
    xyz_2 = xyz.copy()
    xyz_2[:, 2] += h
    return np.concatenate([xyz, xyz_2], axis=0)

def generate_synthetic_points(K, M=4):
    """Generate synthetic points from 
    """
    cube = np.ones((1, 3))
    cube = np.r_[cube, cube * [-1, 1, 1]]
    cube = np.r_[cube, cube * [1, -1, 1]]
    cube = np.r_[cube, cube * [1, 1, -1]] # 8 x 3

    tetra = np.array([
        [0, 0, 1],
        [np.sqrt(8/9), 0, -1/3],
        [-np.sqrt(2/9), np.sqrt(2/3), -1/3],
        [-np.sqrt(2/9), -np.sqrt(2/3), -1/3]
    ])

    points = np.concatenate([
        transform_shape(cube, M),
        transform_shape(tetra, M),
        transform_shape(tetra, M),
        transform_shape(make_cylinder(1, 2, 12), M),
        transform_shape(make_cylinder(1.5, 1, 7), M)
    ], axis=1)

    pixels = points @ K.T
    pixels = pixels[..., :2] / pixels[..., 2:]

    return points, pixels

if __name__ == "__main__":
    M = 4
    K = np.array([
        [320, 0, 320],
        [0, 320, 240],
        [0, 0, 1]
    ])
    pts, p = generate_synthetic_points(K, M)
    np.savetxt("3d_points.txt", pts.ravel('F'))
    np.savetxt("pixels.txt", p.ravel('F'))

    for i in range(M):
        plt.axis("equal")
        plt.xlim([0, 640])
        plt.ylim([0, 480])
        plt.scatter(p[i, :, 0], p[i, :, 1])
        plt.show()

