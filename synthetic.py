import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

def generate_synthetic_points(K, M=4):
    """Generate synthetic points from 
    """
    cube = np.ones((1, 3))
    cube = np.r_[cube, cube * [-1, 1, 1]]
    cube = np.r_[cube, cube * [1, -1, 1]]
    cube = np.r_[cube, cube * [1, 1, -1]] # 8 x 3

    R = Rotation.random(M).as_matrix()
    T = (np.random.rand(M, 3) - 0.5) + [0, 0, 6]
    points = (R @ cube.T).transpose(0, 2, 1)
    points = points + T.reshape(-1, 1, 3)

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
    _, p = generate_synthetic_points(K, M)
    for i in range(M):
        plt.axis("equal")
        plt.xlim([0, 640])
        plt.ylim([0, 480])
        plt.scatter(p[i, :, 0], p[i, :, 1])
        plt.show()

