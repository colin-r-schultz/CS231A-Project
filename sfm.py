import numpy as np

def multibody_sfm(points, K):
    """Compute 3D structure over multiple frames

    M = number of frames
    N = number of points

    Inputs:
        points - M x N x 2 2d pixel positions of N points over M frames
        K - 3x3 intrinsics matrix

    Returns:
        structure - M x N x 3  3d world positions of N points of M frames
    """
    pass