import numpy as np
import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tfg_transformation
from synthetic import generate_synthetic_points
import matplotlib.pyplot as plt
from utils import *
from data import load_dataset

def multibody_sfm(points,  K, iters=3000):
    """Compute 3D structure over multiple frames

    M = number of frames
    N = number of points

    Inputs:
        points - M x N x 2 2d pixel positions of N points over M frames
        segmentation - length N array of object ids [0-O]
        K - 3x3 intrinsics matrix

    Returns:
        structure - N x 3  3d world positions of rigid body
        projections - M x N x 3 2d pixel projections of N moints over M frames
    """
    max_norm = np.max(np.linalg.norm(points[:, 1] - points[:, 0], axis=-1))
    z_guess = K[0, 0] / max_norm
    # print("########Z_GUESS", z_guess)

    M, N, _ = points.shape
    X = tf.Variable(np.zeros((N, 3)), dtype=tf.float64)
    angle = tf.Variable(np.zeros((M, 3)), dtype=tf.float64)
    T = tf.Variable(np.tile([0, 0, z_guess], (M, 1)), dtype=tf.float64)

    K = tf.constant(K, dtype=tf.float64)
    points = tf.constant(points, dtype=tf.float64)

    @tf.function
    def project():
        R = tfg_transformation.rotation_matrix_3d.from_euler(angle)
        X_ = tfg_transformation.rotation_matrix_3d.rotate(tf.reshape(X, (1, N, 3)), tf.reshape(R, (M, 1, 3, 3)))
        X_ += tf.reshape(T, (M, 1, 3))
        pixels = X_ @ tf.transpose(K)
        pixels = pixels[:, :, :2] / pixels[:, :, 2:]

        return pixels

    @tf.function
    def residuals():
        return project() - points

    @tf.function
    def loss():
        res = residuals()
        res_loss = tf.reduce_mean(tf.norm(res, axis=-1))
        centroid_loss = tf.reduce_sum(tf.square(tf.reduce_mean(X, axis=0)))
        loss = res_loss + 0.01 * centroid_loss
        return loss

    var = [X, angle, T]
    opt = tf.keras.optimizers.Adam(0.01)

    # features = get_features()
    # print("features!", features.shape)
    # np.save("features2objs0iters", features)

    for i in range(iters):
        if i % 1000 == 0:
            l = loss()
            print(i, l.numpy())
        opt.minimize(loss, var)


    # features = get_features()
    # print("features!", features.shape)
    # np.save("features2objs10000iters", features)

    res = residuals().numpy()
    # print("loss")
    # print(np.mean(np.linalg.norm(res, axis=-1), axis=0))

    return X.numpy(), project().numpy(), res

if __name__ == "__main__":
    K = np.array([
        [320, 0, 320],
        [0, 320, 240],
        [0, 0, 1]
    ])

    p, ids = load_dataset('test_data/8obj_8frame_1.npz', K)

    pts2, p2, res = multibody_sfm(p, K, iters=10000)
    print(pts2)

    pt_loss = np.mean(np.linalg.norm(res, axis=-1), axis=0)

    plt.scatter(ids, pt_loss)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts2[:, 0], pts2[:, 1], pts2[:, 2])
    ax.set_box_aspect([1,1,1])
    set_axes_equal(ax)
    plt.show()

    M = p.shape[0]
    for i in range(M):
        plt.axis("equal")
        plt.xlim([0, 640])
        plt.ylim([0, 480])
        plt.scatter(p[i, :, 0], p[i, :, 1])
        plt.scatter(p2[i, :, 0], p2[i, :, 1])
        plt.show()