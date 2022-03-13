import numpy as np
import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tfg_transformation
from synthetic import generate_synthetic_points
import matplotlib.pyplot as plt
from utils import *

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
    print("########Z_GUESS", z_guess)

    M, N, _ = points.shape
    X_trainable = tf.Variable(np.zeros((N - 2, 3)), dtype=tf.float64)
    angle = tf.Variable(np.zeros((M, 3)), dtype=tf.float64)
    T = tf.Variable(np.tile([0, 0, z_guess], (M, 1)), dtype=tf.float64)

    X12 = tf.constant([
        [0, 0, 0],
        [0, 0, 1]
    ], tf.float64)
    K = tf.constant(K, dtype=tf.float64)
    points = tf.constant(points, dtype=tf.float64)

    @tf.function
    def project():
        X = tf.concat([X12, X_trainable], axis=0)
        R = tfg_transformation.rotation_matrix_3d.from_euler(angle)
        X = tfg_transformation.rotation_matrix_3d.rotate(tf.reshape(X, (1, N, 3)), tf.reshape(R, (M, 1, 3, 3)))
        X += tf.reshape(T, (M, 1, 3))
        pixels = X @ tf.transpose(K)
        pixels = pixels[:, :, :2] / pixels[:, :, 2:]

        return pixels

    @tf.function
    def residuals():
        return project() - points

    @tf.function
    def loss():
        res = residuals()
        # l = tf.reduce_mean(tf.reduce_sum(tf.square(res), axis=-1))
        l = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(res), axis=-1)))
        return l

    @tf.function
    def get_features():
        with tf.GradientTape(persistent=True) as g:
            X = tf.concat([X12, X_trainable], axis=0)
            angles = tf.tile(tf.reshape(angle, (M, 1, 3)), (1, N, 1))
            g.watch(angles)
            R = tfg_transformation.rotation_matrix_3d.from_euler(angles)
            X = tfg_transformation.rotation_matrix_3d.rotate(tf.reshape(X, (1, N, 3)), R)

            Ts = tf.tile(tf.reshape(T, (M, 1, 3)), (1, N, 1))
            g.watch(Ts)
            X += Ts
            pixels = X @ tf.transpose(K)
            pixels = pixels[:, :, :2] / pixels[:, :, 2:]

            res = pixels - points
            l = tf.reduce_mean(tf.reduce_sum(tf.square(res), axis=-1))

            dl_dR = g.gradient(l, angles)
            dl_dT = g.gradient(l, Ts)

            features = tf.concat([dl_dT, dl_dR], axis=-1)
            features = tf.transpose(features, (1, 0, 2))
            features = tf.reshape(features, (N, M * 6))
            return features
    
    var = [X_trainable, angle, T]
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
    print("loss")
    print(np.mean(np.linalg.norm(res, axis=-1), axis=0))

    X = tf.concat([X12, X_trainable], axis=0)

    return X.numpy(), project().numpy()

if __name__ == "__main__":
    M = 64
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

    pts2, p2 = multibody_sfm(p, K)
    print(pts2)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts2[:, 0], pts2[:, 1], pts2[:, 2])
    ax.set_box_aspect([1,1,1])
    set_axes_equal(ax)
    plt.show()


    
    for i in range(M):
        plt.axis("equal")
        plt.xlim([0, 640])
        plt.ylim([0, 480])
        plt.scatter(p[i, :, 0], p[i, :, 1])
        plt.scatter(p2[i, :, 0], p2[i, :, 1])
        plt.show()