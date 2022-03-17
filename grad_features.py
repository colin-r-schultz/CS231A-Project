import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import v_measure_score
import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tfg_transformation
import matplotlib.pyplot as plt

from data import load_dataset

def sfm_features(points, K):
    """Compute 3D structure over multiple frames
    M = number of frames
    N = number of points
    Inputs:
        points - M x N x 2 2d pixel positions of N points over M frames
        K - 3x3 intrinsics matrix
    Returns:
        structure - M x N x 3  3d world positions of N points of M frames
    """
    M, N, _ = points.shape
    X = tf.Variable(np.zeros((N, 3)), dtype=tf.float64)
    angle = tf.Variable(np.zeros((M, 3)), dtype=tf.float64)
    T = tf.Variable(np.tile([0, 0, 5], (M, 1)), dtype=tf.float64)

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
    def loss():
        res = project() - points
        l = tf.reduce_mean(tf.reduce_sum(tf.square(res), axis=-1))
        # l = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(res), axis=-1)))
        return l

    # not tf.function because we only call this once
    def get_features():
        with tf.GradientTape(persistent=True) as g:
            angles = tf.tile(tf.reshape(angle, (M, 1, 3)), (1, N, 1))
            g.watch(angles)
            R = tfg_transformation.rotation_matrix_3d.from_euler(angles)
            X_ = tfg_transformation.rotation_matrix_3d.rotate(tf.reshape(X, (1, N, 3)), R)

            Ts = tf.tile(tf.reshape(T, (M, 1, 3)), (1, N, 1))
            g.watch(Ts)
            X_ += Ts
            pixels = X_ @ tf.transpose(K)
            pixels = pixels[:, :, :2] / pixels[:, :, 2:]

            res = pixels - points
            l = tf.reduce_mean(tf.reduce_sum(tf.square(res), axis=-1))

        dl_dR = g.gradient(l, angles)
        dl_dT = g.gradient(l, Ts)

        features = tf.concat([dl_dT, dl_dR], axis=-1)
        features = tf.transpose(features, (1, 0, 2))
        features = tf.reshape(features, (N, M * 6))
        return features
    
    var = [X, angle, T]
    opt = tf.keras.optimizers.Adam(0.01)

    # features = get_features()
    # print("features!", features.shape)
    # np.save("features2objs0iters", features)

    for i in range(3000):
        opt.minimize(loss, var)

    features = get_features()

    return features.numpy()


if __name__ == "__main__":
    K = np.array([
        [320, 0, 320],
        [0, 320, 240],
        [0, 0, 1]
    ])
    M = 64
    p, ids = load_dataset("datasets/6cubes_balanced.npz", K, num_frames=M)
    O = np.max(ids) + 1
    features = sfm_features(p, K)
    
    km = KMeans(O)
    clusters = km.fit_predict(features)
    print("Clusters", clusters)
    print(v_measure_score(ids, clusters))
