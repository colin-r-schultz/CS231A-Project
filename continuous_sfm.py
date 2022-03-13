import numpy as np
from collections import Counter
import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tfg_transformation
from data import load_dataset
from synthetic import generate_synthetic_points
import matplotlib.pyplot as plt
from utils import *

def multibody_sfm(points, K, O, iters=3000, init_p=None):
    """Compute 3D structure over multiple frames

    M = number of frames
    N = number of points

    Inputs:
        points - M x N x 2 2d pixel positions of N points over M frames
        segmentation - length N array of object ids [0-O]
        K - 3x3 intrinsics matrix
        O - number of objects

    Returns:
        structure - N x 3  3d world positions of rigid body
        projections - M x N x 3 2d pixel projections of N moints over M frames
    """
    M, N, _ = points.shape
    X = tf.Variable(np.random.randn(N, 3) * 0.1, dtype=tf.float64)
    if init_p is None:
        init_p = np.random.randn(N, O) * 0.01
    P = tf.Variable(init_p, dtype=tf.float64)
    quat = tf.Variable(np.tile([0, 0, 0, 1], (M, O, 1)), dtype=tf.float64)
    T = tf.Variable(np.tile([0, 0, 1], (M, O, 1)), dtype=tf.float64)

    K = tf.constant(K, dtype=tf.float64)
    points = tf.constant(points, dtype=tf.float64)

    @tf.function
    def project():
        R = tfg_transformation.quaternion.normalize(quat)
        X_ = tfg_transformation.quaternion.rotate(tf.reshape(X, (1, N, 1, 3)), tf.reshape(R, (M, 1, O, 4)))
        X_ += tf.reshape(T, (M, 1, O, 3))
        pixels = X_ @ tf.transpose(K)
        pixels = pixels[..., :2] / pixels[..., 2:]

        return pixels

    @tf.function
    def residuals():
        return project() - tf.reshape(points, (M, N, 1, 2))

    @tf.function
    def loss():
        res = residuals()

        P_ = tf.reshape(tf.nn.softmax(P, axis=-1), (1, N, O))
        res_loss = tf.reduce_mean(tf.reduce_sum(P_ * tf.norm(res, axis=-1), axis=-1))
        
        centroid_loss = tf.reduce_sum(tf.square(tf.reduce_mean(X, axis=0)))
        loss = res_loss + 0.01 * centroid_loss
        return loss

    var = [X, P, T, quat]
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
    P_ = tf.nn.softmax(P, axis=-1)
    # print("loss")
    # print(np.mean(np.linalg.norm(res, axis=-1), axis=0))

    return X.numpy(), project().numpy(), P_.numpy()

if __name__ == "__main__":
    K = np.array([
        [320, 0, 320],
        [0, 320, 240],
        [0, 0, 1]
    ])
    p, ids = load_dataset("test_data_2obj.npz", K)
    O = np.max(ids) + 1

    init_p = np.random.randn(p.shape[1], O) * 0.01
    # init_p[(range(p.shape[1]), ids)] = 10

    pts2, p2, prob = multibody_sfm(p, K, O, iters=5000, init_p=init_p)
    print(prob)

    classes = np.argmax(prob, axis=-1)
    print("CLASSES", classes)
    print(ids)

    c = Counter(zip(ids, classes))
    s = [100*c[(xx,yy)] for xx,yy in zip(ids,classes)]
    plt.scatter(ids, classes, s=s)
    plt.show()

    p2 = np.take_along_axis(p2, classes[np.newaxis, ..., np.newaxis, np.newaxis], axis=2)
    p2 = np.squeeze(p2)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts2[:, 0], pts2[:, 1], pts2[:, 2])
    ax.set_box_aspect([1,1,1])
    set_axes_equal(ax)
    plt.show()


    
    for i in range(8):
        plt.axis("equal")
        plt.xlim([0, 640])
        plt.ylim([0, 480])
        plt.scatter(p[i, :, 0], p[i, :, 1])
        plt.scatter(p2[i, :, 0], p2[i, :, 1])
        plt.show()