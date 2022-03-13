import numpy as np
import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tfg_transformation
from synthetic import generate_synthetic_points
import matplotlib.pyplot as plt
from utils import *

def multibody_sfm(points,  K, iters=3000, verbose=False):
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

    M, N, _ = points.shape
    X = tf.Variable(np.zeros((N, 3)), dtype=tf.float64)
    r = tf.convert_to_tensor([[0., 0., 0., 1.]])
    t = tf.convert_to_tensor([[0., 0., 1.]])
    double_q = tfg_transformation.dual_quaternion.from_rotation_translation(r, t)
    RT = tf.Variable(tf.cast(tf.tile(double_q, (M, 1)), dtype=tf.float64))
    

    K = tf.constant(K, dtype=tf.float64)
    points = tf.constant(points, dtype=tf.float64)

    @tf.function
    def project():
        R, T = tfg_transformation.dual_quaternion.to_rotation_translation(RT)
        X_ = tfg_transformation.quaternion.rotate(tf.reshape(X, (1, N, 3)), tf.reshape(R, (M, 1, 4)))
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

    var1 = [X]
    var2 = [RT]
    opt = tf.keras.optimizers.Adam(0.01)

    for i in range(iters):
        if verbose and i % 1000 == 0:
            l = loss()
            print(i, l.numpy())
        if int(i / 100) % 2 == 0:
            opt.minimize(loss, var1)
        else:
            opt.minimize(loss, var2)
        # opt.minimize(loss, var2 + var1)

    res = residuals().numpy()
    return X.numpy(), project().numpy(), res

def visualize(M, K):
    pts, p = generate_synthetic_points(K, M)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[0, :, 0], pts[0, :, 1], pts[0, :, 2])
    ax.set_box_aspect([1,1,1])
    set_axes_equal(ax)
    plt.show()

    pts2, p2, _ = multibody_sfm(p, K, iters=3000)
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


if __name__ == "__main__":
    M = 16
    K = np.array([
        [320, 0, 320],
        [0, 320, 240],
        [0, 0, 1]
    ])
    for i in range(10):
        pts, p = generate_synthetic_points(K, M)
        pts2, p2, res = multibody_sfm(p, K, iters=3000)
        loss = np.mean(np.linalg.norm(res, axis=-1))
        print(i+1, loss)