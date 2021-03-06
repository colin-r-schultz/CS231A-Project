import numpy as np
from sklearn.metrics import v_measure_score
import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tfg_transformation

def multibody_sfm(points, K, O, iters=3000, verbose=False, init_p=None):
    """Compute 3D structure over multiple frames

    M = number of frames
    N = number of points

    Inputs:
        points - M x N x 2 2d pixel positions of N points over M frames
        segmentation - length N array of object ids [0-O]
        K - 3x3 intrinsics matrix
        O - number of objects

    Returns:
        structure - N x 3  3d world positions of points in their own rigid body
        projections - M x N x 3 2d pixel projections of N moints over M frames
        probs - N x O probability of object membership for each of N points
    """
    M, N, _ = points.shape
    X = tf.Variable(np.random.randn(N, 3) * 0.1, dtype=tf.float64)
    if init_p is None:
        init_p = np.random.randn(N, O) * 0.01
    P = tf.Variable(init_p, dtype=tf.float64)
    # quat = tf.Variable(np.tile([0, 0, 0, 1], (M, O, 1)), dtype=tf.float64)
    quat = tf.Variable(np.random.rand(M, O, 4), dtype=tf.float64)
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
        #res_loss = tf.reduce_mean(tf.reduce_sum(P_ * tf.reduce_sum(tf.square(res), axis=-1), axis=-1))
        res_loss = tf.reduce_mean(tf.reduce_sum(P_ * tf.norm(res, axis=-1), axis=-1))
        
        centroid_loss = tf.reduce_sum(tf.square(tf.reduce_mean(X, axis=0)))
        loss = res_loss + 0.01 * centroid_loss
        return loss

    var = [X, T, quat]
    opt = tf.keras.optimizers.Adam(0.01)

    for i in range(iters):
        if verbose and i % 100 == 0:
            l = loss()
            print(i, l.numpy())
        if i == 100:
            var.append(P)
        opt.minimize(loss, var)

    res = residuals().numpy()
    P_ = tf.nn.softmax(P, axis=-1)
    # print("loss")
    # print(np.mean(np.linalg.norm(res, axis=-1), axis=0))

    return X.numpy(), project().numpy(), P_.numpy()

if __name__ == "__main__":

    from data import load_dataset
    import matplotlib.pyplot as plt
    from utils import *
    from collections import Counter
    from sklearn.metrics import v_measure_score

    K = np.array([
        [320, 0, 320],
        [0, 320, 240],
        [0, 0, 1]
    ])
    M = 64
    p, ids = load_dataset("datasets/3mixed_4.npz", K, num_frames=M)
    O = np.max(ids) + 1

    pts2, p2, prob = multibody_sfm(p, K, O, iters=3000, verbose=True)
    print(np.round(prob, 2))

    classes = np.argmax(prob, axis=-1)
    print("CLASSES", classes)
    print("V_MEASUERE", v_measure_score(ids, classes))

    c = Counter(zip(ids, classes))
    s = [100*c[(xx,yy)] for xx,yy in zip(ids,classes)]
    plt.scatter(ids, classes, s=s)
    plt.show()

    p2 = np.take_along_axis(p2, classes[np.newaxis, ..., np.newaxis, np.newaxis], axis=2)
    p2 = np.squeeze(p2)
    

    for o in range(O):
        obj_pts = (classes == o)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pts2[obj_pts, 0], pts2[obj_pts, 1], pts2[obj_pts, 2])
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