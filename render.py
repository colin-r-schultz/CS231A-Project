import cv2
import numpy as np
from scipy.spatial.transform import Slerp, Rotation
from scipy.interpolate import interp1d

from synthetic import transform_shape, get_random_transforms, make_cube, make_cylinder, make_tetra, project_points

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
FOV = 70
FOCAL_LENGTH = IMAGE_WIDTH / (2 * np.tan(np.deg2rad(FOV / 2)))

K = np.array([
    [FOCAL_LENGTH, 0, IMAGE_WIDTH / 2],
    [0, FOCAL_LENGTH, IMAGE_HEIGHT / 2],
    [0, 0, 1]
])

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255)]


def get_color(i):
    global COLORS
    while i >= len(COLORS):
        COLORS.append(np.random.randint(0, 256, 3).tolist())
    return COLORS[i]

def interpolate_transforms(R, T, frames_per=20):
    M = len(R)
    t = np.arange(M)
    R = Rotation.from_matrix(R)
    R_int = Slerp(t, R)
    T_int = interp1d(t, T, axis=0)
    t_new = np.linspace(0, M-1, (M - 1) * frames_per)
    return R_int(t_new).as_matrix(), T_int(t_new)


def load_dataset(fname, K):
    data = np.load(fname)
    points = data["points"]
    ids = data["ids"]
    Rs = data["Rs"]
    Ts = data["Ts"]
    lines = data["lines"]

    objs = []

    O = np.max(ids) + 1
    for i in range(O):
        pts = points[ids == i]
        R, T = interpolate_transforms(Rs[i], Ts[i])
        pts = transform_shape(pts, R, T)
        objs.append(pts)

    points = np.concatenate(objs, axis=1)
    pixels = project_points(points, K)

    return pixels, ids, lines

def render(points, ids, object_lines):
    points = points.astype(int)
    M, N, _ = points.shape
    bg = 255 if object_lines is not None else 0
    frames = np.full((M, IMAGE_HEIGHT, IMAGE_WIDTH, 3), bg, dtype=np.uint8)
    for i in range(M):
        if object_lines is not None:
            for i1, i2 in object_lines:
                cv2.line(frames[i], points[i, i1], points[i, i2], get_color(ids[i1]))
        else:
            for j in range(N):
                cv2.circle(frames[i], points[i, j], 2, (255, 255, 255), -1)
    return frames

if __name__ == "__main__":
    p, ids, lines = load_dataset("datasets/8mixed_4.npz", K)

    frames = render(p, ids, lines)
    frames2 = render(p, ids, None)
    for i in range(frames.shape[0]):
        cv2.imshow("wireframe", frames[i])
        cv2.imshow("keypoints", frames2[i])
        cv2.waitKey(50)