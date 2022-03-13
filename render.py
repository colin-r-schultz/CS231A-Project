import cv2
import numpy as np
from scipy.spatial.transform import Slerp
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
    R_int = Slerp(t, R)
    T_int = interp1d(t, T, axis=0)
    t_new = np.linspace(0, M-1, (M - 1) * frames_per)
    return R_int(t_new), T_int(t_new)


def render(points, object_lines):
    points = points.astype(int)
    M, N, _ = points.shape
    bg = 255 if object_lines is not None else 0
    frames = np.full((M, IMAGE_HEIGHT, IMAGE_WIDTH, 3), bg, dtype=np.uint8)
    for i in range(M):
        if object_lines is not None:
            prev_idx = 0
            for j, lines in enumerate(object_lines):
                lines = lines + prev_idx
                for i1, i2 in lines:
                    cv2.line(frames[i], points[i, i1], points[i, i2], get_color(j))
                prev_idx = np.max(lines) + 1
        else:
            for j in range(N):
                cv2.circle(frames[i], points[i, j], 2, (255, 255, 255), -1)
    return frames

if __name__ == "__main__":
    M = 8
    objs = [
        make_cube(),
        make_cube(0.5),
        make_cube(1.5),
        make_cylinder(1, 2, 6),
        make_cylinder(1.5, 1, 7),
        make_tetra(0.7),
        make_tetra(),
        make_tetra(1.5),
    ]
    points, lines = zip(*objs)
    tforms = [interpolate_transforms(*get_random_transforms(M)) for _ in objs]
    points = np.concatenate(
        [transform_shape(p, *t) for p, t in zip(points, tforms)],
        axis=1
    )
    p = project_points(points, K)

    frames = render(p, lines)
    frames2 = render(p, None)
    for i in range(frames.shape[0]):
        cv2.imshow("wireframe", frames[i])
        cv2.imshow("keypoints", frames2[i])
        cv2.waitKey(50)