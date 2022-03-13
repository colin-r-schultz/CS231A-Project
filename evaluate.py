import os
from pathlib import Path

import numpy as np
from sklearn.metrics import v_measure_score
from continuous_sfm import multibody_sfm

from data import load_dataset
from split_and_merge import split_and_merge

DATA_DIR = Path("datasets")

M = 64

K = np.array([
    [320, 0, 320],
    [0, 320, 240],
    [0, 0, 1]
])

def evaluate(method, n_obj=None, suffix=None):
    """Evaluate a segmentation method

    Args:
        method (function): A function that should take points and K and retrun segmentation
        n_obj (int, optional): If provided, only evaluates on dataset with this many objects
        type (str, optional): If provided, only evaluates on datasets with this suffix
    """
    files = os.listdir(DATA_DIR)
    if n_obj is not None:
        files = filter(lambda f: f.startswith(str(n_obj)), files)
    if suffix is not None:
        files = filter(lambda f: f.endswith(suffix + ".npz"), files)
    scores = []
    for fname in files:
        points, labels = load_dataset(DATA_DIR / fname, K, M)
        segmentation = method(points, K)
        score = v_measure_score(labels, segmentation)
        scores.append(score)
    return scores



if __name__ == "__main__":
    O = 6
    def method(points, K):
        _, _, prob = multibody_sfm(points, K, O, iters=1500)
        seg = np.argmax(prob, axis=-1)
        return seg

    scores = evaluate(method, 6)
    print("####################################")
    print(scores)
    print("median", np.median(scores))