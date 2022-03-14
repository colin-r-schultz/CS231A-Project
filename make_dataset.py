import numpy as np
import random

from synthetic import get_random_transforms, project_points, transform_shape, make_cube, make_cylinder, make_tetra
from data import save_dataset

def make_random_dataset(M, file_num, O=None):
    O = O or np.random.randint(2, 9)

    shapes = ["tetra", "cube", "cylinder"]
    objs = []
    for o in range(O):
        s = random.choice(shapes)
        if s == "tetra":
            objs.append(make_tetra(np.random.rand() + 0.5))
        elif s == "cube":
            objs.append(make_cube(np.random.rand() + 0.5))
        else:
            objs.append(make_cylinder(0.5*np.random.rand() + 0.25, np.random.rand() + 0.5, np.random.randint(3, 13)))
    
    save_dataset(objs, M, "datasets/" + str(O) + "mixed_" + str(file_num))

if __name__ == "__main__":
    M = 128
    # objs = [
    #     make_cube(),
    #     make_cube(0.5),
    #     make_cube(1.5),
    #     make_cylinder(1, 2, 6),
    #     make_cylinder(1.5, 1, 7),
    #     make_tetra(0.7),
    #     make_tetra(),
    #     make_tetra(1.5),
    # ]

    # for i in range(1, 4):
    #     # make_cylinder(0.5, 1.0, 8)
    #     objs = [make_tetra(), make_cube(), make_cylinder(0.5, 1.0, 8)] * i
    #     filename = str(3*i) + "tetras_cubes_cylinders_mixed"
    #     save_dataset(objs, M, "datasets/" + filename)

    for o in range(2, 9):
        for i in range(10):
            make_random_dataset(M, i, O=o)
