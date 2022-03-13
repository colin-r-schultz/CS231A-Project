import numpy as np

from synthetic import get_random_transforms, project_points, transform_shape, make_cube, make_cylinder, make_tetra
from data import save_dataset

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

    for i in range(1, 4):
        # make_cylinder(0.5, 1.0, 8)
        objs = [make_tetra(), make_cube(), make_cylinder(0.5, 1.0, 8)] * i
        filename = str(3*i) + "tetras_cubes_cylinders_mixed"
        save_dataset(objs, M, "datasets/" + filename)