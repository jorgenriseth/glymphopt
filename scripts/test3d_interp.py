import itertools
import subprocess

import dolfin as df
import matplotlib.pyplot as plt
import nibabel
import numpy as np
import pantarei as pr
import ufl
from tqdm import tqdm

from glymphopt.measure import (
    create_index_to_coordinates_affine_map,
    interpolation_measurement,
    quadrature_measurement,
)
from glymphopt.utils import apply_affine, cell_midpoints
from glymphopt.visual import plot_step

expr = " + ".join(
    [
        f"(abs(x[{i}]) <= 1 ? 1 : 0.1) * (exp(-pow((x[{i}] - 0.7)/a1, 2)) + 0.5*exp(-pow((x[{i}] + 0.6)/a1, 2)))"
        for i in range(3)
    ]
)
# Define "true" function
f = df.Expression(expr, degree=3, a1=0.4)


def grid_affine(domain: df.Mesh, shape: tuple[int, int, int]) -> np.ndarray:
    points = domain.coordinates()
    pmin, pmax = points.min(axis=0), points.max(axis=0)
    dim = domain.geometric_dimension()
    T = np.zeros((dim + 1, dim + 1))
    T[-1, -1] = 1.0
    for i, ni in enumerate(shape):
        T[i, i] = (pmax[i] - pmin[i]) / ni
        T[i, -1] = (pmax[i] - pmin[i]) / (2 * ni) + pmin[i]
    return T


# Define MRI-domain/grid, and perform measurement, i.e. map function to array of values
def measure_function(
    pmin: df.Point,
    pmax: df.Point,
    resolution: int | tuple[int, int, int],
) -> nibabel.nifti1.Nifti1Image:
    if isinstance(resolution, int):
        resolution = (resolution, resolution, resolution)
    grid = df.BoxMesh.create(
        [pmin, pmax],
        resolution,
        df.CellType.Type.hexahedron,
    )
    aff = grid_affine(grid, resolution)  # Find affine of the domain

    V = df.FunctionSpace(grid, "DG", 0)
    u = df.interpolate(f, V)
    z = V.tabulate_dof_coordinates()

    ind = np.rint(apply_affine(np.linalg.inv(aff), z)).astype(int)
    D = np.nan * np.zeros(resolution, dtype=np.single)
    i, j, k = ind.T
    D[i, j, k] = u.vector()[:]
    return nibabel.nifti1.Nifti1Image(D, aff)
