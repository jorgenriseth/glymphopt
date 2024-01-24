from pathlib import Path

import dolfin as df
import matplotlib.pyplot as plt
import numpy as np
import pantarei as pr
import ufl

from glymphopt.utils import (
    MeshExpression,
    apply_affine,
    cell_midpoints,
    load_data_interpolator,
)
from glymphopt.visual import plot_step


def quadrature_measurement(
    f: ufl.Coefficient, measure_domain: df.Mesh, degree: int = 3
) -> np.ndarray:
    """Perform measurement adhering to the 'measure_domain' of a dolfin
    expression or function 'f' by performing cellwise quadrature on f and return
    the values as a numpy-array."""
    dx = df.Measure("dx", measure_domain)
    DG0 = df.FunctionSpace(measure_domain, "DG", 0)
    v = df.TestFunction(DG0)
    measures = df.assemble(
        (1.0 / df.CellVolume(measure_domain)) * df.inner(f, v) * dx,
        form_compiler_parameters={"quadrature_degree": degree},
    )
    return np.array(measures)


def interpolation_measurement(
    f: ufl.Coefficient, measure_domain: df.Mesh
) -> np.ndarray:
    """Interpolate expression/or function onto a DG0-function space"""
    DG0 = df.FunctionSpace(measure_domain, "DG", 0)
    u = df.interpolate(f, DG0)
    return np.array(u.vector())


def create_index_to_coordinates_affine_map(domain: df.Mesh):
    """Given a 1D domain, get the constants for an affine map from cell index
    to the cell midpoint."""
    n = domain.num_cells()
    cell_centers = cell_midpoints(domain)
    xmin, xmax = cell_centers.min(), cell_centers.max()
    dim = domain.geometric_dimension()
    T = np.zeros((dim + 1, dim + 1))
    T[:-1, :-1] = (xmax - xmin) / (n - 1)
    T[:-1, -1] = xmin
    T[-1, -1] = 1.0
    return T


def measure_from_file(
    inputfile: Path,
    funcname: str,
    resolution: int,
    timestamp: Path,
) -> None:
    """Given dolfin-function stored in an hdf-file "inputfile", store the DG0-
    discrete measurement with given resolution as in a txt-file."""
    u_data = load_data_interpolator(inputfile, funcname)
    np.savetxt(timestamp, u_data.times)
    domain = pr.MMSInterval(resolution)
    affine = create_index_to_coordinates_affine_map(domain)
    for idx, ci in enumerate(u_data.data):
        Ci = interpolation_measurement(ci, domain)
        np.savez(f"data/concentration_{idx}.npz", fdata=Ci, affine=affine)


def visualize_measurement(filename: Path, affine_path: Path):
    """Simple function for visualization of measurements stored in 'txt'-files"""
    C = np.loadtxt(filename)
    affine = dict(np.load(affine_path))
    ind = np.arange(len(C))

    plt.figure()
    plot_step(apply_affine(affine, ind), C)
    plt.show()


def concentrations_to_T1_maps(
    c: ufl.Coefficient, resolution: int, subdomain_map: dict[int, float]
):
    r1 = 3.2
    domain = pr.MMSInterval(resolution)
    T10_expr = MeshExpression(domain, domain.subdomains, subdomain_map, degree=0)

    DG0 = df.FunctionSpace(domain, "DG", 0)
    T10 = df.interpolate(T10_expr, DG0)

    DG1 = df.FunctionSpace(domain, "DG", 1)
    R1_expr = 1.0 / (1.0 / T10 + r1 * c)
    R1_bdry = df.Constant(1.0 / (1.0 / T10(-1) + r1 * c(-1)))
    R1_dg1 = interpolate_to_functionspace(
        R1_expr, DG1, [df.DirichletBC(DG1, R1_bdry, "on_boundary")]
    )
    T1_expr = 1.0 / R1_dg1
    T1_bdry = df.Constant(1.0 / R1_dg1(-1))
    T1_dg1 = interpolate_to_functionspace(
        T1_expr, DG1, [df.DirichletBC(DG1, T1_bdry, "on_boundary")]
    )
    return interpolation_measurement(T1_dg1, domain)


def T1_measurement_from_file(
    inputfile: Path,
    funcname: str,
    resolution: int,
    T10: dict[int, float],
    timestamp: Path,
) -> None:
    u_data = load_data_interpolator(inputfile, funcname)
    np.savetxt(timestamp, u_data.times)
    domain = pr.MMSInterval(resolution)
    affine = create_index_to_coordinates_affine_map(domain)
    for idx, ci in enumerate(u_data.data):
        T1 = concentrations_to_T1_maps(ci, resolution, T10)
        np.savez(f"data/T1maps_{idx}.npz", {"fdata": T1, "affine": affine})
    return


def interpolate_to_functionspace(
    f: ufl.Coefficient, V: df.FunctionSpace, boundaries: list[df.DirichletBC]
) -> df.Function:
    dx = df.Measure("dx", V.mesh())
    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    a = u * v * dx
    l = f * v * dx

    A = df.assemble(a)
    b = df.assemble(l)
    for bc in boundaries:
        bc.apply(A, b)
    u = df.Function(V)
    df.solve(A, u.vector(), b)
    return u


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", type=Path, required=True)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--timestamps_out", type=Path, required=True)
    parser.add_argument("--measure_T1", action="store_true")

    args = parser.parse_args()
    if args.measure_T1:
        T10 = {1: 1.0, 2: 1.2}
        T1_measurement_from_file(
            args.infile, "concentration", args.resolution, T10, args.timestamps_out
        )
    else:
        measure_from_file(
            args.infile, "concentration", args.resolution, args.timestamps_out
        )
