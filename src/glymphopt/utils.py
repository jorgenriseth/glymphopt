from pathlib import Path

import dolfin as df
import numpy as np
import pantarei as pr
import ufl


def load_data_interpolator(filename: Path, funcname: str) -> pr.DataInterpolator:
    with df.HDF5File(df.MPI.comm_world, str(filename), "r") as hdf:
        domain = pr.read_domain(hdf)
        element = pr.read_element(hdf, funcname)
        timevec = pr.read_timevector(hdf, funcname)
        V = df.FunctionSpace(domain, element)
        N = len(timevec)
        data = [df.Function(V, name=f"{funcname}_{idx}") for idx in range(N)]
        for idx in range(N):
            pr.read_checkpoint(hdf, data[idx], name=funcname, idx=idx)
    return pr.DataInterpolator(data, timevec)


class MeshExpression(df.UserExpression):
    """Assign values from a dict to cells labeled by subdomains."""

    def __init__(
        self,
        mesh: df.Mesh,
        subdomains: df.MeshFunction,
        subdomain_map: dict[int, ufl.Coefficient],
        **kwargs,
    ):
        super().__init__(kwargs)
        self.mesh = mesh
        self.subdomains = subdomains
        self.subdomain_map = subdomain_map

    def eval_cell(self, value, x, ufc_cell):
        cell = df.Cell(self.mesh, ufc_cell.index)
        cell_subdomain = self.subdomains[cell]
        value[0] = self.subdomain_map[cell_subdomain]

    def value_shape(self):
        return ()


def cell_midpoints(mesh: df.Mesh) -> np.ndarray:
    return np.array(
        [cell.midpoint().array()[: cell.dim()] for cell in df.cells(mesh)],
    )


def apply_affine(T: np.ndarray, X: np.ndarray) -> np.ndarray:
    A, b = T[:-1, :-1], T[:-1, -1]
    return X.dot(A) + b
