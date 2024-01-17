from pathlib import Path

import dolfin as df
import matplotlib.pyplot as plt
import numpy as np
import pantarei as pr

from glymphopt.utils import apply_affine


def mri2function(filename: Path, domain: df.Mesh) -> df.Function:
    # Load voxeldata and affine map
    mri = np.load(filename)
    C = mri["fdata"]
    affine = mri["affine"]
    affine_inv = np.linalg.inv(affine)

    # Map from array-data to DG0-function space on regular grid.
    N = len(C)
    grid = pr.MMSInterval(N)
    DG0 = df.FunctionSpace(grid, "DG", 0)
    c_dg0 = df.Function(DG0)
    dofs_coordinates = DG0.tabulate_dof_coordinates()
    ind = np.rint(apply_affine(affine_inv, dofs_coordinates)).astype(int)
    c_dg0.vector()[:] = C[ind].flatten()

    # Project DG0-function onto CG1-function with same grid values.
    CG1 = df.FunctionSpace(grid, "CG", 1)
    c_cg1 = df.Function(CG1)
    c_cg1 = df.project(c_dg0, CG1)

    # Interpolate CG1-function from regular grid, to CG1 on mesh.
    CG1 = df.FunctionSpace(domain, "CG", 1)
    c = df.Function(CG1, name="concentration")
    c.assign(df.interpolate(c_cg1, CG1))
    return c


def mri2hdf(
    inputfiles: list[Path], timestampsfile: Path, outputfile: Path, domain: df.Mesh
) -> None:
    timestamps = np.loadtxt(timestampsfile)
    with df.HDF5File(df.MPI.comm_world, str(outputfile), "w") as hdf:
        pr.write_domain(hdf, domain)
        for ti, filename in zip(timestamps, inputfiles):
            ci = mri2function(filename, domain)
            pr.write_checkpoint(hdf, ci, "concentration", ti)


if __name__ == "__main__":
    import argparse

    from glymphopt.visual import data_visual

    parser = argparse.ArgumentParser()
    parser.add_argument("--inputfiles", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timestamps", type=Path, required=True)
    parser.add_argument("--resolution", type=int, required=True)
    args = parser.parse_args()

    domain = pr.MMSInterval(args.resolution)
    mri2hdf(args.inputfiles, args.timestamps, args.output, domain)

    plt.figure()
    data_visual(args.output, "concentration", ax=plt.gca())
    plt.show()
