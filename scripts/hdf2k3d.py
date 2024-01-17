from pathlib import Path
from typing import Sequence

import dolfin as df
import k3d
import numpy as np
import pantarei as pr
import pyvista as pv

infile = "results/normal_noise/mean1_std1_N5.hdf"
infile = Path(infile)


CELLTYPES = {"interval": pv.CellType.LINE, "triangle": pv.CellType.TRIANGLE}


def create_pv_grid(domain):
    cells = domain.cells()
    celltype = CELLTYPES[domain.cell_name()]
    cell_count = celltype.bit_length()
    cells_expanded = np.hstack([np.full((len(cells), 1), cell_count), cells])

    points = domain.coordinates()
    points3D = np.zeros((len(points), 3))
    points3D[:, : domain.geometric_dimension()] = points

    grid = pv.UnstructuredGrid(
        cells_expanded.flatten(), np.array([celltype] * len(cells)), points3D
    )
    return grid


def read_pv_data(filename: Path, funcname: str, animation_speed: float = 1.0):
    with df.HDF5File(df.MPI.comm_world, str(filename), "r") as hdf:
        domain = pr.read_domain(hdf)
        timevec = pr.read_timevector(hdf, funcname)
        u = pr.read_function(hdf, funcname, domain, idx=len(timevec) - 1)
        num_steps = len(timevec)
        grid = create_pv_grid(domain)
        dofmap = df.vertex_to_dof_map(u.function_space())
        for idx in range(num_steps):
            pr.read_checkpoint(hdf, u, funcname, idx)
            grid[f"{funcname}_{idx}"] = u.vector()[dofmap].reshape(-1, 1)
    return grid, timevec / animation_speed


def pv_data_to_k3d(
    grid: pv.UnstructuredGrid, timevec: np.ndarray, funcname: str, dim: int
) -> k3d.objects.Drawable:
    num_steps = len(timevec)
    if dim == 1:
        k3d_grid = k3d.line(
            vertices=np.array(grid.points, dtype="single"),
            name=funcname,
        )
    else:
        k3d_grid = k3d.vtk_poly_data(grid.extract_geometry(), side="double")
    k3d_grid.attribute = {
        str(timevec[idx]): np.array(grid[f"{funcname}_{idx}"], dtype="single")
        for idx in range(num_steps)
    }
    k3d_grid.vertices = {
        str(timevec[idx]): np.array(
            grid.warp_by_scalar(f"{funcname}_{idx}").points, dtype="single"
        )
        for idx in range(num_steps)
    }
    k3d_grid.color_range = k3d.helpers.minmax(list(k3d_grid.attribute.values()))
    return k3d_grid


def k3d_plot_as_html(output: Path, grids: Sequence[k3d.objects.Drawable]):
    pl = k3d.plot()
    for grid in grids:
        pl += grid

    with open(output, "w") as fp:
        fp.write(pl.get_snapshot(additional_js_code="K3DInstance.startAutoPlay()"))
    return pl


if __name__ == "__main__":
    input = "data/data.hdf"
    input = "data/sim.hdf"
    output = "output.html"

    grid_sim, timevec_sim = read_pv_data(
        Path("data/sim.hdf"), "concentration", animation_speed=0.1
    )
    k3d_sim = pv_data_to_k3d(grid_sim, timevec_sim, "concentration", dim=1)
    grid_data, timevec_data = read_pv_data(
        Path("data/data.hdf"), "concentration", animation_speed=0.1
    )
    k3d_data = pv_data_to_k3d(grid_data, timevec_data, "concentration", dim=1)
    pl = k3d_plot_as_html(Path(output), [k3d_sim, k3d_data])
    print("Output available at: ", output)
