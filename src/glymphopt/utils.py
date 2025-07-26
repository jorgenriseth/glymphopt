from pathlib import Path

import numpy as np
import pyvista as pv


def dolfin_mesh_to_pyvista_ugrid(mesh):
    coordinates = mesh.coordinates()
    cells = mesh.cells()
    num_cells = cells.shape[0]
    num_points_per_cell = 4
    connectivity = np.hstack([np.full((num_cells, 1), num_points_per_cell), cells])
    return pv.UnstructuredGrid(
        connectivity, np.ones(num_cells) * pv.CellType.TETRA, coordinates
    )


def with_suffix(p: Path, newsuffix: str) -> Path:
    p = Path(p)
    return p.parent / f"{p.name.split('.')[0]}{newsuffix}"


def if_finite(kf, form):
    return int(np.isfinite(kf)) * form


def sorted_dict(d):
    return {key: d[key] for key in sorted(d.keys())}


def flatten_dict(nested_dict, start=0):
    result = {}
    for key, value in nested_dict.items():
        if isinstance(value, list):
            for i, item in enumerate(value, start=start):
                result[f"{key}_{i}"] = item
        else:
            result[key] = value
    return result
