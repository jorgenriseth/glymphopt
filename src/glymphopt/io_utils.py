from os import PathLike
from typing import Optional

import h5py
import numpy as np
import ufl
from loguru import logger

import glymphopt.dfa as df
from glymphopt.domain import Domain


def write_domain(hdf: df.HDF5File, domain: df.Mesh) -> None:
    hdf.write(domain, "/domain/mesh")
    if isinstance(domain, Domain):
        hdf.write(domain.subdomains, "/domain/subdomains")
        hdf.write(domain.boundaries, "/domain/boundaries")


def read_domain(hdf: df.HDF5File) -> Domain:
    mesh = df.Mesh(df.MPI.comm_world)
    hdf.read(mesh, "/domain/mesh", False)
    n = mesh.topology().dim()
    subdomains = df.MeshFunction("size_t", mesh, n)
    boundaries = df.MeshFunction("size_t", mesh, n - 1)
    if hdf.has_dataset("/domain/subdomains"):
        hdf.read(subdomains, "/domain/subdomains")
    if hdf.has_dataset("/domain/boundaries"):
        hdf.read(boundaries, "/domain/boundaries")
    return Domain(mesh, subdomains, boundaries)


def write_element(
    hdf: df.HDF5File, funcname: str, function_space: df.FunctionSpace
) -> str:
    signature = function_space.dolfin_element().signature()
    hdf.attributes(funcname)["signature"] = signature
    return signature


def read_element(hdf: df.HDF5File, function_name: str) -> ufl.FiniteElementBase:
    signature = hdf.attributes(function_name)["signature"]
    return read_signature(signature)


def write_function(
    hdf: df.HDF5File,
    function: df.Function,
    name: str,
    filename_overwrite: Optional[PathLike] = None,
):
    if filename_overwrite is not None:
        delete_dataset(hdf, filename_overwrite, name)
    if not hdf.has_dataset("/domain"):
        write_domain(hdf, function.function_space().mesh())
    hdf.write(function, name, 0.0)


def read_function(
    hdf: df.HDF5File, name: str, domain: Optional[df.Mesh] = None, idx: int = 0
):
    if domain is None:
        domain = read_domain(hdf)
    element = read_element(hdf, name)
    V = df.FunctionSpace(domain, element)
    u = df.Function(V, name=name)
    hdf.read(u, f"{name}/vector_{idx}")
    return u


def write_checkpoint(hdf: df.HDF5File, function: df.Function, name: str, t: float):
    hdf.write(function, name, t)


def read_checkpoint(
    hdf: df.HDF5File, u: df.Function, name: str, idx: int
) -> df.Function:
    hdf.read(u, f"{name}/vector_{idx}")
    return u


def read_checkpoint_time(hdf: df.HDF5File, name: str, idx: int) -> float:
    return hdf.attributes(f"{name}/vector_{idx}")["timestamp"]


def read_timevector(hdf: df.HDF5File, function_name: str) -> np.ndarray:
    num_entries = int(hdf.attributes(function_name)["count"])
    time = np.zeros(num_entries)
    for i in range(num_entries):
        time[i] = read_checkpoint_time(hdf, function_name, i)
    return time


def close(hdf: df.HDF5File) -> None:
    worldsize = df.MPI.comm_world.size
    if worldsize > 1:
        logger.debug(
            f"Process {df.MPI.comm_world.rank}/{worldsize-1} waiting for other\
                    processes before closing file."
        )
    hdf.close()
    if df.MPI.comm_world.rank == 0:
        logger.debug(f"File closed, continuing.")


def delete_dataset(hdf: df.HDF5File, filename: PathLike, dataset_name: str):
    if df.MPI.comm_world.rank == 0:
        if hdf.has_dataset(dataset_name):
            with h5py.File(filename, "a") as f:
                logger.info((f"Deleting dataset {dataset_name} form file {filename}"))
                del f[dataset_name]
    df.MPI.comm_world.barrier()


def read_signature(signature: str):
    # Imported here since the signature require functions without namespace
    # but we want to avoid them in global scope.
    from scripts.dfa import (
        FiniteElement,
        MixedElement,
        TensorElement,
        VectorElement,
        hexahedron,
        interval,
        quadrilateral,
        tetrahedron,
        triangle,
    )

    return eval(signature)
