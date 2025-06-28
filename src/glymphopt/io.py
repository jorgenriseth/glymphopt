from pathlib import Path

import dolfin as df
import numpy as np
import pantarei as pr

from gripmodels.utils import with_suffix


def read_augmented_dti(input_hdf):
    with df.HDF5File(df.MPI.comm_world, input_hdf, "r") as hdf:
        domain = pr.read_domain(hdf)
        DTI = pr.read_function(hdf, "DTI", domain)
        MD = pr.read_function(hdf, "MD", domain)

    DTI.rename("DTI", "")
    MD.rename("MD", "")
    DG0 = MD.function_space()
    DG09 = DTI.function_space()

    (gm_cells,) = np.where(domain.subdomains.array() == 1)
    (non_gm_cells,) = np.where(domain.subdomains.array() != 1)

    D = df.Function(DG09, name="DTI-augmented")
    I = df.Constant(np.eye(3))

    for cell_index in gm_cells:
        cell_scalar_dof = DG0.dofmap().cell_dofs(cell_index)[0]
        cell_tensor_dofs = DG09.dofmap().cell_dofs(cell_index)
        D.vector()[cell_tensor_dofs] = MD.vector()[cell_scalar_dof] * I.values()

    for cell_index in non_gm_cells:
        cell_tensor_dofs = DG09.dofmap().cell_dofs(cell_index)
        D.vector()[cell_tensor_dofs] = DTI.vector()[cell_tensor_dofs]
    return D


def read_concentration_data(
    filepath: str, funcname: str
) -> tuple[np.ndarray, list[df.Function]]:
    """Read and store all outputted data"""
    hdf = df.HDF5File(df.MPI.comm_world, filepath, "r")
    tvec = pr.read_timevector(hdf, funcname)
    c = pr.read_function(hdf, funcname, idx=0)
    C = [df.Function(c.function_space()) for _ in range(tvec.size)]
    for idx in range(len(C)):
        pr.read_checkpoint(hdf, C[idx], funcname, idx)

    hdf.close()
    return tvec, C


class LinearDataInterpolator(df.Function):
    def __init__(self, input, funcname, domain, timescale=1.0, valuescale=1.0):
        tvec, C_sas = read_function_data(str(input), domain, funcname)
        super().__init__(C_sas[0].function_space())
        for ci in C_sas:
            ci.vector()[:] *= valuescale
        self.timepoints = tvec * timescale
        self.interpolator = vectordata_interpolator(C_sas, self.timepoints)
        self.update(t=0.0)

    def update(self, t):
        self.vector()[:] = self.interpolator(t)
        return self


def read_function_data(input_hdf, domain, funcname):
    with df.HDF5File(df.MPI.comm_world, str(input_hdf), "r") as hdf:
        celltype = domain.ufl_cell()
        element = df.FiniteElement("CG", celltype, 1)
        V = df.FunctionSpace(domain, element)

        time_data = read_timevector(hdf, funcname)
        N = len(time_data)
        c_data = [df.Function(V, name=f"funcname_{idx}") for idx in range(N)]
        for idx in range(N):
            read_checkpoint(hdf, c_data[idx], name=funcname, idx=idx)
        return np.array(time_data), c_data


def read_timevector(hdf: df.HDF5File, function_name: str):
    num_entries = int(hdf.attributes(function_name)["count"])
    time = np.zeros(num_entries)
    for i in range(num_entries):
        time[i] = read_checkpoint_time(hdf, function_name, i)
    return time


def read_checkpoint(
    hdf: df.HDF5File, u: df.Function, name: str, idx: int
) -> df.Function:
    hdf.read(u, f"{name}/vector_{idx}")
    return u


def read_checkpoint_time(hdf: df.HDF5File, name: str, idx: int) -> float:
    return hdf.attributes(f"{name}/vector_{idx}")["timestamp"]


def hdf2xdmf(input, output, funcname, timescale):
    xdmf = df.XDMFFile(df.MPI.comm_world, output)
    with df.HDF5File(df.MPI.comm_world, input, "r") as hdf:
        tvec = pr.read_timevector(hdf, funcname)
        for idx, ti in enumerate(tvec):
            if idx > 0:
                u = pr.read_function(hdf, funcname)
            else:
                u = read_checkpoint(hdf, u, funcname, idx)  # pyright: ignore
            xdmf.write(u, ti * timescale)
    xdmf.close()


def read_mesh(input_hdf):
    with df.HDF5File(df.MPI.comm_world, input_hdf, "r") as hdf:
        domain = pr.read_domain(hdf)
    return domain


def read_subdomains(domain, data_hdf):
    with df.HDF5File(df.MPI.comm_world, str(data_hdf), "r") as hdf:
        segments = df.MeshFunction("size_t", domain, domain.topology().dim())
        hdf.read(segments, "parcellations")
    return segments


def read_xdmf_mesh(input_xdmf):
    domain = df.Mesh(df.MPI.comm_world)
    with df.XDMFFile(domain.mpi_comm(), input_xdmf) as xdmf:
        xdmf.read(domain)
    return domain


class NullWriter:
    def __init__(self):
        pass

    def write(self, t, u):
        pass


class HDFWriter:
    def __init__(self, output):
        self.hdf = df.HDF5File(df.MPI.comm_world, str(output), "w")

    def write(self, t, c, cT):
        if t == 0:
            pr.write_function(self.hdf, c, "fluid_concentrations")
            pr.write_function(self.hdf, cT, "total_concentration")
        else:
            pr.write_checkpoint(self.hdf, c, "fluid_concentrations", t=t)
            pr.write_checkpoint(self.hdf, cT, "total_concentration", t=t)

    def close(self):
        self.hdf.close()


class XDMFWriterSingle:
    def __init__(self, output):
        self.xdmf = df.XDMFFile(df.MPI.comm_world, str(output))

    def write(self, t, c, _):
        self.xdmf.write(c, t=t)

    def close(self):
        self.xdmf.close()


class XDMFWriter:
    def __init__(self, output, compartments):
        self.xdmfs = {
            **{
                i: df.XDMFFile(
                    df.MPI.comm_world, str(with_suffix(Path(output), f"_{i}.xdmf"))
                )
                for i in compartments
            },
            "total": df.XDMFFile(
                df.MPI.comm_world, str(with_suffix(Path(output), "_total.xdmf"))
            ),
        }
        self.compartments = compartments

    def write(self, t, c, cT):
        for i, ci in zip(self.compartments, c.split()[: len(self.compartments)]):
            if i in self.compartments:
                ci.rename("fluid_concentration", "")
            self.xdmfs[i].write(ci, t=t)
        if cT:
            self.xdmfs["total"].write(cT, t=t)

    def close(self):
        for i in self.xdmfs:
            self.xdmfs[i].close()


class CombinedWriter:
    def __init__(self, hdf_writer, xdmf_writer):
        self.hdf = hdf_writer
        self.xdmf = xdmf_writer

    def write(self, t, c, cT):
        self.hdf.write(t, c, cT)
        self.xdmf.write(t, c, cT)

    def close(self):
        self.hdf.close()
        self.xdmf.close()


class WriteCollection:
    def __init__(self, *args):
        self.writers = args

    def write(self, t, c, cT):
        for writer in self.writers:
            writer.write(t, c, cT)

    def close(self):
        for writer in self.writers:
            writer.close()
