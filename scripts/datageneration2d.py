import numpy as np
import dolfin as df
import tqdm.notebook as tqdm
import pantarei as pr
from dolfin import inner, grad, dot

from threecomp.timestepper import TimeStepper

from threecomp.parameters import parameters_2d_default
from threecomp.interpolation import measure
from threecomp.datagen import BoundaryConcentration


def generate_data(domain_path, outputdir):
    acquisition_times = np.array([0.0, 4.8, 26.3, 48.3, 70.9])

    dt = 0.1
    endtime = np.ceil(acquisition_times)[-1]

    timestepper = TimeStepper(dt, (0.0, endtime))
    print(timestepper.dt, timestepper.num_intervals())
    timepoints = timestepper.vector()

    with df.HDF5File(df.MPI.comm_world, str(domain_path), "r") as hdf:
        domain = pr.read_domain(hdf)

    V = df.FunctionSpace(domain, "CG", 1)
    g = BoundaryConcentration(V)

    coefficients = parameters_2d_default()
    dt = timestepper.dt
    r = coefficients["r"]
    k = coefficients["k"]
    a = coefficients["a"]
    aD = a * df.Identity(2)  # type: ignore
    print(coefficients)

    u0 = df.Function(V)
    dx = df.Measure("dx", domain)
    ds = df.Measure("ds", domain)
    u, v = df.TrialFunction(V), df.TestFunction(V)

    a = inner(u, v) * dx + dt * (
        inner(dot(aD, grad(u)), grad(v)) * dx
        + r * inner(u, v) * dx  # type: ignore
        + k * inner(u, v) * ds  # type: ignore
    )
    L = u0 * v * dx + dt * k * g * v * ds  # type: ignore

    assembler = df.SystemAssembler(a, L)
    A, b = df.PETScMatrix(), df.PETScVector()
    assembler.assemble(A, b)
    solver = df.LUSolver(A)

    Y = [df.Function(V, name="concentration") for _ in range(len(timepoints))]
    for n, tn in enumerate(tqdm.tqdm(timepoints[1:]), start=1):
        u0.assign(Y[n - 1])
        g.update(tn)
        assembler.assemble(b)
        solver.solve(Y[n].vector(), b)

    xdmf_boundary = df.XDMFFile(
        domain.mpi_comm(), f"{outputdir}/true_boundary_concentration.xdmf"
    )
    xdmf_internal = df.XDMFFile(
        domain.mpi_comm(), f"{outputdir}/true_concentration.xdmf"
    )
    for n, tn in enumerate(timepoints):
        xdmf_boundary.write(g.update(tn), t=tn)
        xdmf_internal.write(Y[n], t=tn)
    xdmf_boundary.close()
    xdmf_internal.close()

    Ym = measure(timestepper, Y, acquisition_times)

    xdmf_measured = df.XDMFFile(
        domain.mpi_comm(), f"{outputdir}/measured_concentration.xdmf"
    )
    hdf_measured = df.HDF5File(
        domain.mpi_comm(), f"{outputdir}/concentrations.hdf", "w"
    )
    for i, ti in enumerate(acquisition_times):
        xdmf_measured.write(Ym[i], t=ti)
        if i == 0:
            pr.write_function(hdf_measured, Ym[i], "concentration")
            pr.write_function(hdf_measured, g.update(ti), "boundary_concentration")
        else:
            pr.write_checkpoint(hdf_measured, Ym[i], "concentration", t=ti)
            pr.write_checkpoint(
                hdf_measured, g.update(ti), "boundary_concentration", t=ti
            )

    xdmf_measured.close()
    hdf_measured.close()


if __name__ == "__main__":
    generate_data("resources/brain-2d-domain.hdf", "resources")
