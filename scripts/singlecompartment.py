import click
import dolfin as df
import pantarei as pr


from glymphopt.coefficientvectors import CoefficientVector
from glymphopt.interpolation import LinearDataInterpolator
from glymphopt.io import read_mesh, read_function_data, read_augmented_dti
from glymphopt.parameters import (
    singlecomp_parameters,
    default_twocomp_parameters,
)
from glymphopt.singlecompartment import SingleCompartmentInverseProblem
from glymphopt.utils import with_suffix


@click.command()
@click.option("--input", "-i", type=str, required=True)
@click.option("--output", "-o", type=str, required=True)
@click.option("--dt", type=float, default=3600)
@click.option("--a", type=float)
@click.option("--r", type=float)
@click.option("--k", type=float)
@click.option("--phi", type=float)
@click.option("--visual", is_flag=True)
def main(input, output, visual, **kwargs):
    # Load default parameters, dimensionalize and overwrite coefficients
    default_coefficients = singlecomp_parameters(default_twocomp_parameters())
    overwrite_coefficients = {
        key: val for key, val in kwargs.items() if val is not None
    }
    coefficients = default_coefficients | overwrite_coefficients

    domain = read_mesh(input)
    D = read_augmented_dti(input)
    D.vector()[:] *= coefficients["rho"]

    td, Yd = read_function_data(input, domain, "concentration")
    td, Y_bdry = read_function_data(input, domain, "boundary_concentration")

    coeffconverter = CoefficientVector(coefficients, ("a", "r"))
    g = LinearDataInterpolator(td, Y_bdry, valuescale=1.0)
    problem = SingleCompartmentInverseProblem(
        td, Yd, coeffconverter, g=g, D=D, progress=True
    )
    Y = problem.forward(coeffconverter.to_vector())
    Ym = problem.measure(Y)
    with df.HDF5File(domain.mpi_comm(), output, "w") as hdf:
        for ti, Ym_i in zip(td, Ym):
            pr.write_checkpoint(hdf, Ym_i, "concentration", t=ti)

    if visual:
        with df.XDMFFile(
            domain.mpi_comm(),
            str(with_suffix(output, "_measurements.xdmf")),
        ) as xdmf:
            for ti, Ym_i in zip(td, Ym):
                xdmf.write(Ym_i, t=ti)
        with df.XDMFFile(
            domain.mpi_comm(),
            str(with_suffix(output, "_all.xdmf")),
        ) as xdmf:
            for ti, Yi in zip(problem.timestepper.vector(), Y):
                xdmf.write(Yi, t=ti)


if __name__ == "__main__":
    main()
