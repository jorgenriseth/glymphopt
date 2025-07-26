import click
import numpy as np
import dolfin as df
import pantarei as pr

from dolfin import inner, grad


from glymphopt.coefficientvectors import CoefficientVector
from glymphopt.datageneration import BoundaryConcentration
from glymphopt.interpolation import LinearDataInterpolator
from glymphopt.io import read_mesh, read_function_data, read_augmented_dti
from glymphopt.parameters import (
    singlecomp_parameters,
    default_twocomp_parameters,
)
from glymphopt.timestepper import TimeStepper
from glymphopt.singlecompartment import SingleCompartmentInverseProblem


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
    td, Yd = read_function_data(input, domain, "concentration")
    td, Y_bdry = read_function_data(input, domain, "boundary_concentration")

    D = read_augmented_dti(input)
    D.vector()[:] *= coefficients["rho"]
    g = LinearDataInterpolator(td, Y_bdry, valuescale=1.0)
    coeffconverter = CoefficientVector(coefficients, ("a", "r"))
    problem = SingleCompartmentInverseProblem(
        td, Yd, coeffconverter, g=g, D=D, progress=True
    )
    Y = problem.forward(coeffconverter.to_vector())
    Ym = problem.measure(Y)
    with df.HDF5File(domain.mpi_comm(), output, "w") as hdf:
        for ti, Ym_i in zip(td, Ym):
            pr.write_checkpoint(hdf, Ym_i, "concentration", t=ti)


class Model:
    def __init__(self, V, D=None, g=None):
        D = D or df.Identity(V.mesh().topology().dim())

        domain = V.mesh()
        dx = df.Measure("dx", domain)
        ds = df.Measure("ds", domain)

        u, v = df.TrialFunction(V), df.TestFunction(V)
        self.M = df.assemble(inner(u, v) * dx)
        self.DK = df.assemble(inner(D * grad(u), grad(v)) * dx)
        self.S = df.assemble(inner(u, v) * ds)
        self.g = g or BoundaryConcentration(V)


def gradient_sensitivities(F, x, **kwargs):
    return np.array([F(x, ei, **kwargs) for ei in np.eye(len(x))])


def measure_interval(n: int, td: np.ndarray, timestepper: TimeStepper):
    bins = np.digitize(td, timestepper.vector(), right=True)
    return list(np.where(n == bins)[0])


if __name__ == "__main__":
    main()
