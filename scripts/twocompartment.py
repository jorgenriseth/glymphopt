import click
import dolfin as df
import pantarei as pr

from glymphopt.coefficientvectors import CoefficientVector
from glymphopt.interpolation import LinearDataInterpolator
from glymphopt.io import read_mesh, read_function_data, read_augmented_dti
from glymphopt.twocompartment import MulticompartmentInverseProblem


@click.command()
@click.option("--input", "-i", type=str, required=True)
@click.option("--output", "-o", type=str, required=True)
@click.option("--dt", type=float, default=3600)
@click.option("--gamma", type=float)
@click.option("--t_ep", "t_ep", type=float)
@click.option("--t_pb", "t_pb", type=float)
@click.option("--k_e", "k_e", type=float)
@click.option("--k_p", "k_p", type=float)
def main(input, output, dt, **kwargs):
    # Load default parameters, dimensionalize and overwrite coefficients
    overwrite_coefficients = {
        key: val for key, val in kwargs.items() if val is not None
    }
    domain = read_mesh(input)
    coefficients = {
        "n_e": 0.2,
        "n_p": 0.02,
        "t_ep": 0.029,
        "t_pb": 2e-06,
        "k_e": 1e-05,
        "k_p": 0.0037,
        "rho": 0.113,
        "gamma": 20.0,
        "eta": 0.4,
        "phi": 0.22,
    } | overwrite_coefficients
    coeffconverter = CoefficientVector(coefficients, ("gamma", "t_pb"))

    domain = read_mesh(input)

    D = read_augmented_dti(input)
    D.vector()[:] *= coefficients["rho"]

    td, Yd = read_function_data(input, domain, "concentration")
    td, Y_bdry_tmp = read_function_data(input, domain, "boundary_concentration")

    # Want Y_bdry in same function space as Yd
    Y_bdry = [
        df.Function(Yd[0].function_space(), name="boundary_concentration")
        for _ in Y_bdry_tmp
    ]
    [Y_bdry[i].interpolate(Y_bdry_tmp[i]) for i in range(len(td))]
    g = LinearDataInterpolator(
        td, Y_bdry, valuescale=coefficients["eta"] / coefficients["phi"]
    )

    problem = MulticompartmentInverseProblem(
        td, Yd, coeffconverter, g=g, D=D, dt=dt, progress=True
    )
    coeffconverter = CoefficientVector(coefficients, ("gamma", "t_pb"))
    Y = problem.forward(coeffconverter.to_vector())
    Ym = problem.measure(Y)
    print("Storing...")
    with df.HDF5File(domain.mpi_comm(), output, "w") as hdf:
        for ti, Ym_i in zip(td, Ym):
            pr.write_checkpoint(hdf, Ym_i, "concentration", t=ti)

    print("Finished..")
    print()


def parse_evaluation(evaluation, coeffconverter):
    return {
        **{
            key: evaluation
            for key, evaluation in zip(coeffconverter.vars, evaluation["point"])
        },
        "funceval": evaluation["value"],
    }


if __name__ == "__main__":
    main()
