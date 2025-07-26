import click
import numpy as np
import dolfin as df
import pandas as pd


from glymphopt.coefficientvectors import CoefficientVector
from glymphopt.interpolation import LinearDataInterpolator
from glymphopt.io import read_mesh, read_function_data, read_augmented_dti

from glymphopt.minimize import adaptive_grid_search
from glymphopt.twocompartment import MulticompartmentInverseProblem


@click.command()
@click.option("--input", "-i", type=str, required=True)
@click.option("--output", "-o", type=str, required=True)
@click.option("--iter", "iterations", type=int, default=10)
def main(input, output, iterations):
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
    }
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
        td, Yd, coeffconverter, g=g, D=D, progress=True
    )
    coeffconverter = CoefficientVector(coefficients, ("gamma", "t_pb"))
    best, hist, full_hist = adaptive_grid_search(
        func=problem.F,
        lower_bounds=[1.0, 0.0],
        upper_bounds=[45, 1e-5],
        n_points_per_dim=5,
        n_iterations=iterations,
    )
    records = [parse_evaluation(pointeval, coeffconverter) for pointeval in full_hist]
    data = pd.DataFrame.from_records(records)
    data.to_csv(output, sep=";", index=False)


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
