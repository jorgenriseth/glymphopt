import click
import numpy as np
import dolfin as df
import tqdm
import pandas as pd

from dolfin import inner, grad


from glymphopt.cache import CacheObject, cache_fetch
from glymphopt.coefficientvectors import CoefficientVector
from glymphopt.datageneration import BoundaryConcentration

from glymphopt.interpolation import LinearDataInterpolator
from glymphopt.io import read_mesh, read_function_data, read_augmented_dti
from glymphopt.measure import measure
from glymphopt.operators import (
    matrix_operator,
    bilinear_operator,
    matmul,
)
from glymphopt.minimize import adaptive_grid_search
from glymphopt.singlecompartment import SingleCompartmentInverseProblem
from glymphopt.timestepper import TimeStepper


@click.command()
@click.option("--input", "-i", type=str, required=True)
@click.option("--output", "-o", type=str, required=True)
def main(input, output):
    domain = read_mesh(input)
    td, Yd = read_function_data(input, domain, "concentration")
    td, Y_bdry = read_function_data(input, domain, "boundary_concentration")
    coefficients = {
        "a": 2.0,
        "phi": 0.22,
        "r": 1e-6,
        "k": 1e-2,
        "rho": 0.123,
        "eta": 0.4,
    }

    D = read_augmented_dti(input)
    D.vector()[:] *= coefficients["rho"]

    g = LinearDataInterpolator(td, Y_bdry, valuescale=1.0)

    coeffconverter = CoefficientVector(coefficients, ("a", "r"))
    problem = SingleCompartmentInverseProblem(
        td, Yd, coeffconverter, g=g, D=D, progress=True
    )
    best, hist, full_hist = adaptive_grid_search(
        func=problem.F,
        lower_bounds=[0.1, 0.0],
        upper_bounds=[10.0, 1e-5],
        n_points_per_dim=5,
        n_iterations=10,
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
