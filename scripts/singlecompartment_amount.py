from pathlib import Path

import click
import numpy as np
import pandas as pd

from glymphopt.coefficientvectors import CoefficientVector
from glymphopt.interpolation import LinearDataInterpolator
from glymphopt.io import read_augmented_dti, read_function_data, read_mesh
from glymphopt.operators import matrix_operator
from glymphopt.parameters import (
    default_twocomp_parameters,
    singlecomp_parameters,
)
from glymphopt.singlecompartment import SingleCompartmentInverseProblem

# Mesh coordinates in mm and concentrations in mmol/L: mm^3 -> L gives mmol.
MM3_TO_L = 1e-6


@click.command()
@click.option("--input", "-i", "input_hdf", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--output", "-o", "output_csv", type=click.Path(dir_okay=False, path_type=Path), required=True, help="Simulation CSV with time and amount_mesh for every timestep.")
@click.option("--data-output", type=click.Path(dir_okay=False, path_type=Path), help="Optional CSV with measured mesh amounts at measurement times.")
@click.option("--dt", type=click.FloatRange(min=0, min_open=True), default=3600, show_default=True, help="Timestep in seconds.")
@click.option("--a", type=float)
@click.option("--r", type=float)
@click.option("--k", type=float)
@click.option("--phi", type=float)
def main(input_hdf, output_csv, data_output, dt, **kwargs):
    """Simulate the single-compartment model and save total mesh solute amounts."""
    outputs = [output_csv] + ([data_output] if data_output is not None else [])
    for path in outputs:
        if path.suffix.lower() != ".csv":
            raise click.BadParameter("Output filenames must end in .csv.")
        if path.resolve() == Path(input_hdf).resolve():
            raise click.BadParameter("Output must differ from the input HDF.")
    if data_output is not None and data_output.resolve() == output_csv.resolve():
        raise click.BadParameter("Simulation and measured-data outputs must differ.")

    coefficients = singlecomp_parameters(default_twocomp_parameters()) | {
        key: val for key, val in kwargs.items() if val is not None
    }

    domain = read_mesh(input_hdf)
    D = read_augmented_dti(input_hdf)
    D.vector()[:] *= coefficients["rho"]
    td, Yd = read_function_data(input_hdf, domain, "concentration")
    _, Y_bdry = read_function_data(input_hdf, domain, "boundary_concentration")

    coeffconverter = CoefficientVector(coefficients, ("a", "r"))
    problem = SingleCompartmentInverseProblem(
        td,
        Yd,
        coeffconverter,
        g=LinearDataInterpolator(td, Y_bdry, valuescale=1.0),
        D=D,
        dt=dt,
        progress=True,
        solver_type="iterative",
    )
    Y = problem.forward(coeffconverter.to_vector())

    M_mesh = matrix_operator(problem.model.M(coefficients))
    time = problem.timestepper.vector()
    amount = np.array([M_mesh(Yi.vector()).sum() * MM3_TO_L for Yi in Y])
    try:
        pd.DataFrame({"time": time, "amount_mesh": amount}).to_csv(output_csv, index=False)
    except OSError as error:
        raise click.ClickException(f"Cannot save {output_csv}: {error}") from error
    click.echo(f"Saved {output_csv}")

    if data_output is not None:
        amount_data = np.array([M_mesh(Ydi.vector()).sum() * MM3_TO_L for Ydi in Yd])
        try:
            pd.DataFrame({"time": td, "amount_mesh": amount_data}).to_csv(data_output, index=False)
        except OSError as error:
            raise click.ClickException(f"Cannot save {data_output}: {error}") from error
        click.echo(f"Saved {data_output}")


if __name__ == "__main__":
    main()
