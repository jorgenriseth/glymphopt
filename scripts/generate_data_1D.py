from pathlib import Path
from typing import Callable

import dolfin as df
import matplotlib.pyplot as plt
import numpy as np
import pantarei as pr

from glymphopt.utils import load_data_interpolator
from glymphopt.visual import data_visual


def generate_data(
    output: Path,
    domain: df.Mesh,
    family: str,
    degree: int,
    sampletimes: list[float] | np.ndarray,
    update_function: Callable[[df.Function, float], None],
) -> None:
    V = df.FunctionSpace(domain, family, degree)
    u = df.Function(V, name="concentration")
    with df.HDF5File(df.MPI.comm_world, str(output), "w") as hdf:
        pr.write_domain(hdf, domain)
        update_function(u, sampletimes[0])
        pr.write_function(hdf, u, "concentration")
        for ti in sampletimes[1:]:
            update_function(u, ti)
            pr.write_checkpoint(hdf, u, "concentration", ti)


def sample_from_file(
    u_data: pr.DataInterpolator,
) -> Callable[[df.Function, float], None]:
    def call(u: df.Function, t: float) -> None:
        u.assign(u_data.update(t))
        return

    return call


def cosh_diffusion_clearance_data(
    D: float,
    r: float,
    decayrate: float,
) -> Callable[[df.Function, float], None]:
    """"""
    assert (
        1 / r < decayrate
    ), f"decayrate must be larger than 1/r, got {decayrate} <= {1/r} "
    lambda_ = r - 1 / decayrate
    print(lambda_)
    a = np.sqrt(lambda_ / D)
    time = df.Constant(0.0)
    expr = df.Expression(
        f"{1.0 / np.cosh(a)}*cosh({a}*x[0]) * exp( -{r - lambda_}*t )",
        degree=2,
        t=time,
    )

    def update(u: df.Function, t: float) -> None:
        time.assign(t)
        u.assign(df.interpolate(expr, u.function_space()))

    return update


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sampletimes", type=float, nargs="+")
    parser.add_argument("--fem_family", type=str, default="CG")
    parser.add_argument("--fem_degree", type=int, default=1)
    parser.add_argument("--visual", action="store_true")
    args = parser.parse_args()

    if args.sampletimes is None:
        args.sampletimes = np.linspace(0, 1, 11)
    print(args.sampletimes)

    u_data = load_data_interpolator(args.input, "concentration")
    update_function = sample_from_file(u_data)
    generate_data(
        output=args.output,
        domain=u_data.function_space().mesh(),
        family=args.fem_family,
        degree=args.fem_degree,
        sampletimes=args.sampletimes,
        update_function=update_function,
    )
    if args.visual:
        data_visual(args.output, "concentration")
        plt.show()
