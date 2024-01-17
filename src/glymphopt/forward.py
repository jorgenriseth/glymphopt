import time as pytime
from pathlib import Path
from typing import Any, Callable, Sequence

import dolfin as df
import matplotlib.pyplot as plt
import numpy as np
import pantarei as pr
import ufl
from loguru import logger
from ufl import grad, inner

from glymphopt.visual import data_visual


def diffusion_reaction_form(
    V: df.FunctionSpace,
    coefficients: dict[str, float],
    boundaries: list[pr.BoundaryData],
    u0: df.Function,
    dt: float,
) -> ufl.Form:
    u, v = df.TrialFunction(V), df.TestFunction(V)
    D, r = coefficients["D"], coefficients["r"]
    print(D, r)
    dx = df.Measure("dx", domain=V.mesh())
    F = (u - u0) * v * dx + dt * (inner(D * grad(u), grad(v)) * dx + r * u * v * dx)
    return F + pr.process_boundary_forms(u, v, boundaries)


def solve_timedependent(
    u0: df.Function,
    form: pr.Callable[
        [
            df.FunctionSpace,
            dict[str, float],
            Sequence[pr.BoundaryData],
            df.Function,
            pr.TimeDelta,
        ],
        ufl.Form,
    ],
    coefficients: dict[str, Any],
    update_coefficients: Callable[
        [df.Function, float, dict[str, Any], Sequence[pr.BoundaryData]], None
    ],
    boundaries: list[pr.BoundaryData],
    time: pr.TimeKeeper,
    hdf: df.HDF5File,
    computer: Callable[[df.Function, pr.TimeKeeper], dict[str, Any]],
):
    pr.write_function(hdf, u0, u0.name())
    V = u0.function_space()
    dirichlet_bcs = pr.process_dirichlet(V, boundaries)

    F = form(V, coefficients, boundaries, u0, time.dt)
    a = df.lhs(F)
    l = df.rhs(F)
    A = df.assemble(a)
    tic = pytime.time()
    qoi = computer(u0, 0.0)
    for ti in time.as_vector()[1:]:
        pr.print_progress(float(ti), time.endtime, rank=df.MPI.comm_world.rank)
        update_coefficients(u0, ti, coefficients, boundaries)
        b = df.assemble(l)
        for bc in dirichlet_bcs:
            bc.apply(A, b)
        df.solve(A, u0.vector(), b)
        pr.write_checkpoint(hdf, u0, u0.name(), float(ti))
        qoi = computer(u0, time)

    logger.debug("Time loop finished.")
    toc = pytime.time()
    logger.debug(f"Elapsed time in loop: {toc - tic:.2f} seconds.")
    return qoi


def mass_computer():
    qoi = {
        "mass": np.nan * np.zeros(len(time)),
    }
    idx = 0
    dx = df.Measure("dx", V.mesh())

    def call(uh, t):
        nonlocal idx
        qoi["mass"][idx] = df.assemble(uh * dx)
        idx = idx + 1
        return qoi

    return call


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--D", type=float, default=2.4)
    parser.add_argument("--r", type=float, default=0.0)
    parser.add_argument("--visual", action="store_true")
    args = parser.parse_args()

    time = pr.TimeKeeper(dt=0.01, endtime=1.0)
    coefficients = {"D": args.D, "r": args.r}

    domain = pr.MMSInterval(args.resolution)
    V = df.FunctionSpace(domain, "CG", 1)
    uh = df.Function(V, name="concentration")

    bc = df.Constant(0.0)
    boundaries = [pr.DirichletBoundary(bc, tag="everywhere")]

    def coefficients_update(uh, t, coefficients, boundaries):
        time.assign(t)
        for bc in [x for x in boundaries if isinstance(x, pr.DirichletBoundary)]:
            bc.uD.assign(-np.exp(-t / 0.1) + np.exp(-t / 0.5))
        return

    with df.HDF5File(df.MPI.comm_world, str(args.output), "w") as hdf:
        solve_timedependent(
            u0=uh,
            form=diffusion_reaction_form,
            coefficients=coefficients,
            update_coefficients=coefficients_update,
            boundaries=boundaries,
            time=time,
            hdf=hdf,
            computer=lambda _, __: {},
        )

    if args.visual:
        data_visual(args.output, "concentration")
        plt.show()
