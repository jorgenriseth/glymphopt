from pathlib import Path

import dolfin as df
import numpy as np
import pantarei as pr

from glymphopt.forward import diffusion_reaction_form, solve_timedependent
from glymphopt.visual import data_visual

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--D", type=float, default=2.4)
    parser.add_argument("--r", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--endtime", type=float, default=1.0)
    parser.add_argument("--visual", action="store_true")
    args = parser.parse_args()

    time = pr.TimeKeeper(dt=args.dt, endtime=args.endtime)
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
        import matplotlib.pyplot as plt

        data_visual(args.output, "concentration", N_plots=20)
        plt.show()
