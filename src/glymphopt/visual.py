from pathlib import Path

import dolfin as df
import matplotlib.pyplot as plt
import numpy as np
import pantarei as pr


def data_visual(filepath: Path, funcname: str, N_plots: int = 10, ax=None, **kwargs):
    if ax is None:
        plt.figure()
        ax = plt.gca()

    ymin, ymax = 0.0, 0.0
    with df.HDF5File(df.MPI.comm_world, str(filepath), "r") as hdf:
        timevec = pr.read_timevector(hdf, funcname)
        u = pr.read_function(hdf, funcname)
        plot_every = max(len(timevec) // N_plots, 1)
        for idx, ti in enumerate(timevec):
            if idx % plot_every == 0:
                pr.read_checkpoint(hdf, u, funcname, idx)
                df.plot(u, label=f"$t={ti:.2f}$", **kwargs)
                ymin = min(ymin, u.vector()[:].min())
                ymax = max(ymax, u.vector()[:].max())
    ax.legend()
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(-1, 1)
    return ax


def plot_step(x, y, *args, **kwargs):
    """ "Helper function for easy plotting of step-functions where point
    values represent step mid-points."""
    d = np.diff(x)
    x_ = [x[0] - 0.5 * d[0], *x, x[-1] + 0.5 * d[-1]]
    y_ = [y[0], *y, y[-1]]
    plt.step(x_, y_, where="mid", *args, **kwargs)
    return plt.gca()
