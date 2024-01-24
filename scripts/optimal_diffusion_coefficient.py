from pathlib import Path

import numpy as np
import pantarei as pr
from ufl import grad, inner, lhs, rhs

import glymphopt.dfa as df


class DataInterpolator(df.Function):
    def __init__(self, data, times):
        super().__init__(data[0].function_space())
        self.assign(data[0])
        self.times = times.copy()
        self.interpolator = pr.fenicsfunc_interpolator(data, times)

    def update(self, t: float) -> df.Function:
        self.assign(self.interpolator(t))
        return self


# Load data from file in dolfin-adjoint friendly format.
with df.HDF5File(df.MPI.comm_world, "data/sim_sampled.hdf", "r") as hdf:
    element = pr.read_element(hdf, "concentration")
    domain = df.Mesh(pr.read_domain(hdf))
    time_data = pr.read_timevector(hdf, "concentration")
    N = len(time_data)
    V = df.FunctionSpace(domain, element)
    data = [df.Function(V, name=f"concentration_{idx}") for idx in range(N)]
    for idx in range(N):
        pr.read_checkpoint(hdf, data[idx], name="concentration", idx=idx)


def numpy_mergesort(a, b, unique: bool = True):
    c = np.concatenate((a, b))
    c.sort(kind="mergesort")
    if unique:
        c = np.unique(c)
    return c


def forward(D):
    u_data = DataInterpolator(data, time_data)

    dt_base = 0.01
    dt = df.Constant(dt_base)

    time = pr.TimeKeeper(dt=dt, endtime=time_data[-1])
    time_sim = time.as_vector()

    timevec = numpy_mergesort(time_sim, time_data)
    dt_vec = np.diff(timevec)

    u0 = df.Function(V, name="concentration")
    u0.assign(u_data)

    dirichlet_bcs = [df.DirichletBC(V, u_data, "on_boundary")]
    bc_val = df.Constant(0.0)
    dirichlet_bcs = [df.DirichletBC(V, bc_val, "on_boundary")]

    dx = df.Measure("dx", domain)
    u, v = df.TrialFunction(V), df.TestFunction(V)
    F = (u - u0) * v * dx + dt * inner(D * grad(u), grad(v)) * dx
    a = lhs(F)
    l = rhs(F)
    A = df.assemble(a)

    qoi = {"error": [0.0] * len(time_data)}
    bins = np.digitize(time_data, timevec) - 1
    for idx, ti in enumerate(timevec[1:]):
        dt.assign(dt_vec[idx])
        pr.print_progress(float(ti), time.endtime, rank=df.MPI.comm_world.rank)
        u_data.update(ti)
        bc_val.assign(-np.exp(-ti / 0.1) + np.exp(-ti / 0.5))
        b = df.assemble(l)
        for bc in dirichlet_bcs:
            bc.apply(A, b)
        df.solve(A, u0.vector(), b)

        for bin_idx in np.arange(len(time_data))[bins == idx]:
            qoi["error"][bin_idx] = df.assemble(
                ((u0 - u_data) ** 2 + inner(grad(u0 - u_data), grad(u0 - u_data))) * dx
            )

    return 0.5 * sum(qoi["error"])


D = df.Constant(1.0)
control = df.Control(D)
J = forward(D)
dJdD = df.compute_gradient(J, [control])

h = df.Constant(0.1)
Jhat = df.ReducedFunctional(J, control)
conv_rate = df.taylor_test(Jhat, D, h)


# m = [df.Control(c) for c in ctrls.values()]
rf = df.ReducedFunctional(J, control)
opt_ctrls = df.minimize(rf, options={"maxiter": 100, "gtol": 1e-12})

print("D_true\tD_opt")
print(f"{2.4}\t{float(opt_ctrls)}")
