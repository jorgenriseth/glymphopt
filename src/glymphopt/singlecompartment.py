import click
import numpy as np
import dolfin as df
import tqdm

from dolfin import inner, grad


from glymphopt.cache import CacheObject, cache_fetch
from glymphopt.datageneration import BoundaryConcentration

from glymphopt.io import read_mesh, read_function_data
from glymphopt.operators import (
    matrix_operator,
    bilinear_operator,
    matmul,
)
from glymphopt.timestepper import TimeStepper


class SingleCompartmentInverseProblem:
    def __init__(
        self,
        td,
        Yd,
        coefficientvector,
        g,
        D,
        dt=3600,
        timescale=1.0,
        progress=True,
    ):
        self.silent = not progress
        self.td, self.Yd = td, Yd

        t_start = self.td[0]
        dt = dt * timescale
        N = int(np.ceil(np.round((self.td[-1] - t_start) / dt, 12)))
        t_end = N * dt
        self.timestepper = TimeStepper(dt, (t_start, t_end))

        coefficients = coefficientvector.coefficients

        self.V = self.Yd[0].function_space()
        g = g
        D = D

        self.model = Model(self.V, g=g, D=D)
        self.cache = {
            "state": CacheObject(),
            "adjoint": CacheObject(),
            "g": CacheObject(),
            "sensitivity": CacheObject(),
            "operator": CacheObject(),
        }

        _M_ = bilinear_operator(self.model.M)
        self.Yd_norms = [_M_(yi.vector(), yi.vector()) for yi in self.Yd]
        self.coefficients = coefficientvector

    def measure(self, Y: list[df.Function]) -> list[df.Function]:
        Ym = [df.Function(self.V, name="measured_state") for _ in range(len(self.td))]
        find_intervals = self.timestepper.find_intervals(self.td)
        for i, _ in enumerate(self.td[1:], start=1):
            ni = find_intervals[i]
            Ym[i].assign(Y[ni + 1])
        return Ym

    def F(self, x):
        Y = cache_fetch(self.cache["state"], self.forward, {"x": x}, x=x)
        Ym = self.measure(Y)
        _M_ = bilinear_operator(self.model.M)
        timepoint_errors = [
            _M_(Ym_i.vector() - Yd_i.vector(), Ym_i.vector() - Yd_i.vector()) / norm_i
            for Ym_i, Yd_i, norm_i in zip(Ym[1:], self.Yd[1:], self.Yd_norms[1:])
        ]
        J = 0.5 * sum(timepoint_errors)
        return J

    def forward(self, x):
        coefficients = self.coefficients.from_vector(x)
        a = coefficients["a"]
        r = coefficients["r"]
        k = coefficients["k"]
        eta = coefficients["eta"]

        timestepper = self.timestepper
        dt = timestepper.dt
        timepoints = timestepper.vector()
        Y = [df.Function(self.V, name="state") for _ in range(len(timepoints))]
        Y[0].assign(self.Yd[0])

        model = self.model
        M = model.M
        L = a * model.DK + r * model.M + k * model.S
        G = cache_fetch(self.cache["g"], self.boundary_vectors, {"eta": eta}, eta=eta)
        solver = cache_fetch(
            self.cache["operator"], df.LUSolver, {"x": x}, A=M + dt * L
        )
        Mdot = matrix_operator(M)

        N = self.timestepper.num_intervals()
        for n in tqdm.tqdm(range(N), total=N, disable=self.silent):
            solver.solve(Y[n + 1].vector(), Mdot(Y[n].vector()) + dt * k * G[n + 1])
        return Y

    def boundary_vectors(self, eta):
        model = self.model
        timestepper = self.timestepper
        return [eta * matmul(model.S, model.g(t)) for t in timestepper.vector()]

    def gradF(self, x):
        coefficients = self.coefficients.from_vector(x)
        k = coefficients["k"]
        eta = coefficients["eta"]
        dt = self.timestepper.dt
        model = self.model
        Y = cache_fetch(self.cache["state"], self.forward, {"x": x}, x=x)
        Ym = self.measure(Y)

        P = cache_fetch(self.cache["adjoint"], self.adjoint, {"x": x}, x=x, Ym=Ym)
        G = cache_fetch(self.cache["g"], self.boundary_vectors, {"eta": eta}, eta=eta)

        _M_ = bilinear_operator(model.M)
        _DK_ = bilinear_operator(model.DK)
        _S_ = bilinear_operator(model.S)
        return dt * sum(
            np.array(
                [
                    _DK_(p.vector(), y.vector()),
                    _M_(p.vector(), y.vector()),
                    _S_(p.vector(), y.vector()) - eta * p.vector().inner(g),
                    -k * p.vector().inner(g),
                ]
            )
            for y, p, g in zip(Y[1:], P[:-1], G[1:])
        )

    def adjoint(self, x, Ym) -> list[df.Function]:
        coefficients = self.coefficients.from_vector(x)
        a = coefficients["a"]
        r = coefficients["r"]
        k = coefficients["k"]

        timestepper = self.timestepper
        dt = timestepper.dt
        timepoints = timestepper.vector()

        model = self.model
        M = model.M
        L = a * model.DK + r * model.M + k * model.S
        solver = cache_fetch(
            self.cache["operator"], df.LUSolver, {"x": x}, A=M + dt * L
        )
        P = [df.Function(self.V, name="adjoint") for _ in range(len(timepoints))]
        Mdot = matrix_operator(M)
        num_intervals = timestepper.num_intervals()
        for n in tqdm.tqdm(
            range(num_intervals, 0, -1), total=num_intervals, disable=self.silent
        ):
            nj = measure_interval(n, self.td, self.timestepper)
            jump = sum(
                (
                    matmul(M, (Ym[j].vector() - Yd[j].vector()) / self.Yd_norms[j])
                    for j in nj
                )
            )

            solver.solve(
                P[n - 1].vector(),
                Mdot(P[n].vector()) - jump,
            )
        return P

    def dF(self, x, dx):
        timestepper = self.timestepper

        Y = cache_fetch(self.cache["state"], self.forward, {"x": x}, x=x)
        dY = cache_fetch(
            self.cache["sensitivity"],
            self.sensitivity,
            {"x": x, "dx": dx},
            x=x,
            dx=dx,
            Y=Y,
        )
        Ym = self.measure(Y)
        dYm = self.measure(dY)
        _M_ = bilinear_operator(self.model.M)
        return sum(
            [
                _M_(ym.vector() - yd.vector(), dy.vector()) / norm
                for ym, yd, dy, norm in zip(
                    Ym[1:], self.Yd[1:], dYm[1:], self.Yd_norms[1:]
                )
            ]
        )

    def hess(self, x):
        return np.array([self.hessp(x, ei) for ei in np.eye(len(x))])

    def hessp(self, x, dx):
        dt = self.timestepper.dt
        coefficients = self.coefficients.from_vector(x)
        phi = coefficients["phi"]

        Y = cache_fetch(self.cache["state"], self.forward, {"x": x}, x=x)
        Ym = self.measure(Y)
        dY = cache_fetch(
            self.cache["sensitivity"],
            self.sensitivity,
            {"x": x, "dx": dx},
            x=x,
            dx=dx,
            Y=Y,
        )
        dYm = self.measure(dY)

        P = cache_fetch(self.cache["adjoint"], self.adjoint, {"x": x}, x=x, Ym=Ym)
        dP = self.second_order_adjoint(x, dx, dYm, P)

        model = self.model
        G = cache_fetch(self.cache["g"], self.boundary_vectors, {"phi": phi}, phi=phi)
        _DK_ = bilinear_operator(model.DK)
        _M_ = bilinear_operator(model.M)
        _S_ = bilinear_operator(model.S)
        return dt * sum(
            np.array(
                [
                    _DK_(dp.vector(), y.vector()) + _DK_(p.vector(), dy.vector()),
                    _M_(dp.vector(), y.vector()) + _M_(p.vector(), dy.vector()),
                    _S_(dp.vector(), y.vector())
                    - dp.vector().inner(g)
                    + _S_(p.vector(), dy.vector()),
                ]
            )
            for y, dy, p, dp, g in zip(Y[1:], dY[1:], P[:-1], dP[:-1], G[1:])
        )

    def sensitivity(self, x, dx, Y) -> list[df.Function]:
        coefficients = self.coefficients.from_vector(x)
        a = coefficients["a"]
        r = coefficients["r"]
        k = coefficients["k"]
        dD, dr, dk = dx

        timestepper = self.timestepper
        dt = timestepper.dt
        timepoints = timestepper.vector()
        dY = [df.Function(self.V, name="sensitivity") for _ in range(len(timepoints))]

        model = self.model
        M = model.M
        L = a * model.DK + r * model.M + k * model.S
        solver = cache_fetch(
            self.cache["operator"], df.LUSolver, {"x": x}, A=M + dt * L
        )

        dL = dD * model.DK + dr * model.M + dk * model.S
        Mdot = matrix_operator(M)
        dLdot = matrix_operator(dL)
        Sdot = matrix_operator(model.S)
        G = [matmul(model.S, model.g(t)) for t in timepoints]
        N = self.timestepper.num_intervals()
        for n in tqdm.tqdm(range(N), total=N, disable=self.silent):
            solver.solve(
                dY[n + 1].vector(),
                Mdot(dY[n].vector())
                - dt * dLdot(Y[n + 1].vector())
                + dt * dk * G[n + 1],
            )
        return dY

    def second_order_adjoint(self, x, dx, dYm, P):
        coefficients = self.coefficients.from_vector(x)
        a = coefficients["a"]
        r = coefficients["r"]
        k = coefficients["k"]
        dD, dr, dk = dx

        timestepper = self.timestepper
        dt = timestepper.dt
        timepoints = timestepper.vector()

        dP = [
            df.Function(self.V, name="second-order-adjoint")
            for _ in range(len(timepoints))
        ]

        model = self.model
        M = model.M
        L = a * model.DK + r * model.M + k * model.S
        dL = dD * model.DK + dr * model.M + dk * model.S
        solver = cache_fetch(
            self.cache["operator"], df.LUSolver, {"x": x}, A=M + dt * L
        )

        Mdot = matrix_operator(M)
        dLdot = matrix_operator(dL)
        num_intervals = timestepper.num_intervals()
        for n in tqdm.tqdm(
            range(num_intervals, 0, -1), total=num_intervals, disable=self.silent
        ):
            nj = measure_interval(n, self.td, self.timestepper)
            jump = sum((matmul(M, dYm[j].vector()) / self.Yd_norms[j] for j in nj))
            solver.solve(
                dP[n - 1].vector(),
                Mdot(dP[n].vector()) - dt * dLdot(P[n - 1].vector()) - jump,
            )
        return dP


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
