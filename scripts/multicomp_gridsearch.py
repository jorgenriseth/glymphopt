import click
import numpy as np
import dolfin as df
import tqdm
import pandas as pd

from dolfin import inner, grad, dot


from glymphopt.cache import CacheObject, cache_fetch
from glymphopt.coefficientvectors import CoefficientVector

from glymphopt.interpolation import LinearDataInterpolator
from glymphopt.io import read_mesh, read_function_data, read_augmented_dti
from glymphopt.operators import (
    matrix_operator,
    bilinear_operator,
    matmul,
    mass_matrix,
    diffusion_operator_matrices,
    transfer_matrix,
    boundary_mass_matrices,
    mass_matrices,
)

from glymphopt.minimize import adaptive_grid_search
from glymphopt.timestepper import TimeStepper
from glymphopt.assigners import VolumetricConcentration, SuperspaceAssigner
from glymphopt.measure import LossFunction
from glymphopt.timestepper import TimeStepper


@click.command()
@click.option("--input", "-i", type=str, required=True)
@click.option("--output", "-o", type=str, required=True)
def main(input, output):
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

    data_path = (
        "/home/jorgen/gonzo/mri_processed_data/sub-01/modeling/resolution32/data.hdf"
    )
    domain = read_mesh(data_path)

    D = read_augmented_dti(data_path)
    D.vector()[:] *= coefficients["rho"]

    td, Yd = read_function_data(data_path, domain, "concentration")
    td, Y_bdry_tmp = read_function_data(data_path, domain, "boundary_concentration")

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
        n_iterations=6,
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


def as_vector_element(el, dim):
    return df.VectorElement(
        family=el.family(), cell=el.cell(), degree=el.degree(), dim=dim
    )


class TwocompartmentModel:
    def __init__(self, W, D, g):
        self.W = W
        self.Me, self.Mp = mass_matrices(W)
        self.D = D
        self.DKe, self.DKp = diffusion_operator_matrices(D, W)
        self.Se, self.Sp = boundary_mass_matrices(W)
        self.T = transfer_matrix(W)
        self.g = g

    def M(self, coefficients) -> df.Matrix:
        n_e = coefficients["n_e"]
        n_p = coefficients["n_p"]
        return n_e * self.Me + n_p * self.Mp

    def L(self, coefficients) -> df.Matrix:
        n_e = coefficients["n_e"]
        n_p = coefficients["n_p"]
        t_ep = coefficients["t_ep"]
        t_pb = coefficients["t_pb"]
        k_e = coefficients["k_e"]
        k_p = coefficients["k_p"]
        gamma = coefficients["gamma"]
        Me = self.Me
        Mp = self.Mp
        DKe = self.DKe
        DKp = self.DKp
        T = self.T
        Se = self.Se
        Sp = self.Sp
        return (
            (n_e * DKe + gamma * n_p * DKp)
            + t_ep * T
            + t_pb * Mp
            + (k_e * Se + k_p * Sp)
        )

    def preconditioner(self, coefficients) -> None | df.Matrix:
        return None
        n_e = coefficients["n_e"]
        n_p = coefficients["n_p"]
        t_ep = coefficients["t_ep"]
        t_pb = coefficients["t_pb"]
        k_e = coefficients["k_e"]
        k_p = coefficients["k_p"]
        De = Dp = self.D
        dx = df.Measure("dx", self.W.mesh())
        ds = df.Measure("ds", self.W.mesh())
        ce, cp = df.TrialFunctions(self.W)
        ve, vp = df.TestFunctions(self.W)
        return df.assemble(
            inner(n_e * dot(De, grad(ce)), grad(ve)) * dx
            + inner(n_p * dot(Dp, grad(cp)), grad(vp)) * dx
            + t_pb * cp * vp * dx
            + k_e * ce * ve * ds
            + k_p * cp * vp * ds
        )


class MulticompartmentInverseProblem:
    def __init__(
        self, td, Yd, coefficientvector, g, D, dt=3600, timescale=1.0, progress=True
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
        domain = self.V.mesh()
        self.W = df.FunctionSpace(
            domain, as_vector_element(self.V.ufl_element(), dim=2)
        )

        self.model = TwocompartmentModel(self.W, D=D, g=g)
        self.cache = {
            "state": CacheObject(),
            "adjoint": CacheObject(),
            "g": CacheObject(),
            "sensitivity": CacheObject(),
            "operator": CacheObject(),
        }

        _M_ = bilinear_operator(mass_matrix(self.V))
        self.Yd_norms = [_M_(yi.vector(), yi.vector()) for yi in self.Yd]
        self.coefficients = coefficientvector
        self.superassigner = SuperspaceAssigner(self.V, self.W)

        n_e, n_p = coefficients["n_e"], coefficients["n_p"]
        self.volumetric_concentration = VolumetricConcentration(
            (n_e, n_p), self.V, self.W
        )
        self.loss = LossFunction(self.td, self.Yd)

    def F(self, x):
        Y = cache_fetch(self.cache["state"], self.forward, {"x": x}, x=x)
        Ym = self.measure(Y)
        return self.loss(Ym)

    def measure(self, Y):
        Ym = [df.Function(self.V, name="measured_state") for _ in range(len(self.td))]
        find_intervals = self.timestepper.find_intervals(self.td)
        for i, _ in enumerate(self.td[1:], start=1):
            ni = find_intervals[i]
            Ym[i].assign(self.volumetric_concentration(Y[ni + 1]))
        return Ym

    def forward(self, x):
        coefficients = self.coefficients.from_vector(x)
        ne = coefficients["n_e"]
        np = coefficients["n_p"]

        timestepper = self.timestepper
        dt = timestepper.dt
        timepoints = timestepper.vector()

        Y = [df.Function(self.W, name="state") for _ in range(len(timepoints))]
        Y[0].assign(self.superassigner(self.Yd[0]))

        model = self.model
        M = model.M(coefficients)
        L = model.L(coefficients)
        G = cache_fetch(self.cache["g"], self.boundary_vectors, {"_": ""})
        solver = cache_fetch(
            self.cache["operator"],
            df.KrylovSolver,
            {"x": x},
            A=M + dt * L,
            method="cg",
            preconditioner="jacobi",
        )
        Mdot = matrix_operator(M)
        N = self.timestepper.num_intervals()
        for n in tqdm.tqdm(range(N), total=N, disable=self.silent):
            solver.solve(Y[n + 1].vector(), Mdot(Y[n].vector()) + dt * G[n + 1])
        return Y

    def boundary_vectors(self):
        model = self.model
        timestepper = self.timestepper
        k_e = self.coefficients.dict()["k_e"]
        k_p = self.coefficients.dict()["k_p"]
        S = k_e * model.Se + k_p * model.Sp
        return [
            matmul(S, self.superassigner(model.g.update(t)).vector())
            for t in timestepper.vector()
        ]

    def gradF(self, x):
        coefficients = self.coefficients.from_vector(x)
        phi = coefficients["phi"]
        dt = self.timestepper.dt
        model = self.model
        Y = cache_fetch(self.cache["state"], self.forward, {"x": x}, x=x)
        Ym = self.measure(Y)

        P = cache_fetch(self.cache["adjoint"], self.adjoint, {"x": x}, x=x, Ym=Ym)
        G = cache_fetch(self.cache["g"], self.boundary_vectors, {"phi": phi}, phi=phi)

        _M_ = bilinear_operator(model.M)
        _DK_ = bilinear_operator(model.DK)
        _S_ = bilinear_operator(model.S)
        return dt * sum(
            np.array(
                [
                    _DK_(p.vector(), y.vector()),
                    _M_(p.vector(), y.vector()),
                    _S_(p.vector(), y.vector()) - p.vector().inner(g),
                ]
            )
            for y, p, g in zip(Y[1:], P[:-1], G[1:])
        )


#     def adjoint(self, x, Ym) -> list[df.Function]:
#         self.superassigner
#         coefficients = self.coefficients.from_vector(x)

#         timestepper = self.timestepper
#         dt = timestepper.dt
#         timepoints = timestepper.vector()

#         model = self.model
#         M = model.M
#         L = a * model.DK + r * model.M + k * model.S
#         solver = cache_fetch(self.cache["operator"], df.LUSolver, {"x": x}, A=M + dt * L)
#         P = [
#             df.Function(self.V, name="adjoint")
#             for _ in range(len(timepoints))
#         ]
#         Mdot = matrix_operator(M)
#         num_intervals = timestepper.num_intervals()
#         for n in tqdm.tqdm(range(num_intervals, 0, -1), total=num_intervals, disable=self.silent):
#             nj = measure_interval(n, self.td, self.timestepper)
#             jump = sum((M, self.superassigner(Ym[j] - Yd[j]).vector()) / seld.Yd_norms[j])# (Ym[j].vector() - Yd[j].vector()) / self.Yd_norms[j]) for j in nj))

#             solver.solve(
#                 P[n-1].vector(),
#                 Mdot(P[n].vector()) - jump,
#             )
#         return P


if __name__ == "__main__":
    main()
