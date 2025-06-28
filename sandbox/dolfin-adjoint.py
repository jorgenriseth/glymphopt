import click
import dolfin as df
import numpy as np
import tqdm
import pandas as pd
import scipy as sp
from dolfin import inner, grad

from gripmodels.datagen import BoundaryConcentration
from gripmodels.dirac import DiracImpulses
from gripmodels.timestepper import TimeStepper
from gripmodels.quadrature import trapezoid
from gripmodels.interpolation import measurement_error, measure, LinearDataInterpolator
from gripmodels.io import read_function_data, read_mesh
from gripmodels.parameters import unpack


@click.command("main")
@click.option("--input", "-i", type=str, required=True)
@click.option("--output", "-o", type=str)
@click.option("--timescale", "-t", type=float, default=1.0)
def runsimulation(input: str, output: str, timescale: float):
    D_vec = np.logspace(1e-4, 5, 21)
    costs, gradients = [], []
    problem = InverseDiffusionReactionProblem(input, 1.0)

    h = 1.0
    param_dict = {
        "D": 10.0,
        "r": 1e-1,
    }
    J0 = problem(**param_dict)
    dJ = problem.gradient(**param_dict)
    dJdD0 = dJ["D"]
    dJdr0 = dJ["r"]
    records = []
    h = 1e-4
    while h > 1e-8:
        h *= 0.5
        J = problem(**{"D": param_dict["D"], "r": float(param_dict["r"] + h)})
        records.append(
            {
                "h": h,
                "loss": J,
                "loss0": J0,
                "fd": (J - J0) / h,
                "dJdr": dJdr0,
                "error_r": (J - J0) / h - dJdr0,
            }
        )
    dataframe = pd.DataFrame.from_records(records)
    print(dataframe)
    dataframe.to_csv("records.csv", index=False)


@click.command("main")
@click.option("--input", "-i", type=str, required=True)
@click.option("--output", "-o", type=str)
def optimal_D(input, output):
    problem = InverseDiffusionReactionProblem(input, 1.0)
    x0 = np.array([8.0, 1e-3])

    def F(x):
        param_dict = {"D": float(x[0]), "r": float(x[1]) ** 2}
        return problem(**param_dict)

    def dF(x):
        param_dict = {"D": float(x[0]), "r": float(x[1]) ** 2}
        gradient_dict = problem.gradient(**param_dict)
        return [
            gradient_dict["D"],
            2 * x[1] * gradient_dict["r"],
        ]

    def callback(int_res):
        if not int_res.success:
            print(int_res.message)
            print(int_res.x)
            print(int_res.jac)

    res = sp.optimize.minimize(F, x0, method="Newton-CG", jac=dF)
    print(res)
    print(res.x)


class InverseDiffusionReactionProblem:
    def __init__(self, input: str, timescale: float):
        default_coefficients = dict(
            phi=0.22,
            D=10.0,
            k=1.3,
            r=0.1,
        )
        domain = read_mesh(input)
        self.time_data, self.c_data = read_function_data(input, domain, "concentration")
        self.time_data *= timescale
        self.timesteps = TimeStepper(dt=0.1, endtime=self.time_data[-1])

        c_sas = LinearDataInterpolator(
            input, "boundary_concentration", domain, timescale
        )
        c_sas = BoundaryConcentration(self.c_data[0].function_space())
        self.coefficients = {
            **default_coefficients,
            "c_sas": c_sas,
        }

    def __call__(self, **kwargs):
        V = self.c_data[0].function_space()
        domain = V.mesh()

        self.coefficients = self.coefficients | kwargs
        self.states = forward(V, self.coefficients, self.timesteps)
        self.measured_states = measure(self.timesteps, self.states, self.time_data)
        self.measurement_errors = measurement_error(
            self.time_data, self.measured_states, self.c_data
        )
        dx = df.Measure("dx", domain)
        self.loss = 0.5 * sum(df.assemble(ei**2 * dx) for ei in self.measurement_errors)
        return self.loss

    def gradient(self, **kwargs):
        for var, value in kwargs.items():
            if value != self.coefficients[var]:
                self.loss = self(**kwargs)

        self.adjoint_states = adjoint_solve(
            self.timesteps,
            self.time_data,
            self.measurement_errors,
            self.coefficients,
        )
        domain = self.adjoint_states[0].function_space().mesh()
        dx = df.Measure("dx", domain)

        self.derivatives = {}
        if "D" in kwargs:
            diffusion_derivatives = [
                df.assemble(inner(grad(cn), grad(pn)) * dx)
                for cn, pn in zip(self.states, self.adjoint_states)
            ]
            self.derivatives["D"] = trapezoid(self.timesteps.dt, diffusion_derivatives)
        if "r" in kwargs:
            reaction_derivatives = [
                df.assemble(inner(cn, pn) * dx)
                for cn, pn in zip(self.states, self.adjoint_states)
            ]
            self.derivatives["r"] = trapezoid(self.timesteps.dt, reaction_derivatives)
        return self.derivatives


def forward(V, coefficients, timestepper):
    assert "c_sas" in coefficients, "need to provide boundary-concentration 'c_sas'"
    c_sas = coefficients["c_sas"]

    c0 = df.Function(V, name="c_{n-1}")
    a, L = variational_form(c0, timestepper.dt, coefficients)
    A, b = df.assemble_system(a, L)
    assembler = df.SystemAssembler(a, L)
    assembler.assemble(A, b)
    solver = df.LUSolver(A)
    timesteps = timestepper.vector()
    C = [df.Function(V, name="concentration") for _ in range(len(timesteps))]
    C[0].assign(c0)
    c_sas.update(0.0)
    for n, t in enumerate(tqdm.tqdm(timesteps[1:]), start=1):
        c0.assign(C[n - 1])
        c_sas.update(t)
        assembler.assemble(b)
        solver.solve(C[n].vector(), b)
    return C


def variational_form(c0, dt, coefficients):
    V = c0.function_space()
    domain = V.mesh()
    c, v = df.TrialFunction(V), df.TestFunction(V)
    phi, k, r, D, c_sas = unpack(coefficients, "phi", "k", "r", "D", "c_sas")
    dx = df.Measure("dx", domain)
    ds = df.Measure("ds", domain)
    a = (
        c * v * dx
        + inner(dt * D * grad(c), grad(v)) * dx
        + (dt * r) * c * v * dx
        + (dt * k / phi) * c * v * ds
    )
    L = c0 * v * dx + dt * k * c_sas * v * ds
    return a, L


def adjoint_variational_form(pn, dt, coefficients):
    V = pn.function_space()
    p, h = df.TrialFunction(V), df.TestFunction(V)
    phi, r, k, D, source = unpack(coefficients, "phi", "r", "k", "D", "source")
    dx = df.Measure("dx", V.mesh())
    ds = df.Measure("ds", V.mesh())
    a = (
        p * h * dx
        + inner(dt * D * grad(p), grad(h)) * dx
        + (dt * r) * p * h * dx
        + (dt * k / phi) * p * h * ds
    )
    L = pn * h * dx - source * h * dx
    return a, L


def adjoint_solve(timesteps, time_data, errors, coefficients):
    dt = timesteps.dt
    source = DiracImpulses(timesteps.vector(), time_data, errors)
    V = errors[0].function_space()
    pn = df.Function(V, name="p_n")
    a, L = adjoint_variational_form(pn, dt, coefficients | {"source": source})

    A, b = df.assemble_system(a, L)
    assembler = df.SystemAssembler(a, L)
    assembler.assemble(A, b)
    solver = df.LUSolver(A)

    P = [df.Function(V, name="adjoint") for _ in range(len(timesteps.vector()))]
    for n, t in list(enumerate(tqdm.tqdm(timesteps.vector()[1:]), start=1))[::-1]:
        pn.assign(P[n])
        source.update(t, dt)
        assembler.assemble(b)
        solver.solve(P[n - 1].vector(), b)
    return P


if __name__ == "__main__":
    optimal_D()
    # state = runsimulation()
