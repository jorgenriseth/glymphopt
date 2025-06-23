import abc
from typing import Optional, Self

import dolfin as df
from dolfin import inner, grad


def mass_matrix(V: df.FunctionSpace, dx: Optional[df.Measure] = None):
    u, v = df.TrialFunction(V), df.TestFunction(V)
    dx = dx or df.Measure("dx", V.mesh())
    return df.assemble(inner(u, v) * dx)


def stiffness_matrix(V: df.FunctionSpace):
    u, v = df.TrialFunction(V), df.TestFunction(V)
    dx = df.Measure("dx", V.mesh())
    return df.assemble(inner(grad(u), grad(v)) * dx)


def boundary_mass_matrix(V: df.FunctionSpace):
    u, v = df.TrialFunction(V), df.TestFunction(V)
    ds = df.Measure("ds", V.mesh())
    return df.assemble(inner(u, v) * ds)


def matmul(M: df.cpp.la.Matrix, x: df.cpp.la.Matrix):
    Mx = df.Vector()
    M.mult(x, Mx)
    return Mx


def matrix_operator(M: df.cpp.la.Matrix):
    x_ = df.Vector()

    def call(x: df.cpp.la.Vector):
        M.mult(x, x_)
        return x_

    return call


def bilinear_operator(M: df.cpp.la.Matrix):
    Ax = df.Vector()

    def call(x: df.cpp.la.Vector, y: df.cpp.la.Vector):
        M.mult(x, Ax)
        return Ax.inner(y)

    return call


class UpdatableCoefficient(abc.ABC):
    def update(self, t: float, *args) -> Self:
        return self

    def __call__(self, t: float, *args) -> df.Vector:
        pass


class UpdatableFunction(df.Function, UpdatableCoefficient):
    pass


class UpdatableExpression(df.Expression, UpdatableCoefficient):
    pass


def zero_vector(V: df.FunctionSpace) -> df.Vector:
    v = df.TestFunction(V)
    ds = df.Measure("dx", V)
    return df.assemble(inner(df.Constant(0.0), v) * ds)
