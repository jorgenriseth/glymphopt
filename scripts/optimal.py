import dolfin as df
import matplotlib.pyplot as plt
import numpy as np
import pantarei as pr
from ufl import dx, grad, inner, lhs, rhs

import glymphopt.dfa as df

N = 256
element = df.FiniteElement("Lagrange", df.interval, 2)
mesh = df.IntervalMesh(N, -1.0, 1.0)
V = df.FunctionSpace(mesh, element)

time = df.Constant(0.0)
dt = 1e-3
T = 0.1
timevec = pr.TimeKeeper(dt, T).as_vector()

alpha = 1
D_true = 1.4
r_true = 10 * alpha

a = np.sqrt(alpha / D_true)
scale = 1 / np.cosh(a)
expr = df.Expression(
    f"{scale}*cosh(x[0] * {a}) * exp(- {r_true - alpha}*t )",
    degree=2,
    t=time,
)
u_data = df.Function(V, name="data")
u_data = df.interpolate(expr, V)
ctrls = {
    "D": df.Constant(D_true * 1.5),
    "r": df.Constant(r_true * 1.5),
}

u0 = df.Function(V, name="concentration")
u0.assign(df.interpolate(expr, V))
dx = df.Measure("dx", domain=mesh)
u = df.TrialFunction(V)
v = df.TestFunction(V)

F = (
    (u - u0) / dt * v * dx
    + 0.5 * ctrls["D"] * inner((grad(u) + grad(u0)), grad(v)) * dx
    + 0.5 * ctrls["r"] * (u + u0) * v * dx
)
F = (
    (u - u0) / dt * v * dx
    + ctrls["D"] * inner(grad(u), grad(v)) * dx
    + ctrls["r"] * u * v * dx
)
a, L = lhs(F), rhs(F)

bc = df.DirichletBC(V, u_data, "on_boundary")
Nsteps = int(round(T / dt, 0))
plot_every = Nsteps // 10
print(Nsteps)
A = df.assemble(a)
b0 = df.assemble(L)
u = df.Function(V)
plt.figure()
for idx, ti in enumerate(timevec[1:]):
    b = df.assemble(L)
    time.assign(ti)
    u_data.assign(df.interpolate(expr, V))
    bc.apply(A, b)
    df.solve(A, u0.vector(), b)
    if idx % plot_every == 0:
        df.plot(u0)
plt.ylim(0, 1)
plt.show()
# %%
plt.figure()
# df.plot(u_data, label="data")
df.plot(u0, label="solution")
df.plot(expr, mesh=mesh, label="expr")
plt.ylim(0, 1)
plt.legend()
plt.show()

# %%
print(float(time), ctrls["D"])
j = 0.5 * float(dt) * df.assemble((u_data - u0) ** 2 * dx) / df.norm(u_data, "L2") ** 2

m = [df.Control(c) for c in ctrls.values()]
rf = df.ReducedFunctional(j, m)
opt_ctrls = df.minimize(rf, options={"maxiter": 100, "gtol": 1e-12})

print("D", "    r")
print(" ".join([str(D_true), str(r_true)]))
print(" ".join([str(float(c)) for c in opt_ctrls]))


# # %%
plt.figure()
df.plot(u0, label="solution")
df.plot(u_data, label="data")
# df.plot(expr, mesh=mesh, label="expr")
plt.ylim(0, 1)
plt.legend()
plt.show()
