import subprocess

import dolfin as df
import matplotlib.pyplot as plt
import numpy as np
import pantarei as pr
import ufl

from glymphopt.utils import apply_affine, cell_midpoints
from glymphopt.visual import plot_step

# %% Settings
sim_resolution = 2048
sim_dt = 1e-3
measure_resolution = 16
measure_domain_view = (-2.2, 2.31)
measure_time = 1.0

sim_path = "data/sim_interpolate.hdf"

# %% Simulate data
cmd_simulate = (
    "python scripts/simulate_data.py"
    + f" --output {sim_path}"
    + f" --resolution {sim_resolution}"
    + f" --dt {sim_dt}"
    + " --visual"
)
subprocess.run(cmd_simulate, shell=True).check_returncode()


# %% Measure
def quadrature_measurement(
    f: ufl.Coefficient, measure_domain: df.Mesh, degree: int = 3
) -> np.ndarray:
    """Perform measurement adhering to the 'measure_domain' of a dolfin
    expression or function 'f' by performing cellwise quadrature on f and return
    the values as a numpy-array."""
    dx = df.Measure("dx", measure_domain)
    DG0 = df.FunctionSpace(measure_domain, "DG", 0)
    v = df.TestFunction(DG0)
    measures = df.assemble(
        (1.0 / df.CellVolume(measure_domain)) * df.inner(f, v) * dx,
        form_compiler_parameters={"quadrature_degree": degree},
    )
    return np.array(measures)


def interpolation_measurement(
    f: ufl.Coefficient, measure_domain: df.Mesh
) -> np.ndarray:
    """Interpolate expression/or function onto a DG0-function space"""
    DG0 = df.FunctionSpace(measure_domain, "DG", 0)
    u = df.interpolate(f, DG0)
    return np.array(u.vector())


def interpolate_to_functionspace(
    f: ufl.Coefficient, V: df.FunctionSpace, boundaries: list[df.DirichletBC]
) -> df.Function:
    dx = df.Measure("dx", V.mesh())
    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    a = u * v * dx
    l = f * v * dx

    A = df.assemble(a)
    b = df.assemble(l)
    for bc in boundaries:
        bc.apply(A, b)
    u = df.Function(V)
    df.solve(A, u.vector(), b)
    return u


def in_domain(x, domain):
    first_entity = domain.bounding_box_tree().compute_first_entity_collision(
        df.Point(x)
    )
    return first_entity <= domain.num_entities(domain.geometric_dimension())


class ExtendedFunction(df.UserExpression):
    """Assign values from a dict to cells labeled by subdomains."""

    def __init__(
        self,
        f: ufl.Coefficient,
        mesh: df.Mesh,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mesh = mesh
        self.f = f

    def eval(self, value, x):
        if in_domain(x, self.mesh):
            value[0] = self.f(x)
        else:
            value[0] = 0.0

    def eval_cell(self, value, x, ufc_cell):
        if in_domain(x, self.mesh):
            value[0] = self.f(x)
        else:
            value[0] = 0.0
        print(value)

    def value_shape(self):
        return ()


class InDomain(df.UserExpression):
    """Assign values from a dict to cells labeled by subdomains."""

    def __init__(
        self,
        mesh: df.Mesh,
        **kwargs,
    ):
        super().__init__(kwargs)
        self.mesh = mesh

    def eval_cell(self, value, x, ufc_cell):
        if in_domain(x, self.mesh):
            value[0] = 1.0
        else:
            value[0] = np.nan
        print(value)

    def value_shape(self):
        return ()


def interpolate_to_functionspace(
    f: ufl.Coefficient, V: df.FunctionSpace, boundaries: list[df.DirichletBC]
) -> df.Function:
    dx = df.Measure("dx", V.mesh())
    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    a = u * v * dx
    l = f * v * dx

    A = df.assemble(a)
    b = df.assemble(l)
    for bc in boundaries:
        bc.apply(A, b)
    u = df.Function(V)
    df.solve(A, u.vector(), b)
    return u


measure_domain = df.IntervalMesh(measure_resolution, *measure_domain_view)

x = np.linspace(*measure_domain_view, 401)
DG0 = df.FunctionSpace(measure_domain, "DG", 0)
DG1 = df.FunctionSpace(measure_domain, "DG", 1)

c_sample = pr.interpolate_from_file(sim_path, "concentration", measure_time)
sim_domain = c_sample.function_space().mesh()
c_sample.set_allow_extrapolation(True)


# %%
plt.figure()
f = df.interpolate(InDomain(sim_domain), DG1)
# f.set_allow_extrapolation(True)
plt.plot(x, [f(min(x[-1] - 1e-13, max(x[0] + 1e-13, xi))) for xi in x])
plt.show()


c_extended = interpolate_to_functionspace(c_sample * f, DG1, [])
c_extended.set_allow_extrapolation(True)
# V = DG1
# boundaries = []
# dx = df.Measure("dx", V.mesh())
# u = df.TrialFunction(V)
# v = df.TestFunction(V)
# a = u * v * dx
# l = (f * c_sample) * v * dx
# A = df.assemble(a)
# b = df.assemble(l)
# for bc in boundaries:
#     bc.apply(A, b)
# c_extended = df.Function(V)
# df.solve(A, c_extended.vector(), b)
plt.plot(x, [c_sample(xi) for xi in x], label="c_d")
plt.plot(x, [c_extended(xi) for xi in x])
plt.show()

# %%
D_d = interpolation_measurement(c_extended, measure_domain)
# D_d = quadrature_measurement(c_extended, measure_domain, degree=7)
z = cell_midpoints(measure_domain).flatten()
plot_step(z, D_d, label="D_d")
plt.plot(x, [c_sample(xi) for xi in x], label="c_d")
plt.plot(x, [c_extended(xi) for xi in x], label="c_d_ext")
plt.legend()
plt.show()

# %%
t_idx = 3
imagefile = f"data/concentration_{t_idx}.npz"

with df.HDF5File(df.MPI.comm_world, "data/sim_sampled.hdf", "r") as hdf:
    u = pr.read_function(hdf, "concentration", idx=t_idx)

image_data = np.load(imagefile)
D_d, lambda_d = image_data.values()
n = len(D_d)

grid = df.IntervalMesh(n, -1, 1)
domain = df.IntervalMesh(64, -1.0, 1.0)


def mri2fem_interpolate(D, lambda_, V, datafilter=None):
    u = df.Function(V)
    z = V.tabulate_dof_coordinates()
    ind = np.rint(apply_affine(np.linalg.inv(lambda_), z)).astype(int)
    ind = np.maximum(0, np.minimum(ind, n - 1))
    if datafilter is None:
        D = np.where(np.isnan(D), 0.0, D)
    else:
        D = datafilter(D)
    u.vector()[:] = D[ind].flatten()
    return u


def quadrature_measurement(
    f: ufl.Coefficient, measure_domain: df.Mesh, degree: int = 3
) -> np.ndarray:
    """Perform measurement adhering to the 'measure_domain' of a dolfin
    expression or function 'f' by performing cellwise quadrature on f and return
    the values as a numpy-array."""
    dx = df.Measure("dx", measure_domain)
    DG0 = df.FunctionSpace(measure_domain, "DG", 0)
    v = df.TestFunction(DG0)
    measures = df.assemble(
        (1.0 / df.CellVolume(measure_domain)) * df.inner(f, v) * dx,
        form_compiler_parameters={"quadrature_degree": degree},
    )
    return np.array(measures)


def interpolation_measurement(
    f: ufl.Coefficient, measure_domain: df.Mesh
) -> np.ndarray:
    """Interpolate expression/or function onto a DG0-function space"""
    DG0 = df.FunctionSpace(measure_domain, "DG", 0)
    u = df.interpolate(f, DG0)
    return np.array(u.vector())


V = df.FunctionSpace(domain, "CG", 1)
DG0 = df.FunctionSpace(domain, "DG", 0)

u_h = mri2fem_interpolate(D_d, lambda_d, V)
D_pq = quadrature_measurement(u_h, grid)
D_p = interpolation_measurement(u_h, grid)

ind = np.arange(n)
x_grid = np.linspace(-1, 1, n)
x = np.linspace(-1, 1, 401)
plt.figure()
plot_step(x_grid, D_d)
plot_step(x_grid, D_p)
plt.plot(x, [u_h(xi) for xi in x])

plt.figure()
e_rel = (D_p - D_d) / D_d
plot_step(ind, e_rel)
plt.plot()
plt.show()
