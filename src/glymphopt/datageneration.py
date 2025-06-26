import dolfin as df

from threecomp.operators import UpdatableExpression, UpdatableFunction


class Weibull(df.Expression):
    def __init__(self, amplitude, shape, scale, **kwargs):
        super().__init__(
            "A * k * pow(t / ell, k-1) * exp(-pow(t /ell, k))",
            A=amplitude,
            k=shape,
            ell=scale,
            t=df.Constant(0.0),
            **kwargs,
        )

    def update(self, t):
        self.t.assign(t)


class BoundaryConcentration(UpdatableFunction):
    def __init__(self, V, timescale=1.0):
        super().__init__(V, name="boundary_concentration")
        domain = V.mesh()
        assert hasattr(domain, "subdomains"), (
            "Mesh should have a 'subdomains' attribute"
        )

        c_ventricles = Weibull(2, 2, 10 * timescale, degree=1)
        c_sas = Weibull(1.3, 2, 24 * timescale, degree=1)
        self.expressions = {
            2: c_sas,
            4: c_ventricles,
        }
        dirichlet_bcs = [
            df.DirichletBC(V, expr, domain.boundaries, key)
            for key, expr in self.expressions.items()
        ]
        self.bcs = dirichlet_bcs
        for bc in dirichlet_bcs:
            bc.apply(self.vector())

    def update(self, t, *args):
        [c_i.update(t) for c_i in self.expressions.values()]
        for bc in self.bcs:
            bc.apply(self.vector())
        return self

    def __call__(self, t):
        return self.update(t).vector()
