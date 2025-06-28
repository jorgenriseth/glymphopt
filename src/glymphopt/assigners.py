import dolfin as df


class VolumetricConcentration(df.Function):
    def __init__(self, volume_fractions, V, W):
        assert len(volume_fractions) == W.num_sub_spaces()
        super().__init__(V, name="measured_state")
        self.phis = volume_fractions
        self.basis = [df.Function(V) for _ in range(len(volume_fractions))]
        self.assigners = [
            df.FunctionAssigner(V, W.sub(i)) for i in range(len(volume_fractions))
        ]

    def __call__(self, Y):
        [
            self.assigners[i].assign(self.basis[i], Y.sub(i))
            for i in range(len(self.basis))
        ]
        return sum([self.phis[i] * self.basis[i] for i in range(len(self.basis))])


class WeightedAssigner:
    def __init__(self, weights, V, W):
        assert len(weights) == W.num_sub_spaces()
        self.V = V
        self.W = W

        self.w = weights
        self.assigners = [
            df.FunctionAssigner(W.sub(i), V) for i in range(W.num_sub_spaces())
        ]
        self.subfuncs = [df.Function(W) for _ in range(W.num_sub_spaces())]

    def __call__(self, u):
        [
            self.assigners[i].assign(self.subfuncs[i].sub(i), u)
            for i in range(self.W.num_sub_spaces())
        ]
        return sum(
            [self.w[i] * self.subfuncs[i] for i in range(self.W.num_sub_spaces())]
        )


class SuperspaceAssigner(df.Function):
    def __init__(self, V, W):
        super().__init__(W, name="superspace")
        self.V = V
        self.W = W
        self.assigners = [
            df.FunctionAssigner(W.sub(i), V) for i in range(W.num_sub_spaces())
        ]
        self.u_ = df.Function(V, name="placeholder")

    def __call__(self, u):
        self.u_.assign(u)
        [
            self.assigners[i].assign(self.sub(i), self.u_)
            for i in range(self.W.num_sub_spaces())
        ]
        return self
