class LinearDataInterpolator(df.Function):
    def __init__(self, input, funcname, domain, timescale=1.0, valuescale=1.0):
        tvec, C_sas = read_function_data(str(input), domain, funcname)
        super().__init__(C_sas[0].function_space())
        for ci in C_sas:
            ci.vector()[:] *= valuescale
        self.timepoints = tvec * timescale
        self.interpolator = vectordata_interpolator(C_sas, self.timepoints)
        self.update(t=0.0)

    def __call__(self, t):
        self.vector()[:] = self.interpolator(t)
        return self

    def update(self, t):
        self.vector()[:] = self.interpolator(t)
        return self
