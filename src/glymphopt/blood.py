from typing import Self
import dolfin as df
import numpy as np


def biexponential(t, a, b, c):
    return a * (np.exp(-b * t) - np.exp(-c * t))


def bateman(t, a, b, c):
    return a * np.power(t, b) * np.exp(-c * t)


class BatemanModel(df.Function):
    def __init__(self, function_space, a, b, c):
        super().__init__(function_space, name="blood_concentration")
        self.const_ = df.Constant(0.0)
        self.params_ = {"a": a, "b": b, "c": c}

    def __call__(self, t) -> Self:
        # self.const_.assign(bateman(t, **self.params_))
        # self.assign(self.const_)
        self.vector()[:] = bateman(t, **self.params_)
        return self

    def update(self, t) -> Self:
        return self(t)


class BiexponentialModel(df.Function):
    def __init__(self, function_space, a, b, c):
        super().__init__(function_space, name="blood_concentration")
        self.const_ = df.Constant(0.0)
        self.params_ = {"a": a, "b": b, "c": c}

    def __call__(self, t) -> Self:
        self.const_.assign(biexponential(t, **self.params_))
        self.assign(self.const_)
        return self

    def update(self, t) -> Self:
        return self(t)


class ZeroModel(df.Function):
    def __init__(self, function_space, *args, **kwargs):
        super().__init__(function_space, name="blood_concentration")

    def __call__(self, t) -> Self:
        return self

    def update(self, t) -> Self:
        return self
