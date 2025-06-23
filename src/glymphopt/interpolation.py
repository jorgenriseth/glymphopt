from typing import Callable


import dolfin as df
import numpy as np


class LinearDataInterpolator(df.Function):
    def __init__(self, td, Y, timescale=1.0, valuescale=1.0):
        super().__init__(Y[0].function_space())
        for ci in Y:
            ci.vector()[:] *= valuescale
        self.timepoints = td * timescale
        self.interpolator = vectordata_interpolator(Y, self.timepoints)
        self.update(t=0.0)

    def __call__(self, t):
        self.vector()[:] = self.interpolator(t)
        return self.vector()

    def update(self, t):
        self.vector()[:] = self.interpolator(t)
        return self


def vectordata_interpolator(
    data: list[df.Function], times: np.ndarray
) -> Callable[[float], np.ndarray]:
    dt = times[1:] - times[:-1]
    dvec = [di.vector()[:] for di in data]
    dudt = [(d1 - d0) / dti for d0, d1, dti in zip(dvec[:-1], dvec[1:], dt)]

    def call(t: float) -> np.ndarray:
        if t <= times[0]:
            return dvec[0]
        if t >= times[-1]:
            return dvec[-1]
        bin = np.digitize(t, times) - 1
        return dvec[bin] + dudt[bin] * (t - times[bin])

    return call
