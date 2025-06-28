import dolfin as df
import numpy as np

from glymphopt.timestepper import TimeStepper
from glymphopt.operators import mass_matrix, bilinear_operator


def measure_interval(n: int, td: np.ndarray, timestepper: TimeStepper):
    bins = np.digitize(td, timestepper.vector(), right=True)
    return list(np.where(n == bins)[0])


def measure(
    timesteps: TimeStepper,
    states: list[df.Function],
    measure_times: list[float] | np.ndarray,
):
    V = states[0].function_space()
    Y = [df.Function(V, name="measured_state") for _ in range(len(measure_times))]
    find_intervals = timesteps.find_intervals(measure_times)
    for i, _ in enumerate(measure_times[1:], start=1):
        ni = find_intervals[i]
        # Use stepwise solution, as it is not necessarily straight forward to use
        # the linear interpolant between timepoints.
        # print(i, ni, time[ni], ti, time[ni + 1])
        Y[i].assign(states[ni + 1])
        # Need to rederive this expression if I want to use linear interpolant
        # step_fraction = (ti - time[ni]) / dt
        # Y[i].assign(((1 - step_fraction) * states[ni] + step_fraction * states[ni + 1]))
    return Y


class LossFunction:
    def __init__(self, td, Yd):
        self.td = td
        self.Yd = Yd
        self.V = Yd[0].function_space()
        self.M = mass_matrix(self.V)
        self._M_ = bilinear_operator(self.M)
        self.norms = [self._M_(yi.vector(), yi.vector()) for yi in Yd]

    def __call__(self, Ym):
        timepoint_errors = [
            self._M_(Ym_i.vector() - Yd_i.vector(), Ym_i.vector() - Yd_i.vector())
            / norm_i
            for Ym_i, Yd_i, norm_i in zip(Ym[1:], self.Yd[1:], self.norms[1:])
        ]
        return 0.5 * sum(timepoint_errors)

    def measure(self, timesteps, Y, td, measure_op):
        Y = [df.Function(self.V, name="measured_state") for _ in range(len(td))]
        find_intervals = timesteps.find_intervals(self.td)
        for i, _ in enumerate(td[1:], start=1):
            ni = find_intervals[i]
            Y[i].assign(measure_op(Y[ni + 1]))
        return Y
