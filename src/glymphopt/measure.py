import dolfin as df
import numpy as np

from glymphopt.timestepper import TimeStepper


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
