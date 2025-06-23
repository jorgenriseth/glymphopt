import dolfin as df
import numpy as np

from glymphopt.timesteppers import TimeStepper


def measure(
    timesteps: TimeStepper,
    states: list[df.Function],
    measure_times: list[float] | np.ndarray,
):
    dt = timesteps.dt
    time = timesteps.vector()
    V = states[0].function_space()
    Y = [df.Function(V, name="measured_state") for _ in range(len(measure_times))]
    find_intervals = timesteps.find_intervals(measure_times)
    for i, ti in enumerate(measure_times):
        ni = find_intervals[i]
        step_fraction = (ti - time[ni]) / dt
        Y[i].assign(((1 - step_fraction) * states[ni] + step_fraction * states[ni + 1]))
    return Y
