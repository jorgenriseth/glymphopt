from typing import Sequence
import dataclasses
import numpy as np


@dataclasses.dataclass
class TimeStepper:
    dt: float
    interval: tuple[float, float]

    def vector(self) -> np.ndarray:
        t0, t1 = self.interval
        num_intervals = self.num_intervals()
        return np.linspace(t0, t1, num_intervals + 1)

    def num_intervals(self) -> int:
        t0, t1 = self.interval
        return int(np.ceil(np.round((t1 - t0) / self.dt, 12)))

    def find_intervals(self, timepoints: Sequence | np.ndarray) -> np.ndarray:
        t0, _ = self.interval
        N = self.num_intervals()
        return np.minimum(
            N - 1,
            np.floor((np.array(timepoints) - t0) / self.dt).astype(int),
        )


def subinterval_timestepper(dt_bound: float, t0: float, t1: float) -> TimeStepper:
    num_intervals = int(
        np.ceil(np.round((t1 - t0) / dt_bound, 12))
    )  # Round for machine precision errors.
    dt = np.round((t1 - t0) / num_intervals, 12)
    return TimeStepper(dt=dt, interval=(t0, t1))
