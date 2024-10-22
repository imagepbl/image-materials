from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np
import scipy.stats
import xarray as xr

from prism import Time, Timeline

from time_variable import TimeVariable


class Distribution(Protocol):
    """Protocol for a distribution with a survival function."""
    survival_function: Callable

    def get_parameters(self) -> dict[str, float]:
        ...


@dataclass
class FoldedNormalDistribution:
    """Folded normal distribution with mean and standard deviation."""
    mean: float
    std: float
    survival_function: Callable = scipy.stats.foldnorm.sf

    def get_parameters(self)-> dict[str, float]:
        return {"c": self.mean/self.std, "loc": 0, "scale": self.std}


@dataclass
class WeibullDistribution:
    """Weibull distribution with shape and scale parameters."""
    shape: float
    scale: float
    survival_function: Callable = scipy.stats.weibull_min.sf

    def get_parameters(self) -> dict[str, float]:
        return {"c": self.shape, "scale": self.scale}


@dataclass
class Survival:
    """Survival function for a cohort model."""
    distribution: Distribution
    _time: Timeline
    _matrix: xr.DataArray = field(init=False)

    def __post_init__(self):
        timeslots = self._time.all_timeslots()
        nt = len(timeslots)
        self._matrix = xr.DataArray(
            dims=["time", "cohort"],
            coords=dict(
                time=timeslots,
                cohort=timeslots,
            ),
            data=np.zeros((nt, nt)),
        )

    def __getitem__(self, key: tuple) -> xr.DataArray:
        """Get a cohort at time `t`."""
        t, cohort, *dims = key
        time_index = self._time.time_to_index(t)
        cohort_index = self._time.time_to_index(cohort)
        return self._matrix[time_index, cohort_index]

    def __setitem__(self, t: float, value: np.ndarray):
        """Set a cohort at time `t`."""
        time_index = self._time.time_to_index(t)
        self._matrix[time_index] = value

    def compute(self, time: Time):
        pass

    # def sel(self, args, kwargs):
    #     pass

    # def set_along_timeline(self, at_time: float, data: np.ndarray):
    #     pass


@dataclass
class GenericSurvival[Timed, Cohort, *Dimensions, Unit]:
    """Survival function for a cohort model."""
    distribution: Distribution
    _time: Timeline
    _matrix: TimeVariable[*Dimensions, Unit] = field(init=False)

    def __post_init__(self):
        timeslots = self._time.all_timeslots()
        nt = len(timeslots)
        self._matrix = TimeVariable(
            xr.DataArray(
                dims=["time", "cohort"],
                coords=dict(
                    time=timeslots,
                    cohort=timeslots,
                ),
                data=np.zeros((nt, nt)),
            )
        )

    def __getitem__(self, key: tuple) -> TimeVariable[*Dimensions, Unit]:
        """Get a cohort at time `t`."""
        t, cohort, *dims = key
        time_index = self._time.time_to_index(t)
        cohort_index = self._time.time_to_index(cohort)
        return self._matrix[time_index, cohort_index]

    def __setitem__(self, t: float, value: np.ndarray):
        """Set a cohort at time `t`."""
        time_index = self._time.time_to_index(t)
        self._matrix[time_index] = value

    def compute(self, time: Time):
        pass


@dataclass
class StaticSurvival(Survival):
    """Static survival function for a cohort model."""

    def __setitem__(self, t: float, value: np.ndarray):
        """Set a cohort at time `t`."""
        self._matrix.loc[t, :] = value

    def compute(self, time: Time):
        """Compute the survival function for a given distribution."""
        number_of_future_timesteps = time.number_of_timesteps - time.time_to_index(time.t)
        survival_timesteps = np.arange(number_of_future_timesteps)
        parameters = self.distribution.get_parameters()
        survival = self.distribution.survival_function(survival_timesteps, **parameters)
        self[time.t] = survival


@dataclass
class RecursiveSurvival(Survival):
    """Recursive survival function for a cohort model."""

    def __setitem__(self, t: float, value: np.ndarray):
        """Set a cohort at time `t`."""
        self._matrix.loc[t, :] = value

    def compute(self, time: Time):
        """Compute the survival function for a given distribution."""
        t = time.t
        number_of_future_timesteps = time.number_of_timesteps - time.time_to_index(t)  # TODO: create a method for this in prism?
        survival_timesteps = np.arange(number_of_future_timesteps)
        parameters = self.distribution.get_parameters()
        survival = self.distribution.survival_function(survival_timesteps, **parameters)
        # create survival relative to t - dt
        survival_ratio = np.ones_like(survival)
        survival_ratio[1:] = survival[1:] / survival[:-1]
        self[t] = survival_ratio


survival = Survival(FoldedNormalDistribution(0.5, 0.1), Timeline(0, 1, 0.1))
survival.compute(t)
