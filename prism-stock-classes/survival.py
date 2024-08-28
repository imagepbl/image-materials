import numpy as np
import scipy
from itertools import islice
import xarray as xr
from abc import abstractmethod, ABC

class SurvivalMatrix:
    def __init__(self, survival):
        # TODO: Add docstrings
        self.survival_matrix = xr.DataArray(0.0, dims=("time", "cohort", "mode"),
                                            coords={"time": survival.time_series.to_numpy(),
                                                    "cohort": survival.time_series.to_numpy(),
                                                    "mode": survival.modes})
        self._cached_timesteps = set()
        self.num_timesteps = len(survival.time_series)
        self.survival = survival

    def __getitem__(self, idx):
        # TODO: make the computation dependent on t_idx
        # We know that if t_idx < cohort_idx, the result is 0,
        # So if we compute s[t, :], we know that we only need to compute
        # the columns until cohort_idx == t
        t, cohort = idx

        # TODO: Check if this actually creates the cohort indices properly.
        if isinstance(cohort, slice):
            comp_list = islice(self.survival_matrix.indexes["cohort"], *cohort.indices(self.num_timesteps))
        else:
            comp_list = [cohort]

        # Is the column already computed?
        for cur_cohort in comp_list:
            if cur_cohort not in self._cached_timesteps:
                # Compute the survival matrix for the current cohort_idx
                # The survival coeficients are only relevant for future times t
                # So the values are created in column cohort_idx, for t_idx=cohort_idx, cohort_idx+1, ...
                for mode in self.survival.modes:
                    self.survival_matrix.loc[cur_cohort:, cur_cohort, mode] = self.survival.compute_survival(mode, cur_cohort)
                self._cached_timesteps.add(cur_cohort)
        return self.survival_matrix.loc[t, cohort]


class ScipySurvival(ABC):
    def __init__(self, lifetime_parameters):
        self.lifetime_parameters = lifetime_parameters

    def compute_survival(self, mode, cohort):
        n_coords_left = len(self.time_series.loc[cohort:])
        return self._method(
            np.arange(n_coords_left),
            **self.get_param(mode, cohort)
        )

    @abstractmethod
    def get_param(self, mode, cohort):
        raise NotImplementedError()

    @property
    def modes(self):
        return np.unique([x[0] for x in self.lifetime_parameters.data_vars])

    @property
    def time_series(self):
        return self.lifetime_parameters.coords["year"]

class WeibullSurvival(ScipySurvival):
    _method = scipy.stats.weibull_min.sf

    def get_param(self, mode, cohort):
        return {
            "c": self.lifetime_parameters[mode, "shape"].loc[cohort],
            "scale": self.lifetime_parameters[mode, "scale"].loc[cohort],
            "loc": 0
        }


class FoldedNormalSurvival(ScipySurvival):
    _method = scipy.stats.foldnorm.sf

    def get_param(self, mode, cohort):
        mean = self.lifetime_parameters[mode, "mean"].loc[cohort]
        stdev = self.lifetime_parameters[mode, "stdev"].loc[cohort]
        return {"c": mean/stdev, "scale": stdev, "loc": 0}

def _is_iterable(val):
    try:
        iter(val)
        return True
    except TypeError:
        return False
