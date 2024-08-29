import numpy as np
import scipy
from itertools import islice
import xarray as xr
from abc import abstractmethod, ABC
from functools import cached_property
from collections import defaultdict

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
                self.survival_matrix.loc[cur_cohort:, cur_cohort] = self.survival.compute_survival(cur_cohort)
                self._cached_timesteps.add(cur_cohort)
        return self.survival_matrix.loc[t, cohort]

class WeibullDistribution():
    name = "weibull"
    method = scipy.stats.weibull_min.sf
    params = ["shape", "scale"]

    @staticmethod
    def get_param(param_dict):
        return {
            "c": param_dict["shape"],
            "scale": param_dict["scale"],
            "loc": 0
        }

    @staticmethod
    def has_param(param):
        if "shape" in param and "scale" in param:
            return True
        return False


class FoldedNormalDistribution():
    method = scipy.stats.foldnorm.sf
    params = ["mean", "stdev"]
    name = "folded_norm"

    @staticmethod
    def get_param(param_dict):
        mean, stdev = param_dict["mean"], param_dict["stdev"]
        # mean = self.lifetime_parameters[mode, "mean"].loc[cohort]
        # stdev = self.lifetime_parameters[mode, "stdev"].loc[cohort]
        return {"c": mean/stdev, "scale": stdev, "loc": 0}

    @staticmethod
    def has_param(param):
        if "mean" in param and "stdev" in param:
            return True
        return False


class ScipySurvival(ABC):
    distributions = {"weibull": WeibullDistribution,
                     "folded_norm": FoldedNormalDistribution}

    def __init__(self, lifetime_parameters):
        self.lifetime_parameters = lifetime_parameters
        mode_param = defaultdict(list)
        for mode, par in self.lifetime_parameters.data_vars:
            mode_param[mode].append(par)
    
        self.dist_mode = defaultdict(list)
        for dist_name, dist in self.distributions.items():
            for mode, param in mode_param.items():
                if dist.has_param(param):
                    self.dist_mode[dist_name].append(mode)

    def compute_survival(self, cohort):
        n_coords_left = len(self.time_series.loc[cohort:])
        for dist_name, mode_list in self.dist_mode.items():
            if len(mode_list) == 0:
                continue
            dist = self.distributions[dist_name]
            scipy_param = {"c": [], "scale": [], "loc": []}
            for mode in mode_list:
                mode_param = {param: self.lifetime_parameters[mode, param].loc[cohort]
                              for param in dist.param}
                cur_scipy_param = dist.get_param(**mode_param)
                # for param in dist.param:
                    # scipy_param[]
                    
            
        return self._method(
            np.arange(n_coords_left),
            **self.get_param(mode, cohort)
        )

    @cached_property
    def modes(self):
        return np.unique([x[0] for x in self.lifetime_parameters.data_vars])

    @property
    def time_series(self):
        return self.lifetime_parameters.coords["year"]


def _is_iterable(val):
    try:
        iter(val)
        return True
    except TypeError:
        return False
