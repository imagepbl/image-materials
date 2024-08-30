import numpy as np
import scipy
from itertools import islice
import xarray as xr
from abc import ABC
from functools import cached_property
from collections import defaultdict


class SurvivalMatrix:
    def __init__(self, survival):
        # TODO: Add docstrings
        self.survival_matrix = survival.new_matrix()
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
        return {"c": mean/stdev, "scale": stdev, "loc": 0}

    @staticmethod
    def has_param(param):
        if "mean" in param and "stdev" in param:
            return True
        return False


class ScipySurvival(ABC):
    def __init__(self, lifetime_parameters):
        self.lifetime_parameters = lifetime_parameters

    def new_matrix(self):
        return xr.DataArray(
            0.0, dims=("time", "cohort", "mode"),
            coords={"time": self.time_series.to_numpy(),
                    "cohort": self.time_series.to_numpy(),
                    "mode": self.modes})

    def compute_survival(self, cohort):
        n_coords_left = len(self.time_series.loc[cohort:])
        res_arrays = []
        for param_dict_array, method in self.lifetime_parameters.values():
            first_array = list(param_dict_array.values())[0]
            n_modes = first_array.shape[1]
            index_array = np.empty((n_coords_left, n_modes))
            index_array[:, :] = np.arange(n_coords_left).reshape(-1, 1)
            params = {param_name: (full_array.loc[cohort, :] if isinstance(full_array, xr.DataArray) else full_array)
                      for param_name, full_array in param_dict_array.items()
                      }
            res_numpy_array = method(index_array, **params)
            res_arrays.append(xr.DataArray(res_numpy_array, dims=("time", "mode"),
                                           coords={"time": self.time_series.loc[cohort:],
                                                   "mode": first_array.coords["mode"]}))
        return xr.concat(res_arrays, dim="mode")

    @cached_property
    def modes(self):
        all_modes = []
        for param_dict_array, _ in self.lifetime_parameters.values():
            first_array = list(param_dict_array.values())[0]
            all_modes.extend(str(x) for x in first_array.coords["mode"].to_numpy())
        return all_modes

    @property
    def time_series(self):
        first_dict = list(self.lifetime_parameters.values())[0][0]
        first_array = list(first_dict.values())[0]
        return first_array.coords["time"]


def _is_iterable(val):
    try:
        iter(val)
        return True
    except TypeError:
        return False


ALL_DISTRIBUTIONS = [WeibullDistribution, FoldedNormalDistribution]
NAME_TO_DIST = {dist.name: dist for dist in ALL_DISTRIBUTIONS}


def convert_life_time_vehicles(life_time_vehicles):
    life_time_vehicles = life_time_vehicles.rename({"year": "time"})
    mode_param = defaultdict(list)
    for mode, par in life_time_vehicles.data_vars:
        mode_param[mode].append(par)

    dist_mode = defaultdict(list)
    modes_done = set()
    for dist in ALL_DISTRIBUTIONS:
        for mode, param in mode_param.items():
            if mode not in modes_done and dist.has_param(param):
                dist_mode[dist.name].append(mode)
                modes_done.add(mode)

    scipy_params = {}
    for dist_name, mode_list in dist_mode.items():
        dist = NAME_TO_DIST[dist_name]

        param_arrays = {}
        for param in dist.params:
            array = xr.DataArray(
                0.0, dims=("time", "mode"),
                coords={
                    "time": life_time_vehicles.coords["time"].to_numpy(),
                    "mode": mode_list})
            for mode in mode_list:
                array.loc[:, str(mode)] = life_time_vehicles.data_vars[str(mode), param]
            param_arrays[param] = array
        scipy_params[dist_name] = (dist.get_param(param_arrays), dist.method)
    return scipy_params
