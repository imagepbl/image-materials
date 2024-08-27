import numpy as np
import scipy
from itertools import islice
import xarray as xr

class Survival:
    def __init__(self, time_series):
        # TODO: Add docstrings
        # self.compute_next = compute_next
        self.survival_matrix = xr.DataArray(0.0, dims=("time", "cohort"),
                                            coords={"time": time_series, "cohort": time_series})
        self._cached_timesteps = set()
        self.num_timesteps = len(time_series)

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
                self.survival_matrix.loc[cur_cohort:, cur_cohort] = self.compute_survival(cur_cohort)
                self._cached_timesteps.add(cur_cohort)
        return self.survival_matrix.loc[t, cohort]

    def compute_survival(self, cohort):
        cohort_idx = self.survival_matrix.indexes["cohort"].get_loc(cohort)

        cur_param = {}
        for par, par_val in self.param.items():
            if _is_iterable(par_val):
                cur_param[par] = par_val[cohort_idx]
            else:
                cur_param[par] = par_val
        return self._method(
            np.arange(len(self.survival_matrix.coords["cohort"]) - cohort_idx),
            **cur_param
        )


class WeibullSurvival(Survival):
    _method = scipy.stats.weibull_min.sf

    def __init__(self, time_series, shape, scale):
        self.param = {"c": shape, "scale": scale, "loc": 0}
        super().__init__(time_series)

class FoldedNormalSurvival(Survival):
    _method = scipy.stats.foldnorm.sf

    def __init__(self, time_series, mean, stdev):
        self.param = {"c": mean/stdev, "loc": 0, "scale": stdev}
        super().__init__(time_series)


def _is_iterable(val):
    try:
        iter(val)
        return True
    except TypeError:
        return False
