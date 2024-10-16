"""Test survival classes."""
import numpy as np
import xarray as xr

from imagematerials.constants import SUBTYPE_SEPARATOR
from imagematerials.survival import ScipySurvival, SurvivalMatrix


class ExampleSurvival():
    def __init__(self, n_time):
        self._survival_matrix = xr.DataArray(0.0, dims=("time", "cohort"),
                                             coords={
                                                 "time": np.arange(n_time) + 1900,
                                                 "cohort": np.arange(n_time) + 1900,
                                             })
        self._survival_matrix.loc[self.time_series, self.time_series] = 1.0
        for i_cohort, cur_cohort in enumerate(self.time_series[:-1]):
            cur_cohort = cur_cohort.values
            self._survival_matrix.loc[cur_cohort+1:, cur_cohort] = np.random.rand(len(self.time_series)-i_cohort-1)

    @property
    def time_series(self):
        return self._survival_matrix.coords["time"]

    def compute_survival(self, cohort):
        return self._survival_matrix.loc[cohort:, cohort]

    def new_matrix(self):
        return xr.DataArray(0.0, dims=("time", "cohort"),
                                             coords={
                                                 "time": self.time_series.to_numpy(),
                                                 "cohort": self.time_series.to_numpy(),
                                             })

def test_example_survival_matrix():
    time_len = 20
    survival = ExampleSurvival(time_len)
    survival_matrix = SurvivalMatrix(survival)
    assert survival_matrix.num_timesteps == time_len
    assert len(survival_matrix._cached_timesteps) == 0
    assert survival_matrix.survival_matrix.shape == (time_len, time_len)

    res = survival_matrix[:, :]
    assert res.shape == (time_len, time_len)
    assert len(survival_matrix._cached_timesteps) == time_len
    for i_time in survival_matrix.survival.time_series:
        for i_cohort in survival_matrix.survival.time_series:
            assert survival_matrix[i_time, i_cohort] == res.loc[i_time, i_cohort]
            if i_time == i_cohort:
                assert survival_matrix[i_time, i_cohort] == 1
            elif i_time > i_cohort:
                assert survival_matrix[i_time, i_cohort] < 1
                assert survival_matrix[i_time, i_cohort] > 0
            else:
                assert survival_matrix[i_time, i_cohort] == 0


def create_life_times(prefix, more_life=0):
    time = np.arange(30)+1900
    mode = [f"{prefix}_{i}" for i in range(4)]
    array = xr.DataArray(0.0, dims=["time", "mode", "scipy_param"],
                         coords={"time": time, "mode": mode, "scipy_param": ["c", "scale"]})
    for i_mode, mode_name in enumerate(mode):
        array.loc[:, mode[i_mode], "c"] = 4
        array.loc[:, mode[i_mode], "scale"] = i_mode + 1
    array.attrs["loc"] = 0
    return array


def _check_part(sm_part, i_cohort, n_cohort, n_modes):
    assert sm_part.shape == (n_cohort-i_cohort, n_modes)
    assert np.all(sm_part[i_cohort+1:, :] >= 0 )
    assert np.all(np.isclose(sm_part[0, :], 1))
    assert not np.all(np.isclose(sm_part[1:, :], 1))


def test_scipy_survival():
    lifetime_vehicles = {
        "weibull": create_life_times("weibull"),
        "folded_norm": create_life_times("folded_norm", more_life=10)
    }
    survival = ScipySurvival(lifetime_vehicles)
    assert len(survival.modes) == 8
    assert len(survival.time_series) == 30
    assert survival.new_matrix().shape == (30, 30, 8)
    sm_part = survival.compute_survival(1924).to_numpy()
    _check_part(sm_part, 24, 30, 8)

def test_scipy_survival_subtypes():
    lifetime_vehicles = {
        "weibull": create_life_times("weibull"),
        "folded_norm": create_life_times("folded_norm", more_life=10)
    }
    output_modes = []
    for prefix in lifetime_vehicles:
        for i in range(4):
            for sub_i in range(2):
                output_modes.append(f"{prefix}_{i}{SUBTYPE_SEPARATOR}{sub_i}")
    survival = ScipySurvival(lifetime_vehicles, output_modes=output_modes)
    assert len(survival.modes) == 16
    sm_part = survival.compute_survival(1924).to_numpy()
    _check_part(sm_part, 24, 30, 16)
    assert len(survival.time_series) == 30
    assert survival.new_matrix().shape == (30, 30, 16)
