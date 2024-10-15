# Example on how to create a survival class instance:
#
import numpy as np
import scipy
import scipy.stats
import xarray as xr

from imagematerials.survival import SurvivalMatrix


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


def test_scipy_survival():
    pass

# def test_survival_random():
#     num_timesteps=100
#     compute_survival = lambda num_timesteps, cohort_idx: weibull_survival(num_timesteps, cohort_idx, shape=1, scale=2)
#     surv = Survival(
#         compute_next=compute_survival,
#         num_timesteps=num_timesteps
#     )
#     for _ in range(100):
#         t_idx = np.random.randint(num_timesteps)
#         cohort_idx = np.random.randint(num_timesteps)
#         if t_idx > cohort_idx:
#             assert surv[t_idx, cohort_idx] > 0 and surv[t_idx, cohort_idx] <= 1
#         elif t_idx == cohort_idx:
#             assert surv[t_idx, cohort_idx] == 1
#         else:
#             assert surv[t_idx, cohort_idx] == 0


# def test_survival_slice():
#     num_timesteps=100
#     compute_survival = lambda num_timesteps, cohort_idx: weibull_survival(num_timesteps, cohort_idx, shape=1, scale=2)
#     surv = Survival(
#         compute_next=compute_survival,
#         num_timesteps=num_timesteps
#     )
#     sub_surv = surv[4:10, 10:20]
#     assert np.all(sub_surv == 0)
#     assert sub_surv.shape == (6, 10)

#     sub_surv = surv[20:30, 5:10]
#     assert np.all(sub_surv > 0) and np.all(sub_surv <= 1)

#     sub_surv = surv[:, :]
#     assert np.all(np.diag(sub_surv) == 1)
