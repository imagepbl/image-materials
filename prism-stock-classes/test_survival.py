# Example on how to create a survival class instance:
#
from survival import Survival, weibull_survival, folded_normal
import numpy as np

def test_survival_random():
    num_timesteps=100
    compute_survival = lambda num_timesteps, cohort_idx: weibull_survival(num_timesteps, cohort_idx, shape=1, scale=2)
    surv = Survival(
        compute_next=compute_survival,
        num_timesteps=num_timesteps
    )
    for _ in range(100):
        t_idx = np.random.randint(num_timesteps)
        cohort_idx = np.random.randint(num_timesteps)
        if t_idx > cohort_idx:
            assert surv[t_idx, cohort_idx] > 0 and surv[t_idx, cohort_idx] <= 1
        elif t_idx == cohort_idx:
            assert surv[t_idx, cohort_idx] == 1
        else:
            assert surv[t_idx, cohort_idx] == 0


def test_survival_slice():
    num_timesteps=100
    compute_survival = lambda num_timesteps, cohort_idx: weibull_survival(num_timesteps, cohort_idx, shape=1, scale=2)
    surv = Survival(
        compute_next=compute_survival,
        num_timesteps=num_timesteps
    )
    sub_surv = surv[4:10, 10:20]
    assert np.all(sub_surv == 0)
    assert sub_surv.shape == (6, 10)

    sub_surv = surv[20:30, 5:10]
    assert np.all(sub_surv > 0) and np.all(sub_surv <= 1)

    sub_surv = surv[:, :]
    assert np.all(np.diag(sub_surv) == 1)
