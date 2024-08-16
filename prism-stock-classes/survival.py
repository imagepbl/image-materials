import numpy as np
import scipy

# surv[t_idx, cohort_idx]

# TODO: Move survival functions and classes to separate module.
# TODO: Switch to XArray

class DynamicSurvival:
    def __init__(self, compute_next, num_timesteps):
        # TODO: Add docstrings
        self.compute_next = compute_next
        self.survival_matrix = np.zeros((num_timesteps, num_timesteps))
        self.num_timesteps
        self._cached_timesteps = set()

    def __getitem__(self, idx):
        t_idx, cohort_idx = idx
        # Is the column already computed?
        if cohort_idx not in self._cached_timesteps:
            # TODO: Good explanation here..
            self.survival_matrix[cohort_idx:, cohort_idx] = self.compute_next(self.num_timestep, cohort_idx)
            self._cached_timesteps.add(cohort_idx)
        return self.survival_matrix[t_idx, cohort_idx]


# TODO adapt to new structure of DynamicSurvival

class StaticSurvival:
    def __init__(self, survival_matrix):
        self.survival_matrix = survival_matrix

    def compute(self, t, cohort):
        return self.survival_matrix[t, cohort]


surv = DynamicSurvival(
    compute_next=lambda num_timesteps, cohort_idx: weibull_survival(num_timesteps, cohort_idx, shape=1, scale=2),
    num_timesteps=1000
)

def weibull_survival(num_timesteps, cohort_idx, shape, scale):
    return scipy.stats.weibull_min.sf(
                np.arange(num_timesteps-cohort_idx),
                c=shape[cohort_idx],
                loc=0,
                scale=scale[cohort_idx],
    )

def folded_normal(num_timesteps, cohort_idx, mean, std):
    return scipy.stats.foldnorm.sf(
        np.arange(num_timesteps-cohort_idx),
        c=mean[cohort_idx]/std[cohort_idx],
        loc=0,
        scale=std[cohort_idx],
    )
# TODO: add folded normal

# Remove
@prism.interface
class WeibullSurvival(StaticSurvival):
    shape: float
    scale: float

    def __init__(self, shape, scale, num_timesteps: int):
        # Calculate matrix here
        survival_matrix = np.empty((num_timesteps, num_timesteps))
        for t in range(num_timesteps):
            survival_matrix[t:, t] = scipy.stats.weibull_min.sf(
                np.arange(num_timesteps-t),
                c=shape[t],
                loc=0,
                scale=scale[t],
            )
        super().__init__(survival_matrix)



class FoldedNormalSurvival(Survival):
    mean: float
    std: float

    def __init__(self, mean, std, num_timesteps):
        # Calculate matrix here
        survival_matrix = np.empty((num_timesteps, num_timesteps))
        for t in num_timesteps:
            survival_matrix[t:, :] = scipy.stats.foldnorm.sf(
                np.arange(num_timesteps-t),
                c=mean[t]/std[t],
                loc=0,
                scale=std[t],
            )
        super().__init__(survival_matrix)

