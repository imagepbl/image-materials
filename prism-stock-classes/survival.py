import numpy as np
import scipy
from itertools import islice

# TODO: Switch to XArray

class Survival:
    def __init__(self, compute_next, num_timesteps):
        # TODO: Add docstrings
        self.compute_next = compute_next
        self.survival_matrix = np.zeros((num_timesteps, num_timesteps))
        self.num_timesteps = num_timesteps
        self._cached_timesteps = set()

    def __getitem__(self, idx):
        # TODO: make the computation dependent on t_idx
        # We know that if t_idx < cohort_idx, the result is 0,
        # So if we compute s[t, :], we know that we only need to compute
        # the columns until cohort_idx == t
        t_idx, cohort_idx = idx

        # TODO: Check if this actually creates the cohort indices properly.
        if isinstance(cohort_idx, slice):
            comp_list = islice(range(self.num_timesteps), *cohort_idx.indices(self.num_timesteps))
        else:
            comp_list = [cohort_idx]
            
        # Is the column already computed?
        for cur_cohort_idx in comp_list:
            if cur_cohort_idx not in self._cached_timesteps:
                # Compute the survival matrix for the current cohort_idx
                # The survival coeficients are only relevant for future times t
                # So the values are created in column cohort_idx, for t_idx=cohort_idx, cohort_idx+1, ...
                self.survival_matrix[cur_cohort_idx:, cur_cohort_idx] = self.compute_next(
                    self.num_timesteps, cur_cohort_idx)
                self._cached_timesteps.add(cur_cohort_idx)
        return self.survival_matrix[t_idx, cohort_idx]


def _is_iterable(val):
    try:
        iter(val)
        return True
    except TypeError:
        return False

def weibull_survival(num_timesteps, cohort_idx, shape, scale):
    shape = shape[cohort_idx] if _is_iterable(shape) else shape
    scale = scale[cohort_idx] if _is_iterable(scale) else scale
    return scipy.stats.weibull_min.sf(
                np.arange(num_timesteps-cohort_idx),
                c=shape,
                loc=0,
                scale=scale,
    )

def folded_normal(num_timesteps, cohort_idx, mean, std):
    std = std[cohort_idx] if _is_iterable(std) else std
    mean = mean[cohort_idx] if _is_iterable(mean) else mean
    return scipy.stats.foldnorm.sf(
        np.arange(num_timesteps-cohort_idx),
        c=mean/std,
        loc=0,
        scale=std,
    )
