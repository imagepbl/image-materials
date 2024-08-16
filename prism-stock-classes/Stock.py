import prism
from prism import Q_
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


# To be removed
@prism.interface
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


class Stock:
    def __init__():
        pass
    

    def compute(
            self,
            time: prism.Time,
            input_stock: prism.Array['count'],
            survival: Survival):
        t = time.t
        dt = time.dt

        # determine inflow from mass balance
        stockDiff = input_stock - self.stock_cohort[t].sum('cohort') # Sum remainder of previous cohorts left in this timestep
        self.inflow[t] = prism.switch(
            stockDiff > 0,
            stockDiff / self.survival.loc[{Cohort.dim_label: t, Year.dim_label: t}], 
            # if less than all survives increase inflow to fulfill demand in first year
            default = prism.Array['count'](0.0) # TODO should this be a single number or should this have a dimenaion?
        )
        self.stock_cohort[t] = self.inflow[t] * self.survival.loc[{Cohort.dim_label: t}] # depreciation of inflow at this timestep
        # TODO consider calculating survival for past stocks not future
        self.outflow_cohort[t] = -1 * np.diff(self.stock_cohort[t], dt, axis=0, prepend=0) # TODO potential improvement use np.tril to not include what is above the diagonal
        #TODO check if this results in a negative of positive number

class Survival:


    def compute(lifetime_parameters: dict, timesteps: int):
        survival = np.zeros((timesteps, timesteps))

        unfolded_years = np.arange(0, timesteps)[:, np.newaxis]
        if lifetime_parameters["Type"] == 'FoldedNormal_vector' and lifetime_parameters['Mean'] != 0:  
            # For products with lifetime of 0, sf == 0
            # Folded normal distribution, cf. https://en.wikipedia.org/wiki/Folded_normal_distribution
            # Vectorized calculation for Folded Normal
            survival[:timesteps, :] = scipy.stats.foldnorm.sf(
                unfolded_years,
                lifetime_parameters['Mean'] / lifetime_parameters['StdDev'],
                0,
                scale=lifetime_parameters['StdDev'][:timesteps]
            )

        return survival
