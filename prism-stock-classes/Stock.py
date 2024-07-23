import prism
from prism import Q_
import numpy as np
import scipy


class Survival:
    def __init__(self, survival_matrix):
        self.survival_matrix = survival_matrix

    def compute(t, cohort):
        return self.survival_matrix[t, cohort]


@prism.interface
class WeibullSurvival(Survival):
    shape: float
    scale: float

    def __init__(self, shape, scale, num_timesteps: int):
        # Calculate matrix here
        survival_matrix = np.empty((num_timesteps, num_timesteps))
        for t in range(num_timesteps):
            survival_matrix[t:, :] = scipy.stats.weibull_min.sf(
                np.arange(num_timesteps-t),
                c=shape[t]
                loc=0,
                scale=scale[t],
            )
        super().__init__(survival_matrix)


@prism.interface
class FoldedNormalSurvival(Survival):
    mean: float
    std: float

    def __init__(self, mean, std, num_timesteps):
        # Calculate matrix here
        for t in num_timesteps:
            survival_matrix[t:, :] = scipy.stats.foldnorm.sf(
                np.arange(num_timesteps-t),
                c=mean[t]/std[t]
                loc=0,
                scale=std[t],
            )
        super().__init__(survival_matrix)


class Stock:
    def __init__():

    

    def compute(
    self,
    time: prism.Time,
    input_stock: prism.Array['count'],
    survival: Survival
    
    ):
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






