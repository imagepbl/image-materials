"""
Code is based on: 
Repository for this class, documentation, and tutorials: https://github.com/IndEcol/ODYM

Original author: Stefan Pauliuk, NTNU Trondheim, Norway, later Uni Freiburg, Germany
with contributions from
Sebastiaan Deetman, CML, Leiden, NL
Tomer Fishman, IDC Herzliya, IL
Chris Mutel, PSI, Villingen, CH

Translated to prism python package by Luja von Köckritz and Sebastiaan Deetman
"""

#%%
import prism
from prism import Q_
import numpy as np
import scipy
import pathlib
CURRENT_DIR = pathlib.Path(__file__).parent

# dimensions
Cohort = prism.NewDim('cohort', [2000, 2001, 2002])
Year = prism.NewDim('year', Cohort.coords)

#%%
# TODO move the compute_sf to superclass StockModel
def compute_survival(lifetime_parameters: dict, timesteps: int): 
    """
    Compute survival table for an inflow-driven model.

    Parameters:
    - lifetime_parameters (dict): Dict with parameters for the lifetime distribution.
      - Type (str): 'FoldedNormal' or 'Weibull'.
      - Mean (float): Mean parameter.
      - StdDev (float or array): Standard deviation.
      - Shape (float or array): Shape parameter.
      - Scale (float or array): Scale parameter.
    - timesteps (int): Number of time steps.

    Returns:
    - np.ndarray: Survival table for share of inflow still present at end of each year.

    The survival table self.sf(m, n) denotes share of inflow in year n (age-cohort) still present at end
    of year m (after m-n years). Computation is self.sf(m, n) = ProbDist.sf(m-n), where ProbDist is
    appropriate scipy function for chosen lifetime model. For lifetimes 0, sf is 0, meaning age-cohort
    leaves during same year of inflow. Method does nothing if sf already exists.
    """    

    survival = np.zeros((timesteps, timesteps))
    # Perform specific computations and checks for each lifetime distribution:

    #TODO use type hinting the type of input to the function
    #class called SurvivalFunction - subclasses for the different sf types, values are attributes of the subclass
    #make attribute a time by region array

    if lifetime_parameters["Type"] == 'FoldedNormal':
        # Folded normal distribution, cf. https://en.wikipedia.org/wiki/Folded_normal_distribution
        if lifetime_parameters['Mean'] != 0:  # For products with lifetime of 0, sf == 0
            for m in range(timesteps):  # cohort index
                survival[m:, m] = scipy.stats.foldnorm.sf(
                    np.arange(0, timesteps - m),
                    lifetime_parameters['Mean'] / lifetime_parameters['StdDev'],
                    0, scale=lifetime_parameters['StdDev'][m]
                )
                # NOTE: call this option with parameters of normal distribution mu and sigma of curve
                # BEFORE folding, curve after folding will have different mu and sigma.
             
    if lifetime_parameters["Type"] == 'Weibull':
        # Weibull distribution with standard definition of scale and shape parameters
        if lifetime_parameters['Shape'] != 0:  # For products with lifetime of 0, sf == 0
            for m in range(timesteps):  # cohort index
                survival[m:, m] = scipy.stats.weibull_min.sf(
                    np.arange(0, timesteps - m),
                    c=lifetime_parameters['Shape'][m], loc=0,
                    scale=lifetime_parameters['Scale'][m]
                ) 

    # Chat GPT vectorized version
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

    if lifetime_parameters["Type"] == 'Weibull_vector' and lifetime_parameters['Shape'] != 0:  
        # For products with lifetime of 0, sf == 0
        # Weibull distribution with standard definition of scale and shape parameters
        # Vectorized calculation for Weibull
        survival[:timesteps, :] = scipy.stats.weibull_min.sf(
            unfolded_years,
            c=lifetime_parameters['Shape'][:timesteps],
            loc=0,
            scale=lifetime_parameters['Scale'][:timesteps]
        )

    return survival

#%%
# equivalent function in DSM: compute_stock_driven_model_surplus

# model for stock calculation
@prism.interface
class StockDrivenModel(prism.Model):
    
    inflow: prism.TimeVariable['count'] = prism.export(initial_value = prism.Array['count'](0.0))
    stock_cohort: prism.TimeVariable[Cohort, 'count'] = prism.export(initial_value = prism.Array[Cohort, 'count'](0.0)) 
    outflow_cohort: prism.TimeVariable[Cohort, 'count'] = prism.export(initial_value = prism.Array[Cohort, 'count'](0.0))

    survival: prism.Array[Cohort, Year, 'count'] = prism.Array[Cohort, Year, 'count'](0.0) 
    # TODO sf should be unitless
    # TODO check if this sets the start of sf to the timestep and not at the top

    lifetime_parameters: dict = None   

    def compute_initial_values(self, 
                               timeline: prism.Time):
            # TODO consider adding logging (not high priority)
            survival = compute_survival(lifetime_parameters, timeline.number_of_timeslots) 
            self.survival = prism.Array[Cohort, Year, 'count'](values = survival)   
    
    def compute_values(
        self,
        time: prism.Time,
        input_stock: prism.Array['count']
        
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
        self.outflow_cohort[t] = -1 * np.diff(self.stock_cohort[t], dt, axis=0, prepend=0) # TODO potential improvement use np.tril to not include what is above the diagonal
        #TODO check if this results in a negative of positive number


# Define a timeline.
timeline = prism.Timeline(
    start=Q_(2000, 'year'),
    end=Q_(2002, 'year'),
    stepsize=Q_(1, 'year'))

# Define input.
my_stock = prism.TimeVariable["count"](
    time=timeline, file="../Basic_Stock_Model/demand.dat")

def my_input(t):
    return {
        "input_stock": my_stock[t]}

lifetime_parameters = {'Type': "Normal",'Mean': 0.5, 'StdDev': 1}

# Create a StockModel object.
stockmodel = StockDrivenModel(timeline,lifetime_parameters = lifetime_parameters)

# Simulate the gas reserves.
stockmodel.simulate(timeline, inputs=my_input)

# %%
# To see what happens
for t in timeline.all_timeslots():
    print(t)
    print(stockmodel.stock_c[t])

#TODO add numpy-type docstring (basics) - 80 characters for docstring, 100 for the code #TODO add lines in VS code