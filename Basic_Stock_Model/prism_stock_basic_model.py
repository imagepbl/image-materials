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
#from stockfunctions import StockFunctions 

import pathlib
CURRENT_DIR = pathlib.Path(__file__).parent

# dimensions
Region = prism.NewDim('region', ['USA', 'Mexico', 'Canada']) #TODO ask martijn how to loop through regions
Cohort = prism.NewDim('cohort', [2000, 2001, 2002])

#%%
# TODO move the compute_sf to superclass StockModel
def compute_sf(lt_type, lt, nr_timesteps): # survival functions
    """
    Survival table self.sf(m,n) denotes the share of an inflow in year n (age-cohort) still present at the end of year m (after m-n years).
    The computation is self.sf(m,n) = ProbDist.sf(m-n), where ProbDist is the appropriate scipy function for the lifetime model chosen.
    For lifetimes 0 the sf is also 0, meaning that the age-cohort leaves during the same year of the inflow.
    The method compute outflow_sf returns an array year-by-cohort of the surviving fraction of a flow added to stock in year m (aka cohort m) in in year n. This value equals sf(n,m).
    This is the only method for the inflow-driven model where the lifetime distribution directly enters the computation. All other stock variables are determined by mass balance.
    The shape of the output sf array is NoofYears * NoofYears, and the meaning is years by age-cohorts.
    The method does nothing if the sf alreay exists. For example, sf could be assigned to the dynamic stock model from an exogenous computation to save time.
    """
    sf = np.zeros((nr_timesteps, nr_timesteps))
    # Perform specific computations and checks for each lifetime distribution:

    if lt_type == 'Fixed': # fixed lifetime, age-cohort leaves the stock in the model year when the age specified as 'Mean' is reached.
        for m in range(0, nr_timesteps):  # cohort index
            sf[m::,m] = np.multiply(1, (np.arange(0,nr_timesteps-m) < lt['Mean'])) # converts bool to 0/1
        # Example: if Lt is 3.5 years fixed, product will still be there after 0, 1, 2, and 3 years, gone after 4 years.

    if lt_type == 'Normal': # normally distributed lifetime with mean and standard deviation. Watch out for nonzero values 
        # for negative ages, no correction or truncation done here. Cf. note below.
        if lt['Mean'] != 0:
            for m in range(0, nr_timesteps):  # cohort index
                # For products with lifetime of 0, sf == 0
                sf[m::,m] = scipy.stats.norm.sf(np.arange(0,nr_timesteps-m), loc=lt['Mean'], scale=lt['StdDev'])
                # NOTE: As normal distributions have nonzero pdf for negative ages, which are physically impossible, 
                # these outflow contributions can either be ignored (violates the mass balance) or
                # allocated to the zeroth year of residence, the latter being implemented in the method compute compute_o_c_from_s_c.
                # As alternative, use lognormal or folded normal distribution options.
                
    if lt_type == 'FoldedNormal': # Folded normal distribution, cf. https://en.wikipedia.org/wiki/Folded_normal_distribution
        if lt['Mean'] != 0:  # For products with lifetime of 0, sf == 0
            for m in range(0, nr_timesteps):  # cohort index
                sf[m::,m] = scipy.stats.foldnorm.sf(np.arange(0,len(nr_timesteps)-m), lt['Mean']/lt['StdDev'], 0, scale=lt['StdDev'][m])
                # NOTE: call this option with the parameters of the normal distribution mu and sigma of curve BEFORE folding,
                # curve after folding will have different mu and sigma.
                
    if lt_type == 'LogNormal': # lognormal distribution
        # Here, the mean and stddev of the lognormal curve, 
        # not those o           f the underlying normal distribution, need to be specified! conversion of parameters done here:
        if lt['Mean'] != 0:  # For products with lifetime of 0, sf == 0
            for m in range(0, nr_timesteps):  # cohort index
                # calculate parameter mu    of underlying normal distribution:
                LT_LN = np.log(lt['Mean'][m] / np.sqrt(1 + lt['Mean'][m] * lt['Mean'][m] / (lt['StdDev'][m] * lt['StdDev'][m]))) 
                # calculate parameter sigma of underlying normal distribution:
                SG_LN = np.sqrt(np.log(1 + lt['Mean'][m] * lt['Mean'][m] / (lt['StdDev'][m] * lt['StdDev'][m])))
                # compute survial function
                sf[m::,m] = scipy.stats.lognorm.sf(np.arange(0,nr_timesteps-m), s=SG_LN, loc = 0, scale=np.exp(LT_LN)) 
                # values chosen according to description on
                # https://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.stats.lognorm.html
                # Same result as EXCEL function "=LOGNORM.VERT(x;LT_LN;SG_LN;TRUE)"
                
    if lt_type == 'Weibull': # Weibull distribution with standard definition of scale and shape parameters
        if lt['Shape'] != 0:  # For products with lifetime of 0, sf == 0
            for m in range(0, nr_timesteps):  # cohort index
                sf[m::,m] = scipy.stats.weibull_min.sf(np.arange(0,nr_timesteps-m), c=lt['Shape'][m], loc = 0, scale=lt['Scale'][m])

    return sf

#%%
# equivalent function in DSM: compute_stock_driven_model_surplus
Cohort2 = prism.NewDim('cohort2', Cohort.coords)

# model for stock calculation
@prism.interface
class StockDrivenModel(prism.Model):
    
    inflow: prism.TimeVariable[Region, 'count'] = prism.export(initial_value = prism.Array[Region, 'count'](0.0))
    #stock: prism.TimeVariable[Region, 'count'] = prism.export(initial_value = 0.0)
    stock_c: prism.TimeVariable[Region, Cohort, 'count'] = prism.export(initial_value = prism.Array[Region, Cohort, 'count'](0.0)) # is it possible to have values with int type mixed with float?
    #outflow: prism.TimeVariable[Region, 'count'] = prism.export(initial_value = 0.0) # currently outflow is reported as a negative number
    outflow_c: prism.TimeVariable[Region, Cohort, 'count'] = prism.export(initial_value = prism.Array[Region, Cohort, 'count'](0.0))

    sf: prism.Array[Region, Cohort, Cohort2, 'count'] = prism.Array[Region, Cohort, Cohort2, 'count'](0.0) 
    #sf: prism.TimeVariable[Region, 'kg'] = prism.private(initial_value = 0.0) #stated off with a simplified survivial function
    # TODO sf should be unitless
    # TODO check if this sets the start of sf to the timestep and not at the top

    # start_stock: prism.Array[Region, 'count'] = prism.private()
    lt_type: str = None
    lt: dict = None   

    def compute_initial_values(self, 
                               timeline: prism.Time):
            print("Model initialization")

            #print(self.sf)
            nr_timesteps = timeline.number_of_timeslots
            
            print("Survival function initialization")

            # for region in Region:
            #     self.sf[region] = compute_sf(lt_type, lt_parameters)
            sf = compute_sf(lt_type, lt_parameters, nr_timesteps)
            sf_tmp = prism.Array[Cohort, Cohort2, 'count'](values = sf)
            self.sf = sf_tmp.expand_dims(dim={Region.dim_label: Region.coords})
            # print(self.sf)
            # print(len(self.sf))

            print("Setup complete")
            """
            start_stock = prism.Array[Region, 'count'](0.0)

            # First time step
            if self.sf[0, 0] != 0: # Else, inflow is 0. #TODO ask Martijn if initial step should happen here or in compute initial values
                self.inflow[0] = self.stock_c[0] / self.sf[0, 0]
            self.stock_c[:, 0] = self.inflow[0] * self.sf[0, 0]
            self.outflow_c[0, 0] = self.inflow[0] - self.s_c[0, 0]
            """

    
    
    def compute_values(
        self,
        time: prism.Time,
        input_stock: prism.Array[Region, 'count']
        
    ):
        t = time.t
        t_start = time.start
        dt = time.dt
        
        #self.stock[t] = input_stock

        # determine inflow from mass balance
        stockDiff = input_stock - self.stock_c[t].sum('cohort') # Sum remainder of previous cohorts left in this timestep
        # it = time.time_to_index(t)
        # print(f'it is {it}')
        # print(self.sf.loc[{Cohort.dim_label: t}])
        diagonal = self.sf.loc[{Cohort.dim_label: t, Cohort2.dim_label: t}]
        # print(diagonal)
        zeros = prism.Array[Region, 'count'](0.0)
        self.inflow[t] = prism.switch(
            self.sf.loc[{Cohort.dim_label: t, Cohort2.dim_label: t}] == 0,
            zeros,
            stockDiff > 0,
            stockDiff / diagonal, 
            default = zeros #TODO CHECK what happens if stockdiff is negative
        )
            #if : # if less than all survives
            #    self.inflow[t] = stockDiff / self.sf[it, it]
        print(f"Timestep {t}")
        print(stockDiff)
        print(self.inflow[t])
        print(self.sf.loc[{Cohort.dim_label: t}])
        self.stock_c[t] = self.inflow[t] * self.sf.loc[{Cohort.dim_label: t}] # depreciation of inflow at this timestep
        print(self.stock_c[t])
        self.outflow_c[t] = -1 * np.diff(self.stock_c[t], 1, axis=0, prepend=0) # TODO potential improvement use np.tril to not include what is above the diagonal
        #TODO check if this results in a negative of positive number


# Define a timeline.
timeline = prism.Timeline(
    start=Q_(2000, 'year'),
    end=Q_(2002, 'year'),
    stepsize=Q_(1, 'year'))

# Define input.
my_stock = prism.TimeVariable[Region, "count"](
    time=timeline, file="../Basic_Stock_Model/demand.dat")

def my_input(t):
    return {
        "input_stock": my_stock[t]}

lt_type = "Normal"

lt_parameters = {'Mean': 0.5, 'StdDev': 1}

# Create a StockModel object.
stockmodel = StockDrivenModel(timeline,lt_type = lt_type,lt = lt_parameters)

# Simulate the gas reserves.
stockmodel.simulate(timeline, inputs=my_input)



# %%
# To see what happens
for t in timeline.all_timeslots():
    print(t)
    print(stockmodel.stock_c[t])

