import prism
from prism import Q_
import numpy as np
import scipy

@prism.interface
class Stock:
    """
    Stock class
    
    Dimensions:
    Stock.stock:   mode, region, cohort
    Stock.inflow:  mode, region 
    Stock.outflow: mode, region, cohort
    
    Question: How to specify the different stock modelling options (e.g simple, surplus, etc.?
              Do we still want to define a basic Stock class without cohort-specific tracking? e.g Through inheritance
    """
    
    stock_cohort:   prism.TimeVariable[Region, Mode, Cohort, 'count'] = prism.export(initial_value = prism.Array[Region, Mode, Cohort, 'count'](0.0)) 
    inflow:         prism.TimeVariable[Region, Mode, 'count']         = prism.export(initial_value = prism.Array[Region, Mode, 'count'](0.0))   
    outflow_cohort: prism.TimeVariable[Region, Mode, Cohort, 'count'] = prism.export(initial_value = prism.Array[Region, Mode, Cohort, 'count'](0.0))

    def compute_initial_values():
        # Generate the historic stock by cohort (including historic inflow & outflow by cohort) from a historic total stock
        # OR (if known beforehand) load cohort specific stock & outflow data and associated inflow data.
        # OR (if known beforehand) load cohort specific stock data and derive inflow & outflow accordingly
        
    def compute_values(
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
