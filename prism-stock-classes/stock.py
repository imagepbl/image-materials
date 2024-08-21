import prism
from prism import Q_
import numpy as np
import scipy
from dynamic_stock_calculation import dynamic_stock_model 
from globals.constants import (
    _IMAGE_REGIONS;
    start_simulation;
    all_modes
)
from survival import Survival


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

    def compute_initial_values(self, timeline, survival: Survival, historical_data):
        # Generate the historic stock by cohort (including historic inflow & outflow by cohort) from a historic total stock
        # OR (if known beforehand) load cohort specific stock & outflow data and associated inflow data.
        # OR (if known beforehand) load cohort specific stock data and derive inflow & outflow accordingly
        
        """Compute initial values for the historical data using dynamic_stock_model."""
        for year in timeline:
            if year < start_simulation:  # Historical period cutoff
                for region in _IMAGE_REGIONS:
                    for mode in all_modes:
                        if year == timeline.start:  # First year in the timeline
                            stock_prev = historical_data[region][mode][year]
                            inflow = stock_prev  # Assume all initial stock comes from inflow
                            outflow = 0  # No outflow in the first year
                        else:
                            stock_prev = self.vehicles_stock.sel(Region=region, Mode=mode, Year=year-1)
                            survival_rate = survival.survival_matrix.sel(Region=region, Mode=mode, Cohort=year-1, Year=year)

                            # Stock difference based on historical data
                            stock_diff = historical_data[region][mode][year] - stock_prev * survival_rate

                            # Calculate new stock, inflow, and outflow using dynamic_stock_model
                            new_stock, inflow, outflow = dynamic_stock_model(stock_prev, survival_rate, stock_diff)

                        # Update stock, inflow, and outflow arrays
                        self.vehicles_stock.loc[{"Year": year, "Region": region, "Mode": mode}] = new_stock
                        self.inflow.loc[{"Year": year, "Region": region, "Mode": mode}] = inflow
                        self.outflow.loc[{"Year": year, "Region": region, "Mode": mode}] = outflow
        
    def compute_values(self, time, survival: Survival, required_stock):
        """Compute values for the current time step."""
        year = time.current_time
        for region in _IMAGE_REGIONS:
            for mode in all_modes:
                stock_prev = self.vehicles_stock.sel(Region=region, Mode=mode, Year=year-1)
                survival_rate = survival.survival_matrix.sel(Region=region, Mode=mode, Cohort=year-1, Year=year)
                stock_diff = required_stock[region][mode][year] - stock_prev * survival_rate

                new_stock, inflow, outflow = dynamic_stock_model(stock_prev, survival_rate, stock_diff)

                self.vehicles_stock.loc[{"Year": year, "Region": region, "Mode": mode}] = new_stock
                self.inflow.loc[{"Year": year, "Region": region, "Mode": mode}] = inflow
                self.outflow.loc[{"Year": year, "Region": region, "Mode": mode}] = outflow
