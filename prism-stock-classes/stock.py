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
        
    def compute_values():
        pass
