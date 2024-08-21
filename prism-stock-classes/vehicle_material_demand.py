"""_summary_
"""

from globals.constants import (
    Region
    start_simulation
)
import preprocessing
import prism
from survival import Survival 
from stock import Stock
from dynamic_stock_calculation import dynamic_stock_model 
from material_calculation import material_calculation

@prism.interface
class VehicleMaterialDemand(prism.Model):
    """
    Vehicle Material Demand model integrating stock and survival
    TODO: Describe
    Question: how are input & output data transferred outside of this class
    Where do we need to create the Region(26), Mode & Material dimensions?
    """
    
    required_vehicles_stock:     prism.TimeVariable[Region, Mode, 'count']        = prism.export(initial_value = prism.Array[Region, Mode, 'count'](0.0)) 
    vehicles_materials: prism.TimeVariable[Region, Mode, Material, 'kg'] = prism.export(initial_value = prism.Array[Region, Mode, Material, 'kg'](0.0)) 

    def compute_initial_values(self, timeline, start_simulation):
        vehicles_preprocessing       = preprocessing
        
        # dictionary should provide the type & the survival parameters for all modes 
        # & regions in the first year & the future (2017-2100), 
        # so deselection of historic parameters is needed here:            
        vehicles_survival_dictionary = vehicles_preprocessing[lifetimes_vehicles]  
        self.survival = Survival(vehicles_survival_dictionary)
        
        # separation of historic and modelled stock happens here
        self.historical_data  = vehicles_preprocessing[total_nr_vehicles_total("data from start of dataset to start_simulation")] # some selection needed still
        self.future_data      = vehicles_preprocessing[total_nr_vehicles_total("data after start_simulation")] # some selection needed still
        
        self.stock = Stock()
        self.stock.compute_initial_values(timeline, self.survival, historical_data)
        # self.stock(historical_data, vehicles_survival)

        
    def compute_values(self, time, required_stock):
        # Vehicles: Stock # no clue how this works
        self.survival.compute_next()
        self.stock.compute_values(time, self.survival, required_stock)
        material_calculation()
    
    #def run_dynamic_stock_model():
    #    vehicles_survival.compute_next()
    #    vehicles_stock.compute_next()
        
    
    