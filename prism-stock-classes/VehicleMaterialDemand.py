"""_summary_
"""

from globals.constants import (
    Region
)
import preprocessing
import prism

@prism.interface
class VehicleMaterialDemand(prism.Model):
    '''
    TODO: Describe
    Question: how are input & output data transferred outside of this class
    Where do we need to create the Region(26), Mode & Material dimensions?
    '''
    vehicles_stock:     prism.TimeVariable[Region, Mode, 'count']        = prism.export(initial_value = prism.Array[Region, Mode, 'count'](0.0)) 
    vehicles_materials: prism.TimeVariable[Region, Mode, Material, 'kg'] = prism.export(initial_value = prism.Array[Region, Mode, Material, 'kg'](0.0)) 

    def compute_initial_values():
        vehicles_preprocessing       = preprocessing()
        
        # dictionary should provide the type & the survival parameters for all modes 
        # & regions in the first year & the future (2017-2100), 
        # so deselection of historic parameters is needed here:            
        vehicles_survival_dictionary = vehicles_preprocessing[lifetimes_vehicles]  
        vehicles_survival            = Survival(vehicles_survival_dictionary)
        
        # separation of historic and modelled stock happens here
        vehicles_historic  = vehicles_preprocessing[total_nr_vehicles_total] # some selection needed still
        vehicles_future    = vehicles_preprocessing[total_nr_vehicles_total] # some selection needed still
        vehicles_stock     = Stock(vehicles_historic, vehicles_survival)
        
    def compute_values():
        # Vehicles: Stock # no clue how this works
        dynamic_stock_modelling()
        material_calculation()
    
    def run_dynamic_stock_model():
        vehicles_survival.compute_next()
        vehicles_stock.compute_next()
        
    def material_calculation():
        pass
    