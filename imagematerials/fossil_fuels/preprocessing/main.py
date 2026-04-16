#%% Import general constants and modules 
import pandas as pd
import numpy as np
import os
from pathlib import Path

import pint
import xarray as xr
import prism

from imagematerials.fossil_fuels.preprocessing.ffconstants import FF_TECHNOLOGIES, IMAGE_REGIONS, STANDARD_SCEN_EXTERNAL_DATA, YEAR_FIRST_GRID, SD_LIFETIME

from imagematerials.read_mym import read_mym_df
from imagematerials.util import dataset_to_array, pandas_to_xarray, convert_lifetime
from imagematerials.concepts import KnowledgeGraph, Node, create_fossil_fuel_graph, create_region_graph
from imagematerials.electricity.utils import (
   MNLogit, 
   stock_tail, 
   create_prep_data, 
   interpolate_xr, 
   add_historic_stock, 
   calculate_grid_growth, 
   calculate_fraction_underground, 
   apply_ce_measures_to_elc
)

# from prism.prism.examples.fuel import scenario

path_current = Path().resolve()
path_base = path_current.parents [1]
print("current:", path_current)
print("base:", path_base)

#still not sure what to do with these 
scen_folder = "SSP3_no_policy"
climate_policy_scenario_dir = Path(path_base, "data", "raw", "image", scen_folder)
STANDARD_SCEN_EXTERNAL_DATA = "SSP3_no_policy" 

year_start = 1880
year_end = 2100
year_out = 2100

#Import xarrays from lifetimes, materials and capacity folders 
from imagematerials.fossil_fuels.preprocessing.lifetimes import compute_extraction_lifetimes, compute_processing_lifetimes, compute_transport_lifetimes, compute_pipelines_lifetimes
from imagematerials.fossil_fuels.preprocessing.materials import compute_extraction_materials, compute_processing_materials, compute_transport_materials, compute_pipelines_materials
from imagematerials.fossil_fuels.preprocessing.capacity import compute_extraction_capacity, compute_processing_capacity, compute_transport_capacity, compute_pipelines_capacity

args = (path_base, climate_policy_config, circular_economy_config, scenario, year_start, year_end, year_out)

extraction_lifetime_xr = compute_extraction_lifetimes(*args)
processing_lifetime_xr = compute_processing_lifetimes(*args)
transport_lifetime_xr = compute_transport_lifetimes(*args)
pipelines_lifetime_xr = compute_pipelines_lifetimes(*args)  

extraction_materials_xr = compute_extraction_materials(*args)
processing_materials_xr = compute_processing_materials(*args)
transport_materials_xr = compute_transport_materials(*args)
pipelines_materials_xr = compute_pipelines_materials(*args) 

extractioncap_xr = compute_extraction_capacity(*args)
processingcap_xr = compute_processing_capacity(*args)
transportcap_xr = compute_transport_capacity(*args)
pipelinecap_xr = compute_pipelines_capacity(*args)

#%%Extraction stage
def get_preprocessing_data_extraction(path_base: str, climate_policy_config: dict, circular_economy_config: dict, scenario: str, year_start: int, year_end: int, year_out: int):

    path_external_data_scenario = Path(path_base, "fossil_fuels", "Scenario_data", scenario)
    # test if path_external_data_scenario exists and if not set to standard scenario
    if not path_external_data_scenario.exists():
        path_external_data_scenario = Path(path_base, "fossil_fuels", STANDARD_SCEN_EXTERNAL_DATA)
    
    # The lifetimes are converted to the proper format for the model (dictionary with keys:distribution name, values:datarrays containing distribution parameters)
    extraction_lifetime_xr = convert_lifetime(extraction_lifetime_xr)
        
    # bring preprocessing data into a generic format for the model
    prep_data_extraction = {}
    prep_data_extraction["lifetimes"] = extraction_lifetime_xr
    prep_data_extraction["stocks"] = extractioncap_xr
    prep_data_extraction["material_intensities"] = extraction_materials_xr
    prep_data_extraction["knowledge_graph"] = create_fossil_fuel_graph() 
    # add units
    prep_data_extraction["stocks"] = prism.Q_(prep_data_extraction["stocks"], "kg")
    prep_data_extraction["material_intensities"] = prism.Q_(prep_data_extraction["material_intensities"], "kg/kg/year")
    prep_data_extraction["set_unit_flexible"] = prism.U_(prep_data_extraction["stocks"]) # prism.U_ gives the unit back
        # set_unit_flexible is needed by the model to deal with the fact the in the beginning of the model it doesn't know th data yet and needs to work with a placeholder/flexible unit (see model.py) 

    return prep_data_extraction

#%%Processing stage
def get_preprocessing_data_processing(path_base: str, climate_policy_config: dict, circular_economy_config: dict, scenario: str, year_start: int, year_end: int, year_out: int):

    path_external_data_scenario = Path(path_base, "fossil_fuels", "Scenario_data", scenario)
    # test if path_external_data_scenario exists and if not set to standard scenario
    if not path_external_data_scenario.exists():
        path_external_data_scenario = Path(path_base, "fossil_fuels", STANDARD_SCEN_EXTERNAL_DATA)

    # assert path_external_data_scenario.is_dir()

    #return path_external_data_scenario

     # The lifetimes are converted to the proper format for the model (dictionary with keys:distribution name, values:datarrays containing distribution parameters)
    processing_lifetime_xr = convert_lifetime(processing_lifetime_xr)
        
    # bring preprocessing data into a generic format for the model
    prep_data_processing = {}
    prep_data_processing["lifetimes"] = processing_lifetime_xr
    prep_data_processing["stocks"] = processingcap_xr
    prep_data_processing["material_intensities"] = processing_materials_xr
    #prep_data_processing["knowledge_graph"] = create_fossil_fuel_graph() 
    # add units
    prep_data_processing["stocks"] = prism.Q_(prep_data_processing["stocks"], "kg")
    prep_data_processing["material_intensities"] = prism.Q_(prep_data_processing["material_intensities"], "kg/kg/year")
    prep_data_processing["set_unit_flexible"] = prism.U_(prep_data_processing["stocks"]) # prism.U_ gives the unit back
    #     # set_unit_flexible is needed by the model to deal with the fact the in the beginning of the model it doesn't know th data yet and needs to work with a placeholder/flexible unit (see model.py) 

    return prep_data_processing

#%% Transport stage
def get_preprocessing_data_transport(path_base: str, climate_policy_config: dict, circular_economy_config: dict, scenario: str, year_start: int, year_end: int, year_out: int):

    path_external_data_scenario = Path(path_base, "fossil_fuels", "Scenario_data", scenario)
    # test if path_external_data_scenario exists and if not set to standard scenario
    if not path_external_data_scenario.exists():
        path_external_data_scenario = Path(path_base, "fossil_fuels", STANDARD_SCEN_EXTERNAL_DATA)

    # assert path_external_data_scenario.is_dir()

    #  # bring preprocessing data into a generic format for the model
    prep_data_transport = {}
    prep_data_transport["stocks"] = transportcap_xr
    prep_data_transport["material_intensities"] = transport_materials_xr
    # add units
    prep_data_transport["stocks"] = prism.Q_(prep_data_transport["stocks"], "kg")
    prep_data_transport["set_unit_flexible"] = prism.U_(prep_data_transport["stocks"]) # prism.U_ gives the unit back
        # set_unit_flexible is needed by the model to deal with the fact the in the beginning of the model it doesn't know th data yet and needs to work with a placeholder/flexible unit (see model.py) 

    return prep_data_transport

#%% Pipelines stage (oil, gas) ---------------------------------------------------------------------------------------------------------------------------------
def get_preprocessing_data_pipelines(path_base: str, climate_policy_config: dict, circular_economy_config: dict, scenario: str, year_start: int, year_end: int, year_out: int):

    path_external_data_scenario = Path(path_base, "fossil_fuels", "Scenario_data", scenario)
    # test if path_external_data_scenario exists and if not set to standard scenario
    if not path_external_data_scenario.exists():
        path_external_data_scenario = Path(path_base, "fossil_fuels", STANDARD_SCEN_EXTERNAL_DATA)

    # assert path_external_data_scenario.is_dir()
    #  # bring preprocessing data into a generic format for the model
    prep_data_pipelines = {}
    prep_data_pipelines["stocks"] = pipelinecap_xr
    prep_data_pipelines["knowledge_graph"] = create_fossil_fuel_graph() 
    # add units
    prep_data_pipelines["stocks"] = prism.Q_(prep_data_pipelines["stocks"], "km")
    prep_data_pipelines["set_unit_flexible"] = prism.U_(prep_data_pipelines["stocks"]) # prism.U_ gives the unit back
    #     # set_unit_flexible is needed by the model to deal with the fact the in the beginning of the model it doesn't know th data yet and needs to work with a placeholder/flexible unit (see model.py) 

    return prep_data_pipelines
