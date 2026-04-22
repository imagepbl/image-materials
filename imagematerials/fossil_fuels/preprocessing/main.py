#%% Import general constants and modules 
import pandas as pd
import numpy as np
import os
from pathlib import Path

import pint
import xarray as xr
import prism

from imagematerials.factory import ModelFactory, Sector
from imagematerials.fossil_fuels.preprocessing.ffconstants import (
    # config
    circular_economy_config,
    climate_policy_config,
    scenario,
    scen_folder,

    # paths
    path_base,
    DATA_DIR,
    IMAGE_DIR,
    OUTPUT_DIR,
    CLIMATE_POLICY_SCENARIO_DIR,
    climate_policy_scenario_dir,

    # model constants
    FF_TECHNOLOGIES,
    IMAGE_REGIONS,
    STANDARD_SCEN_EXTERNAL_DATA,
    YEAR_FIRST_GRID,
    SD_LIFETIME
)

# from imagematerials.fossil_fuels.preprocessing.ffconstants import FF_TECHNOLOGIES, IMAGE_REGIONS, STANDARD_SCEN_EXTERNAL_DATA, YEAR_FIRST_GRID, SD_LIFETIME

from imagematerials.model import GenericStocks, MaterialIntensities
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

# from imagematerials.fossil_fuels.preprocessing.ffconstants import (    
#     circular_economy_config,
#     climate_policy_config,
#     scenario,
#     scen_folder,
#     STANDARD_SCEN_EXTERNAL_DATA,
#     BASE_DIR,
#     DATA_DIR,
#     IMAGE_DIR,
#     OUTPUT_DIR,
#     CLIMATE_POLICY_SCENARIO_DIR,
#     path_base,
#     climate_policy_scenario_dir,
# )

# from prism.prism.examples.fuel import scenario
# circular_economy_config = None
# climate_policy_config = "SSP2_baseline" #SSP2_baseline is the only option right now given the existing files of primpersec and final_energy_rt
# scenario = "SSP2_baseline" #SSP2_baseline is the only option right now given the existing files of primpersec and final_energy_rt

# scen_folder = "SSP2_baseline" #SSP2_baseline is the only option right now given the existing files of primpersec and final_energy_rt
# STANDARD_SCEN_EXTERNAL_DATA = "SSP2_baseline" #SSP2_baseline is the only option right now given the existing files of primpersec and final_energy_rt

# BASE_DIR = Path(__file__).resolve()
# while BASE_DIR.name != "image-materials":
#     BASE_DIR = BASE_DIR.parent

# DATA_DIR = BASE_DIR / "data" / "raw" / "fossil_fuels"
# IMAGE_DIR = BASE_DIR / "data" / "raw" / "image"
# OUTPUT_DIR = DATA_DIR / "Scenario_data"
# CLIMATE_POLICY_SCENARIO_DIR = IMAGE_DIR / scen_folder  
# path_base = BASE_DIR / "imagematerials"
# climate_policy_scenario_dir = CLIMATE_POLICY_SCENARIO_DIR

year_start = 1880
year_end = 2100
year_out = 2100
YEAR_START = 2100

VARIANT = "M_CP"
SCEN = "SSP2"

knowledge_graph_region = create_region_graph()
fossil_fuel_knowledge_graph = create_fossil_fuel_graph()

#Import xarrays from lifetimes, materials and capacity folders 
from imagematerials.fossil_fuels.preprocessing.lifetimes import compute_extraction_lifetimes, compute_processing_lifetimes, compute_transport_lifetimes, compute_pipelines_lifetimes
from imagematerials.fossil_fuels.preprocessing.materials import compute_extraction_materials, compute_processing_materials, compute_transport_materials, compute_pipelines_materials
from imagematerials.fossil_fuels.preprocessing.capacity import compute_extraction_capacity, compute_processing_capacity, compute_transport_capacity, compute_pipelines_capacity

#%%Extraction stage
def get_preprocessing_data_extraction(path_base: str, climate_policy_config: dict, circular_economy_config: dict, scenario: str, year_start: int, year_end: int, year_out: int):

    path_external_data_scenario = Path(path_base, "fossil_fuels", "Scenario_data", scenario)
    # test if path_external_data_scenario exists and if not set to standard scenario
    if not path_external_data_scenario.exists():
        path_external_data_scenario = Path(path_base, "fossil_fuels", STANDARD_SCEN_EXTERNAL_DATA)
    
    # The lifetimes are converted to the proper format for the model (dictionary with keys:distribution name, values:datarrays containing distribution parameters)
    extraction_lifetime_xr = compute_extraction_lifetimes (path_base, climate_policy_config, circular_economy_config, scenario, year_start, year_end, year_out)
    # print(type(extraction_lifetime_xr))
    # print(extraction_lifetime_xr)
    extraction_lifetime_xr = convert_lifetime(extraction_lifetime_xr)
    # print(type(extraction_lifetime_xr))
    # print(extraction_lifetime_xr)
        
    extraction_materials_xr = compute_extraction_materials(path_base, climate_policy_config, circular_economy_config, scenario, year_start, year_end, year_out)
    extractioncap_xr = compute_extraction_capacity(path_base, climate_policy_config, circular_economy_config, scenario, year_start, year_end, year_out)

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
    processing_lifetime_xr = compute_processing_lifetimes(path_base, climate_policy_config, circular_economy_config, scenario, year_start, year_end, year_out)
    processing_lifetime_xr = convert_lifetime(processing_lifetime_xr)


    processing_materials_xr = compute_processing_materials(path_base, climate_policy_config, circular_economy_config, scenario, year_start, year_end, year_out)
    processingcap_xr = compute_processing_capacity(path_base, climate_policy_config, circular_economy_config, scenario, year_start, year_end, year_out)
    
   
    # bring preprocessing data into a generic format for the model
    prep_data_processing = {}
    prep_data_processing["lifetimes"] = processing_lifetime_xr
    prep_data_processing["stocks"] = processingcap_xr
    prep_data_processing["material_intensities"] = processing_materials_xr
    prep_data_processing["knowledge_graph"] = create_fossil_fuel_graph() 
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
    transport_lifetime_xr = compute_transport_lifetimes(path_base, climate_policy_config, circular_economy_config, scenario, year_start, year_end, year_out)
    transport_lifetime_xr = convert_lifetime(transport_lifetime_xr)
    transport_materials_xr = compute_transport_materials(path_base, climate_policy_config, circular_economy_config, scenario, year_start, year_end, year_out)
    transportcap_xr = compute_transport_capacity(path_base, climate_policy_config, circular_economy_config, scenario, year_start, year_end, year_out)

    #  # bring preprocessing data into a generic format for the model
    prep_data_transport = {}
    prep_data_transport["lifetimes"] = transport_lifetime_xr
    prep_data_transport["stocks"] = transportcap_xr
    prep_data_transport["material_intensities"] = transport_materials_xr
    prep_data_transport["knowledge_graph"] = create_fossil_fuel_graph() 

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

    pipelines_lifetime_xr = compute_pipelines_lifetimes(path_base, climate_policy_config, circular_economy_config, scenario, year_start, year_end, year_out) 
    pipelinecap_xr = compute_pipelines_capacity(path_base, climate_policy_config, circular_economy_config, scenario, year_start, year_end, year_out)
    pipelines_materials_xr = compute_pipelines_materials(path_base, climate_policy_config, circular_economy_config, scenario, year_start, year_end, year_out) 

    # assert path_external_data_scenario.is_dir()
    #  # bring preprocessing data into a generic format for the model
    prep_data_pipelines = {}
    prep_data_pipelines["lifetimes"] = pipelines_lifetime_xr
    prep_data_pipelines["stocks"] = pipelinecap_xr
    prep_data_pipelines["material_intensities"] = pipelines_materials_xr
    prep_data_pipelines["knowledge_graph"] = create_fossil_fuel_graph() 
    # add units
    prep_data_pipelines["stocks"] = prism.Q_(prep_data_pipelines["stocks"], "km")
    prep_data_pipelines["material_intensities"] = prism.Q_(prep_data_pipelines["material_intensities"], "kg/kg/year")
    prep_data_pipelines["set_unit_flexible"] = prism.U_(prep_data_pipelines["stocks"]) # prism.U_ gives the unit back
    #     # set_unit_flexible is needed by the model to deal with the fact the in the beginning of the model it doesn't know th data yet and needs to work with a placeholder/flexible unit (see model.py) 

    return prep_data_pipelines


# ####################################################################################################################
# #%% Extraction
# ####################################################################################################################
# print(get_preprocessing_data_extraction)

# YEAR_START = 1971   # start year of the simulation period
# YEAR_END = 2100     # end year of the calculations
# YEAR_OUT = 2100     # year of output generation = last year of reporting

# prep_data_extraction = get_preprocessing_data_extraction(path_base, climate_policy_scenario_dir, scen_folder, YEAR_START, YEAR_END, YEAR_OUT)
# # Define the complete timeline, including historic tail
# time_start = prep_data_extraction["stocks"].coords["Time"].min().values
# complete_timeline = prism.Timeline(time_start, YEAR_END, 1)
# simulation_timeline = prism.Timeline(YEAR_START, YEAR_END, 1) #1970

# sec_ff_extraction = Sector("ff_extraction", prep_data_extraction)
# main_model_factory_extraction = ModelFactory(
#     sec_ff_extraction, complete_timeline
#     ).add(GenericStocks
#     ).add(MaterialIntensities
#     ).finish()

# main_model_factory_extraction.simulate(simulation_timeline)

# list(main_model_factory_extraction.ff_extraction)


# ####################################################################################################################
# #%% Processing
# ####################################################################################################################

# YEAR_START = 1971   # start year of the simulation period
# YEAR_END = 2100     # end year of the calculations
# YEAR_OUT = 2100     # year of output generation = last year of reporting

# prep_data_processing = get_preprocessing_data_processing(path_base, SCEN, VARIANT, YEAR_START, YEAR_END, YEAR_OUT)


# # # Define the complete timeline, including historic tail
# time_start = prep_data_processing["stocks"].coords["Time"].min().values
# complete_timeline = prism.Timeline(time_start, YEAR_END, 1)
# simulation_timeline = prism.Timeline(YEAR_START, YEAR_END, 1) #1970

# sec_ff_processing = Sector("ff_processing", prep_data_processing)
# main_model_factory_processing = ModelFactory(
#     sec_ff_processing, complete_timeline
#     ).add(GenericStocks
#     ).add(MaterialIntensities
#     ).finish()

# main_model_factory_processing.simulate(simulation_timeline)
# list(main_model_factory_processing.ff_processing)


# ####################################################################################################################
# #%% Transport
# ####################################################################################################################

# YEAR_FIRST_STOR = 1907 # first use of pumped storage was in 1907 at the Engeweiher pumped storage facility near Schaffhausen, Switzerland (Mitali et al. 2022)
# YEAR_START = 1971  # start year of the simulation period
# YEAR_END = 2100    # end year of the calculations
# YEAR_OUT = 2100    # year of output generation = last year of reporting

# prep_data_transport = get_preprocessing_data_transport(path_base, SCEN, VARIANT, YEAR_START, YEAR_END, YEAR_OUT)

# # # Define the complete timeline, including historic tail
# time_start = prep_data_transport["stocks"].coords["Time"].min().values
# complete_timeline = prism.Timeline(time_start, YEAR_END, 1)
# simulation_timeline = prism.Timeline(YEAR_START, YEAR_END, 1) #1970

# sec_ff_transport = Sector("ff_transport", prep_data_transport)
# main_model_factory_transport = ModelFactory(
#     sec_ff_transport, complete_timeline
#     ).add(GenericStocks
#     ).add(MaterialIntensities
#     ).finish()

# main_model_factory_transport.simulate(simulation_timeline)
# list(main_model_factory_transport.ff_transport)


# ####################################################################################################################
# #%% Pipelines
# ####################################################################################################################

# YEAR_FIRST_STOR = 1907 # first use of pumped storage was in 1907 at the Engeweiher pumped storage facility near Schaffhausen, Switzerland (Mitali et al. 2022)
# YEAR_START = 1971  # start year of the simulation period
# YEAR_END = 2100    # end year of the calculations
# YEAR_OUT = 2100    # year of output generation = last year of reporting

# prep_data_pipelines = get_preprocessing_data_pipelines(path_base, SCEN, VARIANT, YEAR_START, YEAR_END, YEAR_OUT)

# # # Define the complete timeline, including historic tail
# time_start = prep_data_pipelines["stocks"].coords["Time"].min().values
# complete_timeline = prism.Timeline(time_start, YEAR_END, 1)
# simulation_timeline = prism.Timeline(YEAR_START, YEAR_END, 1) #1970

# sec_ff_pipelines = Sector("ff_pipelines", prep_data_pipelines)
# main_model_factory_pipelines = ModelFactory(
#     sec_ff_pipelines, complete_timeline
#     ).add(GenericStocks
#     ).add(MaterialIntensities
#     ).finish()

# main_model_factory_pipelines.simulate(simulation_timeline)
# list(main_model_factory_pipelines.ff_pipelines)


print("main.py ran successfully!")


