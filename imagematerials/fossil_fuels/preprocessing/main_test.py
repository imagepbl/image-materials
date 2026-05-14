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

year_start = 1880
year_end = 2100
year_out = 2100
YEAR_START = 2100

VARIANT = "M_CP"
SCEN = "SSP2"

region_knowledge_graph = create_region_graph()
fossil_fuel_knowledge_graph = create_fossil_fuel_graph()

#Import xarrays from lifetimes, materials and capacity folders 
from imagematerials.fossil_fuels.preprocessing.lifetimes_test import compute_extraction_lifetimes_test, compute_processing_lifetimes_test, compute_transport_lifetimes_test, compute_pipelines_lifetimes_test, compute_storage_lifetimes_test
from imagematerials.fossil_fuels.preprocessing.materials_test import compute_extraction_materials_test, compute_processing_materials_test, compute_transport_materials_test, compute_pipelines_materials_test, compute_storage_materials_test
from imagematerials.fossil_fuels.preprocessing.capacity_test import compute_extraction_capacity_test, compute_processing_capacity_test, compute_storage_capacity_test, compute_transport_capacity_test, compute_pipelines_capacity_test

#%%Extraction stage
def get_preprocessing_data_extraction(path_base: str, climate_policy_config: dict, circular_economy_config: dict, scenario: str, year_start: int, year_end: int, year_out: int):

    path_external_data_scenario = Path(path_base, "fossil_fuels", "Scenario_data", scenario)
    # test if path_external_data_scenario exists and if not set to standard scenario
    if not path_external_data_scenario.exists():
        path_external_data_scenario = Path(path_base, "fossil_fuels", STANDARD_SCEN_EXTERNAL_DATA)
    
    # The lifetimes are converted to the proper format for the model (dictionary with keys:distribution name, values:datarrays containing distribution parameters)
    extraction_lifetime_xr = compute_extraction_lifetimes_test (path_base, climate_policy_config, circular_economy_config, scenario, year_start, year_end, year_out)
    extraction_lifetime_xr = convert_lifetime(extraction_lifetime_xr)

    extraction_materials_xr = compute_extraction_materials_test(path_base, climate_policy_config, circular_economy_config, scenario, year_start, year_end, year_out)
    extractioncap_xr = compute_extraction_capacity_test(path_base, climate_policy_config, circular_economy_config, scenario, year_start, year_end, year_out)

    # bring preprocessing data into a generic format for the model
    prep_data_extraction = {}
    prep_data_extraction["lifetimes"] = extraction_lifetime_xr
    prep_data_extraction["stocks"] = extractioncap_xr
    prep_data_extraction["material_intensities"] = extraction_materials_xr
    prep_data_extraction["knowledge_graph"] = create_fossil_fuel_graph() 
    # add units
    prep_data_extraction["stocks"] = prism.Q_(prep_data_extraction["stocks"], "kg")
    prep_data_extraction["material_intensities"] = prism.Q_(prep_data_extraction["material_intensities"], "kg/kg")
    prep_data_extraction["set_unit_flexible"] = prism.U_(prep_data_extraction["stocks"]) # prism.U_ gives the unit back

    region_knowledge_graph = create_region_graph()
    prep_data_extraction["stocks"] = prep_data_extraction["stocks"].assign_coords(Region=prep_data_extraction["stocks"].coords["Region"].astype(str))
    prep_data_extraction["stocks"] = region_knowledge_graph.rebroadcast_xarray(prep_data_extraction["stocks"], output_coords=IMAGE_REGIONS, dim="Region")
        # set_unit_flexible is needed by the model to deal with the fact the in the beginning of the model it doesn't know th data yet and needs to work with a placeholder/flexible unit (see model.py) 

    return prep_data_extraction

#%%Processing stage (Just coal preparation and oil refinery)
def get_preprocessing_data_processing(path_base: str, climate_policy_config: dict, circular_economy_config: dict, scenario: str, year_start: int, year_end: int, year_out: int):

    path_external_data_scenario = Path(path_base, "fossil_fuels", "Scenario_data", scenario)
    # test if path_external_data_scenario exists and if not set to standard scenario
    if not path_external_data_scenario.exists():
        path_external_data_scenario = Path(path_base, "fossil_fuels", STANDARD_SCEN_EXTERNAL_DATA)

    # assert path_external_data_scenario.is_dir()

    #return path_external_data_scenario

     # The lifetimes are converted to the proper format for the model (dictionary with keys:distribution name, values:datarrays containing distribution parameters)
    processing_lifetime_xr = compute_processing_lifetimes_test(path_base, climate_policy_config, circular_economy_config, scenario, year_start, year_end, year_out)
    processing_lifetime_xr = convert_lifetime(processing_lifetime_xr)

    processing_materials_xr = compute_processing_materials_test(path_base, climate_policy_config, circular_economy_config, scenario, year_start, year_end, year_out)
    processingcap_xr = compute_processing_capacity_test(path_base, climate_policy_config, circular_economy_config, scenario, year_start, year_end, year_out)
    
    # bring preprocessing data into a generic format for the model
    prep_data_processing = {}
    prep_data_processing["lifetimes"] = processing_lifetime_xr
    prep_data_processing["stocks"] = processingcap_xr
    prep_data_processing["material_intensities"] = processing_materials_xr
    prep_data_processing["knowledge_graph"] = create_fossil_fuel_graph() 
    # add units
    prep_data_processing["stocks"] = prism.Q_(prep_data_processing["stocks"], "kg")
    prep_data_processing["material_intensities"] = prism.Q_(prep_data_processing["material_intensities"], "kg/kg")
    prep_data_processing["set_unit_flexible"] = prism.U_(prep_data_processing["stocks"]) # prism.U_ gives the unit back

    region_knowledge_graph = create_region_graph()
    prep_data_processing["stocks"] = prep_data_processing["stocks"].assign_coords(Region=prep_data_processing["stocks"].coords["Region"].astype(str))
    prep_data_processing["stocks"] = region_knowledge_graph.rebroadcast_xarray(prep_data_processing["stocks"], output_coords=IMAGE_REGIONS, dim="Region")
    #     # set_unit_flexible is needed by the model to deal with the fact the in the beginning of the model it doesn't know th data yet and needs to work with a placeholder/flexible unit (see model.py) 

    return prep_data_processing

#%%Processing stage (Just gas processing and oil storage)
def get_preprocessing_data_storage(path_base: str, climate_policy_config: dict, circular_economy_config: dict, scenario: str, year_start: int, year_end: int, year_out: int):

    path_external_data_scenario = Path(path_base, "fossil_fuels", "Scenario_data", scenario)
    # test if path_external_data_scenario exists and if not set to standard scenario
    if not path_external_data_scenario.exists():
        path_external_data_scenario = Path(path_base, "fossil_fuels", STANDARD_SCEN_EXTERNAL_DATA)

    # assert path_external_data_scenario.is_dir()

    #return path_external_data_scenario

     # The lifetimes are converted to the proper format for the model (dictionary with keys:distribution name, values:datarrays containing distribution parameters)
    storage_lifetime_xr = compute_storage_lifetimes_test(path_base, climate_policy_config, circular_economy_config, scenario, year_start, year_end, year_out)
    storage_lifetime_xr = convert_lifetime(storage_lifetime_xr)

    storage_materials_xr = compute_storage_materials_test(path_base, climate_policy_config, circular_economy_config, scenario, year_start, year_end, year_out)
    storagecap_xr = compute_storage_capacity_test(path_base, climate_policy_config, circular_economy_config, scenario, year_start, year_end, year_out)
    # bring preprocessing data into a generic format for the model
    prep_data_storage = {}
    prep_data_storage["lifetimes"] = storage_lifetime_xr
    prep_data_storage["stocks"] = storagecap_xr
    prep_data_storage["material_intensities"] = storage_materials_xr
    prep_data_storage["knowledge_graph"] = create_fossil_fuel_graph() 
    # add units
    prep_data_storage["stocks"] = prism.Q_(prep_data_storage["stocks"], "meter**3")
    prep_data_storage["material_intensities"] = prism.Q_(prep_data_storage["material_intensities"], "kg/meter**3")
    prep_data_storage["set_unit_flexible"] = prism.U_(prep_data_storage["stocks"]) # prism.U_ gives the unit back

    region_knowledge_graph = create_region_graph()
    prep_data_storage["stocks"] = prep_data_storage["stocks"].assign_coords(Region=prep_data_storage["stocks"].coords["Region"].astype(str))
    prep_data_storage["stocks"] = region_knowledge_graph.rebroadcast_xarray(prep_data_storage["stocks"], output_coords=IMAGE_REGIONS, dim="Region")
    #     # set_unit_flexible is needed by the model to deal with the fact the in the beginning of the model it doesn't know th data yet and needs to work with a placeholder/flexible unit (see model.py) 

    return prep_data_storage
#%% Transport stage
def get_preprocessing_data_transport(path_base: str, climate_policy_config: dict, circular_economy_config: dict, scenario: str, year_start: int, year_end: int, year_out: int):

    path_external_data_scenario = Path(path_base, "fossil_fuels", "Scenario_data", scenario)
    # test if path_external_data_scenario exists and if not set to standard scenario
    if not path_external_data_scenario.exists():
        path_external_data_scenario = Path(path_base, "fossil_fuels", STANDARD_SCEN_EXTERNAL_DATA)

    # assert path_external_data_scenario.is_dir()
    transport_lifetime_xr = compute_transport_lifetimes_test(path_base, climate_policy_config, circular_economy_config, scenario, year_start, year_end, year_out)
    transport_lifetime_xr = convert_lifetime(transport_lifetime_xr)
    transport_materials_xr = compute_transport_materials_test(path_base, climate_policy_config, circular_economy_config, scenario, year_start, year_end, year_out)
    transportcap_xr = compute_transport_capacity_test(path_base, climate_policy_config, circular_economy_config, scenario, year_start, year_end, year_out)

    #  # bring preprocessing data into a generic format for the model
    prep_data_transport = {}
    prep_data_transport["lifetimes"] = transport_lifetime_xr
    prep_data_transport["stocks"] = transportcap_xr
    prep_data_transport["material_intensities"] = transport_materials_xr
    prep_data_transport["knowledge_graph"] = create_fossil_fuel_graph() 

    # add units
    prep_data_transport["stocks"] = prism.Q_(prep_data_transport["stocks"], "kg")
    prep_data_transport["material_intensities"] = prism.Q_(prep_data_transport["material_intensities"], "kg/kg")
    prep_data_transport["set_unit_flexible"] = prism.U_(prep_data_transport["stocks"]) # prism.U_ gives the unit back

    region_knowledge_graph = create_region_graph()
    prep_data_transport["stocks"] = prep_data_transport["stocks"].assign_coords(Region=prep_data_transport["stocks"].coords["Region"].astype(str))
    prep_data_transport["stocks"] = region_knowledge_graph.rebroadcast_xarray(prep_data_transport["stocks"], output_coords=IMAGE_REGIONS, dim="Region")
        # set_unit_flexible is needed by the model to deal with the fact the in the beginning of the model it doesn't know th data yet and needs to work with a placeholder/flexible unit (see model.py) 
    return prep_data_transport

#%% Pipelines stage (oil, gas) ---------------------------------------------------------------------------------------------------------------------------------
def get_preprocessing_data_pipelines(path_base: str, climate_policy_config: dict, circular_economy_config: dict, scenario: str, year_start: int, year_end: int, year_out: int):

    path_external_data_scenario = Path(path_base, "fossil_fuels", "Scenario_data", scenario)
    # test if path_external_data_scenario exists and if not set to standard scenario
    if not path_external_data_scenario.exists():
        path_external_data_scenario = Path(path_base, "fossil_fuels", STANDARD_SCEN_EXTERNAL_DATA)

    pipelines_lifetime_xr = compute_pipelines_lifetimes_test(path_base, climate_policy_config, circular_economy_config, scenario, year_start, year_end, year_out) 
    pipelines_lifetime_xr = convert_lifetime(pipelines_lifetime_xr)
    
    pipelinecap_xr = compute_pipelines_capacity_test(path_base, climate_policy_config, circular_economy_config, scenario, year_start, year_end, year_out)
    pipelines_materials_xr = compute_pipelines_materials_test(path_base, climate_policy_config, circular_economy_config, scenario, year_start, year_end, year_out) 

    # assert path_external_data_scenario.is_dir()
    #  # bring preprocessing data into a generic format for the model
    prep_data_pipelines = {}
    prep_data_pipelines["lifetimes"] = pipelines_lifetime_xr
    prep_data_pipelines["stocks"] = pipelinecap_xr
    prep_data_pipelines["material_intensities"] = pipelines_materials_xr
    prep_data_pipelines["knowledge_graph"] = create_fossil_fuel_graph() 
    # add units
    prep_data_pipelines["stocks"] = prism.Q_(prep_data_pipelines["stocks"], "km")
    prep_data_pipelines["material_intensities"] = prism.Q_(prep_data_pipelines["material_intensities"], "kg/km")
    prep_data_pipelines["set_unit_flexible"] = prism.U_(prep_data_pipelines["stocks"]) # prism.U_ gives the unit back


    region_knowledge_graph= create_region_graph()
    prep_data_pipelines["stocks"] = prep_data_pipelines["stocks"].assign_coords(Region=prep_data_pipelines["stocks"].coords["Region"].astype(str))
    prep_data_pipelines["stocks"] = region_knowledge_graph.rebroadcast_xarray(prep_data_pipelines["stocks"], output_coords=IMAGE_REGIONS, dim="Region")
    #     # set_unit_flexible is needed by the model to deal with the fact the in the beginning of the model it doesn't know th data yet and needs to work with a placeholder/flexible unit (see model.py) 

    return prep_data_pipelines


print("main.py_test ran successfully!")


