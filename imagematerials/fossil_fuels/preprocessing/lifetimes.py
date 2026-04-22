#Outline (split into each stage (extraction, processing, transport, pipelines) for each fuel (coal, oil, gas)):
# 1) Import general modules and constants
# 2) read in necessary data files for lifetimes 
# 3) create xarray

#%% Import general modules and constants
import pint
import xarray as xr
import prism
import pandas as pd
import numpy as np
import os
from pathlib import Path


from imagematerials.fossil_fuels.preprocessing.ffconstants import (
    # configs & scenarios
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
    SD_LIFETIME,
)

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

#from prism.prism.examples.fuel import scenario

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

# path_current = Path().resolve()
# path_base = path_current.parents [1]
# print("current:", path_current)
# print("base:", path_base)

#still not sure what to do with these 
# scen_folder = "SSP2_baseline"
# climate_policy_scenario_dir = Path(path_base, "data", "raw", "image", scen_folder)
# STANDARD_SCEN_EXTERNAL_DATA = "SSP2_baseline" 

# BASE_DIR = Path(__file__).resolve().parent
# DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "fossil_fuels"
# OUTPUT_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "fossil_fuels" / "Scenario_data"
# IMAGE_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "image"

# print("current:", path_current)
# print("base:", path_base)
# print("climate policy scenario dir:", climate_policy_scenario_dir)
# print("data dir:", DATA_DIR)    
# print("output dir:", OUTPUT_DIR)
# print("image dir:", IMAGE_DIR)  

year_start = 1880
year_end = 2100
year_out = 2100

knowledge_graph_region = create_region_graph()
fossil_fuel_knowledge_graph = create_fossil_fuel_graph()


# Extraction stage (coal, oil, gas) ---------------------------------------------------------------------------------------------------------------------------------

###########################################################################################################
def compute_extraction_lifetimes(path_base: str, climate_policy_config: dict, circular_economy_config: dict, scenario: str, year_start: int, year_end: int, year_out: int):

    path_external_data_scenario = Path(path_base, "fossil_fuels", "Scenario_data", scenario)
    # test if path_external_data_scenario exists and if not set to standard scenario
    if not path_external_data_scenario.exists():
        path_external_data_scenario = Path(path_base, "fossil_fuels", STANDARD_SCEN_EXTERNAL_DATA)

    # assert path_external_data_scenario.is_dir()

###########################################################################################################
    # Read in files #

    # 1. External Data --------------------------------------------- 

        # lifetimes of extraction tech in years
    extraction_lifetime_data = pd.read_csv(DATA_DIR / "Extraction" / "Extraction_lifetimes.csv",
        index_col=["Year", "Tech Type"]
    ) 

       # Lifetimes -------
    types = extraction_lifetime_data.index.get_level_values("Tech Type").unique().to_numpy()
    values = extraction_lifetime_data["Lifetime"].unstack().reindex(columns=types).to_numpy(dtype=float)
    times = extraction_lifetime_data.index.get_level_values("Year").unique().to_numpy()
    
    scipy_params = ["mean", "stdev"]

    # Build xarray with shape (ScipyParam, Time, Type)
    data_array = np.stack([values, np.full_like(values, SD_LIFETIME)], axis=0)

    #create xarray with dimensions and coordinates
    extraction_lifetime_xr = xr.DataArray(
        data_array,
        dims=["DistributionParams", "Cohort", "Type"],
        coords={
            "DistributionParams": scipy_params,
            "Cohort": times,
            "Type": types
        },
        name="ExtractionLifetime"
    )

    # Expand 2020 across 1880-2100
    year_range = np.arange(1880, 2101)
    value_2020 = extraction_lifetime_xr.sel(Cohort=2020)
    extraction_lifetime_xr = value_2020.expand_dims(Cohort=year_range).transpose("DistributionParams", "Cohort", "Type")

    #Add units 
    extraction_lifetime_xr = prism.Q_(extraction_lifetime_xr, "year")
    #Set order of technology coordinate to match the order in the lifetime and material intensity data, so that we can easily combine the data in the model (since they all have to be in the same order of technologies)
    extraction_order = [
        'Coal Opencast',
        'Coal Underground',
        'Gas Offshore',
        'Gas Onshore',
        'Oil Offshore',
        'Oil Onshore'
    ]
    extraction_lifetime_xr = extraction_lifetime_xr.reindex(Type=extraction_order)

    # print("=== extractionlifetimes_xr ===")
    # print(extraction_lifetime_xr)
    # print("Dims:", extraction_lifetime_xr.dims)
    # print("Sizes:", extraction_lifetime_xr.sizes)
    # print("Cohort:", extraction_lifetime_xr['Cohort'].values)
    # print("Type:", extraction_lifetime_xr['Type'].values)
    # print("Params:", extraction_lifetime_xr['DistributionParams'].values)
    # extraction_lifetime_xr.isnull().sum()

    # #Rebroadcast to standard technology names from TIMER, and convert coordinate type back to python strings (since rebroadcast changes it to numpy strings)
    # extraction_lifetime_xr = fossil_fuel_knowledge_graph.rebroadcast_xarray(extraction_lifetime_xr, output_coords=FF_TECHNOLOGIES, dim="Type") # convert technology names to the standard names from TIMER
    # extraction_lifetime_xr = extraction_lifetime_xr.assign_coords(Type=np.array(extraction_lifetime_xr.Type.values, dtype=object)) # rebroadcast_xarray changes the type of the coordinates to numpy strings (np.str_), so convert back to python strings (str)

    # print("=== extractionlifetimes_xr ===")
    # print(extraction_lifetime_xr)
    # print("Dims:", extraction_lifetime_xr.dims)
    # print("Sizes:", extraction_lifetime_xr.sizes)
    # print("Cohort:", extraction_lifetime_xr['Cohort'].values)
    # print("Type:", extraction_lifetime_xr['Type'].values)
    # print("Params:", extraction_lifetime_xr['DistributionParams'].values)
    # extraction_lifetime_xr.isnull().sum()

    return extraction_lifetime_xr

#%% Processing stage (coal, oil, gas) ---------------------------------------------------------------------------------------------------------------------------------

def compute_processing_lifetimes(path_base: str, climate_policy_config: dict, circular_economy_config: dict, scenario: str, year_start: int, year_end: int, year_out: int):

    path_external_data_scenario = Path(path_base, "fossil_fuels", "Scenario_data", scenario)
    # test if path_external_data_scenario exists and if not set to standard scenario
    if not path_external_data_scenario.exists():
        path_external_data_scenario = Path(path_base, "fossil_fuels", STANDARD_SCEN_EXTERNAL_DATA)

    # assert path_external_data_scenario.is_dir()

    #return path_external_data_scenario
###########################################################################################################
    # Read in files #

    # 1. External Data --------------------------------------------- 

    # lifetimes of processing tech in years
    processing_lifetime_data = pd.read_csv(DATA_DIR / "Processing" / "Processing_lifetimes.csv",
        index_col=["Year", "Tech Type"]
    ) 

           # Lifetimes -------
    values = processing_lifetime_data["Lifetime"].unstack().to_numpy(dtype=float)
    #Create coordinates
    times = processing_lifetime_data.index.levels[0].to_numpy()
    types = processing_lifetime_data.index.get_level_values("Tech Type").unique().to_numpy()
    scipy_params = ["mean", "stdev"]
        # Build full array: shape (ScipyParam, Time, Type)
    data_array = np.stack([values, np.full_like(values, SD_LIFETIME)], axis=0)
        # Create DataArray
    processing_lifetime_xr = xr.DataArray(
            data_array,
            dims=["DistributionParams", "Cohort", "Type"],
            coords={
                "DistributionParams": scipy_params,
                "Cohort": times,
                "Type": types
            },
            name="ProcessingLifetime"
        )
    # Expand 2020 across 1880-2100
    year_range = np.arange(1880, 2101)
    value_2020 = processing_lifetime_xr.sel(Cohort=2020)
    processing_lifetime_xr = value_2020.expand_dims(Cohort=year_range).transpose("DistributionParams", "Cohort", "Type")
    processing_lifetime_xr = prism.Q_(processing_lifetime_xr, "year")
 
    #  #Rebroadcast to standard technology names from TIMER, and convert coordinate type back to python strings (since rebroadcast changes it to numpy strings)
    # processing_lifetime_xr = fossil_fuel_knowledge_graph.rebroadcast_xarray(processing_lifetime_xr, output_coords=FF_TECHNOLOGIES, dim="Type") # convert technology names to the standard names from TIMER
    # processing_lifetime_xr = processing_lifetime_xr.assign_coords(Type=np.array(processing_lifetime_xr.Type.values, dtype=object)) # rebroadcast_xarray changes the type of the coordinates to numpy strings (np.str_), so convert back to python strings (str)

    return processing_lifetime_xr

#%% Transport/Vehicles stage (coal, oil, gas) ---------------------------------------------------------------------------------------------------------------------------------

###########################################################################################################
def compute_transport_lifetimes(path_base: str, climate_policy_config: dict, circular_economy_config: dict, scenario: str, year_start: int, year_end: int, year_out: int):

    path_external_data_scenario = Path(path_base, "fossil_fuels", "Scenario_data", scenario)
    # test if path_external_data_scenario exists and if not set to standard scenario
    if not path_external_data_scenario.exists():
        path_external_data_scenario = Path(path_base, "fossil_fuels", STANDARD_SCEN_EXTERNAL_DATA)

    # assert path_external_data_scenario.is_dir()
###########################################################################################################
    # Read in files #

    # 1. External Data --------------------------------------------- 

    # lifetimes of transport tech in years
    transport_lifetime_data = pd.read_csv(DATA_DIR / "Transport" / "Transport_lifetimes.csv", index_col=["Year", "Tech Type"])

        # Lifetimes -------
    values = transport_lifetime_data["Lifetimes"].unstack().to_numpy(dtype=float)
    #Create coordinates
    times = transport_lifetime_data.index.levels[0].to_numpy()
    types = transport_lifetime_data.index.get_level_values("Tech Type").unique().to_numpy()
    scipy_params = ["mean", "stdev"]
    # Build full array: shape (ScipyParam, Time, Type)
    data_array = np.stack([values, np.full_like(values, SD_LIFETIME)], axis=0)
    # Create DataArray
    transport_lifetime_xr = xr.DataArray(
            data_array,
            dims=["DistributionParams", "Cohort", "Type"],
            coords={
                "DistributionParams": scipy_params,
                "Cohort": times,
                "Type": types
            },
            name="TransportLifetime"
        )

    # Expand 2020 across 1880-2100
    year_range = np.arange(1880, 2101)
    value_2020 = transport_lifetime_xr.sel(Cohort=2020)
    transport_lifetime_xr = value_2020.expand_dims(Cohort=year_range).transpose("DistributionParams", "Cohort", "Type")
    transport_lifetime_xr = prism.Q_(transport_lifetime_xr, "year")

    # #Rebroadcast to standard technology names from TIMER, and convert coordinate type back to python strings (since rebroadcast changes it to numpy strings)
    # transport_lifetime_xr = fossil_fuel_knowledge_graph.rebroadcast_xarray(transport_lifetime_xr, output_coords=FF_TECHNOLOGIES, dim="Type") # convert technology names to the standard names from TIMER
    # transport_lifetime_xr = transport_lifetime_xr.assign_coords(Type=np.array(transport_lifetime_xr.Type.values, dtype=object)) # rebroadcast_xarray changes the type of the coordinates to numpy strings (np.str_), so convert back to python strings (str)

    return transport_lifetime_xr

#%% Pipelines stage (oil, gas) ---------------------------------------------------------------------------------------------------------------------------------
###########################################################################################################

def compute_pipelines_lifetimes(path_base: str, climate_policy_config: dict, circular_economy_config: dict, scenario: str, year_start: int, year_end: int, year_out: int):

    path_external_data_scenario = Path(path_base, "fossil_fuels", "Scenario_data", scenario)
    # test if path_external_data_scenario exists and if not set to standard scenario
    if not path_external_data_scenario.exists():
        path_external_data_scenario = Path(path_base, "fossil_fuels", STANDARD_SCEN_EXTERNAL_DATA)

    # assert path_external_data_scenario.is_dir()
###########################################################################################################
    # Read in files #

    # 1. External Data --------------------------------------------- 

        # lifetimes of transport tech in years
    pipelines_lifetime_data = pd.read_csv(
        DATA_DIR / "Pipelines" / "Pipelines_lifetimes_edit.csv",
        index_col=["Year", "Tech Type"]
    ) 

            ###########################################################################################################
        # Transform to xarray #
    # knowledge_graph_region = create_region_graph()
    # fossil_fuel_knowledge_graph = create_fossil_fuel_graph()

        # Lifetimes -------
    values = pipelines_lifetime_data["Lifetimes"].unstack().to_numpy(dtype=float)
    #Create coordinates
    times = pipelines_lifetime_data.index.levels[0].to_numpy()
    types = pipelines_lifetime_data.index.get_level_values("Tech Type").unique().to_numpy()
    scipy_params = ["mean", "stdev"]
    # Build full array: shape (ScipyParam, Time, Type)
    data_array = np.stack([values, np.full_like(values, SD_LIFETIME)], axis=0)
    # Create DataArray
    pipelines_lifetime_xr = xr.DataArray(
            data_array,
            dims=["DistributionParams", "Cohort", "Type"],
            coords={
                "DistributionParams": scipy_params,
                "Cohort": times,
                "Type": types
            },
            name="PipelinesLifetime"
        )
    pipelines_lifetime_xr = prism.Q_(pipelines_lifetime_xr, "year")

    # Expand 2020 across 1880-2100
    year_range = np.arange(1880, 2101)
    value_2020 = pipelines_lifetime_xr.sel(Cohort=2020)
    pipelines_lifetime_xr = value_2020.expand_dims(Cohort=year_range).transpose("DistributionParams", "Cohort", "Type")

    # # Rebroadcast to standard technology names from TIMER, and convert coordinate type back to python strings (since rebroadcast changes it to numpy strings)
    # pipelines_lifetime_xr = fossil_fuel_knowledge_graph.rebroadcast_xarray(pipelines_lifetime_xr, output_coords=FF_TECHNOLOGIES, dim="Type") # convert technology names to the standard names from TIMER
    # pipelines_lifetime_xr = pipelines_lifetime_xr.assign_coords(Type=np.array(pipelines_lifetime_xr.Type.values, dtype=object)) # rebroadcast_xarray changes the type of the coordinates to numpy strings (np.str_), so convert back to python strings (str)
    
    pipeline_order = [
        'Gas Distribution Pipeline',
        'Gas Transmission Pipeline',
        'Oil Offshore Crude Pipeline',
        'Oil Onshore Crude Pipeline',
        'Oil Offshore Product Pipeline',
        'Oil Onshore Product Pipeline'
    ]

    #Reorder technology coordinate in all three datasets (capacity, lifetime, material intensities) to match the same order of technologies (coal opencast, coal underground, gas offshore, gas onshore, oil offshore, oil onshore), so that we can easily combine the data in the model (since they all have to be in the same order of technologies)
    pipelines_lifetime_xr = pipelines_lifetime_xr.reindex(Type=pipeline_order)

    return pipelines_lifetime_xr


print("lifetimes.py ran successfully!")
