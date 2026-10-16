# import relevant libraries
# import constants and other functions within image materials
# read in materials files for each stage
# convert to xarrays
# return material_intensity_extraction,material_intensity_processing,... 


import pint
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path

import prism

from imagematerials.concepts import create_fossil_fuel_graph, create_region_graph
from imagematerials.fossil_fuels.constants import (
    DATA_DIR,
    STANDARD_SCEN_EXTERNAL_DATA,
)

year_start = 1880
year_end = 2100
year_out = 2100

knowledge_graph_region = create_region_graph()
fossil_fuel_knowledge_graph = create_fossil_fuel_graph()

#%% Extraction stage (coal, oil, gas) ---------------------------------------------------------------------------------------------------------------------------------

###########################################################################################################
def compute_extraction_materials(path_base: str, climate_policy_config: dict, circular_economy_config: dict, scenario: str, year_start: int, year_end: int, year_out: int):

    path_external_data_scenario = Path(path_base, "fossil_fuels", "Scenario_data", scenario)
    # test if path_external_data_scenario exists and if not set to standard scenario
    if not path_external_data_scenario.exists():
        path_external_data_scenario = Path(path_base, "fossil_fuels", STANDARD_SCEN_EXTERNAL_DATA)

    # assert path_external_data_scenario.is_dir()

    ###########################################################################################################
    # Read in files #

    # 1. External Data --------------------------------------------- 

        # material compositions of coal infrastructure (processing, extraction, transport) in kg/kg/year
    extraction_materials_data = pd.read_csv(DATA_DIR / "Extraction" / "Extraction_materials.csv")
 # Material Intensities -------

    # Set index
    extraction_materials_data = extraction_materials_data.set_index(['Year', 'Tech Type'])

    # Convert to 3D array: (Material, Year, Tech)
    extraction_materials_xr = (
        extraction_materials_data
            .to_xarray()            
            .to_array("material") 
            .rename({"Year": "Cohort", "Tech Type": "Type"})
    )

    extraction_materials_xr.name = "ExtractionMaterialIntensities"

    #Expand 2020 values across 1880-2100
    value_2020 = extraction_materials_xr.sel(Cohort=2020)
    year_range = np.arange(1880, 2101)
    extraction_materials_xr = value_2020.expand_dims(Cohort=year_range).transpose("material", "Cohort", "Type")

    #Add units
    extraction_materials_xr = prism.Q_(extraction_materials_xr, "kg/kg")

    return extraction_materials_xr

#%% Processing stage (coal preparation and oil refinery) ---------------------------------------------------------------------------------------------------------------------------------

def compute_processing_materials(path_base: str, climate_policy_config: dict, circular_economy_config: dict, scenario: str, year_start: int, year_end: int, year_out: int):
    path_external_data_scenario = Path(path_base, "fossil_fuels", "Scenario_data", scenario)
    # test if path_external_data_scenario exists and if not set to standard scenario
    if not path_external_data_scenario.exists():
        path_external_data_scenario = Path(path_base, "fossil_fuels", STANDARD_SCEN_EXTERNAL_DATA)

    # assert path_external_data_scenario.is_dir()

    #return path_external_data_scenario
###########################################################################################################
    # Read in files #

    # 1. External Data --------------------------------------------- 
        # material compositions of processing infrastructure in kg/kg/year
    processing_materials_data = pd.read_csv(DATA_DIR / "Processing" / "Processing_materials.csv")
# Material Intensities -------

        # Material Intensities -------
    # Set index
    processing_materials_data = processing_materials_data.set_index(['Year', 'Tech Type'])

    # Convert to 3D array: (Material, Year, Tech)
    processing_materials_xr = (
        processing_materials_data
            .to_xarray()           # keeps Year & Tech Type as coords
            .to_array("material")  # converts columns into a 'Material' dimension
            .rename({"Year": "Cohort", "Tech Type": "Type"})
    )

    # Name the DataArray
    processing_materials_xr.name = "ProcessingMaterialIntensities"

    #Expand 2020 values across 1880-2100
    value_2020 = processing_materials_xr.sel(Cohort=2020)
    year_range = np.arange(1880, 2101)
    processing_materials_xr = value_2020.expand_dims(Cohort=year_range).transpose("material", "Cohort", "Type")

    #Add units
    processing_materials_xr = prism.Q_(processing_materials_xr, "kg/kg")

    return processing_materials_xr

#%% Processing stage (gas processing and oil storage)) ---------------------------------------------------------------------------------------------------------------------------------

def compute_storage_materials(path_base: str, climate_policy_config: dict, circular_economy_config: dict, scenario: str, year_start: int, year_end: int, year_out: int):
    path_external_data_scenario = Path(path_base, "fossil_fuels", "Scenario_data", scenario)
    # test if path_external_data_scenario exists and if not set to standard scenario
    if not path_external_data_scenario.exists():
        path_external_data_scenario = Path(path_base, "fossil_fuels", STANDARD_SCEN_EXTERNAL_DATA)

    # assert path_external_data_scenario.is_dir()

    #return path_external_data_scenario
###########################################################################################################
    # Read in files #

    # 1. External Data --------------------------------------------- 
        # material compositions of processing infrastructure in kg/kg/year
    storage_materials_data = pd.read_csv(DATA_DIR / "Processing" / "Storage_materials.csv")
# Material Intensities -------

        # Material Intensities -------
    # Set index
    storage_materials_data = storage_materials_data.set_index(['Year', 'Tech Type'])

    # Convert to 3D array: (Material, Year, Tech)
    storage_materials_xr = (
        storage_materials_data
            .to_xarray()           # keeps Year & Tech Type as coords
            .to_array("material")  # converts columns into a 'Material' dimension
            .rename({"Year": "Cohort", "Tech Type": "Type"})
    )

    # Name the DataArray
    storage_materials_xr.name = "StorageMaterialIntensities"

    #Expand 2020 values across 1880-2100
    value_2020 = storage_materials_xr.sel(Cohort=2020)
    year_range = np.arange(1880, 2101)
    storage_materials_xr = value_2020.expand_dims(Cohort=year_range).transpose("material", "Cohort", "Type")

    #Add units
    storage_materials_xr = prism.Q_(storage_materials_xr, "kg/meter**3")

    return storage_materials_xr


# #%% Transport/Vehicles stage (coal, oil, gas) ---------------------------------------------------------------------------------------------------------------------------------

# ###########################################################################################################
# def compute_transport_material_fraction(path_base: str, climate_policy_config: dict, circular_economy_config: dict, scenario: str, year_start: int, year_end: int, year_out: int):
def compute_transport_materials(path_base: str, climate_policy_config: dict, circular_economy_config: dict, scenario: str, year_start: int, year_end: int, year_out: int):

    path_external_data_scenario = Path(path_base, "fossil_fuels", "Scenario_data", scenario)
    # test if path_external_data_scenario exists and if not set to standard scenario
    if not path_external_data_scenario.exists():
        path_external_data_scenario = Path(path_base, "fossil_fuels", STANDARD_SCEN_EXTERNAL_DATA)

#     # assert path_external_data_scenario.is_dir()
# ###########################################################################################################
#     # Read in files #

#     # 1. External Data --------------------------------------------- 

    transport_materials_data = pd.read_csv(DATA_DIR / "Transport" / "Transport_materials.csv")

# Set index
    transport_materials_data = transport_materials_data.set_index(['Year', 'Tech Type'])

    # Convert to 3D array: (Material, Year, Tech)
    transport_materials_xr = (
        transport_materials_data
            .to_xarray()           
            .to_array("material")  
            .rename({"Year": "Cohort", "Tech Type": "Type"})
    )
    transport_materials_xr.name = "TransportMaterialIntensities"

    #Expand 2020 values across 1880-2100
    value_2020 = transport_materials_xr.sel(Cohort=2020)
    year_range = np.arange(1880, 2101)
    transport_materials_xr = value_2020.expand_dims(Cohort=year_range).transpose("material", "Cohort", "Type")
    transport_materials_xr = prism.Q_(transport_materials_xr, "kg/kg")

    return transport_materials_xr


#%% Pipelines stage (oil, gas) ---------------------------------------------------------------------------------------------------------------------------------
###########################################################################################################

def compute_pipelines_materials(path_base: str, climate_policy_config: dict, circular_economy_config: dict, scenario: str, year_start: int, year_end: int, year_out: int):

    path_external_data_scenario = Path(path_base, "fossil_fuels", "Scenario_data", scenario)
    # test if path_external_data_scenario exists and if not set to standard scenario
    if not path_external_data_scenario.exists():
        path_external_data_scenario = Path(path_base, "fossil_fuels", STANDARD_SCEN_EXTERNAL_DATA)

    # assert path_external_data_scenario.is_dir()
###########################################################################################################
    # Read in files #

    # 1. External Data --------------------------------------------- 
        # material compositions of transport infrastructure in kg/kg/year
    pipelines_materials_data = pd.read_csv(DATA_DIR / "Pipelines" / "Pipelines_materials_edit.csv")
        # Material Intensities -------
    # Set index
    pipelines_materials_data = pipelines_materials_data.set_index(['Year', 'Tech Type'])

    # Convert to 3D array: (Material, Year, Tech)
    pipelines_materials_xr = (
        pipelines_materials_data
            .to_xarray()           
            .to_array("material")  
            .rename({"Year": "Cohort", "Tech Type": "Type"})
    )
    pipelines_materials_xr.name = "PipelinesMaterialIntensities"

    #Expand 2020 values across 1880-2100
    value_2020 = pipelines_materials_xr.sel(Cohort=2020)
    year_range = np.arange(1880, 2101)
    pipelines_materials_xr = value_2020.expand_dims(Cohort=year_range).transpose("material", "Cohort", "Type")
    pipelines_materials_xr = prism.Q_(pipelines_materials_xr, "kg/km")

    pipeline_order = [
        'Gas Distribution Pipeline',
        'Gas Transmission Pipeline',
        'Oil Offshore Crude Pipeline',
        'Oil Onshore Crude Pipeline',
        'Oil Offshore Product Pipeline',
        'Oil Onshore Product Pipeline'
    ]

    #Reorder technology coordinate in all three datasets (capacity, lifetime, material intensities) to match the same order of technologies (coal opencast, coal underground, gas offshore, gas onshore, oil offshore, oil onshore), so that we can easily combine the data in the model (since they all have to be in the same order of technologies)
    pipelines_materials_xr = pipelines_materials_xr.reindex(Type=pipeline_order)
    
    return pipelines_materials_xr
