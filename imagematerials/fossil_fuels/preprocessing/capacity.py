#
# Outline:
# 1) Import general modules and constants
# 2) Define functions to get preprocessing data for stocks of each supply chain stage (extraction, processing, transport, pipelines)
# 3) Within each step, first read in necessary file and then convert to xarray and then do necessary processing to get it into the right format for the model (e.g. rebroadcasting to standard region and technology names, adding units, etc.)

#%% Import general modules and constants

import pint
import xarray as xr
import prism
import pandas as pd
import numpy as np
import os
from pathlib import Path

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

from imagematerials.fossil_fuels.preprocessing.drivers import coal_infra, oil_infra, gas_infra

extraction_stock_oil, transport_total_oil, oil_pipelines, oil_storage, refinery_stock_oil = oil_infra()
extraction_stock_coal, preparation_stock_coal, transport_total_coal = coal_infra()
extraction_stock_gas, transport_total_gas, gas_pipelines, processing_stock_gas = gas_infra()


from imagematerials.read_mym import read_mym_df
from imagematerials.util import dataset_to_array, pandas_to_xarray, convert_lifetime
from imagematerials.concepts import KnowledgeGraph, Node, create_fossil_fuel_graph, create_region_graph
from imagematerials.electricity.utils import (
   MNLogit,  
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

knowledge_graph_region = create_region_graph()
fossil_fuel_knowledge_graph = create_fossil_fuel_graph()


#%%Extraction stage (coal, oil, gas) ---------------------------------------------------------------------------------------------------------------------------------
def compute_extraction_capacity(path_base: str, climate_policy_config: dict, circular_economy_config: dict, scenario: str, year_start: int, year_end: int, year_out: int):

    path_external_data_scenario = Path(path_base, "fossil_fuels", "Scenario_data", scenario)
    # test if path_external_data_scenario exists and if not set to standard scenario
    if not path_external_data_scenario.exists():
        path_external_data_scenario = Path(path_base, "fossil_fuels", STANDARD_SCEN_EXTERNAL_DATA)

    # 2.FUMA output -----------------------------------------
    #Stock of each type of extraction infrastructure (stock demand per generation technology) per region per year
    #Load data from drivers.py

    coal_drivers = coal_infra()
    extraction_stock_coal = coal_drivers["extraction"]

    oil_drivers = oil_infra()
    extraction_stock_oil = oil_drivers["extraction"]

    gas_drivers = gas_infra()
    extraction_stock_gas = gas_drivers["extraction"]

    # --- NOW reset_index safely ---
    df_extraction_coal = extraction_stock_coal.reset_index()
    df_extraction_oil  = extraction_stock_oil.reset_index()
    df_extraction_gas  = extraction_stock_gas.reset_index()

    #Add fuel column to each df so that we can combine them and then split them again into the different technologies (coal, oil, gas) after melting the df to long format
    df_extraction_coal["fuel"] = "coal"
    df_extraction_oil["fuel"] = "oil"
    df_extraction_gas["fuel"] = "gas"


    # Combine the dataframes for coal, oil, and gas into one dataframe and then melt it to long format so that we have one row per combination of time, region, fuel, technology, and value (stock)
    df_extraction_all = pd.concat(
        [df_extraction_coal, df_extraction_oil, df_extraction_gas],
        ignore_index=True
    )

    df_extraction_all = df_extraction_all.melt(
        id_vars=['type', 'time', 'fuel', 'stage', 'unit'],  
        var_name='DIM_1',
        value_name='value'
    )

    #Combine fuel and type columns to one column "Tech Type"
    df_extraction_all['Tech Type'] = (df_extraction_all['fuel'].str.strip() + ' ' + df_extraction_all['type'].str.strip())

    df_extraction_all['Tech Type'] = (
        df_extraction_all['Tech Type']
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
        .str.title()
        .replace({
            "Coal Open Cast": "Coal Opencast",
            "Gas Off Shore": "Gas Offshore",
            "Gas On Shore": "Gas Onshore"
        })
    )

    #Drop columns that are not needed for the analysis (stage, unit, type, fuel) since we have already combined the relevant information into the "Tech Type" column and we know that all rows are for the extraction stage and in kg units
    df_extraction_all = df_extraction_all.drop(columns=['stage', 'unit', 'type', 'fuel'])  

# Starting stock (capacity) ------
    df_extraction_all = df_extraction_all.loc[~df_extraction_all['DIM_1'].isin([27,28])]  # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies
    df_extraction_all = df_extraction_all.loc[df_extraction_all['time'].isin(range(year_start, year_end + 1)), ['time', 'DIM_1', 'value', 'Tech Type']]  # only keep relevant years and technology columns
    df_extraction_all['DIM_1'] = df_extraction_all['DIM_1'].astype(int)   

    #Extract coordinate labels for years, regions, technologies
    years = sorted(df_extraction_all['time'].unique())
    regions = sorted(df_extraction_all['DIM_1'].unique())
    techtypes = sorted(df_extraction_all['Tech Type'].unique())

    # Convert to 3D array: (Year, Region, Fuel, Type)
    extractioncap_xr = (
        df_extraction_all
        .set_index(['time', 'DIM_1', 'Tech Type'])
        .to_xarray()['value']
        .rename({'time': 'Time', 'DIM_1': 'Region', 'Tech Type': 'Type'})
    )
    #Add units  
    extractioncap_xr = prism.Q_(extractioncap_xr, "kg")

    #Set order of technology coordinate to match the order in the lifetime and material intensity data, so that we can easily combine the data in the model (since they all have to be in the same order of technologies)
    extraction_order = [
        'Coal Opencast',
        'Coal Underground',
        'Gas Offshore',
        'Gas Onshore',
        'Oil Offshore',
        'Oil Onshore'
    ]

    #Reorder technology coordinate in all three datasets (capacity, lifetime, material intensities) to match the same order of technologies (coal opencast, coal underground, gas offshore, gas onshore, oil offshore, oil onshore), so that we can easily combine the data in the model (since they all have to be in the same order of technologies)
    extractioncap_xr = extractioncap_xr.reindex(Type=extraction_order)

    # print("=== extractioncap_xr ===")
    # print(extractioncap_xr)
    # print("\nDims:", extractioncap_xr.dims)
    # print("Sizes:", extractioncap_xr.sizes)
    # print("\nTime labels:", extractioncap_xr['Time'].values)
    # print("Region labels:", extractioncap_xr['Region'].values)
    # print("Type labels:", extractioncap_xr['Type'].values)
    # extractioncap_xr.isnull().sum()
    
    return extractioncap_xr

#%% Processing stage (coal preparation and oil refinery) ---------------------------------------------------------------------------------------------------------------------------------
def compute_processing_capacity(path_base: str, climate_policy_config: dict, circular_economy_config: dict, scenario: str, year_start: int, year_end: int, year_out: int):

    path_external_data_scenario = Path(path_base, "fossil_fuels", "Scenario_data", scenario)
    # test if path_external_data_scenario exists and if not set to standard scenario
    if not path_external_data_scenario.exists():
        path_external_data_scenario = Path(path_base, "fossil_fuels", STANDARD_SCEN_EXTERNAL_DATA)

    # assert path_external_data_scenario.is_dir()

    #return path_external_data_scenario

    # 2. FUMA output  -----------------------------------------

    #Stock of each type of extraction infrastructure (stock demand per generation technology) per region per year

    coal_drivers = coal_infra()
    preparation_stock_coal = coal_drivers["processing"]

    oil_drivers = oil_infra()
    refinery_stock_oil = oil_drivers["refinery"]

    gas_drivers = gas_infra()
    processing_stock_gas = gas_drivers["processing"]

        # --- FIX INDEX NAMES FIRST (IMPORTANT) ---
    preparation_stock_coal.index.set_names(
        ['type', 'time', 'fuel', 'stage', 'unit'],
        inplace=True
    )

    refinery_stock_oil.index.set_names(
        ['type', 'time', 'fuel', 'stage', 'unit'],
        inplace=True
    )

    processing_stock_gas.index.set_names(
        ['type', 'time', 'fuel', 'stage', 'unit'],
        inplace=True
    )    

    df_processing_coal = preparation_stock_coal.reset_index()
    df_processing_refinery_oil = refinery_stock_oil.reset_index()
    df_processing_gas = processing_stock_gas.reset_index()

    df_processing_coal["fuel"] = "coal"
    df_processing_refinery_oil["fuel"] = "oil"  
    df_processing_gas["fuel"] = "gas"

    #Combine the dataframes for coal, oil, and gas into one dataframe and then melt it to long format so that we have one row per combination of time, region, fuel, technology, and value (stock)
    df_processing_all = pd.concat(
        [df_processing_coal, df_processing_refinery_oil, df_processing_gas],
        ignore_index=True
    )

    df_processing_all = df_processing_all.melt(
        id_vars=['type', 'time', 'fuel', 'stage', 'unit'],  
        var_name='DIM_1',
        value_name='value'
    )

    #Combine fuel and type columns to one column "Tech Type"
    df_processing_all['Tech Type'] = df_processing_all['fuel'].str.strip() + ' ' + df_processing_all['type'].str.strip()

    # Clean up the "Tech Type" column to have consistent formatting and naming, and to match the technology names in the lifetime and material intensity data (since we will need to combine these later in the model, so they all need to have the same technology names)
    df_processing_all['Tech Type'] = (
        df_processing_all['Tech Type']
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
        .str.title()
        .replace({
            "coal preparation": "Coal Preparation",
            "gas processing": "Gas Processing",
            "oil refinery": "Oil Refinery",
        })
    )

    # Drop columns that are not needed for the analysis
    df_processing_all = df_processing_all.drop(columns=['stage', 'unit', 'type', 'fuel'])  

# Gcap ------
    # combine the processing data for coal, oil, gas into one df and then put it into this 
    df_processing_all = df_processing_all.loc[~df_processing_all['DIM_1'].isin([27,28])]  # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies
    df_processing_all = df_processing_all.loc[df_processing_all['time'].isin(range(year_start, year_end + 1)), ['time', 'DIM_1', 'value', 'Tech Type']]  # only keep relevant years and technology columns
    df_processing_all['DIM_1'] = df_processing_all['DIM_1'].astype(int)


    # Extract coordinate labels for years, regions, technologies
    regions = sorted(df_processing_all['DIM_1'].unique())
    techtypes = sorted(df_processing_all['Tech Type'].unique())

    # Convert to 3D array: (Year, Region, Fuel, Type)
    processingcap_xr = (
        df_processing_all
        .set_index(['time', 'DIM_1', 'Tech Type'])
        .to_xarray()['value']
        .rename({'time': 'Time', 'DIM_1': 'Region', 'Tech Type': 'Type'})
    )

    #Add units
    processingcap_xr = prism.Q_(processingcap_xr, "kg")

    # print("=== processingcap_xr ===")
    # print(processingcap_xr)
    # print("\nDims:", processingcap_xr.dims)
    # print("Sizes:", processingcap_xr.sizes)
    # print("\nTime labels:", processingcap_xr['Time'].values)
    # print("Region labels:", processingcap_xr['Region'].values)
    # print("Type labels:", processingcap_xr['Type'].values)
    # processingcap_xr.isnull().sum()

    return processingcap_xr

#%% Processing stage (oil storage) ---------------------------------------------------------------------------------------------------------------------------------
def compute_storage_capacity(path_base: str, climate_policy_config: dict, circular_economy_config: dict, scenario: str, year_start: int, year_end: int, year_out: int):

    path_external_data_scenario = Path(path_base, "fossil_fuels", "Scenario_data", scenario)
    # test if path_external_data_scenario exists and if not set to standard scenario
    if not path_external_data_scenario.exists():
        path_external_data_scenario = Path(path_base, "fossil_fuels", STANDARD_SCEN_EXTERNAL_DATA)

    # assert path_external_data_scenario.is_dir()

    #return path_external_data_scenario

    # 2. FUMA output  -----------------------------------------

    #Stock of each type of extraction infrastructure (stock demand per generation technology) per region per year

    oil_drivers = oil_infra()
    oil_storage = oil_drivers["storage"]

    oil_storage.index.set_names(
        ['type', 'time', 'fuel', 'stage', 'unit'],
        inplace=True
    )

    df_storage_all = oil_storage.reset_index() 

    df_storage_all["fuel"] = "oil"


    #Combine the dataframes for coal, oil, and gas into one dataframe and then melt it to long format so that we have one row per combination of time, region, fuel, technology, and value (stock)
    # df_volume_all = pd.concat(
    #     [df_processing_oil],
    #     ignore_index=True
    # )

    # df_storage_all = df_processing_oil.reset_index(drop=True)

    # print(df_processing_all.columns)
    # print(df_processing_all.head())
    # print(df_processing_all['type'].unique())

    df_storage_all = df_storage_all.melt(
        id_vars=['type', 'time', 'fuel', 'stage', 'unit'],  
        var_name='DIM_1',
        value_name='value'
    )

    #Combine fuel and type columns to one column "Tech Type"
    df_storage_all['Tech Type'] = df_storage_all['fuel'].str.strip() + ' ' + df_storage_all['type'].str.strip()

    # Clean up the "Tech Type" column to have consistent formatting and naming, and to match the technology names in the lifetime and material intensity data (since we will need to combine these later in the model, so they all need to have the same technology names)
    df_storage_all['Tech Type'] = (
        df_storage_all['Tech Type']
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
        .str.title()
        )
    

    # Drop columns that are not needed for the analysis
    df_storage_all = df_storage_all.drop(columns=['stage', 'unit', 'type', 'fuel'])  
# Gcap ------
    # combine the processing data for coal, oil, gas into one df and then put it into this 
    df_storage_all = df_storage_all.loc[~df_storage_all['DIM_1'].isin([27,28])]  # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies
    df_storage_all = df_storage_all.loc[df_storage_all['time'].isin(range(year_start, year_end + 1)), ['time', 'DIM_1', 'value', 'Tech Type']]  # only keep relevant years and technology columns
    df_storage_all['DIM_1'] = df_storage_all['DIM_1'].astype(int)


    # Extract coordinate labels for years, regions, technologies
    regions = sorted(df_storage_all['DIM_1'].unique())
    techtypes = sorted(df_storage_all['Tech Type'].unique())

    # Convert to 3D array: (Year, Region, Fuel, Type)
    storagecap_xr = (
        df_storage_all
        .set_index(['time', 'DIM_1', 'Tech Type'])
        .to_xarray()['value']
        .rename({'time': 'Time', 'DIM_1': 'Region', 'Tech Type': 'Type'})
    )
    #Combine oil crude and oil products into one "oil storage" category 
    oil_storage = storagecap_xr.sel(Type=['Oil Crude', 'Oil Products']).sum(dim='Type')
    storagecap_xr = storagecap_xr.drop_sel(Type=['Oil Crude', 'Oil Products'])
    oil_storage = oil_storage.expand_dims({'Type': ['Oil Storage']})
    storagecap_xr = xr.concat([storagecap_xr, oil_storage], dim='Type')

    #Add units
    storagecap_xr = prism.Q_(storagecap_xr, "meter**3")

    # print("=== storagecap_xr ===")
    # print(storagecap_xr)
    # print("\nDims:", storagecap_xr.dims)
    # print("Sizes:", storagecap_xr.sizes)
    # print("\nTime labels:", processingcap_xr['Time'].values)
    # print("Region labels:", processingcap_xr['Region'].values)
    # print("Type labels:", processingcap_xr['Type'].values)
    # processingcap_xr.isnull().sum()

    return storagecap_xr

#%% Transport/Vehicles stage (coal, oil, gas) ---------------------------------------------------------------------------------------------------------------------------------

###########################################################################################################
def compute_transport_capacity(path_base: str, climate_policy_config: dict, circular_economy_config: dict, scenario: str, year_start: int, year_end: int, year_out: int):

    path_external_data_scenario = Path(path_base, "fossil_fuels", "Scenario_data", scenario)
    # test if path_external_data_scenario exists and if not set to standard scenario
    if not path_external_data_scenario.exists():
        path_external_data_scenario = Path(path_base, "fossil_fuels", STANDARD_SCEN_EXTERNAL_DATA)

    # assert path_external_data_scenario.is_dir()

     # 2. FUMA output -----------------------------------------
    #Transport capacity (stock demand per generation technology) 
    coal_drivers = coal_infra()
    transport_total_coal = coal_drivers["transport"]

    oil_drivers = oil_infra()
    transport_total_oil = oil_drivers["transport"]

    gas_drivers = gas_infra()
    transport_total_gas = gas_drivers["transport"]
    
    transport_total_coal.index.set_names(
        ['type', 'time', 'fuel', 'stage', 'unit'],
        inplace=True
    )
    transport_total_oil.index.set_names(
        ['type', 'time', 'fuel', 'stage', 'unit'],
        inplace=True
    )

    transport_total_gas.index.set_names(
        ['type', 'time', 'fuel', 'stage', 'unit'],
        inplace=True
    )

    df_transport_coal = transport_total_coal.reset_index()
    df_transport_oil = transport_total_oil.reset_index()
    df_transport_gas = transport_total_gas.reset_index()

    df_transport_coal["fuel"] = "coal"
    df_transport_oil["fuel"] = "oil"
    df_transport_gas["fuel"] = "gas"

    #Combine the dataframes for coal, oil, and gas into one dataframe and then melt it to long format so that we have one row per combination of time, region, fuel, technology, and value (stock)
    df_transport_all = pd.concat(
        [df_transport_coal, df_transport_oil, df_transport_gas],
        ignore_index=True
    )

    df_transport_all = df_transport_all.melt(
        id_vars=['type', 'time', 'fuel', 'stage', 'unit'],  
        var_name='DIM_1',
        value_name='value'
    )

    #Combine fuel and type columns to one column "Tech Type"
    df_transport_all['Tech Type'] = df_transport_all['fuel'].str.strip() + ' ' + df_transport_all['type'].str.strip()

    # Clean up the "Tech Type" column to have consistent formatting and naming, and to match the technology names in the lifetime and material intensity data (since we will need to combine these later in the model, so they all need to have the same technology names)
    df_transport_all['Tech Type'] = (
        df_transport_all['Tech Type']
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
        .str.title()
        .replace({
            "Coal Inland Ships": "Coal Inland Ship",
            "Coal Ocean Ships": "Coal Ocean Ship",
            "Coal Rail Cargo": "Coal Rail",
            "Coal Trucks": "Coal Truck",
            "Gas Inland Ships": "Gas Inland Ship",
            "Gas Ocean Ships": "Gas Ocean Ship",
            "Gas Rail Cargo": "Gas Rail",
            "Gas Trucks": "Gas Truck",
            "Oil Inland Ships": "Oil Inland Ship",
            "Oil Ocean Ships": "Oil Ocean Ship",
            "Oil Rail Cargo": "Oil Rail",
            "Oil Trucks": "Oil Truck"
        })
    )

    #Drop columns that are not needed for the analysis 
    df_transport_all = df_transport_all.drop(columns=['stage', 'unit', 'type', 'fuel']) 
# Transport capacity ------
    df_transport_all = df_transport_all.loc[~df_transport_all['DIM_1'].isin([27,28])]  # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies
    df_transport_all = df_transport_all.loc[df_transport_all['time'].isin(range(year_start, year_end + 1)), ['time', 'DIM_1', 'value', 'Tech Type']]  # only keep relevant years and technology columns
    df_transport_all['DIM_1'] = df_transport_all['DIM_1'].astype(int)

    # Extract coordinate labels
    years = sorted(df_transport_all['time'].unique())
    regions = sorted(df_transport_all['DIM_1'].unique())
    techtypes = sorted(df_transport_all['Tech Type'].unique())

    transportcap_xr = (
        df_transport_all
        .set_index(['time', 'DIM_1', 'Tech Type'])
        .to_xarray()['value']
        .rename({'time': 'Time', 'DIM_1': 'Region', 'Tech Type': 'Type'})
    )
    #Add units
    transportcap_xr = prism.Q_(transportcap_xr, "kg")

    # print("=== transportcap_xr ===")
    # print(transportcap_xr)
    # print("\nDims:", transportcap_xr.dims)
    # print("Sizes:", transportcap_xr.sizes)
    # print("\nTime labels:", transportcap_xr['Time'].values)
    # print("Region labels:", transportcap_xr['Region'].values)
    # print("Type labels:", transportcap_xr['Type'].values)
    # transportcap_xr.isnull().sum()

    return transportcap_xr

#%% Pipelines stage (oil, gas) ---------------------------------------------------------------------------------------------------------------------------------
###########################################################################################################
on_offshore_oil_pipeline = pd.read_csv(
    DATA_DIR / "DriversFiles" / "onshore_offshore_oil_pipeline.csv", index_col=[0]

)  # Share of 2 different pipeline types (onshore & offshore) global (%)

def compute_pipelines_capacity(path_base: str, climate_policy_config: dict, circular_economy_config: dict, scenario: str, year_start: int, year_end: int, year_out: int):

    path_external_data_scenario = Path(path_base, "fossil_fuels", "Scenario_data", scenario)
    # test if path_external_data_scenario exists and if not set to standard scenario
    if not path_external_data_scenario.exists():
        path_external_data_scenario = Path(path_base, "fossil_fuels", STANDARD_SCEN_EXTERNAL_DATA)

    # assert path_external_data_scenario.is_dir()

    # 2. FUMA output -----------------------------------------

    #Transport capacity (stock demand per generation technology)
    #Load data from drivers.py
    oil_drivers = oil_infra()
    oil_pipelines = oil_drivers["pipelines"]

    gas_drivers = gas_infra()
    gas_pipelines = gas_drivers["pipelines"]

    oil_pipelines.index.set_names(
        ['type', 'time', 'fuel', 'stage', 'unit'],
        inplace=True
    )

    gas_pipelines.index.set_names(
        ['type', 'time', 'fuel', 'stage', 'unit'],
        inplace=True
    )    

    df_pipelines_oil = oil_pipelines.reset_index()
    df_pipelines_gas = gas_pipelines.reset_index()

    df_pipelines_oil["fuel"] = "oil"
    df_pipelines_gas["fuel"] = "gas"

    #Combine the dataframes for oil and gas into one dataframe and then melt it to long format so that we have one row per combination of time, region, fuel, technology, and value (stock)
    df_pipelines_all = pd.concat(
        [df_pipelines_oil, df_pipelines_gas],
        ignore_index=True
    )

    df_pipelines_all = df_pipelines_all.melt(
        id_vars=['type', 'time', 'fuel', 'stage', 'unit'],  
        var_name='DIM_1',
        value_name='value'
    )

    #Combine fuel and type columns to one column "Tech Type"
    df_pipelines_all['Tech Type'] = df_pipelines_all['fuel'].str.strip() + ' ' + df_pipelines_all['type'].str.strip()

    #Drop columns that are not needed for the analysis
    df_pipelines_all = df_pipelines_all.drop(columns=['stage', 'unit', 'type', 'fuel'])  # drop columns that are not needed for the analysis

    # Clean up the "Tech Type" column to have consistent formatting and naming, and to match the technology names in the lifetime and material intensity data (since we will need to combine these later in the model, so they all need to have the same technology names)
    df_pipelines_all['Tech Type'] = (
        df_pipelines_all['Tech Type']
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
        .str.title()
        .replace({
            "Gas Distribution": "Gas Distribution Pipeline",
            "Gas Transmission": "Gas Transmission Pipeline"
            # "Oil Crude": "Oil Crude Pipeline",
            # "Oil Product": "Oil Product Pipeline"
        })
    )

    # Pipeline capacity ------
    df_pipelines_all = df_pipelines_all.loc[~df_pipelines_all['DIM_1'].isin([27,28])]  # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies
    df_pipelines_all = df_pipelines_all.loc[df_pipelines_all['time'].isin(range(year_start, year_end + 1)), ['time', 'DIM_1', 'value', 'Tech Type']]  # only keep relevant years and technology columns
    df_pipelines_all['DIM_1'] = df_pipelines_all['DIM_1'].astype(int)

        # Extract coordinate labels
    years = sorted(df_pipelines_all['time'].unique())
    regions = sorted(df_pipelines_all['DIM_1'].unique())
    techtypes = sorted(df_pipelines_all['Tech Type'].unique())

    # Convert to 3D array: (Year, Region, Fuel, Type)
    pipelinecap_xr = (
        df_pipelines_all
        .set_index(['time', 'DIM_1', 'Tech Type'])
        .to_xarray()['value']
        .rename({'time': 'Time', 'DIM_1': 'Region', 'Tech Type': 'Type'})
    )

    pipelinecap_xr = pipelinecap_xr.assign_coords(
        Type=np.array(pipelinecap_xr.Type.values, dtype=object)
    )

    # Since the oil pipelines in the data are not split into onshore and offshore pipelines, we will split them based on the fractions of onshore and offshore pipelines in the global pipeline network f

    # First we identify the types of oil pipelines in the capacity data (oil crude and oil products) 
    oil_types = ['Oil Crude', 'Oil Product']  


    # Then we create new entries for onshore and offshore oil pipelines based on the fractions in the global pipeline network
    expanded_oil_arrays = []
    for oil_type in oil_types:
        key = oil_type.split()[1] 
        for division, fraction in on_offshore_oil_pipeline[key].items():
            expanded_name = f"Oil {division.capitalize()} {key} Pipeline"
            new_arr = pipelinecap_xr.sel(Type=oil_type) * fraction
            new_arr = new_arr.expand_dims(Type=[expanded_name])
            expanded_oil_arrays.append(new_arr)

    # Add new oil pipeline types to the original xarray by concatenating along the 'Type' dimension
    pipelinecap_xr = xr.concat([pipelinecap_xr] + expanded_oil_arrays, dim='Type')

    # Drop original oil pipeline types since they have now been split into onshore and offshore pipelines
    pipelinecap_xr = pipelinecap_xr.drop_sel(Type=oil_types)

    # Add units
    pipelinecap_xr = prism.Q_(pipelinecap_xr, "km")

    #Correct the order
    pipeline_order = [
        'Gas Distribution Pipeline',
        'Gas Transmission Pipeline',
        'Oil Offshore Crude Pipeline',
        'Oil Onshore Crude Pipeline',
        'Oil Offshore Product Pipeline',
        'Oil Onshore Product Pipeline'
    ]

    #Reorder technology coordinate in all three datasets (capacity, lifetime, material intensities) to match the same order of technologies (coal opencast, coal underground, gas offshore, gas onshore, oil offshore, oil onshore), so that we can easily combine the data in the model (since they all have to be in the same order of technologies)
    pipelinecap_xr = pipelinecap_xr.reindex(Type=pipeline_order)

    # print("=== pipelinecap_xr ===")
    # print(pipelinecap_xr)
    # print("\nDims:", pipelinecap_xr.dims)
    # print("Sizes:", pipelinecap_xr.sizes)
    # print("\nTime labels:", pipelinecap_xr['Time'].values)
    # print("Region labels:", pipelinecap_xr['Region'].values)
    # print("Type labels:", pipelinecap_xr['Type'].values)
    # pipelinecap_xr.isnull().sum()

    return pipelinecap_xr
