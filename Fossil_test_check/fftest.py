 
#%% Outline of the pre-processing steps for fossil fuels (coal, oil, gas) ---------------------------------------------------------------------------------------------------------------------------------

# #1: Import module and constants
# a General imports/packages
# b Image material imports
# c Make variable for pre-processing data 
# d Make variable for path to external data for the scenario, and test if it exists, if not set to standard scenario

#2: Read in files for the extraction stage for each fuel (coal, oil, gas)
# a lifetimes 
# b material intensities (steel, alumnium, copper?, I could add other ones?)
# C image/timer data (this is where I am unsure) but I know we need a step to actually give information about the quantity of this type of infrastructure
# d Transform to xarray and add units using the knowledge graph 

#3: Read in files for the processing stage for each fuel
# a lifetimes 
# b material intensities
# C image/timer data (made from stock calculation (FUMA) output)
# d Transform to xarray and add units using the knowledge graph 

#4: Read in files for the transport stage for each fuel
# a lifetimes 
# b material intensities
# C image/timer data (made from stock calculation (FUMA) output)
# d Transform to xarray and add units using the knowledge graph 

#5: Read in files for the pipelines stage for each fuel
# a lifetimes 
# b material intensities
# C image/timer data (made from stock calculation (FUMA) output)
# d Transform to xarray and add units using the knowledge graph 

#6 interpolate data (i am not sure if this will be necessary)

#7 Make prep_data dictionary with all the pre-processed data, and return it

# 8 Adjust as necessary to fit into the IMAGE framework (so its runs with ELMA)

# 9 Make output figures 


#%% Import modules and constants
import pandas as pd
import numpy as np
from pathlib import Path
import pint
import xarray as xr
import prism

from ffconstants import FF_TECHNOLOGIES, STANDARD_SCEN_EXTERNAL_DATA, YEAR_FIRST_GRID, STD_LIFETIMES_ELECTR

from imagematerials.read_mym import read_mym_df
from imagematerials.util import dataset_to_array, pandas_to_xarray, convert_lifetime
from imagematerials.concepts import create_electricity_graph, create_region_graph
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
scenario = "SSP1_ML"

scen_folder = "SSP1_ML"

path_current = Path().resolve()
path_base = path_current.parent # base path of the project -> image-materials
print("current:", path_current)
print("base:", path_base)
climate_policy_scenario_dir = Path(path_base, "data", "raw", "image", scen_folder)


STANDARD_SCEN_EXTERNAL_DATA = "SSP1_ML"
SCENARIO_DEFAULT = "SSP1_ML"
year_start = 2020
year_end = 2100
year_out = 2100


#%% Extraction stage (coal, oil, gas) ---------------------------------------------------------------------------------------------------------------------------------

###########################################################################################################
###########################################################################################################
# Extraction stage (coal, oil, gas) ---------------------------------------------------------------------------------------------------------------------------------
#def get_preprocessing_data_extraction(path_base: str, climate_policy_config: dict, scenario: str, year_start: int, year_end: int, year_out: int):

   # path_external_data_scenario = Path(path_base, "fossil", scenario)

    # If scenario folder does not exist, fall back to standard scenario
   # if not path_external_data_scenario.exists():
    #    path_external_data_scenario = Path(path_base, "fossil", STANDARD_SCEN_EXTERNAL_DATA)

    #assert path_external_data_scenario.is_dir()

    #return path_external_data_scenario

###########################################################################################################
    # Read in files #

    # 1. External Data --------------------------------------------- 

    # lifetimes of extraction tech in years
extraction_lifetime_data = pd.read_csv(
    'Extraction/FF Intensity & LT - Extraction lifetimes.csv',
    index_col=["Year", "Tech Type"]
) 

print(extraction_lifetime_data.head())
print(extraction_lifetime_data.columns)
print(extraction_lifetime_data.index)
    # material compositions of coal infrastructure (processing, extraction, transport) in kg/kg/year
extraction_materials_data = pd.read_csv('Extraction/FF Intensity & LT - Extraction materials.csv')

#print(extraction_materials_data.columns)
#print(extraction_materials_data.head())

    # 2. IMAGE/TIMER files -----------------------------------------
#Stock of each type of extraction infrastructure (stock demand per generation technology) per region per year

extraction_coal = Path(path_base, "Fossil_test", "stock_calculation", "output", scenario, "coal_extraction_stock_kg.csv")
df_extraction_coal = pd.read_csv(extraction_coal)
extraction_oil = Path(path_base, "Fossil_test", "stock_calculation", "output", scenario, "oil_extraction_stock_kg.csv")
df_extraction_oil = pd.read_csv(extraction_oil)
extraction_gas = Path(path_base, "Fossil_test", "stock_calculation", "output", scenario, "gas_extraction_stock_kg.csv")
df_extraction_gas = pd.read_csv(extraction_gas)
df_extraction_coal["fuel"] = "coal"
df_extraction_oil["fuel"] = "oil"
df_extraction_gas["fuel"] = "gas"

df_extraction_all = pd.concat(
    [df_extraction_coal, df_extraction_oil, df_extraction_gas],
    ignore_index=True
)

df_extraction_all = df_extraction_all.melt(
    id_vars=['type', 'time', 'fuel', 'stage', 'unit'],  
    var_name='DIM_1',
    value_name='value'
)

df_extraction_all = df_extraction_all.drop(columns=['stage', 'unit'])


#print(df_extraction_all.shape)
#print(df_extraction_all.columns.tolist())

#print("COLUMNS:", df_extraction_all.columns)
#print(df_extraction_all.head())

   # test if path_external_data_scenario exists and if not set to standard scenario
#if not path_external_data_scenario.exists():
            #path_external_data_scenario = Path(path_base, "fossil", STANDARD_SCEN_EXTERNAL_DATA)

#assert path_external_data_scenario.is_dir()

    ###########################################################################################################
    # Transform to xarray #
#knowledge_graph_region = create_region_graph()
#knowledge_graph_coal = create_ff_graph()
    

    # Lifetimes -------
values = extraction_lifetime_data["Lifetime"].unstack().to_numpy(dtype=float)
#Create coordinates
times = extraction_lifetime_data.index.get_level_values("Year").unique().to_numpy()
types = extraction_lifetime_data.index.get_level_values("Tech Type").unique().to_numpy()
scipy_params = ["mean", "stdev"]
    # Build full array: shape (ScipyParam, Time, Type)
data_array = np.stack([values, np.full_like(values, np.nan)], axis=0)
    # Create DataArray
extraction_lifetime_xr = xr.DataArray(
        data_array,
        dims=["DistributionParams", "Cohort", "Type"],
        coords={
            "DistributionParams": scipy_params,
            "Cohort": times,
            "Type": [str(r) for r in types]
        },
        name="ExtractionLifetime"
    )
extraction_lifetime_xr = prism.Q_(extraction_lifetime_xr, "year")

#TODO: try to interpolate the extraction lifetime even it is just for the same value the whole time 

    #coal_lifetime_xr = knowledge_graph_electr.rebroadcast_xarray(coal_lifetime_xr, output_coords=EPG_TECHNOLOGIES, dim="Type") # convert technology names to the standard names from TIMER
    #coal_lifetime_xr = coal_lifetime_xr.assign_coords(Type=np.array(coal_lifetime_xr.Type.values, dtype=object)) # rebroadcast_xarray changes the type of the coordinates to numpy strings (np.str_), so convert back to python strings (str)

 # Material Intensities -------

    # Material Intensities -------
   # Set index
extraction_materials_data = extraction_materials_data.set_index(['Year', 'Tech Type'])

   # Convert to 3D array: (Material, Year, Tech)
extraction_materials_xr = (
    extraction_materials_data
        .to_xarray()            
        .to_array("Material") 
        .rename({"Year": "Cohort", "Tech Type": "Type"})
)

# Name the DataArray
extraction_materials_xr.name = "ExtractionMaterialIntensities"

extraction_materials_xr = prism.Q_(extraction_materials_xr, "kg/kg/year")
#print(extraction_materials_xr)
#extraction_materials_xr = knowledge_graph_coal.rebroadcast_xarray(extraction_materials_xr, output_coords=EPG_TECHNOLOGIES, dim="Type")
#extraction_materials_xr = extraction_materials_xr.assign_coords(Type=np.array(extraction_materials_xr.Type.values, dtype=object)) # rebroadcast_xarray changes the type of the coordinates to numpy strings (np.str_), so convert back to python strings (str)

# Starting stock (capacity) ------
#df_extraction_all = df_extraction_all.loc[~df_extraction_all['DIM_1'].isin([27,28])]  # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies
#df_extraction_all = df_extraction_all.loc[df_extraction_all['time'].isin(range(year_start, year_end + 1)), ['time', 'DIM_1', *FF_TECHNOLOGIES]]  # only keep relevant years and technology columns
    # Extract coordinate labels
years = sorted(df_extraction_all['time'].unique())
regions = sorted(df_extraction_all['DIM_1'].unique())
fuels = sorted(df_extraction_all['fuel'].unique())
types = sorted(df_extraction_all['type'].unique())
#print(df_extraction_all['type'].unique())
#df_sorted = df_extraction_all.sort_values(['time', 'DIM_1', 'fuel', 'type'])
    # Convert to 3D array: (Year, Region, Fuel, Type)
#data_array = df_sorted['value'].to_numpy().reshape(len(years), len(regions), len(fuels), len(types))
extractioncap_xr = (
    df_extraction_all
    .set_index(['time', 'DIM_1', 'fuel', 'type'])
    .to_xarray()['value']
    .rename({'time': 'Time', 'DIM_1': 'Region', 'fuel': 'Fuel', 'type': 'Type'})
)
    # Build xarray DataArray
#extractioncap_xr = xr.DataArray(
        #data_array,
        #dims=('Time', 'Region', 'Fuel', 'Type'),
        #coords={
            #'Time': years,
            #'Region': [str(r) for r in regions],
            #'Fuel': [str(r) for r in fuels],
            #'Type': [str(r) for r in types]
       # },
        #name='ExtractionCapacity'
   # )
#extractioncap_xr = prism.Q_(extractioncap_xr, "MW")
#extractioncap_xr = knowledge_graph_region.rebroadcast_xarray(extractioncap_xr, output_coords=IMAGE_REGIONS, dim="Region") 
#extractioncap_xr = knowledge_graph_coal.rebroadcast_xarray(extractioncap_xr, output_coords=FF_TECHNOLOGIES, dim="Type")
extractioncap_xr = extractioncap_xr.assign_coords(Type=np.array(extractioncap_xr.Type.values, dtype=object)) # rebroadcast_xarray changes the type of the coordinates to numpy strings (np.str_), so convert back to python strings (str)

    ###########################################################################################################
    # Interpolate #

    # interpolate_xr: The lifetimes & material intensities are only given for specific years (2020 and 2050), so we linearly interpolate to get values for the years 2020-2050.
    # The values before 2020 are kept constant at the 2020 level, and the values after 2050 are kept constant at the 2050 level.
extraction_lifetime_xr_interp = interpolate_xr(extraction_lifetime_xr, YEAR_FIRST_GRID, year_out)
extraction_lifetime_xr_interp.loc[dict(DistributionParams="stdev")] = extraction_lifetime_xr_interp.loc[dict(DistributionParams="mean")] * STD_LIFETIMES_ELECTR
extraction_materials_xr_interp = interpolate_xr(extraction_materials_xr, YEAR_FIRST_GRID, year_out)

    # TIMER data only start in 1971, so we add a historic tail back to YEAR_FIRST_GRID=1921 #TODO to be adjusted
extractioncap_xr_interp = add_historic_stock(extractioncap_xr, YEAR_FIRST_GRID)

print(extraction_lifetime_xr_interp.head())
print(extraction_materials_xr_interp.head())
print(extractioncap_xr_interp.head())

#%% Processing stage (coal, oil, gas) ---------------------------------------------------------------------------------------------------------------------------------

###########################################################################################################
###########################################################################################################
# Processing stage (coal, oil, gas) ---------------------------------------------------------------------------------------------------------------------------------
def get_preprocessing_data_processing(path_base: str, climate_policy_config: dict, scenario: str, year_start: int, year_end: int, year_out: int):

    path_external_data_scenario = Path(path_base, "fossil", scenario)

    # test if path_external_data_scenario exists and if not set to standard scenario
    if not path_external_data_scenario.exists():
        path_external_data_scenario = Path(path_base, "fossil", STANDARD_SCEN_EXTERNAL_DATA)

    assert path_external_data_scenario.is_dir()

    return path_external_data_scenario
###########################################################################################################
    # Read in files #

    # 1. External Data --------------------------------------------- 

    # lifetimes of processing tech in years
processing_lifetime_data = pd.read_csv(
    'Processing/FF Intensity & LT - Processing lifetimes.csv',
    index_col=["Year", "Tech Type"]
) 
    # material compositions of processing infrastructure in kg/kg/year
processing_materials_data = pd.read_csv('Processing/FF Intensity & LT - Processing materials.csv')

#print(extraction_materials_data.columns)
#print(extraction_materials_data.head())

    # 2. IMAGE/TIMER files -----------------------------------------
#Processing capacity (stock demand per generation technology) in MW peak capacity
#Stock of each type of extraction infrastructure (stock demand per generation technology) per region per year

processing_coal = Path(path_base, "Fossil_test", "stock_calculation", "output", scenario, "coal_preparation_stock_kg.csv")
df_processing_coal = pd.read_csv(processing_coal)
processing_storage_oil = Path(path_base, "Fossil_test", "stock_calculation", "output", scenario, "oil_storage_volume_m3.csv")
df_processing_oil = pd.read_csv(processing_storage_oil)
processing_refinery_oil = Path(path_base, "Fossil_test", "stock_calculation", "output", scenario, "oil_refinery_stock_kg.csv")
df_processing_refinery_oil = pd.read_csv(processing_refinery_oil)
processing_gas = Path(path_base, "Fossil_test", "stock_calculation", "output", scenario, "gas_processing_stock_kg.csv")
df_processing_gas = pd.read_csv(processing_gas)
df_processing_all = pd.concat(
    [df_processing_coal, df_processing_oil, df_processing_refinery_oil, df_processing_gas],
    ignore_index=True
)


   # test if path_external_data_scenario exists and if not set to standard scenario
#if not path_external_data_scenario.exists():
 #        path_external_data_scenario = Path(path_base, "fossil", STANDARD_SCEN_EXTERNAL_DATA)
 
#assert path_external_data_scenario.is_dir()

    ###########################################################################################################
    # Transform to xarray #
#knowledge_graph_region = create_region_graph()
#knowledge_graph_coal = create_ff_graph()
    

    # Lifetimes -------
values = processing_lifetime_data["Lifetime"].unstack().to_numpy(dtype=float)
#Create coordinates
times = processing_lifetime_data.index.levels[0].to_numpy()
types = processing_lifetime_data.index.levels[1].to_numpy()
scipy_params = ["mean", "stdev"]
    # Build full array: shape (ScipyParam, Time, Type)
data_array = np.stack([values, np.full_like(values, np.nan)], axis=0)
    # Create DataArray
processing_lifetime_xr = xr.DataArray(
        data_array,
        dims=["DistributionParams", "Cohort", "Type"],
        coords={
            "DistributionParams": scipy_params,
            "Cohort": times,
            "Type": [str(r) for r in types]
        },
        name="ProcessingLifetime"
    )
processing_lifetime_xr = prism.Q_(processing_lifetime_xr, "year")
    #coal_lifetime_xr = knowledge_graph_electr.rebroadcast_xarray(coal_lifetime_xr, output_coords=EPG_TECHNOLOGIES, dim="Type") # convert technology names to the standard names from TIMER
    #coal_lifetime_xr = coal_lifetime_xr.assign_coords(Type=np.array(coal_lifetime_xr.Type.values, dtype=object)) # rebroadcast_xarray changes the type of the coordinates to numpy strings (np.str_), so convert back to python strings (str)

 # Material Intensities -------

    # Material Intensities -------
   # Set index
processing_materials_data = processing_materials_data.set_index(['Year', 'Tech Type'])

   # Convert to 3D array: (Material, Year, Tech)
processing_materials_xr = (
    processing_materials_data
        .to_xarray()           # keeps Year & Tech Type as coords
        .to_array("Material")  # converts columns into a 'Material' dimension
        .rename({"Year": "Cohort", "Tech Type": "Type"})
)

# Name the DataArray
processing_materials_xr.name = "ProcessingMaterialIntensities"

processing_materials_xr = prism.Q_(processing_materials_xr, "kg/kg/year")
#print(processing_materials_xr)
#processing_materials_xr = knowledge_graph_coal.rebroadcast_xarray(processing_materials_xr, output_coords=EPG_TECHNOLOGIES, dim="Type")
#processing_materials_xr = processing_materials_xr.assign_coords(Type=np.array(processing_materials_xr.Type.values, dtype=object)) # rebroadcast_xarray changes the type of the coordinates to numpy strings (np.str_), so convert back to python strings (str)

# Gcap ------

# combine the processing data for coal, oil, gas into one df and then put it into this 

df_processing_all

df_processing_all = df_processing_all.loc[~df_processing_all['DIM_1'].isin([27,28])]  # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies
df_processing_all  = df_processing_all .loc[df_processing_all ['time'].isin(range(year_start, year_end + 1)), ['time', 'DIM_1', *range(1, len(FF_TECHNOLOGIES)+1)]]  # only keep relevant years and technology columns
    # Extract coordinate labels
years = sorted(df_processing_all ['time'].unique())
regions = sorted(df_processing_all ['DIM_1'].unique())
techs = list(range(1, len(FF_TECHNOLOGIES)+1))
    # Convert to 3D array: (Year, Region, Tech)
data_array = df_processing_all [techs].to_numpy().reshape(len(years), len(regions), len(techs))
    # Build xarray DataArray
processing_capacities_xr = xr.DataArray(
        data_array,
        dims=('Time', 'Region', 'Type'),
        coords={
            'Time': years,
            'Region': [str(r) for r in regions],
            'Type': [str(r) for r in techs]
        },
        name='ProcessingCapacities'
    )
processing_capacities_xr = prism.Q_(processing_capacities_xr, "MW")
#processing_capacities_xr = knowledge_graph_region.rebroadcast_xarray(processing_capacities_xr, output_coords=IMAGE_REGIONS, dim="Region") 
#processing_capacities_xr = knowledge_graph_coal.rebroadcast_xarray(processing_capacities_xr, output_coords=FF_TECHNOLOGIES, dim="Type")
processing_capacities_xr = processing_capacities_xr.assign_coords(Type=np.array(processing_capacities_xr.Type.values, dtype=object)) # rebroadcast_xarray changes the type of the coordinates to numpy strings (np.str_), so convert back to python strings (str)

    ###########################################################################################################
    # Interpolate #

    # interpolate_xr: The lifetimes & material intensities are only given for specific years (2020 and 2050), so we linearly interpolate to get values for the years 2020-2050.
    # The values before 2020 are kept constant at the 2020 level, and the values after 2050 are kept constant at the 2050 level.
processing_lifetime_xr_interp = interpolate_xr(processing_lifetime_xr, YEAR_FIRST_GRID, year_out)
processing_lifetime_xr_interp.loc[dict(DistributionParams="stdev")] = processing_lifetime_xr_interp.loc[dict(DistributionParams="mean")] * STD_LIFETIMES_ELECTR
processing_materials_xr_interp = interpolate_xr(processing_materials_xr, YEAR_FIRST_GRID, year_out)

    # TIMER data only start in 1971, so we add a historic tail back to YEAR_FIRST_GRID=1921 #TODO to be adjusted
processing_capacities_xr_interp = add_historic_stock(processing_capacities_xr, YEAR_FIRST_GRID)

#%% Transport/Vehicles stage (coal, oil, gas) ---------------------------------------------------------------------------------------------------------------------------------

###########################################################################################################
###########################################################################################################
# Transport/Vehicles stage (coal, oil, gas) ---------------------------------------------------------------------------------------------------------------------------------
def get_preprocessing_data_processing(path_base: str, climate_policy_config: dict, scenario: str, year_start: int, year_end: int, year_out: int):

    #path_external_data_scenario = Path(path_base, "fossil", scenario)

    # test if path_external_data_scenario exists and if not set to standard scenario
   # if not path_external_data_scenario.exists():
    #    path_external_data_scenario = Path(path_base, "fossil", STANDARD_SCEN_EXTERNAL_DATA)

    #assert path_external_data_scenario.is_dir()
###########################################################################################################
    # Read in files #

    # 1. External Data --------------------------------------------- 

    # lifetimes of transport tech in years
 transport_lifetime_data = pd.read_csv(
    'Transport/FF Intensity & LT - Transport lifetimes.csv',
    index_col=["Year", "Tech Type"]
) 
    # material compositions of transport infrastructure in kg/kg/year
transport_materials_data = pd.read_csv('Transport/FF Intensity & LT - Transport materials.csv')
transport_lifetime_data = pd.read_csv('Transport/FF Intensity & LT - Transport Lifetime.csv', index_col=["Year", "Tech Type"])

#print(extraction_materials_data.columns)
#print(extraction_materials_data.head())

    # 2. IMAGE/TIMER files -----------------------------------------
#Transport capacity (stock demand per generation technology) in MW peak capacity
transport_coal = Path(path_base, "Fossil_test", "stock_calculation", "output", scenario, "coal_transport_stock_kg.csv")
df_transport_coal = pd.read_csv(transport_coal)
transport_oil = Path(path_base, "Fossil_test", "stock_calculation", "output", scenario, "oil_transport_stock_kg.csv")
df_transport_oil = pd.read_csv(transport_oil)
transport_gas = Path(path_base, "Fossil_test", "stock_calculation", "output", scenario, "gas_transport_stock_kg.csv")
df_transport_gas = pd.read_csv(transport_gas)
df_transport_coal["fuel"] = "coal"
df_transport_oil["fuel"] = "oil"
df_transport_gas["fuel"] = "gas"

df_transport_all = pd.concat(
    [df_transport_coal, df_transport_oil, df_transport_gas],
    ignore_index=True
)

   # test if path_external_data_scenario exists and if not set to standard scenario
# if not path_external_data_scenario.exists():
  #       path_external_data_scenario = Path(path_base, "fossil", STANDARD_SCEN_EXTERNAL_DATA)
# 
# assert path_external_data_scenario.is_dir()

    ###########################################################################################################
    # Transform to xarray #
#knowledge_graph_region = create_region_graph()
#knowledge_graph_coal = create_ff_graph()
    

    # Lifetimes -------
values = transport_lifetime_data["Lifetime"].unstack().to_numpy(dtype=float)
#Create coordinates
times = transport_lifetime_data.index.levels[0].to_numpy()
types = transport_lifetime_data.index.levels[1].to_numpy()
scipy_params = ["mean", "stdev"]
    # Build full array: shape (ScipyParam, Time, Type)
data_array = np.stack([values, np.full_like(values, np.nan)], axis=0)
    # Create DataArray
transport_lifetime_xr = xr.DataArray(
        data_array,
        dims=["DistributionParams", "Cohort", "Type"],
        coords={
            "DistributionParams": scipy_params,
            "Cohort": times,
            "Type": [str(r) for r in types]
        },
        name="TransportLifetime"
    )
transport_lifetime_xr = prism.Q_(transport_lifetime_xr, "year")
    #coal_lifetime_xr = knowledge_graph_electr.rebroadcast_xarray(coal_lifetime_xr, output_coords=EPG_TECHNOLOGIES, dim="Type") # convert technology names to the standard names from TIMER
    #coal_lifetime_xr = coal_lifetime_xr.assign_coords(Type=np.array(coal_lifetime_xr.Type.values, dtype=object)) # rebroadcast_xarray changes the type of the coordinates to numpy strings (np.str_), so convert back to python strings (str)

 # Material Intensities -------

    # Material Intensities -------
   # Set index
transport_materials_data = transport_materials_data.set_index(['Year', 'Tech Type'])

   # Convert to 3D array: (Material, Year, Tech)
transport_materials_xr = (
    transport_materials_data
        .to_xarray()           # keeps Year & Tech Type as coords
        .to_array("Material")  # converts columns into a 'Material' dimension
        .rename({"Year": "Cohort", "Tech Type": "Type"})
)

# Name the DataArray
transport_materials_xr.name = "TransportMaterialIntensities"

transport_materials_xr = prism.Q_(transport_materials_xr, "kg/kg/year")
#print(transport_materials_xr)
#transport_materials_xr = knowledge_graph_coal.rebroadcast_xarray(transport_materials_xr, output_coords=EPG_TECHNOLOGIES, dim="Type")
#transport_materials_xr = transport_materials_xr.assign_coords(Type=np.array(transport_materials_xr.Type.values, dtype=object)) # rebroadcast_xarray changes the type of the coordinates to numpy strings (np.str_), so convert back to python strings (str)

# Transport capacity ------
df_transport_all = df_transport_all.loc[~df_transport_all['Region'].isin([27,28])]  # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies
df_transport_all = df_transport_all.loc[df_transport_all['time'].isin(range(year_start, year_end + 1)), ['time', 'Region', *range(1, len(FF_TECHNOLOGIES) + 1)]]  # only keep relevant years and technology columns
    # Extract coordinate labels
years = sorted(df_transport_all['time'].unique())
regions = sorted(df_transport_all['Region'].unique())
techs = list(range(1, len(FF_TECHNOLOGIES)+1))
    # Convert to 3D array: (Year, Region, Tech)
data_array = df_transport_all[techs].to_numpy().reshape(len(years), len(regions), len(techs))
    # Build xarray DataArray
transportcap_xr = xr.DataArray(
        data_array,
        dims=('Time', 'Region', 'Type'),
        coords={
            'Time': years,
            'Region': [str(r) for r in regions],
            'Type': [str(r) for r in techs]
        },
        name='TransportCap'
    )
transportcap_xr = prism.Q_(transportcap_xr, "MW")
#transportcap_xr = knowledge_graph_region.rebroadcast_xarray(transportcap_xr, output_coords=IMAGE_REGIONS, dim="Region") 
#transportcap_xr = knowledge_graph_coal.rebroadcast_xarray(transportcap_xr, output_coords=FF_TECHNOLOGIES, dim="Type")
transportcap_xr = transportcap_xr.assign_coords(Type=np.array(transportcap_xr.Type.values, dtype=object)) # rebroadcast_xarray changes the type of the coordinates to numpy strings (np.str_), so convert back to python strings (str)

    ###########################################################################################################
    # Interpolate #

    # interpolate_xr: The lifetimes & material intensities are only given for specific years (2020 and 2050), so we linearly interpolate to get values for the years 2020-2050.
    # The values before 2020 are kept constant at the 2020 level, and the values after 2050 are kept constant at the 2050 level.
transport_lifetime_xr_interp = interpolate_xr(transport_lifetime_xr, YEAR_FIRST_GRID, year_out)
transport_lifetime_xr_interp.loc[dict(DistributionParams="stdev")] = transport_lifetime_xr_interp.loc[dict(DistributionParams="mean")] * STD_LIFETIMES_ELECTR
transport_materials_xr_interp = interpolate_xr(transport_materials_xr, YEAR_FIRST_GRID, year_out)

    # TIMER data only start in 1971, so we add a historic tail back to YEAR_FIRST_GRID=1921 #TODO to be adjusted
transportcap_xr_interp = add_historic_stock(transportcap_xr, YEAR_FIRST_GRID)

#%% Pipelines stage (oil, gas) ---------------------------------------------------------------------------------------------------------------------------------

###########################################################################################################
###########################################################################################################
# Pipelines stage (oil, gas) ---------------------------------------------------------------------------------------------------------------------------------
# def get_preprocessing_data_processing(path_base: str, climate_policy_config: dict, scenario: str, year_start: int, year_end: int, year_out: int):

    # path_external_data_scenario = Path(path_base, "fossil", scenario)
    # test if path_external_data_scenario exists and if not set to standard scenario
    # if not path_external_data_scenario.exists():
        # path_external_data_scenario = Path(path_base, "fossil", STANDARD_SCEN_EXTERNAL_DATA)

   #  assert path_external_data_scenario.is_dir()
###########################################################################################################
    # Read in files #

    # 1. External Data --------------------------------------------- 

    # lifetimes of transport tech in years
pipelines_lifetime_data = pd.read_csv(
    'Pipelines/FF Intensity & LT - Pipelines lifetimes.csv',
    index_col=["Year", "Tech Type"]
) 
    # material compositions of transport infrastructure in kg/kg/year
pipelines_materials_data = pd.read_csv('Pipelines/FF Intensity & LT - Pipelines materials.csv')
#print(extraction_materials_data.columns)
#print(extraction_materials_data.head())

    # 2. IMAGE/TIMER files -----------------------------------------
#Transport capacity (stock demand per generation technology) in MW peak capacity
pipelines_oil = Path(path_base, "image-materials","Fossil_test", "stock_calculation", "output", scenario, "oil_pipelines_length_km.csv")
df_pipelines_oil = pd.read_csv(pipelines_oil)
pipelines_gas = Path(path_base, "image-materials","Fossil_test", "stock_calculation", "output", scenario, "gas_pipelines_length_km.csv")
df_pipelines_gas = pd.read_csv(pipelines_gas)

df_pipelines_oil["fuel"] = "oil"
df_pipelines_gas["fuel"] = "gas"

df_pipelines_all = pd.concat(
    [df_pipelines_oil, df_pipelines_gas],
    ignore_index=True
)

   # test if path_external_data_scenario exists and if not set to standard scenario
# if not path_external_data_scenario.exists():
  #       path_external_data_scenario = Path(path_base, "fossil", STANDARD_SCEN_EXTERNAL_DATA)
# 
# assert path_external_data_scenario.is_dir()

    ###########################################################################################################
    # Transform to xarray #
#knowledge_graph_region = create_region_graph()
#knowledge_graph_coal = create_ff_graph()
    

    # Lifetimes -------
values = pipelines_lifetime_data["Lifetime"].unstack().to_numpy(dtype=float)
#Create coordinates
times = pipelines_lifetime_data.index.levels[0].to_numpy()
types = pipelines_lifetime_data.index.levels[1].to_numpy()
scipy_params = ["mean", "stdev"]
    # Build full array: shape (ScipyParam, Time, Type)
data_array = np.stack([values, np.full_like(values, np.nan)], axis=0)
    # Create DataArray
pipelines_lifetime_xr = xr.DataArray(
        data_array,
        dims=["DistributionParams", "Cohort", "Type"],
        coords={
            "DistributionParams": scipy_params,
            "Cohort": times,
            "Type": [str(r) for r in types]
        },
        name="PipelinesLifetime"
    )
pipelines_lifetime_xr = prism.Q_(pipelines_lifetime_xr, "year")
    #coal_lifetime_xr = knowledge_graph_electr.rebroadcast_xarray(coal_lifetime_xr, output_coords=EPG_TECHNOLOGIES, dim="Type") # convert technology names to the standard names from TIMER
    #coal_lifetime_xr = coal_lifetime_xr.assign_coords(Type=np.array(coal_lifetime_xr.Type.values, dtype=object)) # rebroadcast_xarray changes the type of the coordinates to numpy strings (np.str_), so convert back to python strings (str)

 # Material Intensities -------

    # Material Intensities -------
   # Set index
pipelines_materials_data = pipelines_materials_data.set_index(['Year', 'Tech Type'])

   # Convert to 3D array: (Material, Year, Tech)
pipelines_materials_xr = (
    pipelines_materials_data
        .to_xarray()           # keeps Year & Tech Type as coords
        .to_array("Material")  # converts columns into a 'Material' dimension
        .rename({"Year": "Cohort", "Tech Type": "Type"})
)

# Name the DataArray
pipelines_materials_xr.name = "PipelinesMaterialIntensities"

pipelines_materials_xr = prism.Q_(pipelines_materials_xr, "kg/kg/year")
#print(pipelines_materials_xr)
#pipelines_materials_xr = knowledge_graph_coal.rebroadcast_xarray(pipelines_materials_xr, output_coords=EPG_TECHNOLOGIES, dim="Type")
#pipelines_materials_xr = pipelines_materials_xr.assign_coords(Type=np.array(pipelines_materials_xr.Type.values, dtype=object)) # rebroadcast_xarray changes the type of the coordinates to numpy strings (np.str_), so convert back to python strings (str)

# Pipeline capacity ------
df_pipelines_all = df_pipelines_all.loc[~df_pipelines_all['DIM_1'].isin([27,28])]  # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies
df_pipelines_all = df_pipelines_all.loc[df_pipelines_all['time'].isin(range(year_start, year_end + 1)), ['time', 'DIM_1', *range(1, len(FF_TECHNOLOGIES) + 1)]]  # only keep relevant years and technology columns
    # Extract coordinate labels
years = sorted(df_pipelines_all['time'].unique())
regions = sorted(df_pipelines_all['DIM_1'].unique())
techs = list(range(1, len(FF_TECHNOLOGIES)+1))
    # Convert to 3D array: (Year, Region, Tech)
data_array = df_pipelines_all[techs].to_numpy().reshape(len(years), len(regions), len(techs))
    # Build xarray DataArray
pipelinecap_xr = xr.DataArray(
        data_array,
        dims=('Time', 'Region', 'Type'),
        coords={
            'Time': years,
            'Region': [str(r) for r in regions],
            'Type': [str(r) for r in techs]
        },
        name='PipelineCap'
    )
pipelinecap_xr = prism.Q_(pipelinecap_xr, "MW")
#pipelinecap_xr = knowledge_graph_region.rebroadcast_xarray(pipelinecap_xr, output_coords=IMAGE_REGIONS, dim="Region") 
#pipelinecap_xr = knowledge_graph_coal.rebroadcast_xarray(pipelinecap_xr, output_coords=FF_TECHNOLOGIES, dim="Type")
pipelinecap_xr = pipelinecap_xr.assign_coords(Type=np.array(pipelinecap_xr.Type.values, dtype=object)) # rebroadcast_xarray changes the type of the coordinates to numpy strings (np.str_), so convert back to python strings (str)

    ###########################################################################################################
    # Interpolate #

    # interpolate_xr: The lifetimes & material intensities are only given for specific years (2020 and 2050), so we linearly interpolate to get values for the years 2020-2050.
    # The values before 2020 are kept constant at the 2020 level, and the values after 2050 are kept constant at the 2050 level.
pipelines_lifetime_xr_interp = interpolate_xr(pipelines_lifetime_xr, YEAR_FIRST_GRID, year_out)
pipelines_lifetime_xr_interp.loc[dict(DistributionParams="stdev")] = pipelines_lifetime_xr_interp.loc[dict(DistributionParams="mean")] * STD_LIFETIMES_ELECTR
pipelines_materials_xr_interp = interpolate_xr(pipelines_materials_xr, YEAR_FIRST_GRID, year_out)

    # TIMER data only start in 1971, so we add a historic tail back to YEAR_FIRST_GRID=1921 #TODO to be adjusted

pipelinecap_xr_interp = add_historic_stock(pipelinecap_xr, YEAR_FIRST_GRID)