 
#%% Outline of the pre-processing steps for fossil fuels (coal, oil, gas) ---------------------------------------------------------------------------------------------------------------------------------

# #1: Import module and constants
# a General imports/packages
# b Image material imports
# c Set scenario 
# d Make variable for pre-processing data 
# e Make variable for path to external data for the scenario, and test if it exists, if not set to standard scenario

#2: Read in files for the extraction stage for each fuel (coal, oil, gas)
# a lifetimes 
# b material intensities (steel, alumnium, copper?, I could add other ones?)
# C Import capacity files that are made from the stock calculation (FUMA) output (for each fuel: coal, oil, gas) which is labelled as fuma.py and are found under stock_calculation/output/SSP1_ML (or the relevant scenario)
# # d Transform each to xarray
# e add units using the knowledge graph
# f expand materials and lifteimes to all years (2020-2100)

#3: Read in files for the processing stage for each fuel
# a lifetimes 
# b material intensities
# C Import capacity files that are made from the stock calculation (FUMA) output (for each fuel: coal, oil, gas) which is labelled as fuma.py and are found under stock_calculation/output/SSP1_ML (or the relevant scenario)
# d Transform each to xarray and add units using the knowledge graph and expand materials and lifteimes to all years (2020-2100)

#4: Read in files for the transport stage for each fuel
# a lifetimes 
# b material intensities
# C Import capacity files that are made from the stock calculation (FUMA) output (for each fuel: coal, oil, gas) which is labelled as fuma.py and are found under stock_calculation/output/SSP1_ML (or the relevant scenario)
# d Transform each to xarray
# e add units using the knowledge graph
# f expand materials and lifteimes to all years (2020-2100)

#5: Read in files for the pipelines stage for each fuel
# a lifetimes 
# b material intensities
# C Import capacity files that are made from the stock calculation (FUMA) output (for each fuel: coal, oil, gas) which is labelled as fuma.py and are found under stock_calculation/output/SSP1_ML (or the relevant scenario)
# d Transform each to xarray
# e add units using the knowledge graph
# f expand materials and lifteimes to all years (2020-2100)

#7 Make prep_data dictionary with all the pre-processed data, and return it

# 8 Adjust as necessary to fit into the IMAGE framework (so its runs with ELMA)

# 9 Make knowledge graph for fossil fuel so that figure can combine many types of infrastructure for each fuel type

# 10 Make output figures 


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


# SET SCENARIO HERE (Select from SSP1_ML, SSP1_VLLO, SSP2, SSP2_2D, SSP2_ML, SSP2_VLLO, SSP2_VLLO_Lifetech, SSP3_H)
scenario = "SSP3_H"

path_current = Path().resolve()
path_base = path_current.parent # base path of the project -> image-materials
print("current:", path_current)
print("base:", path_base)

#NOt sure what to do with these 
scen_folder = "SSP3_H"
climate_policy_scenario_dir = Path(path_base, "data", "raw", "image", scen_folder)
STANDARD_SCEN_EXTERNAL_DATA = "SSP3_H"
SCENARIO_DEFAULT = "SSP3_H"

#Not 100% on what these do yet 
year_start = 2020
year_end = 2100
year_out = 2100


#%% Extraction stage (coal, oil, gas) ---------------------------------------------------------------------------------------------------------------------------------

###########################################################################################################
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
    index_col=["Year", "TechType"]
) 

    # material compositions of coal infrastructure (processing, extraction, transport) in kg/kg/year
extraction_materials_data = pd.read_csv('Extraction/FF Intensity & LT - Extraction materials.csv')
#print(extraction_materials_data.index.is_unique)
#print(extraction_materials_data.columns)
#print(extraction_materials_data.head())
    # 2.FUMA files -----------------------------------------
#Stock of each type of extraction infrastructure (stock demand per generation technology) per region per year
#files are sourced from the output of the stock calculation (FUMA) for the relevant scenario, which is found under stock_calculation/output/SSP1_ML (or the relevant scenario)
extraction_coal = Path(path_base, "Fossil_test_check", "stock_calculation", "output", scenario, "coal_extraction_stock_kg.csv")
df_extraction_coal = pd.read_csv(extraction_coal)
extraction_oil = Path(path_base, "Fossil_test_check", "stock_calculation", "output", scenario, "oil_extraction_stock_kg.csv")
df_extraction_oil = pd.read_csv(extraction_oil)
extraction_gas = Path(path_base, "Fossil_test_check", "stock_calculation", "output", scenario, "gas_extraction_stock_kg.csv")
df_extraction_gas = pd.read_csv(extraction_gas)

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
df_extraction_all['Tech Type'] = df_extraction_all['fuel'].str.strip() + ' ' + df_extraction_all['type'].str.strip()

#Drop columns that are not needed for the analysis (stage, unit, type, fuel) since we have already combined the relevant information into the "Tech Type" column and we know that all rows are for the extraction stage and in kg units
df_extraction_all = df_extraction_all.drop(columns=['stage', 'unit', 'type', 'fuel'])  

    ###########################################################################################################
    # Transform to xarray #

#still need to make knowledge graphs  
#knowledge_graph_region = create_region_graph()
#knowledge_graph_coal = create_ff_graph()
    

# Lifetimes -------
types = ["Coal Underground","Coal open cast","Oil off shore","Oil on shore","Gas off shore","Gas on shore"]
values = extraction_lifetime_data["Lifetime"].unstack().reindex(columns=types).to_numpy(dtype=float)
times = extraction_lifetime_data.index.get_level_values("Year").unique().to_numpy()
scipy_params = ["mean", "stdev"]

# Build xarray with shape (ScipyParam, Time, Type)
data_array = np.stack([values, np.full_like(values, np.nan)], axis=0)

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

# Expand 2020 across 2020-2100
year_range = np.arange(2020, 2101)
value_2020 = extraction_lifetime_xr.sel(Cohort=2020)
extraction_lifetime_xr = value_2020.expand_dims(Cohort=year_range).transpose("DistributionParams", "Cohort", "Type")

#Add units 
extraction_lifetime_xr = prism.Q_(extraction_lifetime_xr, "year")

#Uncomment to check if the xarray looks correct
#print(extraction_lifetime_xr.head())
#print(types)
#print("First year:", extraction_lifetime_xr.Cohort.values[0])
#print("Last year:", extraction_lifetime_xr.Cohort.values[-1])
#print("Number of years:", len(extraction_lifetime_xr.Cohort))
#print("Shape of DataArray:", extraction_lifetime_xr.shape)

#TODO: check about rebroadcasting to knowledge graphs (we also dont have the knowledge graphs yet)

#extraction_lifetime_xr = knowledge_graph_electr.rebroadcast_xarray(extraction_lifetime_xr, output_coords=FF_TECHNOLOGIES, dim="Type") # convert technology names to the standard names from TIMER
#extraction_lifetime_xr = extraction_lifetime_xr.assign_coords(Type=np.array(extraction_lifetime_xr.Type.values, dtype=object)) # rebroadcast_xarray changes the type of the coordinates to numpy strings (np.str_), so convert back to python strings (str)

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

extraction_materials_xr.name = "ExtractionMaterialIntensities"

#Expand 2020 values across 2020-2100
value_2020 = extraction_materials_xr.sel(Cohort=2020)
year_range = np.arange(2020, 2101)
extraction_materials_xr = value_2020.expand_dims(Cohort=year_range).transpose("Material", "Cohort", "Type")

#Add units
extraction_materials_xr = prism.Q_(extraction_materials_xr, "kg/kg/year")

#Uncomment to check if it worked
#print(extraction_materials_xr.head())
#print(types)
#print("First year:", extraction_materials_xr.Cohort.values[0])
#print("Last year:", extraction_materials_xr.Cohort.values[-1])
#print("Number of years:", len(extraction_materials_xr.Cohort))
#print("Shape of DataArray:", extraction_materials_xr.shape)

#TODO: check about rebroadcasting to knowledge graphs (we also dont have the knowledge graphs yet)
#extraction_materials_xr = knowledge_graph_coal.rebroadcast_xarray(extraction_materials_xr, output_coords=EPG_TECHNOLOGIES, dim="Type")
#extraction_materials_xr = extraction_materials_xr.assign_coords(Type=np.array(extraction_materials_xr.Type.values, dtype=object)) # rebroadcast_xarray changes the type of the coordinates to numpy strings (np.str_), so convert back to python strings (str)

# Starting stock (capacity) ------
df_extraction_all = df_extraction_all.loc[~df_extraction_all['DIM_1'].isin([27,28])]  # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies
df_extraction_all = df_extraction_all.loc[df_extraction_all['time'].isin(range(year_start, year_end + 1)), ['time', 'DIM_1', 'value', 'Tech Type']]  # only keep relevant years and technology columns
   
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

#extractioncap_xr = prism.Q_(extractioncap_xr, "MW")
#extractioncap_xr = knowledge_graph_region.rebroadcast_xarray(extractioncap_xr, output_coords=IMAGE_REGIONS, dim="Region") 
#extractioncap_xr = knowledge_graph_coal.rebroadcast_xarray(extractioncap_xr, output_coords=FF_TECHNOLOGIES, dim="Type")
extractioncap_xr = extractioncap_xr.assign_coords(Type=np.array(extractioncap_xr.Type.values, dtype=object)) # rebroadcast_xarray changes the type of the coordinates to numpy strings (np.str_), so convert back to python strings (str)

 # This is from original model and i am not sure if something like this is needed in the new model?
# TIMER data only start in 1971, so we add a historic tail back to YEAR_FIRST_GRID=1921 
#extractioncap_xr_interp = add_historic_stock(extractioncap_xr, YEAR_FIRST_GRID)

#%% Processing stage (coal, oil, gas) ---------------------------------------------------------------------------------------------------------------------------------

#def get_preprocessing_data_processing(path_base: str, climate_policy_config: dict, scenario: str, year_start: int, year_end: int, year_out: int):

    #path_external_data_scenario = Path(path_base, "fossil", scenario)

    # test if path_external_data_scenario exists and if not set to standard scenario
    #if not path_external_data_scenario.exists():
        #path_external_data_scenario = Path(path_base, "fossil", STANDARD_SCEN_EXTERNAL_DATA)

    #assert path_external_data_scenario.is_dir()

    #return path_external_data_scenario
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


    # 2. FUMA files -----------------------------------------
#Stock of each type of extraction infrastructure (stock demand per generation technology) per region per year

processing_coal = Path(path_base, "Fossil_test_check", "stock_calculation", "output", scenario, "coal_preparation_stock_kg.csv")
df_processing_coal = pd.read_csv(processing_coal)
processing_storage_oil = Path(path_base, "Fossil_test_check", "stock_calculation", "output", scenario, "oil_storage_volume_m3.csv")
df_processing_oil = pd.read_csv(processing_storage_oil)
processing_refinery_oil = Path(path_base, "Fossil_test_check", "stock_calculation", "output", scenario, "oil_refinery_stock_kg.csv")
df_processing_refinery_oil = pd.read_csv(processing_refinery_oil)
processing_gas = Path(path_base, "Fossil_test_check", "stock_calculation", "output", scenario, "gas_processing_stock_kg.csv")
df_processing_gas = pd.read_csv(processing_gas)

#Combine the dataframes for coal, oil, and gas into one dataframe and then melt it to long format so that we have one row per combination of time, region, fuel, technology, and value (stock)
df_processing_all = pd.concat(
    [df_processing_coal, df_processing_oil, df_processing_refinery_oil, df_processing_gas],
    ignore_index=True
)

df_processing_all = df_processing_all.melt(
    id_vars=['type', 'time', 'fuel', 'stage', 'unit'],  
    var_name='DIM_1',
    value_name='value'
)

#Combine fuel and type columns to one column "Tech Type"
df_processing_all['Tech Type'] = df_processing_all['fuel'].str.strip() + ' ' + df_processing_all['type'].str.strip()

# Drop columns that are not needed for the analysis
df_processing_all = df_processing_all.drop(columns=['stage', 'unit', 'type', 'fuel'])  

    ###########################################################################################################
    # Transform to xarray #
#TODO: need to work on knowledge graphs for the processing stage    
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
# Expand 2020 across 2020-2100
year_range = np.arange(2020, 2101)
value_2020 = processing_lifetime_xr.sel(Cohort=2020)
processing_lifetime_xr = value_2020.expand_dims(Cohort=year_range).transpose("DistributionParams", "Cohort", "Type")
processing_lifetime_xr = prism.Q_(processing_lifetime_xr, "year")

#Uncomment to check if the xarray looks correct
#print(processing_lifetime_xr.head())
#print(types)
#print("First year:", processing_lifetime_xr.Cohort.values[0])
#print("Last year:", processing_lifetime_xr.Cohort.values[-1])
#print("Number of years:", len(processing_lifetime_xr.Cohort))
#print("Shape of DataArray:", processing_lifetime_xr.shape)

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

#Expand 2020 values across 2020-2100
value_2020 = processing_materials_xr.sel(Cohort=2020)
year_range = np.arange(2020, 2101)
processing_materials_xr = value_2020.expand_dims(Cohort=year_range).transpose("Material", "Cohort", "Type")

#Add units
processing_materials_xr = prism.Q_(processing_materials_xr, "kg/kg/year")

#Uncomment to check if it worked
#print(processing_materials_xr.head())
# print(types)
# print("First year:", processing_materials_xr.Cohort.values[0])
# print("Last year:", processing_materials_xr.Cohort.values[-1])
# print("Number of years:", len(processing_materials_xr.Cohort))
# print("Shape of DataArray:", processing_materials_xr.shape)

#TODO: check about rebroadcasting to knowledge graphs (we also dont have the knowledge graphs yet)
#processing_materials_xr = knowledge_graph_coal.rebroadcast_xarray(processing_materials_xr, output_coords=EPG_TECHNOLOGIES, dim="Type")
#processing_materials_xr = processing_materials_xr.assign_coords(Type=np.array(processing_materials_xr.Type.values, dtype=object)) # rebroadcast_xarray changes the type of the coordinates to numpy strings (np.str_), so convert back to python strings (str)

# Gcap ------
# idx = pd.MultiIndex.from_frame(df_processing_all[['time', 'DIM_1', 'Tech Type']])
# print(idx.duplicated())
# duplicates = df_processing_all[df_processing_all.duplicated(subset=['time', 'DIM_1', 'Tech Type'], keep=False)]
# print(duplicates)
# dup_counts = df_processing_all.groupby(['time', 'DIM_1', 'Tech Type']).size()
# print(dup_counts[dup_counts > 1])
# print(df_processing_coal[df_processing_coal['time'].isna()].head())
# print(df_processing_oil[df_processing_oil['time'].isna()].head())
# print(df_processing_refinery_oil[df_processing_refinery_oil['time'].isna()].head())
# print(df_processing_gas[df_processing_gas['time'].isna()].head())



# combine the processing data for coal, oil, gas into one df and then put it into this 
df_processing_all = df_processing_all.loc[~df_processing_all['DIM_1'].isin([27,28])]  # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies
df_processsing_all = df_processing_all.loc[df_processing_all['time'].isin(range(year_start, year_end + 1)), ['time', 'DIM_1', 'value', 'Tech Type']]  # only keep relevant years and technology columns
   
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
processingcap_xr = prism.Q_(processingcap_xr, "kg?")
#processing_capacities_xr = knowledge_graph_region.rebroadcast_xarray(processing_capacities_xr, output_coords=IMAGE_REGIONS, dim="Region") 
#processing_capacities_xr = knowledge_graph_coal.rebroadcast_xarray(processing_capacities_xr, output_coords=FF_TECHNOLOGIES, dim="Type")
processingcap_xr = processingcap_xr.assign_coords(Type=np.array(processingcap_xr.Type.values, dtype=object)) # rebroadcast_xarray changes the type of the coordinates to numpy strings (np.str_), so convert back to python strings (str)

#Not sure if this is needed 
# TIMER data only start in 1971, so we add a historic tail back to YEAR_FIRST_GRID=1921 #TODO to be adjusted
#processingcap_xr_interp = add_historic_stock(processingcap_xr, YEAR_FIRST_GRID)

#%% Transport/Vehicles stage (coal, oil, gas) ---------------------------------------------------------------------------------------------------------------------------------

###########################################################################################################
#def get_preprocessing_data_processing(path_base: str, climate_policy_config: dict, scenario: str, year_start: int, year_end: int, year_out: int):

    #path_external_data_scenario = Path(path_base, "fossil", scenario)

    # test if path_external_data_scenario exists and if not set to standard scenario
# if not path_external_data_scenario.exists():
#    path_external_data_scenario = Path(path_base, "fossil", STANDARD_SCEN_EXTERNAL_DATA)

#assert path_external_data_scenario.is_dir()
###########################################################################################################
# Read in files #

# 1. External Data --------------------------------------------- 

# lifetimes of transport tech in years
transport_lifetime_data = pd.read_csv('Transport/FF Intensity & LT - Transport lifetimes.csv',index_col=["Year", "Tech Type"]) 

# material compositions of transport infrastructure in kg/kg/year
transport_materials_data = pd.read_csv('Transport/FF Intensity & LT - Transport materials.csv')


    # 2. IMAGE/TIMER files -----------------------------------------
#Transport capacity (stock demand per generation technology) in MW peak capacity
transport_coal = Path(path_base, "Fossil_test_check", "stock_calculation", "output", scenario, "coal_transport_stock_kg.csv")
df_transport_coal = pd.read_csv(transport_coal)
transport_oil = Path(path_base, "Fossil_test_check", "stock_calculation", "output", scenario, "oil_transport_stock_kg.csv")
df_transport_oil = pd.read_csv(transport_oil)
transport_gas = Path(path_base, "Fossil_test_check", "stock_calculation", "output", scenario, "gas_transport_stock_kg.csv")
df_transport_gas = pd.read_csv(transport_gas)

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

#Drop columns that are not needed for the analysis 
df_transport_all = df_transport_all.drop(columns=['stage', 'unit', 'type', 'fuel']) 

    ###########################################################################################################
    # Transform to xarray #
#TODO: need to work on knowledge graphs for the transport stage
#knowledge_graph_region = create_region_graph()
#knowledge_graph_coal = create_ff_graph()
    
    # Lifetimes -------
values = transport_lifetime_data["Lifetimes"].unstack().to_numpy(dtype=float)
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

# Expand 2020 across 2020-2100
year_range = np.arange(2020, 2101)
value_2020 = transport_lifetime_xr.sel(Cohort=2020)
transport_lifetime_xr = value_2020.expand_dims(Cohort=year_range).transpose("DistributionParams", "Cohort", "Type")
transport_lifetime_xr = prism.Q_(transport_lifetime_xr, "year")

#Uncomment to check if the xarray looks correct
# print(transport_lifetime_xr.head())
# print(types)
# print("First year:", transport_lifetime_xr.Cohort.values[0])
# print("Last year:", transport_lifetime_xr.Cohort.values[-1])
# print("Number of years:", len(transport_lifetime_xr.Cohort))
# print("Shape of DataArray:", transport_lifetime_xr.shape)

#transport_lifetime_xr = knowledge_graph_electr.rebroadcast_xarray(coal_transport_xr, output_coords=EPG_TECHNOLOGIES, dim="Type") # convert technology names to the standard names from TIMER
#transport_lifetime_xr = coal_lifetime_xr.assign_coords(Type=np.array(transport_lifetime_xr.Type.values, dtype=object)) # rebroadcast_xarray changes the type of the coordinates to numpy strings (np.str_), so convert back to python strings (str)

 # Material Intensities -------

   # Set index
transport_materials_data = transport_materials_data.set_index(['Year', 'Tech Type'])

# Convert to 3D array: (Material, Year, Tech)
transport_materials_xr = (
    transport_materials_data
        .to_xarray()           
        .to_array("Material")  
        .rename({"Year": "Cohort", "Tech Type": "Type"})
)
transport_materials_xr.name = "TransportMaterialIntensities"

#Expand 2020 values across 2020-2100
value_2020 = transport_materials_xr.sel(Cohort=2020)
year_range = np.arange(2020, 2101)
transport_materials_xr = value_2020.expand_dims(Cohort=year_range).transpose("Material", "Cohort", "Type")
transport_materials_xr = prism.Q_(transport_materials_xr, "kg/kg/year")

# Uncomment to check if it worked
# print(transport_materials_xr.head())
# print(types)
# print("First year:", transport_materials_xr.Cohort.values[0])
# print("Last year:", transport_materials_xr.Cohort.values[-1])
# print("Number of years:", len(transport_materials_xr.Cohort))
# print("Shape of DataArray:", transport_materials_xr.shape)

#TODO: check about rebroadcasting to knowledge graphs (we also dont have the knowledge graphs yet)
#transport_materials_xr = knowledge_graph_coal.rebroadcast_xarray(transport_materials_xr, output_coords=EPG_TECHNOLOGIES, dim="Type")
#transport_materials_xr = transport_materials_xr.assign_coords(Type=np.array(transport_materials_xr.Type.values, dtype=object)) # rebroadcast_xarray changes the type of the coordinates to numpy strings (np.str_), so convert back to python strings (str)

# Transport capacity ------
df_transport_all = df_transport_all.loc[~df_transport_all['DIM_1'].isin([27,28])]  # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies
df_transport_all = df_transport_all.loc[df_transport_all['time'].isin(range(year_start, year_end + 1)), ['time', 'DIM_1', 'value', 'Tech Type']]  # only keep relevant years and technology columns
   
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

transportcap_xr = prism.Q_(transportcap_xr, "MW")
#transportcap_xr = knowledge_graph_region.rebroadcast_xarray(transportcap_xr, output_coords=IMAGE_REGIONS, dim="Region") 
#transportcap_xr = knowledge_graph_coal.rebroadcast_xarray(transportcap_xr, output_coords=FF_TECHNOLOGIES, dim="Type")
transportcap_xr = transportcap_xr.assign_coords(Type=np.array(transportcap_xr.Type.values, dtype=object)) # rebroadcast_xarray changes the type of the coordinates to numpy strings (np.str_), so convert back to python strings (str)

#Not sure if this is needed
# TIMER data only start in 1971, so we add a historic tail back to YEAR_FIRST_GRID=1921 #TODO to be adjusted
#transportcap_xr_interp = add_historic_stock(transportcap_xr, YEAR_FIRST_GRID)

#%% Pipelines stage (oil, gas) ---------------------------------------------------------------------------------------------------------------------------------
###########################################################################################################

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

    # 2. IMAGE/TIMER files -----------------------------------------
#Transport capacity (stock demand per generation technology) in MW peak capacity
pipelines_oil = Path(path_base, "Fossil_test_check", "stock_calculation", "output", scenario, "oil_pipelines_length_km.csv")
df_pipelines_oil = pd.read_csv(pipelines_oil)
pipelines_gas = Path(path_base, "Fossil_test_check", "stock_calculation", "output", scenario, "gas_pipelines_length_km.csv")
df_pipelines_gas = pd.read_csv(pipelines_gas)

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

    ###########################################################################################################
    # Transform to xarray #
#TODO: need to work on knowledge graphs 
#knowledge_graph_region = create_region_graph()
#knowledge_graph_coal = create_ff_graph()

    # Lifetimes -------
values = pipelines_lifetime_data["Lifetimes"].unstack().to_numpy(dtype=float)
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

# Expand 2020 across 2020-2100
year_range = np.arange(2020, 2101)
value_2020 = pipelines_lifetime_xr.sel(Cohort=2020)
pipelines_lifetime_xr = value_2020.expand_dims(Cohort=year_range).transpose("DistributionParams", "Cohort", "Type")

#Uncomment to check if the xarray looks correct
# print(pipelines_lifetime_xr.head())
# print(types)
# print("First year:", pipelines_lifetime_xr.Cohort.values[0])
# print("Last year:", pipelines_lifetime_xr.Cohort.values[-1])
# print("Number of years:", len(pipelines_lifetime_xr.Cohort))
# print("Shape of DataArray:", pipelines_lifetime_xr.shape)

#TODO: check about rebroadcasting to knowledge graphs (we also dont have the knowledge graphs yet)
#pipelines_lifetime_xr = knowledge_graph_electr.rebroadcast_xarray(pipelines_lifetime_xr, output_coords=EPG_TECHNOLOGIES, dim="Type") # convert technology names to the standard names from TIMER
#pipelines_lifetime_xr = pipelines_lifetime_xr.assign_coords(Type=np.array(pipelines_lifetime_xr.Type.values, dtype=object)) # rebroadcast_xarray changes the type of the coordinates to numpy strings (np.str_), so convert back to python strings (str)

 # Material Intensities -------
   # Set index
pipelines_materials_data = pipelines_materials_data.set_index(['Year', 'Tech Type'])

   # Convert to 3D array: (Material, Year, Tech)
pipelines_materials_xr = (
    pipelines_materials_data
        .to_xarray()           
        .to_array("Material")  
        .rename({"Year": "Cohort", "Tech Type": "Type"})
)
pipelines_materials_xr.name = "PipelinesMaterialIntensities"

#Expand 2020 values across 2020-2100
value_2020 = pipelines_materials_xr.sel(Cohort=2020)
year_range = np.arange(2020, 2101)
pipelines_materials_xr = value_2020.expand_dims(Cohort=year_range).transpose("Material", "Cohort", "Type")
pipelines_materials_xr = prism.Q_(pipelines_materials_xr, "kg/kg/year")

# Uncomment to check if it worked
# print(pipelines_materials_xr.head())
# print(types)
# print("First year:", pipelines_materials_xr.Cohort.values[0])
# print("Last year:", pipelines_materials_xr.Cohort.values[-1])
# print("Number of years:", len(pipelines_materials_xr.Cohort))
# print("Shape of DataArray:", pipelines_materials_xr.shape)

pipelines_materials_xr = prism.Q_(pipelines_materials_xr, "kg/kg/year")

#TODO: check about rebroadcasting to knowledge graphs (we also dont have the knowledge graphs yet)
#pipelines_materials_xr = knowledge_graph_coal.rebroadcast_xarray(pipelines_materials_xr, output_coords=EPG_TECHNOLOGIES, dim="Type")
#pipelines_materials_xr = pipelines_materials_xr.assign_coords(Type=np.array(pipelines_materials_xr.Type.values, dtype=object)) # rebroadcast_xarray changes the type of the coordinates to numpy strings (np.str_), so convert back to python strings (str)

# Pipeline capacity ------
df_pipelines_all = df_pipelines_all.loc[~df_pipelines_all['DIM_1'].isin([27,28])]  # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies
df_pipelines_all = df_pipelines_all.loc[df_pipelines_all['time'].isin(range(year_start, year_end + 1)), ['time', 'DIM_1', 'value', 'Tech Type']]  # only keep relevant years and technology columns
    
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
pipelinecap_xr = prism.Q_(pipelinecap_xr, "MW")
#pipelinecap_xr = knowledge_graph_region.rebroadcast_xarray(pipelinecap_xr, output_coords=IMAGE_REGIONS, dim="Region") 
#pipelinecap_xr = knowledge_graph_coal.rebroadcast_xarray(pipelinecap_xr, output_coords=FF_TECHNOLOGIES, dim="Type")
pipelinecap_xr = pipelinecap_xr.assign_coords(Type=np.array(pipelinecap_xr.Type.values, dtype=object)) # rebroadcast_xarray changes the type of the coordinates to numpy strings (np.str_), so convert back to python strings (str)

# Not sure if this is needed
#TIMER data only start in 1971, so we add a historic tail back to YEAR_FIRST_GRID=1921 #TODO to be adjusted
#pipelinecap_xr_interp = add_historic_stock(pipelinecap_xr, YEAR_FIRST_GRID)

#%% Bring everything together in a prep_data file ---------------------------------------------------------------------------------------------------------------------------------
###########################################################################################################

#     # Prep_data File # for each stage (extraction, processing, transport, pipelines) 
# # Extraction stage (coal, oil, gas) ---------------------------------------------------------------------------------------------------------------------------------
# # The lifetimes are converted to the proper format for the model (dictionary with keys:distribution name, values:datarrays containing distribution parameters)
# extraction_lifetime_xr = convert_lifetime(extraction_lifetime_xr)
    
#  # bring preprocessing data into a generic format for the model
# prep_data = {}
# prep_data["lifetimes"] = extraction_lifetime_xr
# prep_data["stocks"] = extractioncap_xr
# prep_data["material_intensities"] = extraction_materials_xr
# #prep_data["knowledge_graph"] = create_electricity_graph() #TODO:make knowledge graph for FF
# # add units
# prep_data["stocks"] = prism.Q_(prep_data["stocks"], "")
# prep_data["material_intensities"] = prism.Q_(prep_data["material_intensities"], "")
# prep_data["set_unit_flexible"] = prism.U_(prep_data["stocks"]) # prism.U_ gives the unit back
#     # set_unit_flexible is needed by the model to deal with the fact the in the beginning of the model it doesn't know th data yet and needs to work with a placeholder/flexible unit (see model.py) 

#return prep_data

###########################################################################################################
# # Processing stage (coal, oil, gas) ---------------------------------------------------------------------------------------------------------------------------------
# # The lifetimes are converted to the proper format for the model (dictionary with keys:distribution name, values:datarrays containing distribution parameters)
# processing_lifetime_xr = convert_lifetime(processing_lifetime_xr)
    
#  # bring preprocessing data into a generic format for the model
# prep_data = {}
# prep_data["lifetimes"] = processing_lifetime_xr
# prep_data["stocks"] = processingcap_xr
# prep_data["material_intensities"] = processing_materials_xr
# #prep_data["knowledge_graph"] = create_electricity_graph() #TODO:make knowledge graph for FF
# # add units
# prep_data["stocks"] = prism.Q_(prep_data["stocks"], "")
# prep_data["material_intensities"] = prism.Q_(prep_data["material_intensities"], "")
# prep_data["set_unit_flexible"] = prism.U_(prep_data["stocks"]) # prism.U_ gives the unit back
#     # set_unit_flexible is needed by the model to deal with the fact the in the beginning of the model it doesn't know th data yet and needs to work with a placeholder/flexible unit (see model.py) 

# #return prep_data

# ###########################################################################################################
# # Transportation stage (coal, oil, gas) ---------------------------------------------------------------------------------------------------------------------------------
# # The lifetimes are converted to the proper format for the model (dictionary with keys:distribution name, values:datarrays containing distribution parameters)
# transport_lifetime_xr = convert_lifetime(transport_lifetime_xr)
    
#  # bring preprocessing data into a generic format for the model
# prep_data = {}
# prep_data["lifetimes"] = transport_lifetime_xr
# prep_data["stocks"] = transportcap_xr
# prep_data["material_intensities"] = transport_materials_xr
# #prep_data["knowledge_graph"] = create_electricity_graph() #TODO:make knowledge graph for FF
# # add units
# prep_data["stocks"] = prism.Q_(prep_data["stocks"], "")
# prep_data["material_intensities"] = prism.Q_(prep_data["material_intensities"], "")
# prep_data["set_unit_flexible"] = prism.U_(prep_data["stocks"]) # prism.U_ gives the unit back
#     # set_unit_flexible is needed by the model to deal with the fact the in the beginning of the model it doesn't know th data yet and needs to work with a placeholder/flexible unit (see model.py) 

# #return prep_data


# ###########################################################################################################
# # Processing stage (coal, oil, gas) ---------------------------------------------------------------------------------------------------------------------------------
# # The lifetimes are converted to the proper format for the model (dictionary with keys:distribution name, values:datarrays containing distribution parameters)
# processing_lifetime_xr = convert_lifetime(processing_lifetime_xr)
    
#  # bring preprocessing data into a generic format for the model
# prep_data = {}
# prep_data["lifetimes"] = processing_lifetime_xr
# prep_data["stocks"] = processingcap_xr
# prep_data["material_intensities"] = processing_materials_xr
# #prep_data["knowledge_graph"] = create_electricity_graph() #TODO:make knowledge graph for FF
# # add units
# prep_data["stocks"] = prism.Q_(prep_data["stocks"], "")
# prep_data["material_intensities"] = prism.Q_(prep_data["material_intensities"], "")
# prep_data["set_unit_flexible"] = prism.U_(prep_data["stocks"]) # prism.U_ gives the unit back
#     # set_unit_flexible is needed by the model to deal with the fact the in the beginning of the model it doesn't know th data yet and needs to work with a placeholder/flexible unit (see model.py) 

# #return prep_data

# ###########################################################################################################
# # Pipelines stage (oil, gas) ---------------------------------------------------------------------------------------------------------------------------------
# # The lifetimes are converted to the proper format for the model (dictionary with keys:distribution name, values:datarrays containing distribution parameters)
# pipelines_lifetime_xr = convert_lifetime(pipelines_lifetime_xr)
    
#  # bring preprocessing data into a generic format for the model
# prep_data = {}
# prep_data["lifetimes"] = pipelines_lifetime_xr
# prep_data["stocks"] = pipelinecap_xr
# prep_data["material_intensities"] = pipelines_materials_xr
# #prep_data["knowledge_graph"] = create_electricity_graph() #TODO:make knowledge graph for FF
# # add units
# prep_data["stocks"] = prism.Q_(prep_data["stocks"], "")
# prep_data["material_intensities"] = prism.Q_(prep_data["material_intensities"], "")
# prep_data["set_unit_flexible"] = prism.U_(prep_data["stocks"]) # prism.U_ gives the unit back
#     # set_unit_flexible is needed by the model to deal with the fact the in the beginning of the model it doesn't know th data yet and needs to work with a placeholder/flexible unit (see model.py) 

# #return prep_data


print("Model finished successfully!")