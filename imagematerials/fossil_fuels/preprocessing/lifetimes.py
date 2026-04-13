#Outline (split into each stage (extraction, processing, transport, pipelines) for each fuel (coal, oil, gas)):
# 1) Import general modules and constants
# 2) read in necessary data files for lifetimes 
# 3) create xarray

#%% Import general modules and constants
import pint
import xarray as xr
import prism

from ffconstants import FF_TECHNOLOGIES, IMAGE_REGIONS, STANDARD_SCEN_EXTERNAL_DATA, YEAR_FIRST_GRID, Standard_deviation_lifetime

from imagematerials.fossil_fuels.fftest import create_fossil_fuel_graph
from imagematerials.read_mym import read_mym_df
from imagematerials.util import dataset_to_array, pandas_to_xarray, convert_lifetime
from imagematerials.concepts import KnowledgeGraph, Node, create_electricity_graph, create_region_graph
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

# Extraction stage (coal, oil, gas) ---------------------------------------------------------------------------------------------------------------------------------

###########################################################################################################
def get_preprocessing_data_extraction(path_base: str, climate_policy_config: dict, circular_economy_config: dict, scenario: str, year_start: int, year_end: int, year_out: int):

    path_external_data_scenario = Path(path_base, "fossil_fuels", "Scenario_data", scenario)
    # test if path_external_data_scenario exists and if not set to standard scenario
    if not path_external_data_scenario.exists():
        path_external_data_scenario = Path(path_base, "fossil_fuels", STANDARD_SCEN_EXTERNAL_DATA)

    # assert path_external_data_scenario.is_dir()

###########################################################################################################
    # Read in files #

    # 1. External Data --------------------------------------------- 

        # lifetimes of extraction tech in years
    extraction_lifetime_data = pd.read_csv(path_base / "data" / "raw" / "fossil_fuels" / "Extraction" / "Extraction_lifetimes.csv",
        index_col=["Year", "Tech Type"]
    ) 

       # Lifetimes -------
    types = ["Coal Underground","Coal Opencast","Oil Offshore","Oil Onshore","Gas Offshore","Gas Onshore"]
    values = extraction_lifetime_data["Lifetime"].unstack().reindex(columns=types).to_numpy(dtype=float)
    times = extraction_lifetime_data.index.get_level_values("Year").unique().to_numpy()
    scipy_params = ["mean", "stdev"]

    # Build xarray with shape (ScipyParam, Time, Type)
    data_array = np.stack([values, np.full_like(values, Standard_deviation_lifetime)], axis=0)

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

    #Rebroadcast to standard technology names from TIMER, and convert coordinate type back to python strings (since rebroadcast changes it to numpy strings)
    extraction_lifetime_xr = fossil_fuel_knowledge_graph.rebroadcast_xarray(extraction_lifetime_xr, output_coords=FF_TECHNOLOGIES, dim="Type") # convert technology names to the standard names from TIMER
    extraction_lifetime_xr = extraction_lifetime_xr.assign_coords(Type=np.array(extraction_lifetime_xr.Type.values, dtype=object)) # rebroadcast_xarray changes the type of the coordinates to numpy strings (np.str_), so convert back to python strings (str)

   # Extraction stage (coal, oil, gas) ---------------------------------------------------------------------------------------------------------------------------------
    # The lifetimes are converted to the proper format for the model (dictionary with keys:distribution name, values:datarrays containing distribution parameters)
    extraction_lifetime_xr = convert_lifetime(extraction_lifetime_xr)
        
    # bring preprocessing data into a generic format for the model
    prep_data_extraction = {}
    prep_data_extraction["lifetimes"] = extraction_lifetime_xr
    prep_data_extraction["knowledge_graph"] = create_fossil_fuel_graph() 

    return prep_data_extraction

#%% Processing stage (coal, oil, gas) ---------------------------------------------------------------------------------------------------------------------------------

def get_preprocessing_data_processing(path_base: str, climate_policy_config: dict, circular_economy_config: dict, scenario: str, year_start: int, year_end: int, year_out: int):

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
    processing_lifetime_data = pd.read_csv(
        path_base / "data" / "raw" / "fossil_fuels" / "Processing" / "Processing_lifetimes.csv",
        index_col=["Year", "Tech Type"]
    ) 

           # Lifetimes -------
    values = processing_lifetime_data["Lifetime"].unstack().to_numpy(dtype=float)
    #Create coordinates
    times = processing_lifetime_data.index.levels[0].to_numpy()
    types = processing_lifetime_data.index.levels[1].to_numpy()
    scipy_params = ["mean", "stdev"]
        # Build full array: shape (ScipyParam, Time, Type)
    data_array = np.stack([values, np.full_like(values, Standard_deviation_lifetime)], axis=0)
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
    # Expand 2020 across 1880-2100
    year_range = np.arange(1880, 2101)
    value_2020 = processing_lifetime_xr.sel(Cohort=2020)
    processing_lifetime_xr = value_2020.expand_dims(Cohort=year_range).transpose("DistributionParams", "Cohort", "Type")
    processing_lifetime_xr = prism.Q_(processing_lifetime_xr, "year")

        # Processing stage (coal, oil, gas) ---------------------------------------------------------------------------------------------------------------------------------
    # The lifetimes are converted to the proper format for the model (dictionary with keys:distribution name, values:datarrays containing distribution parameters)
    processing_lifetime_xr = convert_lifetime(processing_lifetime_xr)
        
    # bring preprocessing data into a generic format for the model
    prep_data_processing = {}
    prep_data_processing["lifetimes"] = processing_lifetime_xr
    prep_data_processing["knowledge_graph"] = create_fossil_fuel_graph() 
    
    return prep_data_processing

#%% Transport/Vehicles stage (coal, oil, gas) ---------------------------------------------------------------------------------------------------------------------------------

###########################################################################################################
def get_preprocessing_data_transport(path_base: str, climate_policy_config: dict, circular_economy_config: dict, scenario: str, year_start: int, year_end: int, year_out: int):

    path_external_data_scenario = Path(path_base, "fossil_fuels", "Scenario_data", scenario)
    # test if path_external_data_scenario exists and if not set to standard scenario
    if not path_external_data_scenario.exists():
        path_external_data_scenario = Path(path_base, "fossil_fuels", STANDARD_SCEN_EXTERNAL_DATA)

    # assert path_external_data_scenario.is_dir()
###########################################################################################################
    # Read in files #

    # 1. External Data --------------------------------------------- 

    # lifetimes of transport tech in years
    transport_lifetime_data = pd.read_csv(path_base / "data" / "raw" / "fossil_fuels" / "Transport" / "Transport_lifetimes.csv", index_col=["Year", "Tech Type"])

        # Lifetimes -------
    values = transport_lifetime_data["Lifetimes"].unstack().to_numpy(dtype=float)
    #Create coordinates
    times = transport_lifetime_data.index.levels[0].to_numpy()
    types = transport_lifetime_data.index.levels[1].to_numpy()
    scipy_params = ["mean", "stdev"]
    # Build full array: shape (ScipyParam, Time, Type)
    data_array = np.stack([values, np.full_like(values, Standard_deviation_lifetime)], axis=0)
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

    # Expand 2020 across 1880-2100
    year_range = np.arange(1880, 2101)
    value_2020 = transport_lifetime_xr.sel(Cohort=2020)
    transport_lifetime_xr = value_2020.expand_dims(Cohort=year_range).transpose("DistributionParams", "Cohort", "Type")
    transport_lifetime_xr = prism.Q_(transport_lifetime_xr, "year")

    #Rebroadcast to standard technology names from TIMER, and convert coordinate type back to python strings (since rebroadcast changes it to numpy strings)
    transport_lifetime_xr = fossil_fuel_knowledge_graph.rebroadcast_xarray(transport_lifetime_xr, output_coords=FF_TECHNOLOGIES, dim="Type") # convert technology names to the standard names from TIMER
    transport_lifetime_xr = transport_lifetime_xr.assign_coords(Type=np.array(transport_lifetime_xr.Type.values, dtype=object)) # rebroadcast_xarray changes the type of the coordinates to numpy strings (np.str_), so convert back to python strings (str)

    # Transportation stage (coal, oil, gas) ---------------------------------------------------------------------------------------------------------------------------------
    # The lifetimes are converted to the proper format for the model (dictionary with keys:distribution name, values:datarrays containing distribution parameters)
    transport_lifetime_xr = convert_lifetime(transport_lifetime_xr)
        
    #  # bring preprocessing data into a generic format for the model
    prep_data_transport = {}
    prep_data_transport["lifetimes"] = transport_lifetime_xr
    prep_data_transport["knowledge_graph"] = create_electricity_graph() 
 
    return prep_data_transport

#%% Pipelines stage (oil, gas) ---------------------------------------------------------------------------------------------------------------------------------
###########################################################################################################

def get_preprocessing_data_pipelines(path_base: str, climate_policy_config: dict, circular_economy_config: dict, scenario: str, year_start: int, year_end: int, year_out: int):

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
        path_base / "data" / "raw" / "fossil_fuels" / "Pipelines" / "Pipelines_lifetimes_edit.csv",
        index_col=["Year", "Tech Type"]
    ) 

            ###########################################################################################################
        # Transform to xarray #
    knowledge_graph_region = create_region_graph()
    fossil_fuel_knowledge_graph = create_fossil_fuel_graph()

        # Lifetimes -------
    values = pipelines_lifetime_data["Lifetimes"].unstack().to_numpy(dtype=float)
    #Create coordinates
    times = pipelines_lifetime_data.index.levels[0].to_numpy()
    types = pipelines_lifetime_data.index.levels[1].to_numpy()
    scipy_params = ["mean", "stdev"]
    # Build full array: shape (ScipyParam, Time, Type)
    data_array = np.stack([values, np.full_like(values, Standard_deviation_lifetime)], axis=0)
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

    # Expand 2020 across 1880-2100
    year_range = np.arange(1880, 2101)
    value_2020 = pipelines_lifetime_xr.sel(Cohort=2020)
    pipelines_lifetime_xr = value_2020.expand_dims(Cohort=year_range).transpose("DistributionParams", "Cohort", "Type")

    # Rebroadcast to standard technology names from TIMER, and convert coordinate type back to python strings (since rebroadcast changes it to numpy strings)
    pipelines_lifetime_xr = fossil_fuel_knowledge_graph.rebroadcast_xarray(pipelines_lifetime_xr, output_coords=FF_TECHNOLOGIES, dim="Type") # convert technology names to the standard names from TIMER
    pipelines_lifetime_xr = pipelines_lifetime_xr.assign_coords(Type=np.array(pipelines_lifetime_xr.Type.values, dtype=object)) # rebroadcast_xarray changes the type of the coordinates to numpy strings (np.str_), so convert back to python strings (str)

# The lifetimes are converted to the proper format for the model (dictionary with keys:distribution name, values:datarrays containing distribution parameters)
    pipelines_lifetime_xr = convert_lifetime(pipelines_lifetime_xr)
        
    #  # bring preprocessing data into a generic format for the model
    prep_data_pipelines = {}
    prep_data_pipelines["lifetimes"] = pipelines_lifetime_xr
    prep_data_pipelines["knowledge_graph"] = create_fossil_fuel_graph() 

    return prep_data_pipelines