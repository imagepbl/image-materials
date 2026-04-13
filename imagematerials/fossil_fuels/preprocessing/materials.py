# import relevant libraries
# import constants and other functions within image materials
# read in materials files for each stage
# convert to xarrays
# return material_intensity_extraction,material_intensity_processing,... 


import pint
import xarray as xr
import prism

from ffconstants import FF_TECHNOLOGIES, IMAGE_REGIONS, STANDARD_SCEN_EXTERNAL_DATA, YEAR_FIRST_GRID, Standard_deviation_lifetime

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

#%% Extraction stage (coal, oil, gas) ---------------------------------------------------------------------------------------------------------------------------------

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

        # material compositions of coal infrastructure (processing, extraction, transport) in kg/kg/year
    extraction_materials_data = pd.read_csv(path_base / "data" / "raw" / "fossil_fuels" / "Extraction" / "Extraction_materials.csv")
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

    #Expand 2020 values across 1880-2100
    value_2020 = extraction_materials_xr.sel(Cohort=2020)
    year_range = np.arange(1880, 2101)
    extraction_materials_xr = value_2020.expand_dims(Cohort=year_range).transpose("Material", "Cohort", "Type")

    #Add units
    extraction_materials_xr = prism.Q_(extraction_materials_xr, "kg/kg/year")

    #Rebroadcast to standard technology names from TIMER, and convert coordinate type back to python strings (since rebroadcast changes it to numpy strings)
    extraction_materials_xr = fossil_fuel_knowledge_graph.rebroadcast_xarray(extraction_materials_xr, output_coords=FF_TECHNOLOGIES, dim="Type")
    extraction_materials_xr = extraction_materials_xr.assign_coords(Type=np.array(extraction_materials_xr.Type.values, dtype=object)) # rebroadcast_xarray changes the type of the coordinates to numpy strings (np.str_), so convert back to python strings (str)

  # bring preprocessing data into a generic format for the model
    prep_data_extraction = {}
    prep_data_extraction["material_intensities"] = extraction_materials_xr
    prep_data_extraction["knowledge_graph"] = create_fossil_fuel_graph() 
    # add units
    prep_data_extraction["material_intensities"] = prism.Q_(prep_data_extraction["material_intensities"], "kg/kg/year")
        # set_unit_flexible is needed by the model to deal with the fact the in the beginning of the model it doesn't know th data yet and needs to work with a placeholder/flexible unit (see model.py) 

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
        # material compositions of processing infrastructure in kg/kg/year
    processing_materials_data = pd.read_csv(path_base / "data" / "raw" / "fossil_fuels" / "Processing" / "Processing_materials.csv")
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

    #Expand 2020 values across 1880-2100
    value_2020 = processing_materials_xr.sel(Cohort=2020)
    year_range = np.arange(1880, 2101)
    processing_materials_xr = value_2020.expand_dims(Cohort=year_range).transpose("Material", "Cohort", "Type")

    #Add units
    processing_materials_xr = prism.Q_(processing_materials_xr, "kg/kg/year")

    #Rebroadcast to standard technology names from TIMER, and convert coordinate type back to python strings (since rebroadcast changes it to numpy strings)
    processing_materials_xr = fossil_fuel_knowledge_graph.rebroadcast_xarray(processing_materials_xr, output_coords=FF_TECHNOLOGIES, dim="Type")
    processing_materials_xr = processing_materials_xr.assign_coords(Type=np.array(processing_materials_xr.Type.values, dtype=object)) # rebroadcast_xarray changes the type of the coordinates to numpy strings (np.str_), so convert back to python strings (str)

   # bring preprocessing data into a generic format for the model
    prep_data_processing = {}
    prep_data_processing["material_intensities"] = processing_materials_xr
    #prep_data_processing["knowledge_graph"] = create_fossil_fuel_graph() 
    # add units
 
    prep_data_processing["material_intensities"] = prism.Q_(prep_data_processing["material_intensities"], "kg/kg/year")
 
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
   # material compositions of transport infrastructure in kg/kg/year
    transport_materials_data = pd.read_csv(path_base / "data" / "raw" / "fossil_fuels" / "Transport" / "Transport_materials.csv")
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

    #Expand 2020 values across 1880-2100
    value_2020 = transport_materials_xr.sel(Cohort=2020)
    year_range = np.arange(1880, 2101)
    transport_materials_xr = value_2020.expand_dims(Cohort=year_range).transpose("Material", "Cohort", "Type")
    transport_materials_xr = prism.Q_(transport_materials_xr, "kg/kg/year")

    #Rebroadcast to standard technology names from TIMER, and convert coordinate type back to python strings (since rebroadcast changes it to numpy strings)
    transport_materials_xr = fossil_fuel_knowledge_graph.rebroadcast_xarray(transport_materials_xr, output_coords=FF_TECHNOLOGIES, dim="Type")
    transport_materials_xr = transport_materials_xr.assign_coords(Type=np.array(transport_materials_xr.Type.values, dtype=object)) # rebroadcast_xarray changes the type of the coordinates to numpy strings (np.str_), so convert back to python strings (str)

    #  # bring preprocessing data into a generic format for the model
    prep_data_transport = {}
    prep_data_transport["material_intensities"] = transport_materials_xr
    prep_data_transport["knowledge_graph"] = create_electricity_graph() 
    # add units
    prep_data_transport["material_intensities"] = prism.Q_(prep_data_transport["material_intensities"], "kg/kg/year")

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
        # material compositions of transport infrastructure in kg/kg/year
    pipelines_materials_data = pd.read_csv(path_base / "data" / "raw" / "fossil_fuels" / "Pipelines" / "Pipelines_materials_edit.csv")
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

    #Expand 2020 values across 1880-2100
    value_2020 = pipelines_materials_xr.sel(Cohort=2020)
    year_range = np.arange(1880, 2101)
    pipelines_materials_xr = value_2020.expand_dims(Cohort=year_range).transpose("Material", "Cohort", "Type")
    pipelines_materials_xr = prism.Q_(pipelines_materials_xr, "kg/kg/year")

    #Rebroadcast to standard technology names from TIMER, and convert coordinate type back to python strings (since rebroadcast changes it to numpy strings)
    pipelines_materials_xr = fossil_fuel_knowledge_graph.rebroadcast_xarray(pipelines_materials_xr, output_coords=FF_TECHNOLOGIES, dim="Type")
    pipelines_materials_xr = pipelines_materials_xr.assign_coords(Type=np.array(pipelines_materials_xr.Type.values, dtype=object)) # rebroadcast_xarray changes the type of the coordinates to numpy strings (np.str_), so convert back to python strings (str)
   
    #  # bring preprocessing data into a generic format for the model
    prep_data_pipelines = {}
    prep_data_pipelines["material_intensities"] = pipelines_materials_xr
    prep_data_pipelines["knowledge_graph"] = create_fossil_fuel_graph() 
    # add units
    prep_data_pipelines["material_intensities"] = prism.Q_(prep_data_pipelines["material_intensities"], "kg/km/year")

    return prep_data_pipelines