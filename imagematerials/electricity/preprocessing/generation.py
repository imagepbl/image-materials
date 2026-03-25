#%% Import modules and constants
import pandas as pd
import numpy as np
from pathlib import Path
import pint
import xarray as xr
from importlib.resources import files

import prism
from imagematerials.read_mym import read_mym_df
from imagematerials.util import dataset_to_array, pandas_to_xarray, convert_lifetime
from imagematerials.concepts import create_electricity_graph, create_region_graph
from imagematerials.electricity.utils import (
    MNLogit, 
    stock_tail, 
    create_prep_data, 
    logistic, 
    quadratic,
    interpolate_xr, 
    add_historic_stock, 
    calculate_grid_growth, 
    calculate_fraction_underground,  
    normalize_selected_techs,
    calculate_storage_market_shares
)
from imagematerials.electricity.preprocessing.circular_economy_measures import (
    apply_ce_measures_to_elc_generation
)

from imagematerials.constants import IMAGE_REGIONS

from imagematerials.electricity.constants import (
    STANDARD_SCEN_EXTERNAL_DATA,
    YEAR_FIRST_GRID,
    SENS_ANALYSIS,
    EPG_TECHNOLOGIES,
    STD_LIFETIMES_ELECTR,
    EV_BATTERIES,
    unit_mapping
)

# for ttesting, remove later:
END_YEAR = 2100
INTERMEDIATE_YEAR = 2080


#############################################################################################################
#############################################################################################################


def get_preprocessing_data_gen(path_base: str, climate_policy_config: dict, circular_economy_config: dict, scenario: str, year_start: int, year_end: int, year_out: int):
    """ Prepare preprocessing input data for the electricity generation sub-module.

    This function reads scenario-dependent power generation data from a combination of IMAGE/TIMER 
    outputs (generation capacity) and external data files (lifetimes, material intensities). It 
    converts them into standardized xarray structures, applies interpolation and optional circular 
    economy (CE) measures, and returns all inputs in a unified format required by the stock model.

    Parameters
    ----------
    path_base : str
        Base directory containing external electricity grid data.
    climate_policy_config : dict
        Configuration dictionary providing paths to IMAGE/TIMER input files.
    circular_economy_config : dict or None
        Configuration for circular economy measures. If provided, material
        intensities and lifetimes are adjusted accordingly.
    scenario : str
        Name of the electricity data scenario; falls back to the standard
        scenario if not available. While the climate_policy_config directs to the IMAGE/TIMER
        scenario files used, this parameter specifies which set of external data files are 
        used. The scenarios used here should be consistent with those used in climate_policy_config.
        #TODO: come up with a better way to ensure consistency between the two scenario specifications.
    year_start : int
        First simulation year.
    year_end : int
        Last simulation year.
    year_out : int
        Last year that is transmitted to the stock model (year_out <= year_end). 

    Returns
    -------
    prep_data : dict
        Dictionary containing preprocessed model inputs:
        - "lifetimes": dict of lifetime distributions per technology
        - "stocks": xarray.DataArray of generation capacity (MW)
        - "material_intensities": xarray.DataArray (g/MW)
        - "knowledge_graph": electricity knowledge graph object
        - "set_unit_flexible": unit placeholder used internally by the model
    """

    path_external_data_scenario = Path(path_base, "electricity", scenario)
    # test if path_external_data_scenario exists and if not set to standard scenario
    if not path_external_data_scenario.exists():
        path_external_data_scenario = Path(path_base, "electricity", STANDARD_SCEN_EXTERNAL_DATA)

    # assert path_external_data_scenario.is_dir()

    ###########################################################################################################
    # Read in files #

    # 1. External Data --------------------------------------------- 

    # lifetimes of Gcap tech (original data according to van Vuuren 2006, PhD Thesis)
    gcap_lifetime_data = pd.read_csv(path_external_data_scenario / 'LTTechnical_dynamic.csv', index_col=['Year','DIM_1'])        
    
    # material compositions of electricity generation tecnologies (g/MW)
    gcap_materials_data = pd.read_csv(path_external_data_scenario / 'composition_generation.csv',index_col=[0,1]).transpose()

    # 2. IMAGE/TIMER files -----------------------------------------
    # Generation capacity (stock demand per generation technology) in MW peak capacity
    gcap_data = read_mym_df(
        climate_policy_config["config_file_path"] /
        climate_policy_config["data_files"]['GCap']
    )
    ###########################################################################################################
    # Transform to xarray #
    knowledge_graph_region = create_region_graph()
    knowledge_graph_electr = create_electricity_graph()
    

    # Lifetimes -------
    values = gcap_lifetime_data["TechnicalLT"].unstack().to_numpy(dtype=float)
    # Create coordinates
    times = gcap_lifetime_data.index.levels[0].to_numpy()
    types = gcap_lifetime_data.index.levels[1].to_numpy()
    scipy_params = ["mean", "stdev"]
    # Build full array: shape (ScipyParam, Time, Type)
    data_array = np.stack([values, np.full_like(values, np.nan)], axis=0)
    # Create DataArray
    gcap_lifetime_xr = xr.DataArray(
        data_array,
        dims=["DistributionParams", "Cohort", "Type"],
        coords={
            "DistributionParams": scipy_params,
            "Cohort": times,
            "Type": [str(r) for r in types]
        },
        name="GenerationLifetime"
    )
    gcap_lifetime_xr = prism.Q_(gcap_lifetime_xr, "year")
    gcap_lifetime_xr = knowledge_graph_electr.rebroadcast_xarray(gcap_lifetime_xr, output_coords=EPG_TECHNOLOGIES, dim="Type") # convert technology names to the standard names from TIMER
    gcap_lifetime_xr = gcap_lifetime_xr.assign_coords(Type=np.array(gcap_lifetime_xr.Type.values, dtype=object)) # rebroadcast_xarray changes the type of the coordinates to numpy strings (np.str_), so convert back to python strings (str)


    # Material Intensities -------
    # Extract coordinate labels
    gcap_materials_data.columns = gcap_materials_data.columns.rename([None, None]) # remove column MultiIndex name "g/MW" as it causes issues when converting to xarray
    years = sorted(gcap_materials_data.columns.get_level_values(0).unique())
    techs = gcap_materials_data.columns.get_level_values(1).unique()
    materials = gcap_materials_data.index
    # Convert to 3D array: (Material, Year, Tech)
    data_array = gcap_materials_data.to_numpy().reshape(len(materials),len(years), len(techs))
    # Build xarray DataArray
    gcap_materials_xr = xr.DataArray(
        data_array,
        dims=('material', 'Cohort', 'Type'),
        coords={
            'material': materials,
            'Cohort': years,
            'Type': techs,
        },
        name='GenerationMaterialIntensities'
    )
    gcap_materials_xr = prism.Q_(gcap_materials_xr, "g/MW")
    gcap_materials_xr = knowledge_graph_electr.rebroadcast_xarray(gcap_materials_xr, output_coords=EPG_TECHNOLOGIES, dim="Type")
    gcap_materials_xr = gcap_materials_xr.assign_coords(Type=np.array(gcap_materials_xr.Type.values, dtype=object)) # rebroadcast_xarray changes the type of the coordinates to numpy strings (np.str_), so convert back to python strings (str)

    # Gcap ------
    gcap_data = gcap_data.loc[~gcap_data['DIM_1'].isin([27,28])]  # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies
    gcap_data = gcap_data.loc[gcap_data['time'].isin(range(year_start, year_end + 1)), ['time', 'DIM_1', *range(1, len(EPG_TECHNOLOGIES) + 1)]]  # only keep relevant years and technology columns
    # Extract coordinate labels
    years = sorted(gcap_data['time'].unique())
    regions = sorted(gcap_data['DIM_1'].unique())
    techs = list(range(1, len(EPG_TECHNOLOGIES)+1))
    # Convert to 3D array: (Year, Region, Tech)
    data_array = gcap_data[techs].to_numpy().reshape(len(years), len(regions), len(techs))
    # Build xarray DataArray
    gcap_xr = xr.DataArray(
        data_array,
        dims=('Time', 'Region', 'Type'),
        coords={
            'Time': years,
            'Region': [str(r) for r in regions],
            'Type': [str(r) for r in techs]
        },
        name='GCap'
    )
    gcap_xr = prism.Q_(gcap_xr, "MW")
    gcap_xr = knowledge_graph_region.rebroadcast_xarray(gcap_xr, output_coords=IMAGE_REGIONS, dim="Region") 
    gcap_xr = knowledge_graph_electr.rebroadcast_xarray(gcap_xr, output_coords=EPG_TECHNOLOGIES, dim="Type")
    gcap_xr = gcap_xr.assign_coords(Type=np.array(gcap_xr.Type.values, dtype=object)) # rebroadcast_xarray changes the type of the coordinates to numpy strings (np.str_), so convert back to python strings (str)

    ###########################################################################################################
    # Interpolate #

    # interpolate_xr: The lifetimes & material intensities are only given for specific years (2020 and 2050), so we linearly interpolate to get values for the years 2020-2050.
    # The values before 2020 are kept constant at the 2020 level, and the values after 2050 are kept constant at the 2050 level.
    gcap_lifetime_xr_interp = interpolate_xr(gcap_lifetime_xr, YEAR_FIRST_GRID, year_out)
    gcap_lifetime_xr_interp.loc[dict(DistributionParams="stdev")] = gcap_lifetime_xr_interp.loc[dict(DistributionParams="mean")] * STD_LIFETIMES_ELECTR
    gcap_materials_xr_interp = interpolate_xr(gcap_materials_xr, YEAR_FIRST_GRID, year_out)

    # TIMER data only start in 1971, so we add a historic tail back to YEAR_FIRST_GRID=1921 #TODO to be adjusted
    gcap_xr_interp = add_historic_stock(gcap_xr, YEAR_FIRST_GRID)


    ###########################################################################################################
    # CE measures #

    # Depending on circular economy scenario, apply different measures
    if circular_economy_config is not None:
        gcap_materials_xr_interp, gcap_lifetime_xr_interp = apply_ce_measures_to_elc_generation(gcap_materials_xr_interp,
                                                                                                gcap_lifetime_xr_interp,
                                                                                                circular_economy_config)
        

    ###########################################################################################################
    # Prep_data File #

    # The lifetimes are converted to the proper format for the model (dictionary with keys:distribution name, values:datarrays containing distribution parameters)
    gcap_lifetime_xr_interp = convert_lifetime(gcap_lifetime_xr_interp)
    
    # bring preprocessing data into a generic format for the model
    prep_data = {}
    prep_data["lifetimes"] = gcap_lifetime_xr_interp
    prep_data["stocks"] = gcap_xr_interp
    prep_data["material_intensities"] = gcap_materials_xr_interp
    prep_data["knowledge_graph"] = create_electricity_graph()
    # add units
    prep_data["stocks"] = prism.Q_(prep_data["stocks"], "MW")
    prep_data["material_intensities"] = prism.Q_(prep_data["material_intensities"], "g/MW")
    prep_data["set_unit_flexible"] = prism.U_(prep_data["stocks"]) # prism.U_ gives the unit back
    # set_unit_flexible is needed by the model to deal with the fact the in the beginning of the model it doesn't know th data yet and needs to work with a placeholder/flexible unit (see model.py) 

    return prep_data