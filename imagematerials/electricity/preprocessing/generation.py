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
    apply_ce_measures_to_elc, 
    normalize_selected_techs,
    calculate_storage_market_shares
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
        if "narrow_product" in circular_economy_config.keys():
            ce_scen = "narrow_product"
            target_year          = circular_economy_config[ce_scen]['electricity']['target_year']
            base_year            = circular_economy_config[ce_scen]['electricity']['base_year']
            implementation_rate  = circular_economy_config[ce_scen]['electricity']['implementation_rate']
            gen_weight_change_pc = circular_economy_config[ce_scen]['electricity']['generation']['weight_change_pc']

            gcap_materials_xr_interp = apply_ce_measures_to_elc(
                gcap_materials_xr_interp,
                base_year           = base_year,
                target_year         = target_year,
                change              = gen_weight_change_pc,
                implementation_rate = implementation_rate
            )
            print("narrow|lightweighting applied to ", gcap_materials_xr_interp.name)

        if "slow" in circular_economy_config.keys():
            ce_scen = "slow"
            target_year          = circular_economy_config[ce_scen]['electricity']['target_year']
            base_year            = circular_economy_config[ce_scen]['electricity']['base_year']
            implementation_rate  = circular_economy_config[ce_scen]['electricity']['implementation_rate']
            gen_lifetime_change_pc = circular_economy_config[ce_scen]['electricity']['generation']['lifetime_increase_percent']

            gcap_lifetime_xr_interp = apply_ce_measures_to_elc(
                gcap_lifetime_xr_interp,
                base_year           = base_year,
                target_year         = target_year,
                change              = gen_lifetime_change_pc,
                implementation_rate = implementation_rate,
                data_type           = "lifetime"
            )
            print("slow|lifetime increase applied to ", gcap_lifetime_xr_interp.name)
        

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