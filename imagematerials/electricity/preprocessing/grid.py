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
from imagematerials.electricity.preprocessing.circular_economy_measures import (
    apply_ce_measures_to_elc_grid
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

def get_preprocessing_data_grid(path_base: str, climate_policy_config: dict, circular_economy_config: dict, scenario: str, year_start: int, year_end: int, year_out: int):
    """ Prepare preprocessing input data for the electricity grid sub-module.

    This function reads static and scenario-dependent grid data (line lengths,
    lifetimes, material intensities, and grid additions) from a combination of IMAGE/TIMER 
    outputs and external data files. It derives all variables required for subsequent stock 
    modeling of grid lines and grid additions (transformers and substations). For this, it 
    interpolates them over time, extends them with historic stock, and converts them into 
    the generic preprocessing format used by the stock model. The function allows for the 
    integration of climate policy and circular economy policy scenarios via the provided 
    configuration inputs.

    Parameters
    ----------
    path_base : str
        Base directory containing external electricity grid data.
    climate_policy_config : dict
        Configuration dictionary providing paths to IMAGE/TIMER input files
        (e.g. generation capacity and GDP per capita).
    circular_economy_config : dict
        Circular economy configuration (currently not used but kept for interface consistency).
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
    prep_data_lines : dict
        Preprocessing data for grid lines, containing lifetimes, stock time
        series, material intensities, the electricity knowledge graph, and a
        flexible unit placeholder.
    prep_data_additions : dict
        Preprocessing data for grid additions (transformers and substations),
        structured analogously to `prep_data_lines`.
    """    

    path_external_data_standard = Path(path_base,  "electricity", "standard_data")
    path_external_data_scenario = Path(path_base,  "electricity", scenario)
    
    # test if path_external_data_scenario exists and if not set to standard scenario
    if not path_external_data_scenario.exists():
        path_external_data_scenario = Path(path_base, "electricity", STANDARD_SCEN_EXTERNAL_DATA)

    # assert path_external_data_standard.is_dir()
    # assert path_external_data_scenario.is_dir() 

    ###########################################################################################################
    # Read in files #

    # lenght of the High-voltage (Hv) lines in the grid, based on Open Street Map (OSM) analysis (km)
    grid_length_hv_data    = pd.read_csv(path_external_data_standard /'grid_length_Hv.csv', index_col=0, names=None).transpose()    
    # Ratio between the length of Medium-voltage (Mv) and Low-voltage (Lv) lines in relation to Hv lines (km Lv /km Hv) & (km Mv/ km Hv)
    ratio_hv               = pd.read_csv(path_external_data_standard / 'grid_ratio_voltage_lines.csv', index_col=0)
    # Regression constants in the linear function used to determine the relation between income & the percentage of underground power-lines (% underground = mult * gdp/cap + add)
    ratio_underground      = pd.read_csv(path_external_data_standard / 'grid_ratio_underground.csv', index_col=[0,1])
    # Transformers & substations per km of grid (Hv, Mv & Lv, in units/km)
    ratio_grid_additions   = pd.read_csv(path_external_data_standard / 'grid_additions.csv', index_col=0)


    # dynamic or scenario-dependent data (lifetimes & material intensity)

    # Average lifetime in years of grid elements
    grid_lifetime_data     = pd.read_csv(path_external_data_scenario  / 'operational_lifetime_grid.csv', index_col=0)
    # Material intensity of grid lines (Hv, Mv & Lv; specific for underground vs. aboveground lines) (kg/km)
    materials_lines_data        = pd.read_csv(path_external_data_scenario / 'Materials_grid_dynamic.csv', index_col=[0,1])
    # Material Intensity for additional infrastructure required for grid connections, such as transformers & substations (kg/unit)
    materials_additions_data    = pd.read_csv(path_external_data_scenario / 'Materials_grid_additions.csv', index_col=[0,1])


    # 2. IMAGE/TIMER files ====================================================================================

    # Generation capacity (stock & inflow/new) in MW peak capacity, FILES from TIMER
    # gcap_data = read_mym_df(path_image_output / 'GCap.out')
    gcap_data = read_mym_df(climate_policy_config["config_file_path"] / climate_policy_config["data_files"]['GCap'])

    # GDP per capita (US-dollar 2005, ppp), used to derive underground-aboveground ratio based on income levels
    # gdp_pc_data: pd.DataFrame   = read_mym_df(Path(path_base, "data", "raw", "image", scen_folder, "Socioeconomic", "gdp_pc.scn"))
    gdp_pc_data = read_mym_df(climate_policy_config["config_file_path"] / climate_policy_config["data_files"]['gdp_pc'])

    ###########################################################################################################
    # Transform to xarray #

    knowledge_graph_region = create_region_graph()
    knowledge_graph_electr = create_electricity_graph()


    # # Grid Lines -----------
    grid_length_hv_data.columns.name = None # drop column name ('km')
    # line_types = ["hv_lines_overhead","hv_lines_underground","mv_lines_overhead","mv_lines_underground", "lv_lines_overhead","lv_lines_underground"]  
    line_types = ["HV - Lines - Overhead","HV - Lines - Underground","MV - Lines - Overhead","MV - Lines - Underground", "LV - Lines - Overhead","LV - Lines - Underground"]
    years = np.arange(year_start,year_out+1,1)
    grid_lines = xr.DataArray(
        np.full((len(years), len(grid_length_hv_data.columns), len(line_types)), np.nan),
        dims=("Time","Region","Type"),
        coords={
            "Time": years.astype(int),
            "Region": grid_length_hv_data.columns.astype(str),
            "Type": line_types
        },
        name="GridLength"
    )
    # fill all years with the data read in for HV in 2016, as they will be later scaled by multiplying with a growth factor relative to the year 2016
    grid_lines.loc[{"Time":slice(year_start, year_end), "Type":"HV - Lines - Overhead"}] = grid_length_hv_data.values.reshape(-1)
    grid_lines = prism.Q_(grid_lines, "km")
    grid_lines = knowledge_graph_region.rebroadcast_xarray(grid_lines, output_coords=IMAGE_REGIONS, dim="Region") # convert region names to the standard names from IMAGE


    # # Grid Additions -----------
    additions_types = ["HV - Transformers","HV - Substations","MV - Transformers","MV - Substations", "LV - Transformers","LV - Substations"]
    years = np.arange(year_start,year_out+1,1)
    grid_additions = xr.DataArray(
        np.full((len(years), len(IMAGE_REGIONS), len(additions_types)), np.nan),
        dims=("Time","Region","Type"),
        coords={
            "Time": years.astype(int),
            "Region": IMAGE_REGIONS,
            "Type": additions_types
        },
        name="GridAdditions"
    )
    grid_additions = prism.Q_(grid_additions, "count")


    # # Lifetimes -------
    # data only for lines, substations and transformer -> bring in knowledge_graph format: HV - Lines, MV - Lines, LV - Lines, HV - Transformers, etc.
    expanded_data = {}
    tech_types = []
    for tech in ["Lines", "Transformers", "Substations"]:
        for level in ["HV", "MV", "LV"]:
            new_col = f"{level} - {tech}"
            tech_types.append(new_col)
            expanded_data[new_col] = grid_lifetime_data[tech]
    grid_lifetime_data = pd.DataFrame(expanded_data, index=grid_lifetime_data.index)

    values = grid_lifetime_data.to_numpy(dtype=float)
    # Create coordinates
    times = grid_lifetime_data.index.to_numpy()
    scipy_params = ["mean", "stdev"]
    # Build full array: shape (ScipyParam, Time, Type)
    data_array = np.stack([values, np.full_like(values, np.nan)], axis=0)
    # Create DataArray
    grid_lifetime = xr.DataArray(
        data_array,
        dims=["DistributionParams", "Cohort", "Type"],
        coords={
            "DistributionParams": scipy_params,
            "Cohort": times.astype(int),
            "Type": [str(r) for r in tech_types]
        },
        name="GridLifetime"
    )
    grid_lifetime = prism.Q_(grid_lifetime, "year")


    # # Material Intensities -------

    # Lines ---
    materials_lines = materials_lines_data.reset_index().rename(columns={'year': 'Cohort', 'kg/km': 'Type'})
    # turn "HV Overhead" -> "HV - Lines - Overhead"
    materials_lines['Type'] = materials_lines['Type'].str.split(n=1).apply(lambda x: f"{x[0]} - Lines - {x[1]}")
    # convert to xarray: material columns become a new 'material' dimension
    materials_lines = materials_lines.set_index(['Cohort', 'Type']).to_xarray()
    materials_lines = materials_lines.to_array(dim='material').rename("GridMaterialIntensitiesLines")
    materials_lines = materials_lines.transpose('Cohort', 'Type', 'material') # reorder dimensions
    materials_lines = prism.Q_(materials_lines, "kg/km")
    materials_lines = materials_lines.reindex(Type=grid_lines.Type)  # ensure same order of types as in stock dataarray

    # Additions ---
    materials_additions = materials_additions_data.reset_index().rename(columns={'year': 'Cohort', 'kg/unit': 'Type'})
    # turn "HV Overhead" -> "HV - Lines - Overhead"
    materials_additions['Type'] = materials_additions['Type'].str.split(n=1).apply(lambda x: f"{x[0]} - {x[1]}")
    # convert to xarray: material columns become a new 'material' dimension
    materials_additions = materials_additions.set_index(['Cohort', 'Type']).to_xarray()
    materials_additions = materials_additions.to_array(dim='material').rename("GridMaterialIntensitiesAdditions")
    materials_additions = materials_additions.transpose('Cohort', 'Type', 'material') # reorder dimensions
    materials_additions = prism.Q_(materials_additions, "kg/count")
    materials_additions = materials_additions.reindex(Type=grid_additions.Type)  # ensure same order of types as in stock dataarray


    # # Gcap ------
    gcap_data = gcap_data.loc[~gcap_data['DIM_1'].isin([27,28])]  # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies
    gcap_data = gcap_data.loc[gcap_data['time'].isin(range(year_start, year_end + 1)), ['time', 'DIM_1', *range(1, len(EPG_TECHNOLOGIES) + 1)]]  # only keep relevant years and technology columns
    # Extract coordinate labels
    years = sorted(gcap_data['time'].unique())
    regions = sorted(gcap_data['DIM_1'].unique())
    techs = list(range(1, len(EPG_TECHNOLOGIES)+1))
    # Convert to 3D array: (Year, Region, Tech)
    data_array = gcap_data[techs].to_numpy().reshape(len(years), len(regions), len(techs))
    # Build xarray DataArray
    gcap = xr.DataArray(
        data_array,
        dims=('Time', 'Region', 'Type'),
        coords={
            'Time': years,
            'Region': [str(r) for r in regions],
            'Type': [str(r) for r in techs]
        },
        name='GCap'
    )
    gcap = prism.Q_(gcap, "MW")
    gcap = knowledge_graph_region.rebroadcast_xarray(gcap, output_coords=IMAGE_REGIONS, dim="Region") 
    gcap = knowledge_graph_electr.rebroadcast_xarray(gcap, output_coords=EPG_TECHNOLOGIES, dim="Type")
    gcap = gcap.assign_coords(Type=np.array(gcap.Type.values, dtype=object)) # rebroadcast_xarray changes the type of the coordinates to numpy strings (np.str_), so convert back to python strings (str)


    # # GDP ------
    gdp_pc_data = gdp_pc_data.iloc[:, :len(IMAGE_REGIONS)] # exclude empty column (27) and totals (28)
    # Extract coordinate labels
    years = sorted(gdp_pc_data.index)
    regions = sorted(gdp_pc_data.columns)
    # Convert to DD array: (Year, Region, Tech)
    data_array = gdp_pc_data.to_numpy() #.reshape(len(years), len(regions))
    # Build xarray DataArray
    gdp_pc = xr.DataArray(
        data_array,
        dims=('Time', 'Region'),
        coords={
            'Time': years,
            'Region': [str(r) for r in regions]
        },
        name='GDPPerCapita'
    )
    gdp_pc = prism.Q_(gdp_pc, "USD/person")
    gdp_pc = knowledge_graph_region.rebroadcast_xarray(gdp_pc, output_coords=IMAGE_REGIONS, dim="Region") 


    # # line ratios ------
    # region specific ratio between the length of Hv and Mv and LV networks
    ds_ratio_hv = ratio_hv.to_xarray()
    ds_ratio_hv = ds_ratio_hv.swap_dims({'km/km': 'Region'}) # change the dimension
    ds_ratio_hv = ds_ratio_hv.rename({'km/km': 'Region'}) # rename the coordinate
    ds_ratio_hv = ds_ratio_hv.set_index(Region='Region')
    ds_ratio_hv["HV to MV"] = prism.Q_(ds_ratio_hv["HV to MV"], "km/km")
    ds_ratio_hv["HV to LV"] = prism.Q_(ds_ratio_hv["HV to LV"], "km/km")
    ds_ratio_hv = xr.Dataset({ # rebroadcast both xarrays in dataset to IMAGE regions
        name: knowledge_graph_region.rebroadcast_xarray(da, output_coords=IMAGE_REGIONS, dim="Region")
        for name, da in ds_ratio_hv.data_vars.items()
    }, attrs=ds_ratio_hv.attrs)


    ###########################################################################################################
    # Calculate variables #

    # 1. calculate growth factors for the grid lines ------------------------------------------------------------

    grid_growth = calculate_grid_growth(gcap, grid_lines)
    
    # 2. calculate grid lengths (overhead) with ratios & growth factors -----------------------------------------

    # calculate extend of MV and LV networks in 2016 based on Hv network and fixed ratios
    grid_lines.loc[{"Type": "MV - Lines - Overhead"}] = grid_lines.loc[{"Type": "HV - Lines - Overhead"}] * ds_ratio_hv["HV to MV"]
    grid_lines.loc[{"Type": "LV - Lines - Overhead"}] = grid_lines.loc[{"Type": "HV - Lines - Overhead"}] * ds_ratio_hv["HV to LV"]
    # scale line lengths based on growth factors
    grid_lines = (grid_lines * grid_growth).rename(grid_lines.name).assign_attrs(grid_lines.attrs) # calculate line lengths over time based on growth factors; restore name and attributes (lost during multiplication)


    # 3. calculate underground lines ----------------------------------------------------------------------------
    
    # determine length of underground and aboveground grid lines based on GDP/capita: fraction_underground = mult * gdp_pc + add
    # Based on insights from Kalt et al. 2021, we adjust the underground ratios downwards for non-European regions

    fraction_lines_above_below = calculate_fraction_underground(grid_lines, gdp_pc, ratio_underground)

    for level in ["HV", "MV", "LV"]:
        over = f"{level} - Lines - Overhead"
        under = f"{level} - Lines - Underground"
        grid_lines.loc[dict(Type=under)] = grid_lines.sel(Type=over) # copy the aboveground length into the underground length, both contain now the total length, to be multiplied with the fractions next

    grid_lines = (grid_lines * fraction_lines_above_below).rename(grid_lines.name).assign_attrs(grid_lines.attrs) # calculate line lengths over time based on growth factors; restore name and attributes (lost during multiplication)


    # 4. calculate number of substations & transformers based on line lengths & fixed ratios -------------------------------------------------------------
    
    ureg = grid_lines.data._REGISTRY # extract unit registry from pint xarray dataarray to be able to define units for the calculations below
    for level in ["HV", "MV", "LV"]:
        grid_additions.loc[dict(Type=f"{level} - Transformers")] = (grid_lines.sel(Type=f"{level} - Lines - Overhead") + grid_lines.sel(Type=f"{level} - Lines - Underground")) * (ratio_grid_additions.loc['Transformers',level] * (ureg.count/ureg.kilometer))  # pandas dataframe does not support units itself -> workaround: manually multiply with units count/km
        grid_additions.loc[dict(Type=f"{level} - Substations")]  = (grid_lines.sel(Type=f"{level} - Lines - Overhead") + grid_lines.sel(Type=f"{level} - Lines - Underground")) * (ratio_grid_additions.loc['Substations',level] * (ureg.count/ureg.kilometer))  


    ###########################################################################################################
    # Interpolate #

    grid_lifetime_interp        = interpolate_xr(grid_lifetime, YEAR_FIRST_GRID, year_out)
    materials_lines_interp      = interpolate_xr(materials_lines, YEAR_FIRST_GRID, year_out)
    materials_additions_interp  = interpolate_xr(materials_additions, YEAR_FIRST_GRID, year_out)

    grid_lines_interp       = add_historic_stock(grid_lines, YEAR_FIRST_GRID)
    grid_additions_interp   = add_historic_stock(grid_additions, YEAR_FIRST_GRID)


    # calculate standard deviation as a fixed fraction of the mean lifetime
    grid_lifetime_interp.loc[{"DistributionParams": "stdev"}] = grid_lifetime_interp.sel({"DistributionParams": "mean"}) * STD_LIFETIMES_ELECTR


    ###########################################################################################################
    # CE measures #

    # Depending on circular economy scenario, apply different measures
    if circular_economy_config is not None:

        materials_additions_interp, grid_lifetime_interp = apply_ce_measures_to_elc_grid(materials_additions_interp,
                                                                                          grid_lifetime_interp,
                                                                                          circular_economy_config)

    ###########################################################################################################
    # Prep_data File #

    # the lifetimes are converted to the proper format for the model (dictionary with keys:distribution name, values:datarrays containing distribution parameters)
    grid_lifetime_interp_conv = convert_lifetime(grid_lifetime_interp)

    # bring preprocessing data into a generic format for the model
    prep_data_lines = {}
    prep_data_lines["lifetimes"]            = grid_lifetime_interp_conv
    prep_data_lines["stocks"]               = grid_lines_interp
    prep_data_lines["material_intensities"] = materials_lines_interp
    prep_data_lines["knowledge_graph"]      = create_electricity_graph()
    prep_data_lines["set_unit_flexible"]    = str(prism.U_(prep_data_lines["stocks"])) # add unit (prism.U_ gives the unit back)
    # set_unit_flexible is needed by the model to deal with the fact the in the beginning of the model it doesn't know th data yet and needs to work with a placeholder/flexible unit (see model.py) 
    prep_data_additions = {}
    prep_data_additions["lifetimes"]            = grid_lifetime_interp_conv
    prep_data_additions["stocks"]               = grid_additions_interp
    prep_data_additions["material_intensities"] = materials_additions_interp
    prep_data_additions["knowledge_graph"]      = create_electricity_graph()
    prep_data_additions["set_unit_flexible"]    = str(prism.U_(prep_data_additions["stocks"])) # add unit (prism.U_ gives the unit back)

    return prep_data_lines, prep_data_additions