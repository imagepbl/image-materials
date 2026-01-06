#%%
import pandas as pd
import numpy as np
import os
import scipy
import warnings
from pathlib import Path
import pint
import xarray as xr
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from matplotlib.ticker import ScalarFormatter

import prism
from imagematerials.distribution import ALL_DISTRIBUTIONS, NAME_TO_DIST
from imagematerials.read_mym import read_mym_df
from imagematerials.util import dataset_to_array, pandas_to_xarray, convert_lifetime
from imagematerials.model import GenericMainModel, GenericStocks, SharesInflowStocks, Maintenance, GenericMaterials, MaterialIntensities
from imagematerials.factory import ModelFactory, Sector
from imagematerials.concepts import create_electricity_graph, create_region_graph
from imagematerials.electricity.utils import add_historic_stock, interpolate_xr, flexible_plot_1panel, calculate_grid_growth, calculate_fraction_underground

from imagematerials.constants import (
    IMAGE_REGIONS,
)

from imagematerials.electricity.constants import (
    YEAR_FIRST_GRID,
    STANDARD_SCEN_EXTERNAL_DATA,
    SENS_ANALYSIS,
    EPG_TECHNOLOGIES,
    EPG_TECHNOLOGIES_VRE,
    STD_LIFETIMES_ELECTR,
    DICT_MATERIALS_COLORS,
    DICT_GRID_COLORS,
    DICT_GRID_STYLES_1,
    DICT_GRID_STYLES_2,
)

from imagematerials.electricity.electr_external_data import (
    df_iea_cu_aps,
    df_iea_cu_nzs,
    df_iea_co_aps,
    df_iea_mn_aps,
    df_iea_ni_aps
)

SCEN = "SSP2"
VARIANT = "VLHO"
# VARIANT = "M_CP"
# VARIANT = "BL"
# VARIANT = "450"
# Define paths ----------------------------------------------------------------------
#YOUR_DIR = "C:\\Users\\Admin\\surfdrive\\Projects\\IRP\\GRO23\\Modelling\\2060\\ELMA"   # Change the running directory here
# os.chdir(YOUR_DIR)
scen_folder = SCEN + "_" + VARIANT
# path_base = Path().resolve() # TODO absolute path of file "preprocessing.py" ? current solution can differ depending on IDE used (?) 
path_current = Path().resolve()
path_base = path_current.parent.parent # base path of the project -> image-materials

path_image_output = Path(path_base, "data", "raw", "image", scen_folder, "EnergyServices")
# path_image_output = Path(path_base, "data", "raw", "image", scen_folder)

# TEST---
path_image_output_SSP2_BL = Path(path_base, "data", "raw", "image", "SSP2_BL")
path_image_output_SSP2_450 = Path(path_base, "data", "raw", "image", "SSP2_450")
path_image_output_SSP2_M_CP = Path(path_base, "data", "raw", "image", "SSP2_M_CP", "EnergyServices")
path_image_output_SSP2_VLHO = Path(path_base, "data", "raw", "image", "SSP2_VLHO", "EnergyServices")
#-----------
path_external_data_standard = Path(path_base, "data", "raw", "electricity", "standard_data")
path_external_data_scenario = Path(path_base, "data", "raw", "electricity", scen_folder)
# test if path_external_data_scenario exists and if not set to standard scenario
if not path_external_data_scenario.exists():
    path_external_data_scenario = Path(path_base, "data", "raw", "electricity", STANDARD_SCEN_EXTERNAL_DATA)
print(f"Path to image output: {path_image_output}")

assert path_image_output.is_dir()
assert path_external_data_standard.is_dir()
assert path_external_data_scenario.is_dir()

# create the folder out_test if it does not exist
if not (path_base / 'imagematerials' / 'electricity' / 'out_test').is_dir():
    (path_base / 'imagematerials' / 'electricity' / 'out_test').mkdir(parents=True)
    
    
idx = pd.IndexSlice             # needed for slicing multi-index

    

    

###########################################################################################################
###########################################################################################################
#%% 3) TRANSMISSION GRID 
###########################################################################################################
###########################################################################################################

def get_preprocessing_data_grid(base_dir: str, SCEN, VARIANT, year_start, year_end, year_out): #, climate_policy_config: dict, circular_economy_config: dict

    scen_folder = SCEN + "_" + VARIANT
    path_image_output = Path(path_base, "data", "raw", "image", scen_folder, "EnergyServices")
    path_external_data_standard = Path(path_base, "data", "raw", "electricity", "standard_data")
    path_external_data_scenario = Path(path_base, "data", "raw", "electricity", scen_folder)
    
    # test if path_external_data_scenario exists and if not set to standard scenario
    if not path_external_data_scenario.exists():
        path_external_data_scenario = Path(path_base, "data", "raw", "electricity", STANDARD_SCEN_EXTERNAL_DATA)

    assert path_image_output.is_dir()
    assert path_external_data_standard.is_dir()
    assert path_external_data_scenario.is_dir() 

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
    gcap_data: pd.DataFrame     = read_mym_df(path_image_output / 'Gcap.out')
    # GDP per capita (US-dollar 2005, ppp), used to derive underground-aboveground ratio based on income levels
    gdp_pc_data: pd.DataFrame   = read_mym_df(Path(path_base, "data", "raw", "image", scen_folder, "Socioeconomic", "gdp_pc.scn"))
    

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
        name="Lifetime"
    )
    grid_lifetime = prism.Q_(grid_lifetime, "year")


    # # Material Intensities -------

    # Lines ---
    materials_lines = materials_lines_data.reset_index().rename(columns={'year': 'Cohort', 'kg/km': 'Type'})
    # turn "HV Overhead" -> "HV - Lines - Overhead"
    materials_lines['Type'] = materials_lines['Type'].str.split(n=1).apply(lambda x: f"{x[0]} - Lines - {x[1]}")
    # convert to xarray: material columns become a new 'material' dimension
    materials_lines = materials_lines.set_index(['Cohort', 'Type']).to_xarray()
    materials_lines = materials_lines.to_array(dim='material').rename("GridMaterialsLines")
    materials_lines = materials_lines.transpose('Cohort', 'Type', 'material') # reorder dimensions
    materials_lines = prism.Q_(materials_lines, "kg/km")
    materials_lines = materials_lines.reindex(Type=grid_lines.Type)  # ensure same order of types as in stock dataarray

    # Additions ---
    materials_additions = materials_additions_data.reset_index().rename(columns={'year': 'Cohort', 'kg/unit': 'Type'})
    # turn "HV Overhead" -> "HV - Lines - Overhead"
    materials_additions['Type'] = materials_additions['Type'].str.split(n=1).apply(lambda x: f"{x[0]} - {x[1]}")
    # convert to xarray: material columns become a new 'material' dimension
    materials_additions = materials_additions.set_index(['Cohort', 'Type']).to_xarray()
    materials_additions = materials_additions.to_array(dim='material').rename("GridMaterialsAdditions")
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


    ###########################################################################################################
    # Prep_data File #

    # calculate standard deviation as a fixed fraction of the mean lifetime
    grid_lifetime_interp.loc[{"DistributionParams": "stdev"}] = grid_lifetime_interp.sel({"DistributionParams": "mean"}) * STD_LIFETIMES_ELECTR
    # the lifetimes are converted to the proper format for the model (dictionary with keys:distribution name, values:datarrays containing distribution parameters)
    grid_lifetime_interp_conv = convert_lifetime(grid_lifetime_interp)


    # bring preprocessing data into a generic format for the model

    prep_data_lines = {}
    prep_data_lines["lifetimes"]            = grid_lifetime_interp_conv
    prep_data_lines["stocks"]               = grid_lines_interp
    prep_data_lines["material_intensities"] = materials_lines_interp
    prep_data_lines["knowledge_graph"]      = create_electricity_graph()
    prep_data_lines["set_unit_flexible"]    = prism.U_(prep_data_lines["stocks"]) # add unit (prism.U_ gives the unit back)
    # set_unit_flexible is needed by the model to deal with the fact the in the beginning of the model it doesn't know th data yet and needs to work with a placeholder/flexible unit (see model.py) 
    prep_data_additions = {}
    prep_data_additions["lifetimes"]            = grid_lifetime_interp_conv
    prep_data_additions["stocks"]               = grid_additions_interp
    prep_data_additions["material_intensities"] = materials_additions_interp
    prep_data_additions["knowledge_graph"]      = create_electricity_graph()
    prep_data_additions["set_unit_flexible"]    = prism.U_(prep_data_additions["stocks"]) # add unit (prism.U_ gives the unit back)

    return prep_data_lines, prep_data_additions



#----------------------------------------------------------------------------------------------------------
###########################################################################################################
#%%% 3.1) Read in files
###########################################################################################################
#----------------------------------------------------------------------------------------------------------

# TODO: remove these numbers from preprocessing
YEAR_START = 1971   # start year of the simulation period
YEAR_END = 2100     # end year of the calculations
YEAR_OUT = 2100     # year of output generation = last year of reporting

# 1. External Data --------------------------------------------- 

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
gcap_data: pd.DataFrame     = read_mym_df(path_image_output / 'Gcap.out')

# GDP per capita (US-dollar 2005, ppp), used to derive underground-aboveground ratio based on income levels
gdp_pc_data: pd.DataFrame   = read_mym_df(Path(path_base, "data", "raw", "image", scen_folder, "Socioeconomic", "gdp_pc.scn"))


###########################################################################################################
#%%% Transform to xarray #
###########################################################################################################

knowledge_graph_region = create_region_graph()
knowledge_graph_electr = create_electricity_graph()

# # Grid Lines -----------
grid_length_hv_data.columns.name = None # drop column name ('km')
# line_types = ["hv_lines_overhead","hv_lines_underground","mv_lines_overhead","mv_lines_underground", "lv_lines_overhead","lv_lines_underground"]  
line_types = ["HV - Lines - Overhead","HV - Lines - Underground","MV - Lines - Overhead","MV - Lines - Underground", "LV - Lines - Overhead","LV - Lines - Underground"]
years = np.arange(YEAR_START,YEAR_OUT+1,1)
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
grid_lines.loc[{"Time":slice(YEAR_START, YEAR_END), "Type":"HV - Lines - Overhead"}] = grid_length_hv_data.values.reshape(-1)
grid_lines = prism.Q_(grid_lines, "km")
grid_lines = knowledge_graph_region.rebroadcast_xarray(grid_lines, output_coords=IMAGE_REGIONS, dim="Region") # convert region names to the standard names from IMAGE



# # Grid Additions -----------
additions_types = ["HV - Transformers","HV - Substations","MV - Transformers","MV - Substations", "LV - Transformers","LV - Substations"]
years = np.arange(YEAR_START,YEAR_OUT+1,1)
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
    name="Lifetime"
)
grid_lifetime = prism.Q_(grid_lifetime, "year")


# # Material Intensities -------

# Lines ---
materials_lines = materials_lines_data.reset_index().rename(columns={'year': 'Cohort', 'kg/km': 'Type'})
# turn "HV Overhead" -> "HV - Lines - Overhead"
materials_lines['Type'] = materials_lines['Type'].str.split(n=1).apply(lambda x: f"{x[0]} - Lines - {x[1]}")
# convert to xarray: material columns become a new 'material' dimension
materials_lines = materials_lines.set_index(['Cohort', 'Type']).to_xarray()
materials_lines = materials_lines.to_array(dim='material').rename("GridMaterialsLines")
materials_lines = materials_lines.transpose('Cohort', 'Type', 'material') # reorder dimensions
materials_lines = prism.Q_(materials_lines, "kg/km")
materials_lines = materials_lines.reindex(Type=grid_lines.Type)  # ensure same order of types as in stock dataarray

# Additions ---
materials_additions = materials_additions_data.reset_index().rename(columns={'year': 'Cohort', 'kg/unit': 'Type'})
# turn "HV Overhead" -> "HV - Lines - Overhead"
materials_additions['Type'] = materials_additions['Type'].str.split(n=1).apply(lambda x: f"{x[0]} - {x[1]}")
# convert to xarray: material columns become a new 'material' dimension
materials_additions = materials_additions.set_index(['Cohort', 'Type']).to_xarray()
materials_additions = materials_additions.to_array(dim='material').rename("GridMaterialsAdditions")
materials_additions = materials_additions.transpose('Cohort', 'Type', 'material') # reorder dimensions
materials_additions = prism.Q_(materials_additions, "kg/count")
materials_additions = materials_additions.reindex(Type=grid_additions.Type)  # ensure same order of types as in stock dataarray


# # Gcap ------
gcap_data = gcap_data.loc[~gcap_data['DIM_1'].isin([27,28])]  # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies
gcap_data = gcap_data.loc[gcap_data['time'].isin(range(YEAR_START, YEAR_END + 1)), ['time', 'DIM_1', *range(1, len(EPG_TECHNOLOGIES) + 1)]]  # only keep relevant years and technology columns
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
#%%% Calculate variables #
###########################################################################################################

# 1. calculate growth factors for the grid lines -------------------------------------------------------------

# regional total (peak) generation capacity is used as a proxy for the grid growth
gcap_total = gcap.sum(dim='Type')
gcap_growth = gcap_total / gcap_total.loc[2016]        # define growth according to 2016 as base year

# copy growth factor for all voltage levels (overhead)
grid_growth = gcap_growth.expand_dims(Type=["HV - Lines - Overhead","MV - Lines - Overhead","LV - Lines - Overhead"]).copy()
# add coordinates for underground lines to match grid length dataarray (set to NaN, as underground lines are later calculated based on aboveground lines & fixed ratios)
grid_growth_expanded = grid_growth.broadcast_like(grid_lines).copy().rename("GridGrowthFactor")

# for HV lines: additional growth is presumed after 2020 based on the fraction of variable renewable energy (vre) generation capacity (solar & wind) (used to be only in the sensitivity variant, but now used in the base case as well)
vre_fraction = gcap.sel(Type=EPG_TECHNOLOGIES_VRE).sum(dim='Type') / gcap_total
# Compute additional/reduced growth
add_growth = vre_fraction * 1             # if value is e.g. 0.2 = 20% additional HV lines per doubling of vre gcap
red_growth = (1 - vre_fraction) * 0.7     
add_growth = add_growth.where(add_growth.Time >= 2020, 0)  # Set pre-2020 values to 0
red_growth = red_growth.where(red_growth.Time >= 2020, 0)
# Create a ramp factor (line length addition/reduction is gradually introduced from 2020 towards 2050)
ramp_factor = xr.DataArray(
    np.clip((add_growth.Time - 2020) / 30, 0, 1),
    coords={"Time": add_growth.Time},
    dims=["Time"]
)
add_growth = add_growth * ramp_factor # Apply ramp factor
red_growth = red_growth * ramp_factor
grid_growth_expanded.loc[{"Type": "HV - Lines - Overhead"}] = grid_growth_expanded.loc[{"Type": "HV - Lines - Overhead"}] + add_growth - red_growth


# 2. calculate grid lengths (overhead) with ratios & growth factors -------------------------------------------------------------

# calculate extend of MV and LV networks in 2016 based on Hv network and fixed ratios
grid_lines.loc[{"Type": "MV - Lines - Overhead"}] = grid_lines.loc[{"Type": "HV - Lines - Overhead"}] * ds_ratio_hv["HV to MV"]
grid_lines.loc[{"Type": "LV - Lines - Overhead"}] = grid_lines.loc[{"Type": "HV - Lines - Overhead"}] * ds_ratio_hv["HV to LV"]
# scale line lengths based on growth factors
grid_lines = (grid_lines * grid_growth_expanded).rename(grid_lines.name).assign_attrs(grid_lines.attrs) # calculate line lengths over time based on growth factors; restore name and attributes (lost during multiplication)


# 3. calculate underground lines -------------------------------------------------
# determine length of underground and aboveground grid lines based on GDP/capita: fraction_underground = mult * gdp_pc + add
# Based on insights from Kalt et al. 2021, we adjust the underground ratios downwards for non-European regions

fraction_lines_above_below = xr.full_like(grid_lines, np.nan).rename("FractionUndergroundAboveground")

gdp_pc_unitless = gdp_pc.pint.dequantify() #work-around for now
for region in IMAGE_REGIONS:
    if region in ['WEU','CEU']:
        select_proxy = 'Europe'
    else:
        select_proxy = 'Other'

    fraction_lines_above_below.loc[{"Region": region, "Type": "HV - Lines - Underground"}] = (gdp_pc_unitless.sel(Region=region) * ratio_underground.loc[idx[select_proxy,'mult'],'HV'] + ratio_underground.loc[idx[select_proxy,'add'],'HV'])/100 # /100 to convert from % to fraction
    fraction_lines_above_below.loc[{"Region": region, "Type": "MV - Lines - Underground"}] = (gdp_pc_unitless.sel(Region=region) * ratio_underground.loc[idx[select_proxy,'mult'],'MV'] + ratio_underground.loc[idx[select_proxy,'add'],'MV'])/100
    fraction_lines_above_below.loc[{"Region": region, "Type": "LV - Lines - Underground"}] = (gdp_pc_unitless.sel(Region=region) * ratio_underground.loc[idx[select_proxy,'mult'],'LV'] + ratio_underground.loc[idx[select_proxy,'add'],'LV'])/100
# fraction must be between 0 and 1
fraction_lines_above_below = fraction_lines_above_below.clip(min=0, max=1)

# MIND! the HV lines found in OSM (+national sources) are considered as the total of the aboveground line length + the underground line length
# currently the total is saved in xarray grid_lines as the aboveground fraction only -> need to split into aboveground & underground based on the calculated ratios
# for this, we copy the aboveground length into the underground length & then multiply both with the coresponding fraction to get the correct lengths
for level in ["HV", "MV", "LV"]:
    over = f"{level} - Lines - Overhead"
    under = f"{level} - Lines - Underground"
    grid_lines.loc[dict(Type=under)] = grid_lines.sel(Type=over) # under = over
    fraction_lines_above_below.loc[dict(Type=over)] = 1-fraction_lines_above_below.loc[dict(Type=under)] # above = 1 - under

grid_lines = (grid_lines * fraction_lines_above_below).rename(grid_lines.name).assign_attrs(grid_lines.attrs) # calculate line lengths over time based on growth factors; restore name and attributes (lost during multiplication)


# 4. calculate number of substations & transformers based on line lengths & fixed ratios -------------------------------------------------------------
ureg = grid_lines.data._REGISTRY # extract unit registry from pint xarray dataarray to be able to define units for the calculations below
for level in ["HV", "MV", "LV"]:
    grid_additions.loc[dict(Type=f"{level} - Transformers")] = (grid_lines.sel(Type=f"{level} - Lines - Overhead") + grid_lines.sel(Type=f"{level} - Lines - Underground")) * (ratio_grid_additions.loc['Transformers',level] * (ureg.count/ureg.kilometer))  # pandas dataframe does not support units itself -> workaround: manually multiply with units count/km
    grid_additions.loc[dict(Type=f"{level} - Substations")]  = (grid_lines.sel(Type=f"{level} - Lines - Overhead") + grid_lines.sel(Type=f"{level} - Lines - Underground")) * (ratio_grid_additions.loc['Substations',level] * (ureg.count/ureg.kilometer))  



###########################################################################################################
#%%% Interpolate #
###########################################################################################################


grid_lifetime_interp = interpolate_xr(grid_lifetime, YEAR_FIRST_GRID, YEAR_OUT)
materials_lines_interp = interpolate_xr(materials_lines, YEAR_FIRST_GRID, YEAR_OUT)
materials_additions_interp = interpolate_xr(materials_additions, YEAR_FIRST_GRID, YEAR_OUT)

grid_lines_interp = add_historic_stock(grid_lines, YEAR_FIRST_GRID)
grid_additions_interp = add_historic_stock(grid_additions, YEAR_FIRST_GRID)





# flexible_plot_1panel(
#     da=grid_additions_interp,
#     x_dim="Time",
#     varying_dims=["Type", "Region"],
#     fixed={"Type": [1, 2, 3, 4,5], "Region": [3]}, #
#     plot_type='scatter'
# )



###########################################################################################################
#%%% 3.3) Prep_data File
###########################################################################################################

# calculate standard deviation as a fixed fraction of the mean lifetime
grid_lifetime_interp.loc[{"DistributionParams": "stdev"}] = grid_lifetime_interp.sel({"DistributionParams": "mean"}) * STD_LIFETIMES_ELECTR
# the lifetimes are converted to the proper format for the model (dictionary with keys:distribution name, values:datarrays containing distribution parameters)
grid_lifetime_interp_conv = convert_lifetime(grid_lifetime_interp)


# bring preprocessing data into a generic format for the model
prep_data_lines = {}
prep_data_lines["lifetimes"] = grid_lifetime_interp_conv
prep_data_lines["stocks"] = grid_lines_interp
prep_data_lines["material_intensities"] = materials_lines_interp
prep_data_lines["knowledge_graph"] = create_electricity_graph()
prep_data_lines["set_unit_flexible"] = prism.U_(prep_data_lines["stocks"]) # add unit (prism.U_ gives the unit back)
# set_unit_flexible is needed by the model to deal with the fact the in the beginning of the model it doesn't know th data yet and needs to work with a placeholder/flexible unit (see model.py) 

prep_data_additions = {}
prep_data_additions["lifetimes"] = grid_lifetime_interp_conv
prep_data_additions["stocks"] = grid_additions_interp
prep_data_additions["material_intensities"] = materials_additions_interp
prep_data_additions["knowledge_graph"] = create_electricity_graph()
prep_data_additions["set_unit_flexible"] = prism.U_(prep_data_additions["stocks"]) # add unit (prism.U_ gives the unit back)



#----------------------------------------------------------------------------------------------------------
###########################################################################################################
#%%% 3.4) Run Stock Model New
###########################################################################################################
# TODO: move this to electricity.py
#----------------------------------------------------------------------------------------------------------

YEAR_START = 1971   # start year of the simulation period
YEAR_END = 2100     # end year of the calculations
YEAR_OUT = 2100     # year of output generation = last year of reporting

prep_data_lines, prep_data_additions = get_preprocessing_data_grid(path_base, SCEN, VARIANT, YEAR_START, YEAR_END, YEAR_OUT)

#%%%% LINES ----------------------------------------------------
# prep_data = create_prep_data(results_dict_lines, conversion_table, unit_mapping)






# # Define the complete timeline, including historic tail
time_start = prep_data_lines["stocks"].coords["Time"].min().values
complete_timeline = prism.Timeline(time_start, YEAR_END, 1) #YEAR_END
simulation_timeline = prism.Timeline(YEAR_START, YEAR_END, 1) #1970


sec_electr_grid_lines = Sector("elc_grid_lines", prep_data_lines)
factory_lines = ModelFactory(
    sec_electr_grid_lines, complete_timeline
    ).add(GenericStocks
    ).add(MaterialIntensities
    )

model_lines = factory_lines.finish()

model_lines.simulate(simulation_timeline)
list(model_lines.elc_grid_lines)



#%%%% ADDITIONS -------------------------------------------------------------------------------------
# prep_data = create_prep_data(results_dict_add, conversion_table, unit_mapping)

# # Define the complete timeline, including historic tail
time_start = prep_data_additions["stocks"].coords["Time"].min().values
complete_timeline = prism.Timeline(time_start, YEAR_END, 1)
simulation_timeline = prism.Timeline(YEAR_START, YEAR_END, 1) #1970


sec_electr_grid_add = Sector("elc_grid_add", prep_data_additions)

factory_add = ModelFactory(
    sec_electr_grid_add, complete_timeline
    ).add(GenericStocks
    ).add(MaterialIntensities
    )

model_add = factory_add.finish()

model_add.simulate(simulation_timeline)
list(model_add.elc_grid_add)


# Define the coordinates of all dimensions.
# Region = list(prep_data["stocks"].coords["Region"].values)
# Time = [t for t in complete_timeline]
# Cohort = Time
# Type = list(prep_data["stocks"].coords["Type"].values)
# material = list(prep_data["material_intensities"].coords["material"].values)

# Create
# main_model_normal = GenericMainModel(
#     complete_timeline, Region=Region, Time=Time, Cohort=Cohort, Type=Type, prep_data=prep_data,
#     compute_materials=True, compute_battery_materials=False, compute_maintenance_materials=False, 
#     material=material)

path_test_plots = Path(path_base, "imagematerials", "electricity", "out_test", "Test", "Figures") #scen_folder


###########################################################################################################
#%%% Comparisons 
###########################################################################################################

path_test = Path(path_base, "imagematerials", "electricity", "out_test", scen_folder, "StockModelOutputs")

def sanitize_attrs(da):
    """ Sanitize the attributes of a DataArray and its coordinates for safe serialization.

    This function converts all attribute values that are not of type str, int, or float 
    into strings. It applies this transformation to both the DataArray's `.attrs` and 
    each coordinate's `.attrs`. This is useful when saving xarray objects to formats 
    like NetCDF, which require attribute values to be basic serializable types.

    Parameters
    ----------
    da : xarray.DataArray
        The input DataArray whose attributes need to be sanitized.

    Returns
    -------
    xarray.DataArray
        A copy of the input DataArray with sanitized attributes.

    Notes
    -----
    - This function does not modify the original DataArray in-place; it returns a copy.
    - It preserves the data, coordinates, and dimensions of the original DataArray.
    """
    # use as:
    # da_example = sanitize_attrs(model_lines.inflow.to_array())
    # da_example.to_netcdf(path_test / "grid_lines_inflow_v0.nc")

    da = da.copy()
    da.attrs = {k: str(v) if not isinstance(v, (str, int, float)) else v
                for k, v in da.attrs.items()}
    for c in da.coords:
        da.coords[c].attrs = {
            k: str(v) if not isinstance(v, (str, int, float)) else v
            for k, v in da.coords[c].attrs.items()
        }
    return da

def compare_da(path, da_new):
    """ Compare a saved DataArray to a new one, ignoring units.

    Parameters
    ----------
    path : str or Path
        Path to the saved DataArray file.
    da_new : xarray.DataArray
        The new DataArray to compare.

    Returns
    -------
    equal : bool
        True if the DataArrays match numerically after removing units.
    diff_nonzero : xarray.DataArray or None
        Differences where values differ; None if equal.
    """
    # use as:
    # compare_da(path_test / "grid_lines_inflow_v0.nc", model_lines.inflow.to_array())
    
    da_old = xr.open_dataarray(path)

    da_old_clean = da_old.pint.dequantify()
    da_new_clean = da_new.pint.dequantify()

    equal = da_old_clean.equals(da_new_clean)
    if not equal:
        diff = da_new_clean - da_old_clean
        diff_nonzero = diff.where(diff != 0, drop=True)
        return equal, diff_nonzero
    return equal, None

# da_old = sanitize_attrs(model_lines.inflow.to_array())
# da_old.to_netcdf(path_test / "grid_lines_inflow_v0.nc")
# da_old = sanitize_attrs(model_lines.inflow_materials.to_array())
# da_old.to_netcdf(path_test / "grid_lines_inflow_materials_v0.nc")
# da_old = sanitize_attrs(model_lines.stock_by_cohort_materials)
# da_old.to_netcdf(path_test / "grid_lines_stock_materials_v0.nc")

# da_old = sanitize_attrs(model_add.inflow.to_array())
# da_old.to_netcdf(path_test / "grid_additions_inflow_v0.nc")
# da_old = sanitize_attrs(model_add.inflow_materials.to_array())
# da_old.to_netcdf(path_test / "grid_additions_inflow_materials_v0.nc")
# da_old = sanitize_attrs(model_add.stock_by_cohort_materials)
# da_old.to_netcdf(path_test / "grid_additions_stock_materials_v0.nc")


compare_da(path_test / "grid_lines_inflow_v0.nc", model_lines.inflow.to_array())
compare_da(path_test / "grid_lines_inflow_materials_v0.nc", model_lines.inflow_materials.to_array())
compare_da(path_test / "grid_lines_stock_materials_v0.nc", model_lines.stock_by_cohort_materials)
compare_da(path_test / "grid_additions_inflow_v0.nc", model_add.inflow.to_array())
compare_da(path_test / "grid_additions_inflow_materials_v0.nc", model_add.inflow_materials.to_array())
compare_da(path_test / "grid_additions_stock_materials_v0.nc", model_add.stock_by_cohort_materials)




###########################################################################################################
#%%% Stocks 
###########################################################################################################

#%%%% 2 models ---------------------------------------------------

data_lines  = model_lines.stocks.copy().sum(dim="Region")
data_add    = model_add.stocks.copy().sum(dim="Region")

data_lines        = data_lines.sel(Time=slice(1925, 2060))
data_add          = data_add.sel(Time=slice(1925, 2060))

# data        = xr.concat([data_lines, data_add], dim='Type')
# data_plot   = data.sum(dim="Region")

types_top    = ['HV - Lines - Overhead', 'HV - Lines - Underground', 'MV - Lines - Overhead', 'MV - Lines - Underground', 
                'LV - Lines - Overhead', 'LV - Lines - Underground'] #, 'LV - Transformers', 'LV - Substations',
types_bottom = ['HV - Substations', 'HV - Transformers', 'MV - Substations', 'MV - Transformers']



fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 8))
linewidth = 2
s_legend = 12
s_label = 14

# Top row:
for t in types_top:
    line, = axes[0].plot(data_lines.Time, data_lines.sel(Type=t), label=t, color=DICT_GRID_STYLES_2[t][0], linestyle=DICT_GRID_STYLES_2[t][1], linewidth=linewidth)

axes[0].set_ylabel("Stocks (# counts/km)", fontsize=s_label)
axes[0].grid(alpha=0.3, linestyle='--')
axes[0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
axes[0].tick_params(axis='both', which='major', labelsize=s_legend) # set font size of axis ticks
axes[0].legend(loc='upper left', fontsize=s_legend) #handles=handles, labels=labels, 

# Bottom row:
for t in types_bottom:
    axes[1].plot(data_add.Time, data_add.sel(Type=t), label=t, color=DICT_GRID_STYLES_2[t][0], linestyle=DICT_GRID_STYLES_2[t][1], linewidth=linewidth)

axes[1].set_xlabel("Time", fontsize=s_label)
axes[1].set_ylabel("Stocks (# counts)", fontsize=s_label)
axes[1].grid(alpha=0.3, linestyle='--')
axes[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
axes[1].tick_params(axis='both', which='major', labelsize=s_legend) # set font size of axis ticks
axes[1].legend(loc='upper left', fontsize=s_legend)

plt.suptitle(f"{scen_folder}: Electricity Grid - Stocks", fontsize=16)

plt.tight_layout()
# fig.savefig(path_test_plots / "Grid_stocks_world.png", dpi=300)
plt.show()



###########################################################################################################
#%%% Inflow 
###########################################################################################################

#%%%% 2 models ---------------------------------------------------

type_variable = "inflow" 
data_lines  = model_lines.inflow.to_array().copy().sum(dim="Region")
data_add    = model_add.inflow.to_array().copy().sum(dim="Region")

data_lines        = data_lines.sel(time=slice(1925, 2060))
data_add          = data_add.sel(time=slice(1925, 2060))

# data        = xr.concat([data_lines, data_add], dim='Type')
# data_plot   = data.sum(dim="Region")

types_top    = ['HV - Lines - Overhead', 'HV - Lines - Underground', 'MV - Lines - Overhead', 'MV - Lines - Underground', 
                'LV - Lines - Overhead', 'LV - Lines - Underground'] #, 'LV - Transformers', 'LV - Substations',
types_bottom = ['HV - Substations', 'HV - Transformers', 'MV - Substations', 'MV - Transformers']



fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 8))
linewidth = 2
s_legend = 12
s_label = 14

# Top row:
for t in types_top:
    line, = axes[0].plot(data_lines.time, data_lines.sel(Type=t), label=t, color=DICT_GRID_STYLES_2[t][0], linestyle=DICT_GRID_STYLES_2[t][1], linewidth=linewidth)

axes[0].set_ylabel(f"{type_variable} (# counts/km)", fontsize=s_label)
axes[0].grid(alpha=0.3, linestyle='--')
axes[0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
axes[0].tick_params(axis='both', which='major', labelsize=s_legend) # set font size of axis ticks
axes[0].legend(loc='upper left', fontsize=s_legend) #handles=handles, labels=labels, 

# Bottom row:
for t in types_bottom:
    axes[1].plot(data_add.time, data_add.sel(Type=t), label=t, color=DICT_GRID_STYLES_2[t][0], linestyle=DICT_GRID_STYLES_2[t][1], linewidth=linewidth)

axes[1].set_xlabel("Time", fontsize=s_label)
axes[1].set_ylabel(f"{type_variable} (# counts)", fontsize=s_label)
axes[1].grid(alpha=0.3, linestyle='--')
axes[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
axes[1].tick_params(axis='both', which='major', labelsize=s_legend) # set font size of axis ticks
axes[1].legend(loc='upper left', fontsize=s_legend)

plt.suptitle(f"{scen_folder}: Electricity Grid - {type_variable}", fontsize=16)

plt.tight_layout()
# fig.savefig(path_test_plots / "Grid_stocks_world.png", dpi=300)
plt.show()


###########################################################################################################
#%%% Stocks Materials
###########################################################################################################



#%%%% 2 model ---------------------------------------------------

materials = ["steel", "concrete", "aluminium", "copper"]

data_lines  = model_lines.stock_by_cohort_materials.copy().sum(dim="Region").pint.to("t")  # Convert kg -> tonnes
data_add    = model_add.stock_by_cohort_materials.copy().sum(dim="Region").pint.to("t")  # Convert kg -> tonnes

data        = xr.concat([data_lines, data_add], dim='Type')
data        = data.sel(Time=slice(1971, 2060))

data_sum    = data.sum(dim='Type')

lines_sum           = data.sel(Type=[t for t in data.Type.values if 'Lines' in t]).sum(dim='Type') # Get group sums by keyword and sum over types (sum over HV, MV and LV (and overground/underground for lines))
transformers_sum    = data.sel(Type=[t for t in data.Type.values if 'Transformers' in t]).sum(dim='Type')
substations_sum     = data.sel(Type=[t for t in data.Type.values if 'Substations' in t]).sum(dim='Type')


fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 10))
linewidth = 2
s_legend = 12
s_label = 14

for i, mat in enumerate(materials):
    lines_sum.sel(material=mat).plot(ax=axes[i], color = DICT_GRID_COLORS['Lines'], label="Lines")
    transformers_sum.sel(material=mat).plot(ax=axes[i], color = DICT_GRID_COLORS['Transformers'], label="Transformers")
    substations_sum.sel(material=mat).plot(ax=axes[i], color = DICT_GRID_COLORS['Substations'], label="Substations")
    data_sum.sel(material=mat).plot(ax=axes[i], label="Total", color='red', alpha=0.8, linestyle='--', linewidth=3)

    # if mat == "Cu":
    #     # Add IEA data for copper
    #     axes[i].plot(df_iea_lt_cu.index, df_iea_lt_cu['Cu'], label="IEA L&T", color='#00a5cf', linestyle=':', linewidth=4)

    # if mat == "Aluminium":
    #     # Add IEA data for aluminium
    #     axes[i].plot(df_iea_lt_alu.index, df_iea_lt_alu['Aluminium'], label="IEA L&T", color='#00a5cf', linestyle=':', linewidth=4)

    # if mat == "Steel":
    #     # Add IEA data for steel
    #     axes[i].plot(df_iea_t_steel.index, df_iea_t_steel['Steel'], label="IEA T", color='#00a5cf', linestyle=':', linewidth=4)
    
    axes[i].grid(alpha=0.3, linestyle='--')
    axes[i].ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
    axes[i].tick_params(axis='both', which='major', labelsize=s_legend) # set font size of axis ticks
    axes[i].set_title(f"{mat}")
    axes[i].set_xlabel(" ")
    axes[i].set_ylabel("Stock [t]", fontsize=s_label)
    axes[i].legend()

axes[-1].set_xlabel("Time", fontsize=s_label)

plt.suptitle(f"{scen_folder}: Electricity Grid Stocks Materials", fontsize=16)
plt.tight_layout()
# fig.savefig(path_test_plots / "Grid_stocks-materials_world.svg")
# fig.savefig(path_test_plots / "Grid_stocks-materials_world_1971.pdf")
# fig.savefig(path_test_plots / "Grid_stocks-materials_world_1971.png")
# fig.savefig(path_test_plots / "Grid_stocks-materials_world_1971.svg")
plt.show()


###########################################################################################################
#%% INFLOW Materials
###########################################################################################################

#================================================================================
#%%%% Per TECH - World


regions = ['Brazil', 'C.Europe', 'China'] 
materials = ["steel", "concrete", "aluminium", "copper"]

#%%%% 2 model ---------------------------------------------------

data_lines  = model_lines.inflow_materials.to_array().sum(dim="Region").pint.to("t")  # Convert kg -> tonnes
data_add    = model_add.inflow_materials.to_array().sum(dim="Region").pint.to("t")  # Convert kg -> tonnes

data_lines  = data_lines.sel(time=slice(1971, 2060))
data_add    = data_add.sel(time=slice(1971, 2060))
data        = xr.concat([data_lines, data_add], dim='Type')
data_sum    = data.sum(dim='Type')

lines_sum           = data_lines.sel(Type=[t for t in data_lines.Type.values if 'Lines' in t]).sum(dim='Type') # Get group sums by keyword and sum over types (sum over HV, MV and LV (and overground/underground for lines))
transformers_sum    = data_add.sel(Type=[t for t in data_add.Type.values if 'Transformers' in t]).sum(dim='Type')
substations_sum     = data_add.sel(Type=[t for t in data_add.Type.values if 'Substations' in t]).sum(dim='Type')


fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 10))
linewidth = 2
s_legend = 12
s_label = 14

for i, mat in enumerate(materials):
    lines_sum.sel(material=mat).plot(ax=axes[i], color = DICT_GRID_COLORS['Lines'], label="Lines")
    transformers_sum.sel(material=mat).plot(ax=axes[i], color = DICT_GRID_COLORS['Transformers'], label="Transformers")
    substations_sum.sel(material=mat).plot(ax=axes[i], color = DICT_GRID_COLORS['Substations'], label="Substations")
    data_sum.sel(material=mat).plot(ax=axes[i], label="Total", color='red', alpha=0.8, linestyle='--', linewidth=3)

    # if mat == "Cu":
    #     # Add IEA data for copper
    #     axes[i].plot(df_iea_lt_cu.index, df_iea_lt_cu['Cu'], label="IEA L&T", color='#00a5cf', linestyle=':', linewidth=4)

    # if mat == "Aluminium":
    #     # Add IEA data for aluminium
    #     axes[i].plot(df_iea_lt_alu.index, df_iea_lt_alu['Aluminium'], label="IEA L&T", color='#00a5cf', linestyle=':', linewidth=4)

    # if mat == "Steel":
    #     # Add IEA data for steel
    #     axes[i].plot(df_iea_t_steel.index, df_iea_t_steel['Steel'], label="IEA T", color='#00a5cf', linestyle=':', linewidth=4)
    
    axes[i].grid(alpha=0.3, linestyle='--')
    axes[i].ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
    axes[i].tick_params(axis='both', which='major', labelsize=s_legend) # set font size of axis ticks
    axes[i].set_title(f"{mat}")
    axes[i].set_xlabel(" ")
    axes[i].set_ylabel("Inflow [t]", fontsize=s_label)
    axes[i].legend()

axes[-1].set_xlabel("Time", fontsize=s_label)

plt.suptitle(f"{scen_folder}: Electricity Grid Inflow Materials", fontsize=16)
plt.tight_layout()
# fig.savefig(path_test_plots / "Grid_inflow-materials_world.svg")
# fig.savefig(path_test_plots / "Grid_inflow-materials_world_1971.pdf")
# fig.savefig(path_test_plots / "Grid_inflow-materials_world_1971.png")
# fig.savefig(path_test_plots / "Grid_inflow-materials_world_1971.svg")
plt.show()


