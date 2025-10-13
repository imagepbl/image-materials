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

# path_current = Path(__file__).resolve().parent # absolute path of file
# path_base = path_current.parent.parent # base path of the project -> image-materials
# sys.path.append(str(path_base))

import prism
from imagematerials.distribution import ALL_DISTRIBUTIONS, NAME_TO_DIST
from imagematerials.read_mym import read_mym_df
from imagematerials.util import dataset_to_array, pandas_to_xarray, convert_life_time_vehicles
from imagematerials.model import GenericMainModel, GenericStocks, SharesInflowStocks, Maintenance, GenericMaterials, MaterialIntensities
from imagematerials.factory import ModelFactory, Sector
from imagematerials.concepts import create_electricity_graph, create_region_graph
from imagematerials.electricity.utils import MNLogit, stock_tail, create_prep_data, stock_share_calc


from imagematerials.electricity.constants import (
    YEAR_FIRST,
    YEAR_FIRST_GRID,
    YEAR_SWITCH,
    STANDARD_SCEN_EXTERNAL_DATA,
    SENS_ANALYSIS,
    REGIONS,
    TIMER_REGIONS,
    TECH_GEN,
    STD_LIFETIMES_ELECTR,
    MEGA_TO_TERA,
    PKMS_TO_VKMS,
    TONNES_TO_KGS,
    LOAD_FACTOR,
    BEV_CAPACITY_CURRENT,
    PHEV_CAPACITY_CURRENT,
    unit_mapping,
    DICT_GENTECH_TO_CATEGORY,
    DICT_GENTECH_STYLES,
    DICT_STOR_STYLES,
    DICT_GENTECHCAT_COLORS,
    DICT_MATERIALS_COLORS,
    DICT_GRID_COLORS,
    DICT_GRID_STYLES_1,
    DICT_GRID_STYLES_2
)

from imagematerials.electricity.electr_external_data import (
    df_iea_cu_aps,
    df_iea_cu_nzs,
    df_iea_co_aps,
    df_iea_mn_aps,
    df_iea_ni_aps
)
SCEN = "SSP2"
# VARIANT = "VLHO"
VARIANT = "M_CP"
# Define paths ----------------------------------------------------------------------
#YOUR_DIR = "C:\\Users\\Admin\\surfdrive\\Projects\\IRP\\GRO23\\Modelling\\2060\\ELMA"   # Change the running directory here
# os.chdir(YOUR_DIR)
scen_folder = SCEN + "_" + VARIANT
# path_base = Path().resolve() # TODO absolute path of file "preprocessing.py" ? current solution can differ depending on IDE used (?) 
path_current = Path().resolve()
path_base = path_current.parent.parent # base path of the project -> image-materials

path_image_output = Path(path_base, "data", "raw", "image", scen_folder, "EnergyServices")
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





#----------------------------------------------------------------------------------------------------------
###########################################################################################################
#%%% 1.1) Read in files
###########################################################################################################
#----------------------------------------------------------------------------------------------------------

YEAR_START = 1971   # start year of the simulation period
YEAR_END = 2100     # end year of the calculations
YEAR_OUT = 2100     # year of output generation = last year of reporting

knowledge_graph = create_region_graph()

# 1. External Data ======================================================================================== 

# lifetimes of Gcap tech (original data according to van Vuuren 2006, PhD Thesis)
gcap_lifetime = pd.read_csv(path_external_data_scenario / 'LTTechnical_dynamic.csv', index_col=['Year','DIM_1'])        

# material compositions of electricity generation tecnologies (g/MW)
gcap_materials_data = pd.read_csv(path_external_data_scenario / 'composition_generation.csv',index_col=[0,1]).transpose()


# 2. IMAGE/TIMER files ====================================================================================

# Generation capacity (stock demand per generation technology) in MW peak capacity
gcap_data = read_mym_df(path_image_output / 'Gcap.out')


#----------------------------------------------------------------------------------------------------------
###########################################################################################################
#%%% 1.2) Prepare model specific variables
###########################################################################################################
#----------------------------------------------------------------------------------------------------------

# transform to xarray -----------------------------------------

# Material Intensities -------
# Extract coordinate labels
gcap_materials_data.columns = gcap_materials_data.columns.rename([None, None]) # remove column MultiIndex name "g/MW" as it causes issues when converting to xarray
years = sorted(gcap_materials_data.columns.get_level_values(0).unique())
techs = gcap_materials_data.columns.get_level_values(1).unique()
materials = gcap_materials_data.index
# Convert to 3D array: (Material, Year, Tech)
data_array = gcap_materials_data.to_numpy().reshape(len(materials), len(years), len(techs))
# Build xarray DataArray
gcap_materials_xr = xr.DataArray(
    data_array,
    dims=('Material', 'Time', 'Type'),
    coords={
        'Material': materials,
        'Time': years,
        'Type': techs
    },
    name='MaterialIntensities'
)
gcap_materials_xr = prism.Q_(gcap_materials_xr, "g/MW")

# Gcap ------
gcap_data = gcap_data.loc[~gcap_data['DIM_1'].isin([27,28])]    # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies
gcap_data = gcap_data.loc[gcap_data['time'].isin(range(YEAR_START, YEAR_END + 1)),
                     ['time', 'DIM_1', *range(1, TECH_GEN + 1)]]  # only keep relevant years and technology columns
# Extract numeric columns (technologies)
tech_cols = list(range(1, TECH_GEN+1))
# Pivot to 3D array
gcap_array = gcap_data[tech_cols].to_numpy().reshape(
    len(gcap_data['time'].unique()),
    len(gcap_data['DIM_1'].unique()),
    len(tech_cols)
)
# Create xarray DataArray
gcap_xr = xr.DataArray(
    gcap_array,
    dims=('Time', 'Region', 'Type'),
    coords={
        'Time': sorted(gcap_data['time'].unique()),
        'Region': [str(r) for r in sorted(gcap_data['DIM_1'].unique())],
        'Type': tech_cols
    },
    name='GCap'
)
gcap_xr = prism.Q_(gcap_xr, "MW")
gcap_xr = knowledge_graph.rebroadcast_xarray(gcap_xr, output_coords=TIMER_REGIONS, dim="Region")

# -------------------------------------------------------------


# region_list = list(kilometrage.columns.values)   

# gcap_tech_list = list(composition_generation.loc[:,idx[2020,:]].droplevel(axis=1, level=0).columns)    #list of names of the generation technologies (workaround to retain original order)
# gcap_material_list = list(composition_generation.index.values)  #list of materials the generation technologies

# gcap_data = gcap_data.loc[~gcap_data['DIM_1'].isin([27,28])]    # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies
# gcap = pd.pivot_table(gcap_data[gcap_data['time'].isin(list(range(YEAR_START,YEAR_END+1)))], index=['time','DIM_1'], values=list(range(1,TECH_GEN+1)))  #gcap as multi-index (index = years & regions (26); columns = technologies (34));  the last column in gcap_data (= totals) is now removed

# # renaming multi-index dataframe: generation capacity, based on the regions in grid_length_Hv & technologies as given
# gcap.index = pd.MultiIndex.from_product([list(range(YEAR_START,YEAR_END+1)), region_list], names=['years', 'regions'])
# gcap.columns = gcap_tech_list


# Interpolate material intensities (dynamic content for gcap & storage technologies between 1926 to 2100, based on data files)
index = pd.MultiIndex.from_product([list(range(YEAR_FIRST_GRID, YEAR_OUT+1)), list(composition_generation.index)])
gcap_materials_interpol = pd.DataFrame(index=index, columns=composition_generation.columns.levels[1])

# material intensities for gcap
for cat in list(composition_generation.columns.levels[1]):
   gcap_materials_1st   = composition_generation.loc[:,idx[composition_generation.columns[0][0],cat]]
   gcap_materials_interpol.loc[idx[YEAR_FIRST_GRID ,:],cat] = gcap_materials_1st.to_numpy()                # set the first year (1926) values to the first available values in the dataset (for the year 2000) 
   gcap_materials_interpol.loc[idx[composition_generation.columns.levels[0].min(),:],cat] = composition_generation.loc[:, idx[composition_generation.columns.levels[0].min(),cat]].to_numpy()                # set the middle year (2000) values to the first available values in the dataset (for the year 2000) 
   gcap_materials_interpol.loc[idx[composition_generation.columns.levels[0].max(),:],cat] = composition_generation.loc[:, idx[composition_generation.columns.levels[0].max(),cat]].to_numpy()                # set the last year (2100) values to the last available values in the dataset (for the year 2050) 
   gcap_materials_interpol.loc[idx[:,:],cat] = gcap_materials_interpol.loc[idx[:,:],cat].unstack().astype('float64').interpolate().stack()

# interpolate Gcap (technical) lifetime data
gcap_lifetime.index = gcap_lifetime.index.set_levels(gcap_tech_list, level=1)
gcap_lifetime = gcap_lifetime.unstack().droplevel(axis=1, level=0)
gcap_lifetime = gcap_lifetime.reindex(list(range(YEAR_FIRST_GRID,YEAR_OUT+1)), axis=0).interpolate(limit_direction='both')

# Calculate the historic tail to the Gcap (stock) 
gcap_new = pd.DataFrame(index=pd.MultiIndex.from_product([range(YEAR_FIRST_GRID,YEAR_OUT+1), TIMER_REGIONS], names=['years', 'regions']), columns=gcap.columns)
for tech in gcap_tech_list:
    gcap_new.loc[idx[:,:],tech] = stock_tail(gcap.loc[idx[:,:],tech].unstack(level=1), YEAR_OUT).stack()


# Bring dataframes into correct shape for the results_dict
# I need:
#   A. with only Types: stocks Types (GW), lifetimes Types, material intensities Types
#   B. with also SubTypes: stocks Types (GW), lifetimes SubTypes, material intensities SubTypes, market shares SubTypes

# A.
# stocks: (years, regions) index and technologies as columns -> years as index and (technology, region) as columns
gcap_stock = gcap_new.unstack(level='regions')

# lifetimes
df_mean = gcap_lifetime.copy()
df_stdev = df_mean * STD_LIFETIMES_ELECTR
df_mean.columns = [(col, 'mean') for col in df_mean.columns] # Rename columns to multi-level tuples
df_stdev.columns = [(col, 'stdev') for col in df_stdev.columns]
gcap_lifetime_distr = pd.concat([df_mean, df_stdev], axis=1) # Concatenate along columns

# MIs: (years, material) index and technologies as columns -> years as index and (technology, Material) as columns
gcap_materials_interpol.index.names = ["Year", "Material"]
# gcap_materials_interpol = gcap_materials_interpol.loc[:, ~(gcap_materials_interpol == 0.0).all()] # delete empty columns
gcap_types_materials = gcap_materials_interpol.unstack(level='Material')


#----------------------------------------------------------------------------------------------------------
###########################################################################################################
#%%% 1.3) Prep_data File
###########################################################################################################
#----------------------------------------------------------------------------------------------------------


# Conversion table for all coordinates, to be removed/adapted after input tables are fixed.
conversion_table = {
    "gcap_stock": (["Time"], ["Type", "Region"],),
    "gcap_types_materials": (["Cohort"], ["Type", "material"],)
    # "gcap_materials_interpol": (["Cohort"], ["Type", "SubType", "material"], {"Type": ["Type", "SubType"]})
}

results_dict = {
        'gcap_stock': gcap_stock,
        'gcap_types_materials': gcap_types_materials,
        'gcap_lifetime_distr': gcap_lifetime_distr,
}


prep_data = create_prep_data(results_dict, conversion_table, unit_mapping)
prep_data["stocks"] = prism.Q_(prep_data["stocks"], "MW")
prep_data["material_intensities"] = prism.Q_(prep_data["material_intensities"], "g/MW")
prep_data["set_unit_flexible"] = prism.U_(prep_data["stocks"]) # prism.U_ gives the unit back


