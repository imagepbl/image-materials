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
from imagematerials.model import GenericMainModel, GenericMaterials, GenericStocks, Maintenance, MaterialIntensities
from imagematerials.factory import ModelFactory, Sector
from imagematerials.concepts import create_electricity_graph
from imagematerials.electricity.utils import MNLogit, stock_tail, create_prep_data,print_df_info


from imagematerials.electricity.constants import (
    YEAR_START,
    YEAR_FIRST,
    YEAR_FIRST_GRID,
    YEAR_END,
    YEAR_OUT,
    YEAR_SWITCH,
    YEAR_LAST,
    # COHORTS, # necessary?
    SCEN,
    VARIANT,
    SENS_ANALYSIS,
    REGIONS,
    TECH_GEN,
    STD_LIFETIMES_ELECTR,
    MEGA_TO_TERA,
    PKMS_TO_VKMS,
    TONNES_TO_KGS,
    LOAD_FACTOR,
    BEV_CAPACITY_CURRENT,
    PHEV_CAPACITY_CURRENT,
    unit_mapping,
    gen_tech_to_category,
    dict_gentech_styles,
    dict_gentechcat_colors,
    dict_materials_colors,
    dict_grid_colors,
    dict_grid_styles,
    dict_grid_styles2
)

from imagematerials.electricity.electr_external_data import (
    df_iea_cu_aps,
    df_iea_cu_nzs,
    df_iea_co_aps,
    df_iea_mn_aps,
    df_iea_ni_aps
)
VARIANT = "VLHO"
# VARIANT = "M_CP"
# Define paths ----------------------------------------------------------------------
#YOUR_DIR = "C:\\Users\\Admin\\surfdrive\\Projects\\IRP\\GRO23\\Modelling\\2060\\ELMA"   # Change the running directory here
# os.chdir(YOUR_DIR)
scen_folder = SCEN + "_" + VARIANT
# path_base = Path().resolve() # TODO absolute path of file "preprocessing.py" ? current solution can differ depending on IDE used (?) 
path_current = Path().resolve()
path_base = path_current.parent.parent # base path of the project -> image-materials

# path_image_output = Path(path_base, "data", "raw", SCEN, "EnergyServices")
path_image_output = Path(path_base, "data", "raw", scen_folder, "EnergyServices")
path_external_data_standard = Path(path_base, "data", "raw", "electricity", "standard_data")
path_external_data_scenario = Path(path_base, "data", "raw", "electricity", scen_folder)

assert path_image_output.is_dir()
assert path_external_data_standard.is_dir()
assert path_external_data_scenario.is_dir()

# create the folder out_test if it does not exist
if not (path_base / 'imagematerials' / 'electricity' / 'out_test').is_dir():
    (path_base / 'imagematerials' / 'electricity' / 'out_test').mkdir(parents=True)



years = YEAR_END - YEAR_START  + 1


# from past.builtins import execfile
# execfile('read_mym.py')
idx = pd.IndexSlice             # needed for slicing multi-index


# TODO: decide on variable naming convention
# V1: sector_variableinquestion_additionalinformation (gcap_lifetime_interpolated)
# V2: variableinquestion_sector_additionalinformation (lifetime_gcap_interpolated)

###########################################################################################################
###########################################################################################################
#%% 1) Generation 
###########################################################################################################
###########################################################################################################



def get_preprocessing_data_gen(base_dir: str, SCEN, VARIANT): #, climate_policy_config: dict, circular_economy_config: dict

    scen_folder = SCEN + "_" + VARIANT
    path_image_output = Path(path_base, "data", "raw", scen_folder, "EnergyServices")
    path_external_data_standard = Path(path_base, "data", "raw", "electricity", "standard_data")
    path_external_data_scenario = Path(path_base, "data", "raw", "electricity", scen_folder)

    assert path_image_output.is_dir()
    assert path_external_data_standard.is_dir()
    assert path_external_data_scenario.is_dir()

    years = YEAR_END - YEAR_START  + 1

    idx = pd.IndexSlice   

    ###########################################################################################################
    # Read in files #

    # 1. External Data --------------------------------------------- 

    # lifetimes of Gcap tech (original data according to van Vuuren 2006, PhD Thesis)
    gcap_lifetime = pd.read_csv(path_external_data_scenario / 'LTTechnical_dynamic.csv', index_col=['Year','DIM_1'])        
    # material compositions (generation capacity)
    composition_generation = pd.read_csv(path_external_data_scenario / 'composition_generation.csv',index_col=[0,1]).transpose()  # in gram/MW
    kilometrage = pd.read_csv(path_external_data_scenario / 'kilometrage.csv', index_col='t')  # to get region list without running storage - TODO: get regions from different source

    # 2. IMAGE/TIMER files -----------------------------------------

    # Generation capacity (stock & inflow/new) in MW peak capacity, FILES from TIMER
    gcap_data = read_mym_df(path_image_output / 'Gcap.out')

    ###########################################################################################################
    # Prepare model specific variables #
    
    region_list = list(kilometrage.columns.values)   

    gcap_tech_list = list(composition_generation.loc[:,idx[2020,:]].droplevel(axis=1, level=0).columns)    #list of names of the generation technologies (workaround to retain original order)
    gcap_material_list = list(composition_generation.index.values)  #list of materials the generation technologies

    gcap_data = gcap_data.loc[~gcap_data['DIM_1'].isin([27,28])]    # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies
    gcap = pd.pivot_table(gcap_data[gcap_data['time'].isin(list(range(YEAR_START,YEAR_END+1)))], index=['time','DIM_1'], values=list(range(1,TECH_GEN+1)))  #gcap as multi-index (index = years & regions (26); columns = technologies (34));  the last column in gcap_data (= totals) is now removed

    # renaming multi-index dataframe: generation capacity, based on the regions in grid_length_Hv & technologies as given
    gcap.index = pd.MultiIndex.from_product([list(range(YEAR_START,YEAR_END+1)), region_list], names=['years', 'regions'])
    gcap.columns = gcap_tech_list

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
    gcap_new = pd.DataFrame(index=pd.MultiIndex.from_product([range(YEAR_FIRST_GRID,YEAR_OUT+1), region_list], names=['years', 'regions']), columns=gcap.columns)
    for tech in gcap_tech_list:
        gcap_new.loc[idx[:,:],tech] = stock_tail(gcap.loc[idx[:,:],tech].unstack(level=1)).stack()


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


    ###########################################################################################################
    # Prep_data File #
    
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

    return prep_data