# -*- coding: utf-8 -*-
"""
Created on Fri May 23 11:08:07 2025

@author: MvEng
"""

import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import Union

import pandas as pd
import pint
import xarray as xr
import numpy as np
import prism
from prism import Q_


from imagematerials.distribution import ALL_DISTRIBUTIONS, NAME_TO_DIST
from imagematerials.read_mym import read_mym_df
from imagematerials.util import dataset_to_array, pandas_to_xarray
from imagematerials.constants import _IMAGE_REGIONS

##ADJUST FOR INFRASTRUCTURE SPECIFIC
from imagematerials.infrastructure.constants import (
    END_YEAR,
    FOLDER,
    LIGHT_COMMERCIAL_VEHICLE_SHARE,
    LOAD_FACTOR,
    MEGA_TO_TERA,
    PKMS_TO_VKMS,
    PROJECT,
    REGIONS,
    SCEN,
    SHIPS_YEARS_RANGE,
    START_YEAR,
    TONNES_TO_KGS,
    pkms_label,
    tkms_label,

    years_range,
)
from imagematerials.infrastructure.modelling_functions import prep_area, prep_hdi, prep_pop_gdp, urban_vkm, conversion_to_vkm



def preprocess(base_dir: str):
    """Wrapper function for the preprocessing part of the VEMA script.

    Args:
        base_dir (Optional[str], optional): _description_. Defaults to os.getcwd().
    
    Returns:
        _type_: _description_
    """
    base_path = Path(base_dir)

    # %%
    base_input_data_path = base_path.joinpath("infrastructure")
    standard_input_data_path = base_input_data_path.joinpath("standard_data")
    image_folder = base_path.joinpath(SCEN)
    # standard_output_folder = base_path.joinpath("..", "..", "output", PROJECT,
    #                                           FOLDER)

    # st = time.time()

    idx = pd.IndexSlice          # needed for slicing multi-index

    # Reading all csv files for vehicles and ships that are external to IMAGE
    # 1) scenario independent data
    #Load in a file to load in the specific IMAGE region
    region_load_in = pd.read_csv(datapath + 'region_load.csv', index_col=0, names=None).transpose()
    # Set the first column as the index, but keep it as a column
    region_list = list(region_load_in.columns.values)    
    #Load in HDI per IMAGE region
    hdi = pd.read_excel(standard_input_data_path. joinpath("HDI_SSP2.xlsx'"))  # Load HDI data #change to local file
    sorted_IMAGE_hdi = prep_hdi(hdi)

    #Load in urban areas per IMAGE region
    area = pd.read_excel(standard_input_data_path. joinpath("urban-area.xlsx"))  # Load Urban Area data # change to local file
    IMAGE_urban_area_transposed, IMAGE_rural_area_transposed = prep_area(area)
    
    # Load GDP and population drivers (convert to IMAGE input for integration)
    gdp_cap = pd.read_csv('SSP2\\SSP2_BL\\gdp_pc.csv')
    updated_drivers = pd.read_excel(standard_input_data_path. joinpath("SSP2-IIASA.xlsx"))
    # Remove the 'GDP-cap' DataFrame from the dictionary
    if 'GDP-cap' in updated_drivers:
        del updated_drivers['GDP-cap']
    # Group by 'IMAGE' and sum the values for each DataFrame
    IMAGE_drivers = {sheet_name: df.groupby('IMAGE').sum() for sheet_name, df in updated_drivers.items()}
    # Ensure the necessary DataFrames exist in the dictionary
    if 'GDP' in IMAGE_drivers and 'Urban population' in IMAGE_drivers and 'Rural population' in IMAGE_drivers:
        # Retrieve the required DataFrames
        gdp_df = IMAGE_drivers['GDP']
        urban_pop_df = IMAGE_drivers['Urban population']
        rural_pop_df = IMAGE_drivers['Rural population']
        
    gdp_cap_new, urban_population, rural_population = prep_pop_gdp(gdp_cap,gdp_df,urban_pop_df,rural_pop_df)

    #Smoothed out economic crises - gdp drops have a significant effect on inflows and outflows whereas infrastructure development is continous - less economic shock responsive
    #Convert to a smoothing function from IMAGE input
    #Currently overwrites the gdp_cap_new as it is supposed to calculate
    gdp_cap_new = pd.read_excel(standard_input_data_path. joinpath("gdp_new.xlsx"))
    
    #Calculate densities - and rural - urban shares
    urban_pop_density = urban_population* 1000000 / IMAGE_urban_area_transposed
    rural_pop_density = rural_population* 1000000 / IMAGE_rural_area_transposed
    pop_density = ((urban_population+rural_population) * 1000000) / (IMAGE_urban_area_transposed+IMAGE_rural_area_transposed)    
    rural_share = rural_population / (rural_population+ urban_population)
    urban_share = 1 - rural_share 
    gdp_total = gdp_cap_new * ((rural_population + urban_population) * 1000000)

    #Load in pkm/tkm files
    updated_vkm_pkm = read_mym_df('C:\\Academic\\Chapter2\\2. dMFA\\ELMA\\grid_data\\trp_trvl_pkm.out')    
    updated_vkm_tkm = read_mym_df('C:\\Academic\\Chapter2\\2. dMFA\\ELMA\\grid_data\\trp_frgt_Tkm.out') 
    
    vkm_pivoted, vkm_rail_pivoted = conversion_to_vkm (updated_vkm_pkm, updated_vkm_tkm, region_list)
    
    #Load in gdp_cap vkm_cap based on regressions
    gdp_urban_cap, gdp_rural_cap, vkm_urban_IMAGE_cap, vkm_rural_IMAGE_cap = urban_vkm(urban_share, rural_share, gdp_total, urban_population, rural_population, vkm_pivoted)
    
    #Sorting all dataframes, order of column names (IMAGE)
    sorted_urban_population = urban_population.sort_index(axis=1)
    sorted_rural_population = rural_population.sort_index(axis=1)
    
    sorted_urban_pop_density = urban_pop_density.sort_index(axis=1)
    sorted_rural_pop_density = rural_pop_density.sort_index(axis=1)
    sorted_pop_density = pop_density.sort_index(axis=1)
    
    sorted_gdp_urban_cap = gdp_urban_cap.sort_index(axis=1)
    sorted_gdp_rural_cap = gdp_rural_cap.sort_index(axis=1)
    
    sorted_vkm_urban_cap = vkm_urban_IMAGE_cap.sort_index(axis=1)
    sorted_vkm_rural_cap = vkm_rural_IMAGE_cap.sort_index(axis=1)
    
    sorted_vkm_rail_tot = vkm_rail_pivoted.sort_index(axis=1)
    sorted_vkm_rail_cap = sorted_vkm_rail_tot / ((sorted_urban_population+sorted_rural_population * 1000000))
    









        