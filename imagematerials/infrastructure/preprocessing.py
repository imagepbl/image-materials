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
from imagematerials.infrastructure.modelling_functions import prep_area, prep_hdi


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
    gdp_cap_new = pd.read_excel(standard_input_data_path. joinpath("gdp_new.xlsx")
    

    urban_pop_density, rural_pop_density, pop_density = densities( IMAGE_urban_area_transposed, IMAGE_rural_area_transposed, urban_population, rural_population)                            









        