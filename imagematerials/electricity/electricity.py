# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 08:59:30 2020
@author: Sebasiaan Deetman (deetman@cml.leidenuniv.nl)

This module is used to calculate the materials involved in the electricity generation capacity and electricity storage capacity
Input:  1) scenario files from the IMAGE Integrated Assessment Model
        2) data files on material intensities, costs, lifetimes and weights
Output: Total global material use (stocks, inflow & outflow) in the electricity sector for the period 2000-2050 based on:
        1) the second Shared socio-economic pathway (SSP2) Baseline
        2) the second Shared socio-economic pathway (SSP2) 2-degree Climate Policy scenario
		
4 senitivity settings are defined:
1) 'default'    is the default model setting, used for the outcomes as described in the main text
2) 'high_stor'  defines a pessimistic setting with regard to storage demand (high) and availability (low)
3) 'high_grid'  defines alternative assumptions with respect to the growth of the grid (not relevant here, see grid_materials.py)


Sebastiaans Electricity_sector.py should be splitted into preprocessing.py and electricity.py.
This is still work in progress. The preprocessing.py file is already available and some parts are already transferred.

"""
###########################################################################################################
#%% define imports, counters & settings
###########################################################################################################

import pandas as pd
import numpy as np
import xarray as xr
import scipy
import warnings
from pathlib import Path


from imagematerials.electricity.preprocessing import (
    get_preprocessing_data_gen,
    get_preprocessing_data_grid
    # get_preprocessing_data_stor
)
# from imagematerials.util import import_from_netcdf, export_to_netcdf
from imagematerials.model import GenericMainModel, GenericMaterials, GenericStocks, Maintenance, MaterialIntensities
from imagematerials.factory import ModelFactory, Sector
import prism

from imagematerials.electricity.constants import (
    SCEN,
    VARIANT
)

# VARIANT = "VLHO"
# SCEN = "SSP2"
scen_folder = SCEN + "_" + VARIANT
# path_base = Path().resolve() # TODO absolute path of file "preprocessing.py" ? current solution can differ depending on IDE used (?) 
path_current = Path().resolve()
path_base = path_current.parent.parent # base path of the project -> image-materials



####################################################################################################################
#%% Generation
####################################################################################################################


YEAR_START = 1971   # start year of the simulation period
YEAR_END = 2100     # end year of the calculations
YEAR_OUT = 2100     # year of output generation = last year of reporting

prep_data = get_preprocessing_data_gen(path_base, SCEN, VARIANT, YEAR_START, YEAR_END, YEAR_OUT)

# Define the complete timeline, including historic tail
time_start = prep_data["stocks"].coords["Time"].min().values
complete_timeline = prism.Timeline(time_start, YEAR_END, 1)
simulation_timeline = prism.Timeline(YEAR_START, YEAR_END, 1) #1970


sec_electr_gen = Sector("electr_gen", prep_data)

main_model_factory = ModelFactory(
    sec_electr_gen, complete_timeline
    ).add(GenericStocks
    ).add(MaterialIntensities
    ).finish()

main_model_factory.simulate(simulation_timeline)

list(main_model_factory.electr_gen)





####################################################################################################################
#%% Grid
####################################################################################################################

prep_data_lines, prep_data_add = get_preprocessing_data_grid(path_base, SCEN, VARIANT)


# LINES ----------------------------------------------------
# prep_data = create_prep_data(results_dict_lines, conversion_table, unit_mapping)

# # Define the complete timeline, including historic tail
time_start = prep_data_lines["stocks"].coords["Time"].min().values
time_end = 2060
complete_timeline = prism.Timeline(time_start, time_end, 1)
simulation_timeline = prism.Timeline(1970, time_end, 1)

sec_electr_grid_lines = Sector("electr_grid_lines", prep_data_lines)

main_model_factory_lines = ModelFactory(
    sec_electr_grid_lines, complete_timeline
    ).add(GenericStocks
    ).add(MaterialIntensities
    ).finish()

main_model_factory_lines.simulate(simulation_timeline)
list(main_model_factory_lines.electr_grid_lines)


# ADDITIONS -------------------------------------------------------------------------------------
# prep_data = create_prep_data(results_dict_add, conversion_table, unit_mapping)

# # Define the complete timeline, including historic tail
time_start = prep_data_add["stocks"].coords["Time"].min().values
time_end = 2060
complete_timeline = prism.Timeline(time_start, time_end, 1)
simulation_timeline = prism.Timeline(1970, time_end, 1)

sec_electr_grid_add = Sector("electr_grid_add", prep_data_add)

main_model_factory_add = ModelFactory(
    sec_electr_grid_add, complete_timeline
    ).add(GenericStocks
    ).add(MaterialIntensities
    ).finish()

main_model_factory_add.simulate(simulation_timeline)
list(main_model_factory_add.electr_grid_add)







# %%
