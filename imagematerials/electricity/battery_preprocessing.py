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
from itertools import product

# path_current = Path(__file__).resolve().parent # absolute path of file
# path_base = path_current.parent.parent # base path of the project -> image-materials
# sys.path.append(str(path_base))

import prism
from imagematerials.distribution import ALL_DISTRIBUTIONS, NAME_TO_DIST
from imagematerials.read_mym import read_mym_df
from imagematerials.util import dataset_to_array, pandas_to_xarray, convert_lifetime
from imagematerials.model import GenericMainModel, GenericStocks, SharesInflowStocks, Maintenance, GenericMaterials, MaterialIntensities
from imagematerials.factory import ModelFactory, Sector
from imagematerials.concepts import create_electricity_graph, create_region_graph, create_vehicle_graph
from imagematerials.electricity.utils import MNLogit, stock_tail, create_prep_data, logistic, quadratic
from imagematerials.vehicles.modelling_functions import interpolate

from imagematerials.constants import IMAGE_REGIONS
from imagematerials.vehicles.constants import (
    typical_modes,
    drive_trains
)
from imagematerials.electricity.constants import (
    STANDARD_SCEN_EXTERNAL_DATA,
    YEAR_FIRST,
    YEAR_FIRST_GRID,
    YEAR_SWITCH,
    SENS_ANALYSIS,
    EPG_TECHNOLOGIES,
    STD_LIFETIMES_ELECTR,
    LOAD_FACTOR,
    BEV_CAPACITY_CURRENT,
    PHEV_CAPACITY_CURRENT,
    unit_mapping
)

YEAR_START = 1971
YEAR_END = 2100
YEAR_OUT = 2100

SCEN = "SSP2"
# VARIANT = "VLHO"
VARIANT = "M_CP"
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


idx = pd.IndexSlice   

#----------------------------------------------------------------------------------------------------------
###########################################################################################################
#%%% 2.1) Read in files
###########################################################################################################
#----------------------------------------------------------------------------------------------------------

# 1. External Data ======================================================================================== 

# storage costs according to IRENA storage report & other sources in the SI
storage_costs = pd.read_csv(path_external_data_standard / 'storage_cost.csv', index_col=0).transpose()

# assumed malus & bonus of storage costs (malus for advanced technologies, still under development; bonus for batteries currently used in EVs, we assume that a large volume of used EV batteries will be available and used for dedicated electricity storage, thus lowering costs), only the bonus remains by 2030
storage_malus = pd.read_csv(path_external_data_standard / 'storage_malus.csv', index_col=0).transpose()

# assumptions on the long-term price decline after 2050. Prices are in $ct / kWh electricity cycled (the fraction of the annual growth rate (determined based on 2018-2030) that will be applied after 2030, ranging from 0.25 to 1 - 0.25 means the price decline is not expected to continue strongly, while 1 means that the same (2018-2030) annual price decline is also applied between 2030 and 2050)
storage_ltdecline = pd.Series(pd.read_csv(path_external_data_standard / 'storage_ltdecline.csv',index_col=0,  header=None).transpose().iloc[0])

# energy density assumptions (kg/kWh storage capacity - mass required to store one unit of energy — more mass per energy = worse performance)
storage_density = pd.read_csv(path_external_data_standard / 'storage_density_kg_per_kwh.csv',index_col=0).transpose()

# lifetime of storage technologies (in yrs). The lifetime is assumed to be 1.5* the number of cycles divided by the number of days in a year (assuming diurnal use, and 50% extra cycles before replacement, representing continued use below 80% remaining capacity) OR the maximum lifetime in years, which-ever comes first 
storage_lifetime = pd.read_csv(path_external_data_standard / 'storage_lifetime.csv',index_col=0).transpose()

# material compositions (storage) in wt%
storage_materials = pd.read_csv(path_external_data_standard / 'storage_materials_dynamic.csv',index_col=[0,1]).transpose()  # wt% of total battery weight for various materials, total battery weight is given by the density file above

# Using the 250 Wh/kg on the kWh of the various batteries a weight (in kg) of the battery per vehicle category is determined
# TODO: where is this data from? kWh per battery type?
battery_weights_data = pd.read_csv(path_external_data_standard / "battery_weights_kg.csv", index_col=[0,1])

# usable capacity of EV batteries for V2G applications (fraction of the total battery capacity that can be used for V2G)
if SENS_ANALYSIS == 'high_stor':
   # pessimistic sensitivity variant (meaning more additional storage is needed) -> smaller fraction of the EV capacities is usable as storage compared to the normal case
#    capacity_usable_PHEV = 0.025   # 2.5% of capacity of PHEV is usable as storage (in the pessimistic sensitivity variant)
#    capacity_usable_BEV  = 0.05    # 5  % of capacity of BEVs is usable as storage (in the pessimistic sensitivity variant)
    ev_capacity_usable_rel = pd.read_csv(path_external_data_standard / 'ev_battery_capacity_usable_for_v2g_variant_high_storage.csv')
else:
    ev_capacity_usable_rel = pd.read_csv(path_external_data_standard / 'ev_battery_capacity_usable_for_v2g.csv')

# fraction of EVs available for V2G (considering that not all EVs are capable of bi-directional loading, economic incentives are still missing, and not all owners are willing to provide V2G services)
ev_fraction_v2g_data = pd.read_csv(path_external_data_standard / 'ev_fraction_available_for_v2g.csv', index_col=[0])


# ##########################################################################################################
# %% Prepare general variables
# ##########################################################################################################

# Calculations used in both 'vehicle battery storage' and 'other storage'

##################
# Interpolations #
##################

# storage = storage.iloc[:, :26]    # drop global total column and empty (27) column

# # J: in high storage scenario the storage demand linearly increases between 2021 and 2050 compared to its original value until it is double by 2050, and then remains constant
# if SENS_ANALYSIS == 'high_stor':
#     storage_multiplier = storage
#     for year in range(2021,2051):
#         storage_multiplier.loc[year] = storage.loc[year] * (1 + (1/30*(year-2020)))
#     for year in range(2051,YEAR_END+1):
#         storage_multiplier.loc[year] = storage.loc[year] * 2


# turn index to integer for sorting during the next step
storage_costs.index = storage_costs.index.astype('int64')
storage_malus.index = storage_malus.index.astype('int64')
storage_density.index = storage_density.index.astype('int64')

# to interpolate between 2018 and 2030, first create empty rows (NaN values) 
storage_start = storage_costs.first_valid_index()
storage_end =   storage_costs.last_valid_index()
for i in range(storage_start+1,storage_end):
    storage_costs = pd.concat([storage_costs, pd.DataFrame(index=[i])]) #, ignore_index=True
    storage_malus = pd.concat([storage_malus, pd.DataFrame(index=[i])])         # mind: the malus needs to be defined for the same years as the cost indications
    storage_density = pd.concat([storage_density, pd.DataFrame(index=[i])])     # mind: the density needs to be defined for the same years as the cost indications
    
# then, do the actual interpolation on the sorted dataframes                                                    
storage_costs_interpol = storage_costs.sort_index(axis=0).interpolate(axis=0)#.index.astype('int64')
storage_malus_interpol = storage_malus.sort_index(axis=0).interpolate(axis=0)
storage_density_interpol = storage_density.sort_index(axis=0).interpolate(axis=0)  # density calculation continue with the material calculations

# energy density ---
# fix the energy density (kg/kwh) of storage technologies after 2030
for year in range(2030+1,YEAR_OUT+1):
    # storage_density_interpol = storage_density_interpol.append(pd.Series(storage_density_interpol.loc[storage_density_interpol.last_valid_index()], name=year))
    row = storage_density_interpol.loc[[storage_density_interpol.last_valid_index()]]
    row.index = [year]
    storage_density_interpol = pd.concat([storage_density_interpol, row])
# assumed fixed energy densities before 2018
for year in reversed(range(YEAR_FIRST_GRID,storage_start)): # was YEAR_SWITCH, storage_start
    # storage_density_interpol = storage_density_interpol.append(pd.Series(storage_density_interpol.loc[storage_density_interpol.first_valid_index()], name=year)).sort_index(axis=0)
    row = storage_density_interpol.loc[[storage_density_interpol.first_valid_index()]]
    row.index = [year]
    storage_density_interpol = pd.concat([storage_density_interpol, row]).sort_index(axis=0)

# storage material intensity ---
# Interpolate material intensities (dynamic content for gcap & storage technologies between 1926 to 2100, based on data files)
index = pd.MultiIndex.from_product([list(range(YEAR_FIRST_GRID, YEAR_OUT+1)), list(storage_materials.index)])
stor_materials_interpol = pd.DataFrame(index=index, columns=storage_materials.columns.levels[1])
# material intensities for storage
for cat in list(storage_materials.columns.levels[1]):
    stor_materials_1st   = storage_materials.loc[:,idx[storage_materials.columns[0][0],cat]]
    stor_materials_interpol.loc[idx[YEAR_FIRST_GRID ,:],cat] = stor_materials_1st.to_numpy()  # set the first year (1926) values to the first available values in the dataset (for the year 2000) 
    stor_materials_interpol.loc[idx[storage_materials.columns.levels[0].min(),:],cat] = storage_materials.loc[:, idx[storage_materials.columns.levels[0].min(),cat]].to_numpy() # set the middle year (2000) values to the first available values in the dataset (for the year 2000) 
    stor_materials_interpol.loc[idx[storage_materials.columns.levels[0].max(),:],cat] = storage_materials.loc[:, idx[storage_materials.columns.levels[0].max(),cat]].to_numpy() # set the last year (2100) values to the last available values in the dataset (for the year 2050) 
    stor_materials_interpol.loc[idx[:,:],cat] = stor_materials_interpol.loc[idx[:,:],cat].unstack().astype('float64').interpolate().stack()


#################
# Market Shares #
#################

# Determine MARKET SHARE of the storage capacity using a multi-nomial logit function

# storage costs ---
# determine the annual % decline of the costs based on the 2018-2030 data (original, before applying the malus)
decline = ((storage_costs_interpol.loc[storage_start,:]-storage_costs_interpol.loc[storage_end,:])/(storage_end-storage_start))/storage_costs_interpol.loc[storage_start,:]
decline_used = decline*storage_ltdecline #TODO: what is happening here? Why?
# storage_ltdecline is a single number and should describe the long-term decline after 2030 relative to the 2018-2030 decline

storage_costs_new = storage_costs_interpol * storage_malus_interpol
# calculate the development from 2030 to 2050 (using annual price decline)
for year in range(storage_end+1,2050+1):
    # print(year)
    # storage_costs_new = storage_costs_new.append(pd.Series(storage_costs_new.loc[storage_costs_new.last_valid_index()]*(1-decline_used), name=year))
    # storage_costs_new = pd.concat([storage_costs_new, pd.Series(storage_costs_new.loc[storage_costs_new.last_valid_index()] * (1 - decline_used), name=year)])
    row = pd.DataFrame([storage_costs_new.loc[storage_costs_new.last_valid_index()] * (1 - decline_used)])
    storage_costs_new.loc[year] = row.iloc[0]
# for historic price development, assume 2x AVERAGE annual price decline on all technologies, except lead-acid (so that lead-acid gets a relative price advantage from 1970-2018)
for year in reversed(range(YEAR_FIRST_GRID,storage_start)):
    # storage_costs_new = storage_costs_new.append(pd.Series(storage_costs_new.loc[storage_costs_new.first_valid_index()]*(1+(2*decline_used.mean())), name=year)).sort_index(axis=0)
    row = pd.DataFrame([storage_costs_new.loc[storage_costs_new.first_valid_index()]*(1+(2*decline_used.mean()))])
    storage_costs_new.loc[year] = row.iloc[0]
    storage_costs_new.sort_index(axis=0, inplace=True) 

storage_costs_new.sort_index(axis=0, inplace=True) 
storage_costs_new.loc[1971:2017,'Deep-cycle Lead-Acid'] = storage_costs_new.loc[2018,'Deep-cycle Lead-Acid'] # restore the exception (set to constant 2018 values)


# market shares ---
# use the storage price development in the logit model to get market shares
storage_market_share = MNLogit(storage_costs_new, -0.2) #assumes input of an ordered dataframe with rows as years and columns as technologies, values as prices. Logitpar is the calibrated Logit parameter (usually a nagetive number between 0 and 1)

# fix the market share of storage technologies after 2050
for year in range(2050+1,YEAR_OUT+1):
    # storage_market_share = storage_market_share.append(pd.Series(storage_market_share.loc[storage_market_share.last_valid_index()], name=year))
    row = pd.DataFrame([storage_market_share.loc[storage_market_share.last_valid_index()]])
    storage_market_share.loc[year] = row.iloc[0]
# fix the market share of storage technologies before YEAR_START
for year in range(YEAR_FIRST_GRID,YEAR_START):
    # storage_market_share = storage_market_share.append(pd.Series(storage_market_share.loc[storage_market_share.last_valid_index()], name=year))
    row = pd.DataFrame([storage_market_share.loc[storage_market_share.first_valid_index()]])
    storage_market_share.loc[year] = row.iloc[0]
    
storage_market_share = storage_market_share.sort_index(axis=0)

#First we calculate the share of the inflow using only a few of the technologies in the storage market share
#The selection represents only the batteries that are suitable for EV & mobile applications
list_ev_batteries = ['NiMH', 'LMO', 'NMC', 'NCA', 'LFP', 'Lithium Sulfur', 'Lithium Ceramic', 'Lithium-air']
#normalize the selection of market shares, so that total market share is 1 again (taking the relative share in the selected battery techs)
market_share_EVs = storage_market_share[list_ev_batteries].div(storage_market_share[list_ev_batteries].sum(axis=1), axis=0)


###################
# Battery Weights #
###################

# interpolate & complete series for battery weights, shares & composition
# too
battery_weights = interpolate(battery_weights_data.unstack())


###################
#%% Capacity #
###################

# TODO: make this to an xarray? 

# if SENS_ANALYSIS == 'high_stor':
#    capacity_usable_PHEV = 0.025   # 2.5% of capacity of PHEV is usable as storage (in the pessimistic sensitivity variant)
#    capacity_usable_BEV  = 0.05    # 5  % of capacity of BEVs is usable as storage (in the pessimistic sensitivity variant)
# else: 
#    capacity_usable_PHEV = 0.05    # 5% of capacity of PHEV is usable as storage
#    capacity_usable_BEV  = 0.10    # 10% of capacity of BEVs is usable as storage




x = ev_fraction_v2g_data.reindex(range(ev_fraction_v2g_data.index[0],ev_fraction_v2g_data.index[-1]+1)).interpolate(method='linear')
y = logistic(x, L=x.iloc[-1].values)
# y = quadratic(x)
ev_fraction_v2g = ev_fraction_v2g_data.reindex(range(YEAR_FIRST_GRID,YEAR_OUT+1)).interpolate(method='linear') # create dataframe with full index; values before first data points will be Nans, between data points interpolated linearly, after last data point will be last known value
ev_fraction_v2g.loc[:ev_fraction_v2g_data.index[0]] = 0 # set values before first data point to 0
ev_fraction_v2g.loc[ev_fraction_v2g_data.index[0]:ev_fraction_v2g_data.index[1]] = y # set values between (originally) first and last data point to quadratic/logistic interpolation

# plt.figure()
# plt.plot(ev_fraction_v2g_test.loc[2010:2060].index,ev_fraction_v2g_test.loc[2010:2060], label='')
# plt.scatter(ev_fraction_v2g_test.loc[2023:2026].index,ev_fraction_v2g_test.iloc[:, 0].loc[2023:2026], label='')
# plt.scatter(ev_fraction_v2g_test.loc[2048:2052].index,ev_fraction_v2g_test.iloc[:, 0].loc[2048:2052], label='')


knowledge_graph_vhc = create_vehicle_graph()
# create list of all vehicle types (combinations of vehicles type (Cars, Medium Freight Trucks,...) and drive trains (ICE, BEV,...))
vehicle_list = [f"{super_type} - {sub_type}" for super_type, sub_type in product(typical_modes, drive_trains)]

# Create an empty expanded df
vhc_fraction_v2g = pd.DataFrame(index=ev_fraction_v2g.index)
# transfer the values based on the drive train
for item in vehicle_list:
    if item.endswith("BEV"):
        vhc_fraction_v2g[item] = ev_fraction_v2g["BEV"]
    elif item.endswith("PHEV"):
        vhc_fraction_v2g[item] = ev_fraction_v2g["PHEV"]
    else:
        vhc_fraction_v2g[item] = 0 # other drive trains are not V2G capable
# Build xarray DataArray
years = vhc_fraction_v2g.index.to_numpy()
techs = vhc_fraction_v2g.columns
data_array = vhc_fraction_v2g.to_numpy()
vhc_fraction_v2g_xr = xr.DataArray(
    data_array,
    dims=('Time', 'Type'),
    coords={
        'Time': years,
        'Type': techs
    },
    name='VehicleFractionV2g'
)
vhc_fraction_v2g_xr = prism.Q_(vhc_fraction_v2g_xr, "fraction")


# With a pre-determined battery capacity in 2018, we assume an increasing capacity (as an effect of an increased density) based on a fixed weight assumption
BEV_dynamic_capacity = (weighted_average_density[2018] * BEV_CAPACITY_CURRENT) / weighted_average_density
PHEV_dynamic_capacity = (weighted_average_density[2018] * PHEV_CAPACITY_CURRENT) / weighted_average_density

# Define the V2G settings over time (assuming slow adoption to the maximum usable capacity)
year_v2g_start = 2025-1 # -1 leads to usable capacity in 2025
year_full_capacity = 2040
v2g_usable = pd.Series(index=list(range(YEAR_START,YEAR_OUT+1)), name='years')    #fraction of the cars ready and willing to use v2g
for year in list(range(YEAR_START,YEAR_OUT+1)):
    if (year <= year_v2g_start):
        v2g_usable[year] = 0
    elif (year >= year_full_capacity):
        v2g_usable[year] = 1
v2g_usable = v2g_usable.interpolate()

#total capacity per car connected to v2g (in %)
max_capacity_BEV = v2g_usable * capacity_usable_BEV  
max_capacity_PHEV = v2g_usable * capacity_usable_PHEV   

#usable capacity per car connected to v2g (in kWh)
usable_capacity_BEV = max_capacity_BEV.mul(BEV_dynamic_capacity[:YEAR_OUT])     
usable_capacity_PHEV = max_capacity_PHEV.mul(PHEV_dynamic_capacity[:YEAR_OUT])  




# # Vehicle storage in MWh #TODO: move to battery model
# storage_BEV = storage_PHEV = pd.DataFrame().reindex_like(vehicles_BEV)
# for region in region_list:
#     storage_PHEV[region] = vehicles_PHEV[region].mul(usable_capacity_PHEV) /1000
#     storage_BEV[region] = vehicles_BEV[region].mul(usable_capacity_BEV) /1000
# storage_vehicles = storage_BEV + storage_PHEV



########################
# Material Intensities #
########################
# MIs: (years, material) index and technologies as columns -> years as index and (technology, Material) as columns

# 2nd level of the MultiIndex are materials (1971,'Aluminium')
ev_battery_materials = stor_materials_interpol.copy()
ev_battery_materials.index.names = ["year", "material"]  # assign names
ev_battery_materials = ev_battery_materials.reset_index(level=["year", "material"])
# Pivot so we get (technology, material) as columns, years as index
ev_battery_materials = ev_battery_materials.melt(id_vars=["year", "material"], var_name="technology", value_name="value")
ev_battery_materials = ev_battery_materials.pivot_table(index="year", columns=["technology", "material"], values="value")
# Ensure proper MultiIndex column names
ev_battery_materials.columns = pd.MultiIndex.from_tuples(ev_battery_materials.columns, names=["technology", "material"])
xr_ev_battery_materials = dataset_to_array(pandas_to_xarray(ev_battery_materials, unit_mapping), *(["Cohort"], ["Type", "material"],))
xr_ev_battery_materials = xr_ev_battery_materials.sel(Type=list_ev_batteries)
xr_storage_density_interpol = dataset_to_array(pandas_to_xarray(storage_density_interpol, unit_mapping), *(["Cohort"], ["Type"],))
xr_ev_density_interpol = xr_storage_density_interpol.sel(Type=list_ev_batteries)
# oth_storage_materialintens = xr_ev_battery_materials * xr_storage_density_interpol



# ##########################################################################################################
# %% Prep_data
# ##########################################################################################################

# Conversion table for all coordinates, to be removed/adapted after input tables are fixed.
conversion_table = {
    "battery_shares": (["Cohort"], ["battery"],),
    "battery_weights": (["Cohort"], ["Type", "SubType"], {"Type": ["Type", "SubType"]}),
    "ev_battery_materials": (["Cohort"], ["battery", "material"],),
    "ev_energy_density": (["Cohort"], ["battery"],),
}
## "gcap_materials_interpol": (["Cohort"], ["Type", "SubType", "material"], {"Type": ["Type", "SubType"]})

results_dict = {
        'battery_shares': market_share_EVs,
        'battery_weights': battery_weights,
        'ev_battery_materials': xr_ev_battery_materials,
        'ev_energy_density': xr_ev_density_interpol,
}




preprocessing_results_xarray = create_prep_data(results_dict, conversion_table, unit_mapping)

# Set default battery weight to 0 # TODO: do I need this? (Luja has it in vehicles preprocessing)
# xr_default_battery = xr.DataArray(
#     0.0, 
#     dims=("Cohort", "Type"),
#     coords={
#         "Cohort": preprocessing_results_xarray["battery_weights"].coords["Cohort"],
#         "Type": ["Vehicles"]
#     }
# )
# preprocessing_results_xarray["battery_weights"] = prism.Q_(xr.concat((preprocessing_results_xarray["battery_weights"], xr_default_battery), dim="Type"), "kg")


# prep_data_oth_storage = create_prep_data(results_dict, conversion_table, unit_mapping)
# prep_data_oth_storage["stocks"] = prism.Q_(prep_data_oth_storage["stocks"], "MWh")
# prep_data_oth_storage["material_intensities"] = prism.Q_(prep_data_oth_storage["material_intensities"], "kg/kWh")
# prep_data_oth_storage["shares"] = prism.Q_(prep_data_oth_storage["shares"], "share")
# prep_data_oth_storage["set_unit_flexible"] = prism.U_(prep_data_oth_storage["stocks"]) # prism.U_ gives the unit back