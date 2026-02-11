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
from imagematerials.util import dataset_to_array, pandas_to_xarray, convert_lifetime
from imagematerials.model import GenericMainModel, GenericStocks, SharesInflowStocks, Maintenance, GenericMaterials, MaterialIntensities
from imagematerials.factory import ModelFactory, Sector
from imagematerials.concepts import create_electricity_graph, create_region_graph
from imagematerials.electricity.utils import MNLogit, stock_tail, create_prep_data, stock_share_calc


from imagematerials.constants import IMAGE_REGIONS

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

# read in the storage costs according to IRENA storage report & other sources in the SI
storage_costs = pd.read_csv(path_external_data_standard / 'storage_cost.csv', index_col=0).transpose()

# read in the assumed malus & bonus of storage costs (malus for advanced technologies, still under development; bonus for batteries currently used in EVs, we assume that a large volume of used EV batteries will be available and used for dedicated electricity storage, thus lowering costs), only the bonus remains by 2030
storage_malus = pd.read_csv(path_external_data_standard / 'storage_malus.csv', index_col=0).transpose()

#read in the assumptions on the long-term price decline after 2050. Prices are in $ct / kWh electricity cycled (the fraction of the annual growth rate (determined based on 2018-2030) that will be applied after 2030, ranging from 0.25 to 1 - 0.25 means the price decline is not expected to continue strongly, while 1 means that the same (2018-2030) annual price decline is also applied between 2030 and 2050)
storage_ltdecline = pd.Series(pd.read_csv(path_external_data_standard / 'storage_ltdecline.csv',index_col=0,  header=None).transpose().iloc[0])

#read in the energy density assumptions (kg/kWh storage capacity - mass required to store one unit of energy — more mass per energy = worse performance)
storage_density = pd.read_csv(path_external_data_standard / 'storage_density_kg_per_kwh.csv',index_col=0).transpose()

#read in the lifetime of storage technologies (in yrs). The lifetime is assumed to be 1.5* the number of cycles divided by the number of days in a year (assuming diurnal use, and 50% extra cycles before replacement, representing continued use below 80% remaining capacity) OR the maximum lifetime in years, which-ever comes first 
storage_lifetime = pd.read_csv(path_external_data_standard / 'storage_lifetime.csv',index_col=0).transpose()

kilometrage = pd.read_csv(path_external_data_scenario / 'kilometrage.csv', index_col='t') #annual car mileage in kms/yr, based  mostly  on  Pauliuk  et  al.  (2012a)

# material compositions (storage) in wt%
storage_materials = pd.read_csv(path_external_data_standard / 'storage_materials_dynamic.csv',index_col=[0,1], usecols=lambda col: col != "unit").transpose()  # wt% of total battery weight for various materials, total battery weight is given by the density file above

# Hydro-dam power capacity (also MW) within 5 regions reported by the IHA (international Hydropwer Association)
phs_projections = pd.read_csv(path_external_data_standard / 'PHS.csv', index_col='t')   # pumped hydro storage capacity (MW)

# Using the 250 Wh/kg on the kWh of the various batteries a weight (in kg) of the battery per vehicle category is determined
battery_weights = pd.read_csv(path_external_data_standard / "battery_weights_kg.csv", index_col=[0,1])


# 2. IMAGE/TIMER files ====================================================================================

# read TIMER installed storage capacity (MWh, reservoir)
storage = read_mym_df(path_image_output.joinpath("StorResTot.out"))   #storage capacity in MWh (reservoir, so energy capacity, not power capacity, the latter is used later on in the pumped hydro storage calculations)
storage_0 = read_mym_df(path_image_output.joinpath("StorResTot.out"))   #storage capacity in MWh (reservoir, so energy capacity, not power capacity, the latter is used later on in the pumped hydro storage calculations)
 
#storage capacity in MW (power capacity), to compare it to Pumped hydro storage projections (also given in MW, power capacity)
storage_power = read_mym_df(path_image_output / 'StorCapTot.out')  

# Generation capacity (stock & inflow/new) in MW peak capacity, FILES from TIMER
gcap_data = read_mym_df(path_image_output / 'Gcap.out') # needed to get hydro power for storage
gcap_data = gcap_data.iloc[:, :26]


# ----------------------------------------------------------------------------------------------------------
# ##########################################################################################################
# %% 2.2) Prepare general variables
# ##########################################################################################################
# ----------------------------------------------------------------------------------------------------------

# Calculations used in both 'vehicle battery storage' and 'other storage'

##################
# Interpolations #
##################

storage = storage.iloc[:, :26]    # drop global total column and empty (27) column

# J: in high storage scenario the storage demand linearly increases between 2021 and 2050 compared to its original value until it is double by 2050, and then remains constant
if SENS_ANALYSIS == 'high_stor':
    storage_multiplier = storage
    for year in range(2021,2051):
        storage_multiplier.loc[year] = storage.loc[year] * (1 + (1/30*(year-2020)))
    for year in range(2051,YEAR_END+1):
        storage_multiplier.loc[year] = storage.loc[year] * 2


# turn index to integer for sorting during the next step
storage_costs.index = storage_costs.index.astype('int64')
storage_malus.index = storage_malus.index.astype('int64')
storage_density.index = storage_density.index.astype('int64')
storage_lifetime.index = storage_lifetime.index.astype('int64')

# to interpolate between 2018 and 2030, first create empty rows (NaN values) 
storage_start = storage_costs.first_valid_index()
storage_end =   storage_costs.last_valid_index()
for i in range(storage_start+1,storage_end):
    storage_costs = pd.concat([storage_costs, pd.DataFrame(index=[i])]) #, ignore_index=True
    storage_malus = pd.concat([storage_malus, pd.DataFrame(index=[i])])         # mind: the malus needs to be defined for the same years as the cost indications
    storage_density = pd.concat([storage_density, pd.DataFrame(index=[i])])     # mind: the density needs to be defined for the same years as the cost indications
    storage_lifetime = pd.concat([storage_lifetime, pd.DataFrame(index=[i])])   # mind: the lifetime needs to be defined for the same years as the cost indications
    
# then, do the actual interpolation on the sorted dataframes                                                    
storage_costs_interpol = storage_costs.sort_index(axis=0).interpolate(axis=0)#.index.astype('int64')
storage_malus_interpol = storage_malus.sort_index(axis=0).interpolate(axis=0)
storage_density_interpol = storage_density.sort_index(axis=0).interpolate(axis=0)  # density calculation continue with the material calculations
storage_lifetime_interpol = storage_lifetime.sort_index(axis=0).interpolate(axis=0)  # lifetime calculation continue with the material calculations

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

#############
# Lifetimes #
#############

# First the lifetime of storage technologies needs to be defined over time, before running the dynamic stock function
# before 2018
for year in reversed(range(YEAR_FIRST_GRID,storage_start)):
    # storage_lifetime_interpol = pd.concat([storage_lifetime_interpol, pd.Series(storage_lifetime_interpol.loc[storage_lifetime_interpol.first_valid_index()], name=year)])
    row = pd.DataFrame([storage_lifetime_interpol.loc[storage_lifetime_interpol.first_valid_index()]])
    storage_lifetime_interpol.loc[year] = row.iloc[0]
# after 2030
for year in range(2030+1,YEAR_OUT+1):
    # storage_lifetime_interpol = pd.concat([storage_lifetime_interpol, pd.Series(storage_lifetime_interpol.loc[storage_lifetime_interpol.last_valid_index()], name=year)])
    row = pd.DataFrame([storage_lifetime_interpol.loc[storage_lifetime_interpol.last_valid_index()]])
    storage_lifetime_interpol.loc[year] = row.iloc[0]

storage_lifetime_interpol = storage_lifetime_interpol.sort_index(axis=0)
# drop the PHS from the interpolated lifetime frame, as the PHS is calculated separately
storage_lifetime_interpol = storage_lifetime_interpol.drop(columns=['PHS'])

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

# total = storage_market_share.sum(axis=1)
region_list = list(kilometrage.columns.values)   
storage.columns = region_list

#%% 2.4) Hydro Power & Other Storage
###########################################################################################################

# OPTION: NO V2G ---------------------------------------------------------------
# TODO: this is a temporary solution, until the coupling with the vehicle sector and the battery module calculations are implemented/integrated
storage_vehicles = pd.DataFrame(0, index=storage.index, columns=storage.columns)  # set vehicle storage to zero when not using V2G
storage_vehicles = storage_vehicles.loc[:YEAR_OUT]
#-------------------------------------------------------------------------------

# Take the TIMER Hydro-dam capacity (MW) & compare it to Pumped hydro capacity (MW) projections from the International Hydropower Association

Gcap_hydro = gcap_data[['time','DIM_1', 7]].pivot_table(index='time', columns='DIM_1')   # IMAGE-TIMER Hydro dam capacity (power, in MW)
Gcap_hydro = Gcap_hydro.iloc[:, :26]
region_list = list(kilometrage.columns.values)  # get a list with region names
Gcap_hydro.columns = region_list
Gcap_hydro = Gcap_hydro.loc[:YEAR_OUT]

# storage capacity in MW (power capacity), to compare it to Pumped hydro storage projections (also given in MW, power capacity)              
# storage_power.drop(storage_power.iloc[:, -2:], inplace = True, axis = 1) # error prone
storage_power = storage_power.iloc[:, :26]  
storage_power.columns = region_list
storage_power = storage_power.loc[:YEAR_OUT]

#Disaggregate the Pumped hydro-storgae projections to 26 IMAGE regions according to the relative Hydro-dam power capacity (also MW) within 5 regions reported by the IHA (international Hydropwer Association)
phs_regions = [[10,11],[19],[1],[22],[0,2,3,4,5,6,7,8,9,12,13,14,15,16,17,18,20,21,23,24,25]]   # subregions in IHA data for Europe, China, US, Japan, RoW, MIND: region refers to IMAGE region MINUS 1
phs_projections_IMAGE = pd.DataFrame(index=Gcap_hydro.index, columns=Gcap_hydro.columns)        # empty dataframe

for column in range(0,len(phs_regions)):
    sum_data = Gcap_hydro.iloc[:,phs_regions[column]].sum(axis=1) # first, get the sum of all hydropower in the IHA regions (to divide over in second step)
    for region in range(0,len(IMAGE_REGIONS)):
        if region in phs_regions[column]:
            phs_projections_IMAGE.iloc[:,region] = phs_projections.iloc[:,column] * (Gcap_hydro.iloc[:,region]/sum_data) 
            # J: allocate share of the phs_projections to each IMAGE region based on the share of that region on the generation capacity of the IHA region it is part of
            # J: (Gcap_hydro.iloc[:,region]/sum_data) is between 0 and 1, so the phs_projections are disaggregated to IMAGE regions

# Then fill the years after 2030 (end of IHA projections) according to the Gcap annual growth rate (assuming a fixed percentage of Hydro dams will be built with Pumped hydro capabilities after )
if SENS_ANALYSIS == 'high_stor':
    phs_projections_IMAGE.loc[2030:YEAR_OUT] =  phs_projections_IMAGE.loc[2030] * (Gcap_hydro.loc[2030:YEAR_OUT]/Gcap_hydro.loc[2030:YEAR_OUT])  # no growth after 2030 in the high_stor sensitivity variant
else:
    phs_projections_IMAGE.loc[2030:YEAR_OUT] =  phs_projections_IMAGE.loc[2030] * (Gcap_hydro.loc[2030:YEAR_OUT]/Gcap_hydro.loc[2030])


# Calculate the fractions of the storage capacity that is provided through pumped hydro-storage, electric vehicles or other storage (larger than 1 means the capacity superseeds the demand for energy storage, in terms of power in MW or enery in MWh) 
phs_storage_fraction = phs_projections_IMAGE.divide(storage_power.loc[:YEAR_OUT]).clip(upper=1) # the phs storage fraction deployed to fulfill storage demand, both phs & storage_power here are expressed in MW
# absolute storage capacity (MWh)
phs_storage_theoretical = phs_projections_IMAGE.divide(storage_power) * storage.loc[:YEAR_OUT] # ??? theoretically available PHS storage (MWh; fraction * total) only used in the graphs that show surplus capacity
phs_storage = phs_storage_fraction * storage.loc[:YEAR_OUT]

# derive inflow & outflow (in MWh) for PHS, for later use in the material calculations 
PHS_kg_perkWh = 26.8   # kg per kWh storage capacity (as weight addition to existing hydro plants to make them pumped) 
phs_storage_stock_tail = stock_tail(phs_storage.astype(float), YEAR_OUT)
storage_lifetime_PHS = storage_lifetime['PHS'].reindex(list(range(YEAR_FIRST_GRID,YEAR_OUT+1)), axis=0).interpolate(limit_direction='both')


#------------------------------------------------------------------------------------------------------------------------------------------------------------


storage_remaining = storage.loc[:YEAR_OUT] * (1 - phs_storage_fraction)

# For now: assume no other storage before 1971 -> TODO: check this
for year in range(YEAR_FIRST_GRID,YEAR_START):
    storage_remaining.loc[year] = 0
storage_remaining = storage_remaining.sort_index(axis=0)



#%% PLOTS TEST ###################################################################################


path_comparison_data = Path(path_base, "data", "raw", "electricity")
df_power_s =    pd.read_csv(Path(path_comparison_data,"lit_grid_storage_stock_power_GW.csv"), index_col = [0])
df_energy_s =   pd.read_csv(Path(path_comparison_data,"lit_grid_storage_stock_energy_GWh.csv"), index_col = [0])


# PHS Power (GW) --------------------------------------------------
df_sel = df_power_s[(df_power_s["technology"] == "PHS")]
phs_projections_0 = pd.read_csv(path_external_data_standard / 'PHS.csv', index_col='t') 
phs_projections_0["global"] = phs_projections_0.sum(axis=1)/1000
phs_projections["global"] = phs_projections.sum(axis=1)/1000
phs_projections_IMAGE["global"] = phs_projections_IMAGE.sum(axis=1)/1000
storage_power["global"] = storage_power.sum(axis=1)/1000


fig, ax = plt.subplots(figsize=(10,6))
# ax.plot(phs_projections_0.index, phs_projections_0["global"], label="PHS Projections (IHA)", color='blue')
ax.scatter(phs_projections.index, phs_projections["global"], label="IHA (used in model)", color='purple', s=20)
ax.plot(phs_projections_IMAGE.index, phs_projections_IMAGE["global"], label="IMAGE-Materials", color='orange', linestyle ='--')
ax.plot(storage_power.index, storage_power["global"], label="SSP2_M_CP TIMER StorCapTot", color='black')
for row in range(len(df_sel)):
    src = df_sel["source"].iloc[row]
    scen = df_sel["scenario"].iloc[row]
    plt.scatter(df_sel.index[row], df_sel["value"].iloc[row], label=f"{src} – {scen}", linewidth=1)

ax.hlines(y=0, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1],color='grey', linestyle=':', alpha=0.6) 
ax.set_xlim(2000,2050)
ax.set_ylim(-200,4000)
ax.set_xlabel("Year")
ax.set_ylabel("PHS Capacity (GW)")
ax.set_title("PHS Power Capacity (GW)")
ax.grid(alpha=0.3)
ax.legend()


# PHS Energy (GWh) --------------------------------------------------
df_sel = df_energy_s[(df_energy_s["technology"] == "PHS")]
df_sel_total = df_energy_s[(df_energy_s["category"] == "total")]
phs_storage["global"] = phs_storage.sum(axis=1)/1000 # IMAGE-Materials
storage["global"] = storage.sum(axis=1)/1000 # TIMER


fig, ax = plt.subplots(figsize=(10,6))
# ax.plot(phs_projections_0.index, phs_projections_0["global"], label="PHS Projections (IHA)", color='blue')
# ax.scatter(phs_projections.index, phs_projections["global"], label="IHA (used in model)", color='purple', s=20)
ax.plot(phs_storage.index, phs_storage["global"], label="IMAGE-Materials PHS", color='orange', linestyle ='--')
ax.plot(storage.index, storage["global"], label="SSP2_M_CP TIMER StorResTot", color='black')
for row in range(len(df_sel)):
    src = df_sel["source"].iloc[row]
    scen = df_sel["scenario"].iloc[row]
    plt.scatter(df_sel.index[row], df_sel["value"].iloc[row], label=f"{src} – {scen}", linewidth=1)
for row in range(len(df_sel_total)):
    src = df_sel_total["source"].iloc[row]
    scen = df_sel_total["scenario"].iloc[row]
    plt.scatter(df_sel_total.index[row], df_sel_total["value"].iloc[row], label=f"{src} – {scen} (total storage)", linewidth=1, marker='x')

ax.set_xlim(2000,2100)
ax.set_ylim(-200,55000)
ax.set_xlabel("Year")
ax.set_ylabel("PHS Capacity (GWh)")
ax.set_title("PHS Energy Capacity (GWh)")
ax.grid(alpha=0.3)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=12, frameon=False)
# ax.legend()



# THIS INTO AdjustElectricityStocks MODEL? ------------------------------------------------
# if SENS_ANALYSIS == 'high_stor':
#     oth_storage_fraction = 0.5 * storage_remaining 
#     oth_storage_fraction += ((storage_remaining * 0.5) - storage_vehicles).clip(lower=0)    
#     oth_storage_fraction = oth_storage_fraction.divide(storage).where(oth_storage_fraction > 0, 0).clip(lower=0) 
#     evs_storage_fraction = 1 - (phs_storage_fraction + oth_storage_fraction)     # electric vehicle storage (BEV + PHEV) capacity and total storage demand are expressed as MWh
# else: 
#     oth_storage_fraction = (storage_remaining - storage_vehicles).clip(lower=0)    
#     oth_storage_fraction = oth_storage_fraction.divide(storage.loc[:YEAR_OUT]).where(oth_storage_fraction > 0, 0).clip(lower=0)      
#     evs_storage_fraction = 1 - (phs_storage_fraction + oth_storage_fraction)     # electric vehicle storage (BEV + PHEV) capacity and total storage demand are expressed as MWh
# checksum = phs_storage_fraction + evs_storage_fraction + oth_storage_fraction   # should be 1 for all fields

# absolute storage capacity (MWh)
# evs_storage = evs_storage_fraction * storage.loc[:YEAR_OUT]
# oth_storage = oth_storage_fraction * storage.loc[:YEAR_OUT]
#----------------------------------------------------------------------------------------




###########################################################################################################
#%%% EV

if SENS_ANALYSIS == 'high_stor':
   capacity_usable_PHEV = 0.025   # 2.5% of capacity of PHEV is usable as storage (in the pessimistic sensitivity variant)
   capacity_usable_BEV  = 0.05    # 5  % of capacity of BEVs is usable as storage (in the pessimistic sensitivity variant)
else: 
   capacity_usable_PHEV = 0.05    # 5% of capacity of PHEV is usable as storage
   capacity_usable_BEV  = 0.10    # 10% of capacity of BEVs is usable as storage


#First we calculate the share of the inflow using only a few of the technologies in the storage market share
#The selection represents only the batteries that are suitable for EV & mobile applications
EV_battery_list = ['NiMH', 'LMO', 'NMC', 'NCA', 'LFP', 'Lithium Sulfur', 'Lithium Ceramic', 'Lithium-air']
#normalize the selection of market shares, so that total market share is 1 again (taking the relative share in the selected battery techs)
market_share_EVs = storage_market_share[EV_battery_list].div(storage_market_share[EV_battery_list].sum(axis=1), axis=0)



###########################################################################################################
#%%% 2.4.1) Prep_data File
###########################################################################################################


# PHS -----------------------------------------------------------------------------------------------------

phs_stock = phs_storage_stock_tail.copy()

# Bring dataframes into correct shape for the results_dict

# stocks: years as index and regions as columns -> years as index and (technology, region) as columns
# Current columns are regions
regions = phs_stock.columns.tolist()

# Create a MultiIndex with technology "PHS" for all columns
multi_cols = pd.MultiIndex.from_tuples([("PHS", r) for r in regions], names=["technology", "region"])
# Assign to the DataFrame
phs_stock.columns = multi_cols

# lifetimes
df_mean = storage_lifetime_PHS.copy().to_frame()
df_stdev = df_mean * STD_LIFETIMES_ELECTR
df_mean.columns = [(col, 'mean') for col in df_mean.columns] # Rename columns to multi-level tuples
df_stdev.columns = [(col, 'stdev') for col in df_stdev.columns]
phs_lifetime_distr = pd.concat([df_mean, df_stdev], axis=1) # Concatenate along columns
phs_lifetime_distr.index.name = 'year'

# MIs: (years, material) index and technologies as columns -> years as index and (technology, Material) as columns
phs_materials = stor_materials_interpol.loc[idx[:,:],'PHS'].unstack() * PHS_kg_perkWh * 1000 # wt% * kg/kWh * 1000 kWh/MWh = kg/MWh
# Current columns are materials
materials = phs_materials.columns.tolist()
# Create a MultiIndex with technology "PHS" for all columns
multi_cols = pd.MultiIndex.from_tuples([("PHS", m) for m in materials], names=["technology", "material"])
# Assign to the DataFrame
phs_materials.columns = multi_cols

# Conversion table for all coordinates, to be removed/adapted after input tables are fixed.
conversion_table = {
    "phs_stock": (["Time"], ["Type", "Region"],),
    "phs_materials": (["Cohort"], ["Type", "material"],)
    # "gcap_materials_interpol": (["Cohort"], ["Type", "SubType", "material"], {"Type": ["Type", "SubType"]})
}

results_dict = {
        'phs_stock': phs_stock,
        'phs_materials': phs_materials,
        'phs_lifetime_distr': phs_lifetime_distr,
}

prep_data_phs = create_prep_data(results_dict, conversion_table, unit_mapping)
prep_data_phs["stocks"] = prism.Q_(prep_data_phs["stocks"], "MWh")
prep_data_phs["material_intensities"] = prism.Q_(prep_data_phs["material_intensities"], "kg/MWh")
prep_data_phs["set_unit_flexible"] = prism.U_(prep_data_phs["stocks"])


# Other storage--------------------------------------------------------------------------------------------

oth_storage_stock = storage_remaining.copy()

# Bring dataframes into correct shape for the results_dict

# stocks: (years, regions) index and technologies as columns -> years as index and (technology, region) as columns
# Current columns are regions
regions = storage_remaining.columns.tolist()
# stor_tech = list(storage_lifetime_interpol.columns)
# Create a MultiIndex with technology "Other Storage" for all columns
multi_cols = pd.MultiIndex.from_tuples([("Other Storage", r) for r in regions], names=["technology", "region"])
# multi_cols = pd.MultiIndex.from_product([stor_tech, regions], names=["technology", "region"])
# Assign to the DataFrame
oth_storage_stock.columns = multi_cols

# lifetimes
df_mean = storage_lifetime_interpol.copy() #.to_frame()
df_stdev = df_mean * STD_LIFETIMES_ELECTR
df_mean.columns = [(col, 'mean') for col in df_mean.columns] # Rename columns to multi-level tuples
df_stdev.columns = [(col, 'stdev') for col in df_stdev.columns]
oth_storage_lifetime_distr = pd.concat([df_mean, df_stdev], axis=1) # Concatenate along columns
oth_storage_lifetime_distr.index.name = 'year'

# MIs: (years, material) index and technologies as columns -> years as index and (technology, Material) as columns
stor_tech = list(storage_lifetime_interpol.columns)
# 2nd level of the MultiIndex are materials (1971,'Aluminium')
oth_storage_materials = stor_materials_interpol.copy()
oth_storage_materials.index.names = ["year", "material"]  # assign names
oth_storage_materials = oth_storage_materials.reset_index(level=["year", "material"])
# Pivot so we get (technology, material) as columns, years as index
oth_storage_materials = oth_storage_materials.melt(id_vars=["year", "material"], var_name="technology", value_name="value")
oth_storage_materials = oth_storage_materials.pivot_table(index="year", columns=["technology", "material"], values="value")
# Ensure proper MultiIndex column names
oth_storage_materials.columns = pd.MultiIndex.from_tuples(oth_storage_materials.columns, names=["technology", "material"])
xr_oth_storage_materials = dataset_to_array(pandas_to_xarray(oth_storage_materials, unit_mapping), *(["Cohort"], ["Type", "material"],))
xr_storage_density_interpol = dataset_to_array(pandas_to_xarray(storage_density_interpol, unit_mapping), *(["Cohort"], ["Type"],))
oth_storage_materialintens = xr_oth_storage_materials * xr_storage_density_interpol

# Conversion table for all coordinates, to be removed/adapted after input tables are fixed.
conversion_table = {
    "oth_storage_stock": (["Time"], ["SuperType", "Region"],),
    "oth_storage_materials": (["Cohort"], ["Type", "material"],), #SubType
    "oth_storage_shares": (["Cohort"], ["Type",]) #SubType
}
## "gcap_materials_interpol": (["Cohort"], ["Type", "SubType", "material"], {"Type": ["Type", "SubType"]})

results_dict = {
        'oth_storage_stock': oth_storage_stock,
        'oth_storage_materials': oth_storage_materialintens,
        'oth_storage_lifetime_distr': oth_storage_lifetime_distr,
        'oth_storage_shares': storage_market_share
}

prep_data_oth_storage = create_prep_data(results_dict, conversion_table, unit_mapping)
prep_data_oth_storage["stocks"] = prism.Q_(prep_data_oth_storage["stocks"], "MWh")
prep_data_oth_storage["material_intensities"] = prism.Q_(prep_data_oth_storage["material_intensities"], "kg/kWh")
prep_data_oth_storage["shares"] = prism.Q_(prep_data_oth_storage["shares"], "share")
prep_data_oth_storage["set_unit_flexible"] = prism.U_(prep_data_oth_storage["stocks"]) # prism.U_ gives the unit back


##########################################################
#%% Plots check

stor2 = storage_0/1000 # MWh->GWh
stor2["total"] = stor2.sum(axis=1)

storage_timer = storage/1000 # MWh->GWh
storage_timer["total"] = storage_timer.sum(axis=1)
resstor = prep_data_oth_storage["stocks"].sum(["Region", "SuperType"]).pint.to("GWh")
phs = prep_data_phs["stocks"].sum(["Region", "Type"]).pint.to("GWh")

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(resstor["Time"], resstor, label="Res. storage", c="darkorange")
ax.plot(phs["Time"], phs, label="PHS", c="purple")
ax.plot(storage_timer.index, storage_timer["total"], linewidth=1.5, c="black", label="TIMER")
ax.plot(stor2.index, stor2["total"], linewidth=1.5, c="red", label="TIMER - right after read in")

ax.set_xlabel("Time", fontsize=14)
ax.set_ylabel(f"Storage (GWh)", fontsize=14)
ax.set_title("Storage plotted in preprocessing", fontweight="bold")
ax.legend()
plt.tight_layout()