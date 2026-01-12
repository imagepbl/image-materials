# %%
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
from imagematerials.vehicles.modelling_functions import interpolate
from imagematerials.electricity.utils import (
    MNLogit, 
    create_prep_data, 
    logistic, 
    quadratic, 
    interpolate_xr, 
    add_historic_stock, 
    normalize_selected_techs,
)


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
    EV_BATTERIES,
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

prism.unit_registry.load_definitions(path_base / "imagematerials" / "units.txt")
# C:\Users\Judit\PhD\Coding\image-materials\imagematerials\units.txt
idx = pd.IndexSlice   

#----------------------------------------------------------------------------------------------------------
###########################################################################################################
# %% I. Read in files
###########################################################################################################
#----------------------------------------------------------------------------------------------------------

# 1. External Data ======================================================================================== 

# storage costs according to IRENA storage report & other sources in the SI ($ct/kWh electricity cycled)
storage_costs = pd.read_csv(path_external_data_standard / 'storage_cost.csv', index_col=0).transpose()

# assumed malus & bonus of storage costs (malus for advanced technologies, still under development; bonus for batteries currently used in EVs, we assume that a large volume of used EV batteries will be available and used for dedicated electricity storage, thus lowering costs), only the bonus remains by 2030
costs_correction = pd.read_csv(path_external_data_standard / 'storage_malus.csv', index_col=0).transpose()

# assumptions on the long-term price decline after 2050. the fraction of the annual growth rate (determined based on 2018-2030) that will be applied after 2030, ranging from 0.25 to 1 - 0.25 means the price decline is not expected to continue strongly, while 1 means that the same (2018-2030) annual price decline is also applied between 2030 and 2050)
cost_decline_longterm_correction = pd.Series(pd.read_csv(path_external_data_standard / 'storage_ltdecline.csv',index_col=0,  header=None).transpose().iloc[0]) 

# INVERS energy density (note that the unit is kg/kWh, the invers of energy density: storage capacity - mass required to store one unit of energy —> more mass per energy = worse performance)
# TODO: read in directly IRENA data?
energy_density = pd.read_csv(path_external_data_standard / 'storage_density_kg_per_kwh.csv',index_col=0).transpose()

# lifetime of storage technologies (in yrs). The lifetime is assumed to be 1.5* the number of cycles divided by the number of days in a year (assuming diurnal use, and 50% extra cycles before replacement, representing continued use below 80% remaining capacity) OR the maximum lifetime in years, which-ever comes first 
storage_lifetime = pd.read_csv(path_external_data_standard / 'storage_lifetime.csv',index_col=0).transpose()

# material compositions (storage) in wt%
storage_materials = pd.read_csv(path_external_data_standard / 'storage_materials_dynamic.csv',index_col=[0,1]).transpose()  # wt% of total battery weight for various materials, total battery weight is given by the density file above

# Using the 250 Wh/kg on the kWh of the various batteries a weight (in kg) of the battery per vehicle category is determined
# TODO: where is this data from? kWh per battery type?
battery_weights_data = pd.read_csv(path_external_data_standard / "battery_weights_kg.csv", index_col=[0,1])

# usable capacity of EV batteries for V2G applications (relative: fraction of the total battery capacity that can be used for V2G)
if SENS_ANALYSIS == 'high_stor':
   # pessimistic sensitivity variant (meaning more additional storage is needed) -> smaller fraction of the EV capacities is usable as storage compared to the normal case
#    capacity_usable_PHEV = 0.025   # 2.5% of capacity of PHEV is usable as storage (in the pessimistic sensitivity variant)
#    capacity_usable_BEV  = 0.05    # 5  % of capacity of BEVs is usable as storage (in the pessimistic sensitivity variant)
    ev_capacity_fraction_v2g = pd.read_csv(path_external_data_standard / 'ev_battery_capacity_usable_for_v2g_variant_high_storage.csv')
else:
    ev_capacity_fraction_v2g = pd.read_csv(path_external_data_standard / 'ev_battery_capacity_usable_for_v2g.csv')

# fraction of EVs available for V2G (considering that not all EVs are capable of bi-directional loading, economic incentives are still missing, and not all owners are willing to provide V2G services)
ev_fraction_v2g_data = pd.read_csv(path_external_data_standard / 'ev_fraction_available_for_v2g.csv', index_col=[0])


# ##########################################################################################################
# %% Prepare general variables
# ##########################################################################################################


###########################################################################################################
# %%% Transform to xarray #

# create list of all vehicle types (combinations of vehicles type (Cars, Medium Freight Trucks,...) and drive trains (ICE, BEV,...))
vehicle_list = [f"{super_type} - {sub_type}" for super_type, sub_type in product(typical_modes, drive_trains)]


# 1. Market Shares -----------------------------------------------------------------------------

# 1.1 storage costs
years = storage_costs.index.astype(int) #.astype(int) to convert years from strings to integers
techs = storage_costs.columns
data_array = storage_costs.to_numpy()
xr_storage_costs = xr.DataArray(
    data_array,
    dims=('Cohort', 'Type'),
    coords={
        'Cohort': years,
        'Type': techs
    },
    name='StorageCosts'
)
xr_storage_costs = prism.Q_(xr_storage_costs, "USD_cent/kWh")

# 1.2 storage costs correction (malus/bonus multiplicative factor) 
years = costs_correction.index.astype(int)
techs = costs_correction.columns
data_array = costs_correction.to_numpy()
xr_costs_correction = xr.DataArray(
    data_array,
    dims=('Cohort', 'Type'),
    coords={
        'Cohort': years,
        'Type': techs
    },
    name='StorageCostsCorrection'
)
xr_costs_correction = prism.Q_(xr_costs_correction, "dimensionless")

# 1.3 storage costs longterm decline factor
techs = cost_decline_longterm_correction.index.rename(None)
data_array = cost_decline_longterm_correction.to_numpy()
xr_cost_decline_longterm_correction = xr.DataArray(
    data_array,
    dims=('Type',),
    coords={
        'Type': techs
    },
    name='StorageCostsDeclineLongterm'
)
xr_cost_decline_longterm_correction = prism.Q_(xr_cost_decline_longterm_correction, "fraction")

# 2. battery weigths: first some formatting and then conversion to xarray DataArray (see below)

# 3. material intensities -----------------------------------------------------------------------
storage_materials.columns = storage_materials.columns.rename([None, None]) # remove column MultiIndex name "g/MW" as it causes issues when converting to xarray
years = sorted(storage_materials.columns.get_level_values(0).unique())
techs = storage_materials.columns.get_level_values(1).unique()
materials = storage_materials.index
# Convert to 3D array: (Material, Year, Tech)
data_array = storage_materials.to_numpy().reshape(len(materials), len(years), len(techs))
# Build xarray DataArray
xr_storage_materials = xr.DataArray(
    data_array,
    dims=('material', 'Cohort', 'Type'),
    coords={
        'material': materials,
        'Cohort': years,
        'Type': techs
    },
    name='MaterialFractions'
)
xr_storage_materials = prism.Q_(xr_storage_materials, "fraction")


# 4. energy density -----------------------------------------------------------------------------
years = energy_density.index.astype(int)
energy_density.columns.name = None # remove header "kg/kWh" to avoid issues
techs = energy_density.columns
data_array = energy_density.to_numpy()
xr_energy_density = xr.DataArray(
    data_array,
    dims=('Cohort', 'Type'),
    coords={
        'Cohort': years,
        'Type': techs
    },
    name='EnergyDensity'
)
xr_energy_density = prism.Q_(xr_energy_density, "kg/kWh")


# 5. EV fraction available for V2G --------------------------------------------------------------------
# For this variable first the interpolations are done and then the conversion to xarray DataArray
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

# Create an empty expanded df
vhc_fraction_v2g = pd.DataFrame(index=ev_fraction_v2g.index)
# transfer the values for the relevant vehicle types (for now: Cars - BEV and Cars - PHEV)
for item in vehicle_list: # TODO: is this even necessary? If not all coordinates are defined, maybe they are just ignored in later calculations?
    if item in ev_fraction_v2g.columns:
        vhc_fraction_v2g[item] = ev_fraction_v2g[item]
    else:
        vhc_fraction_v2g[item] = 0 # other drive trains are not V2G capable
# Build xarray DataArray
years = vhc_fraction_v2g.index.astype(int).rename(None)
techs = vhc_fraction_v2g.columns
data_array = vhc_fraction_v2g.to_numpy()
xr_vhc_fraction_v2g = xr.DataArray(
    data_array,
    dims=('Time', 'Type'),
    coords={
        'Time': years,
        'Type': techs
    },
    name='VehicleFractionV2G'
)
xr_vhc_fraction_v2g = prism.Q_(xr_vhc_fraction_v2g, "fraction")

# 6. capacity used for V2G -----------------------------------------------------------------------------
# Build xarray DataArray
techs = ev_capacity_fraction_v2g.columns
data_array = ev_capacity_fraction_v2g.to_numpy().ravel()  # flatten to 1D
xr_capacity_fraction_v2g = xr.DataArray(
    data_array,
    dims=('Type'),
    coords={
        'Type': techs
    },
    name='CapacityFractionV2g'
)
xr_capacity_fraction_v2g = prism.Q_(xr_capacity_fraction_v2g, "fraction")




###########################################################################################################
# %%% Interpolate #


# 1. Market Shares -----------------------------------------------------------------------------

t_start = xr_storage_costs.Cohort.values[0]
t_end   = xr_storage_costs.Cohort.values[-1]

# interpolate from first to last vailable year within the data, then extend to  YEAR_START and 2050 (keep values constant before first and after last year)
xr_storage_costs      = interpolate_xr(xr_storage_costs, YEAR_START, 2050)
xr_costs_correction   = interpolate_xr(xr_costs_correction, YEAR_START, 2050)

# determine the annual % decline of the costs based on the 2018-2030 data (original, before applying the malus)
xr_cost_decline = (((xr_storage_costs.loc[t_start,:]-xr_storage_costs.loc[t_end,:])/(t_end-t_start))/xr_storage_costs.loc[t_start,:]).drop_vars('Cohort')
xr_cost_decline_longterm = xr_cost_decline*xr_cost_decline_longterm_correction # cost decline after 2030 = cost decline 2018-2030 * correction factor
# cost_decline_longterm_correction is a single number and should describe the long-term decline after 2030 relative to the 2018-2030 decline

# storage_costs_new = storage_costs_interpol * costs_correction_interpol
xr_storage_costs_cor = xr_storage_costs * xr_costs_correction

# ---------- future development ----------
# calculate the development from 2030 to 2050 (using annual price decline)
# vectorized approach for "for t in range(t_end, 2050+1): ..."
years_fwd = xr_storage_costs_cor.Cohort.where(
    xr_storage_costs_cor.Cohort > t_end, drop=True
)
n_fwd = years_fwd - t_end
factors_fwd = (1 - xr_cost_decline_longterm) ** n_fwd

xr_storage_costs_cor.loc[dict(Cohort=years_fwd)] = (
    xr_storage_costs_cor.sel(Cohort=t_end) * factors_fwd
)

# ---------- past development ----------
# for historic price development, assume 2x AVERAGE annual price decline on all technologies
years_bwd = xr_storage_costs_cor.Cohort.where(
    xr_storage_costs_cor.Cohort < t_start, drop=True
)
n_bwd = t_start - years_bwd
factors_bwd = (1 + 2 * xr_cost_decline_longterm.mean()) ** n_bwd

xr_storage_costs_cor.loc[dict(Cohort=years_bwd)] = (
    xr_storage_costs_cor.sel(Cohort=t_start) * factors_bwd
)
# restore values for lead-acid (set to constant 2018 values) -> exception: so that lead-acid gets a relative price advantage from 1970-2018
xr_storage_costs_cor.loc[dict(Cohort=years_bwd, Type="Deep-cycle Lead-Acid")] = (
    xr_storage_costs_cor.sel(Cohort=t_start, Type="Deep-cycle Lead-Acid")
)

# plt.figure()
# xr_storage_costs_cor.sel(
#     Type=['Flywheel'] #, 'Compressed Air'
# ).plot.scatter(x='Cohort', label='Flywheel', color='black')
# xr_storage_costs_cor.sel(
#     Type=['Deep-cycle Lead-Acid']
# ).plot.scatter(x='Cohort', label='Deep-cycle Lead-Acid', color='red')
# plt.legend()
# plt.show()


# market shares ---
# use the storage price development in the logit model to get market shares
storage_market_share = MNLogit(xr_storage_costs_cor, -0.2) #assumes input of an ordered dataframe with rows as years and columns as technologies, values as prices. Logitpar is the calibrated Logit parameter (usually a nagetive number between 0 and 1)
# fix the market share of storage technologies before YEAR_START and after 2050
storage_market_share_interp = interpolate_xr(storage_market_share, YEAR_FIRST_GRID, YEAR_OUT)
# As not all storage technologies are suitable for EV & mobile applications, select only those technologies.
# normalize the selection of EV battery technologies, so that total market share is 1 again (taking the relative share in the selected battery techs)
market_share_EVs = normalize_selected_techs(storage_market_share, EV_BATTERIES) # TODO: this should be done differently as market shares of EV batteries probably differ from their market shares in total storage market


# 2. Battery Weights ----------------------------------------------------------------
battery_weights = interpolate(battery_weights_data.unstack())

# 3. Material Intensities ----------------------------------------------------------------
xr_storage_materials_interp = interpolate_xr(xr_storage_materials, YEAR_FIRST_GRID, YEAR_OUT)

#  4. Energy Density ----------------------------------------------------------------
xr_energy_density_interp = interpolate_xr(xr_energy_density, YEAR_FIRST_GRID, YEAR_OUT)


#%%%% old
# turn index to integer for sorting during the next step
# storage_costs.index = storage_costs.index.astype('int64')
# costs_correction.index = costs_correction.index.astype('int64')
# energy_density.index = energy_density.index.astype('int64')

# to interpolate between 2018 and 2030, first create empty rows (NaN values) 
# storage_start = storage_costs.first_valid_index()
# storage_end =   storage_costs.last_valid_index()
# for i in range(storage_start+1,storage_end):
#     storage_costs = pd.concat([storage_costs, pd.DataFrame(index=[i])]) #, ignore_index=True
#     costs_correction = pd.concat([costs_correction, pd.DataFrame(index=[i])])         # mind: the malus needs to be defined for the same years as the cost indications
#     energy_density = pd.concat([energy_density, pd.DataFrame(index=[i])])     # mind: the density needs to be defined for the same years as the cost indications
    
# then, do the actual interpolation on the sorted dataframes                                                    
# storage_costs_interpol = storage_costs.sort_index(axis=0).interpolate(axis=0)#.index.astype('int64')
# costs_correction_interpol = costs_correction.sort_index(axis=0).interpolate(axis=0)
# energy_density_interpol = energy_density.sort_index(axis=0).interpolate(axis=0)  # density calculation continue with the material calculations

# energy density ---
# fix the energy density (kg/kwh) of storage technologies after 2030
# for year in range(2030+1,YEAR_OUT+1):
#     # energy_density_interpol = energy_density_interpol.append(pd.Series(energy_density_interpol.loc[energy_density_interpol.last_valid_index()], name=year))
#     row = energy_density_interpol.loc[[energy_density_interpol.last_valid_index()]]
#     row.index = [year]
#     energy_density_interpol = pd.concat([energy_density_interpol, row])
# # assumed fixed energy densities before 2018
# for year in reversed(range(YEAR_FIRST_GRID,storage_start)): # was YEAR_SWITCH, storage_start
#     # energy_density_interpol = energy_density_interpol.append(pd.Series(energy_density_interpol.loc[energy_density_interpol.first_valid_index()], name=year)).sort_index(axis=0)
#     row = energy_density_interpol.loc[[energy_density_interpol.first_valid_index()]]
#     row.index = [year]
#     energy_density_interpol = pd.concat([energy_density_interpol, row]).sort_index(axis=0)

# storage material intensity ---
# Interpolate material intensities (dynamic content for gcap & storage technologies between 1926 to 2100, based on data files)
# index = pd.MultiIndex.from_product([list(range(YEAR_FIRST_GRID, YEAR_OUT+1)), list(storage_materials.index)])
# stor_materials_interpol = pd.DataFrame(index=index, columns=storage_materials.columns.levels[1])
# # material intensities for storage
# for cat in list(storage_materials.columns.levels[1]):
#     stor_materials_1st   = storage_materials.loc[:,idx[storage_materials.columns[0][0],cat]]
#     stor_materials_interpol.loc[idx[YEAR_FIRST_GRID ,:],cat] = stor_materials_1st.to_numpy()  # set the first year (1926) values to the first available values in the dataset (for the year 2000) 
#     stor_materials_interpol.loc[idx[storage_materials.columns.levels[0].min(),:],cat] = storage_materials.loc[:, idx[storage_materials.columns.levels[0].min(),cat]].to_numpy() # set the middle year (2000) values to the first available values in the dataset (for the year 2000) 
#     stor_materials_interpol.loc[idx[storage_materials.columns.levels[0].max(),:],cat] = storage_materials.loc[:, idx[storage_materials.columns.levels[0].max(),cat]].to_numpy() # set the last year (2100) values to the last available values in the dataset (for the year 2050) 
#     stor_materials_interpol.loc[idx[:,:],cat] = stor_materials_interpol.loc[idx[:,:],cat].unstack().astype('float64').interpolate().stack()


#################
#%%%% Market Shares #
#################

# Determine MARKET SHARE of the storage capacity using a multi-nomial logit function

# storage costs ---
# determine the annual % decline of the costs based on the 2018-2030 data (original, before applying the malus)
# decline = ((storage_costs_interpol.loc[storage_start,:]-storage_costs_interpol.loc[storage_end,:])/(storage_end-storage_start))/storage_costs_interpol.loc[storage_start,:]
# decline_used = decline*costs_decline_longterm #TODO: what is happening here? Why?
# # costs_decline_longterm is a single number and should describe the long-term decline after 2030 relative to the 2018-2030 decline

# storage_costs_new = storage_costs_interpol * costs_correction_interpol
# # calculate the development from 2030 to 2050 (using annual price decline)
# for year in range(storage_end+1,2050+1):
#     # print(year)
#     # storage_costs_new = storage_costs_new.append(pd.Series(storage_costs_new.loc[storage_costs_new.last_valid_index()]*(1-decline_used), name=year))
#     # storage_costs_new = pd.concat([storage_costs_new, pd.Series(storage_costs_new.loc[storage_costs_new.last_valid_index()] * (1 - decline_used), name=year)])
#     row = pd.DataFrame([storage_costs_new.loc[storage_costs_new.last_valid_index()] * (1 - decline_used)])
#     storage_costs_new.loc[year] = row.iloc[0]
# # for historic price development, assume 2x AVERAGE annual price decline on all technologies, except lead-acid (so that lead-acid gets a relative price advantage from 1970-2018)
# for year in reversed(range(YEAR_START,storage_start)):
#     # storage_costs_new = storage_costs_new.append(pd.Series(storage_costs_new.loc[storage_costs_new.first_valid_index()]*(1+(2*decline_used.mean())), name=year)).sort_index(axis=0)
#     row = pd.DataFrame([storage_costs_new.loc[storage_costs_new.first_valid_index()]*(1+(2*decline_used.mean()))])
#     storage_costs_new.loc[year] = row.iloc[0]
#     storage_costs_new.sort_index(axis=0, inplace=True) 

# storage_costs_new.sort_index(axis=0, inplace=True) 
# storage_costs_new.loc[1971:2017,'Deep-cycle Lead-Acid'] = storage_costs_new.loc[2018,'Deep-cycle Lead-Acid'] # restore the exception (set to constant 2018 values)


# # market shares ---
# # use the storage price development in the logit model to get market shares
# storage_market_share = MNLogit(storage_costs_new, -0.2) #assumes input of an ordered dataframe with rows as years and columns as technologies, values as prices. Logitpar is the calibrated Logit parameter (usually a nagetive number between 0 and 1)

# fix the market share of storage technologies after 2050
# for year in range(2050+1,YEAR_OUT+1):
#     # storage_market_share = storage_market_share.append(pd.Series(storage_market_share.loc[storage_market_share.last_valid_index()], name=year))
#     row = pd.DataFrame([storage_market_share.loc[storage_market_share.last_valid_index()]])
#     storage_market_share.loc[year] = row.iloc[0]
# # fix the market share of storage technologies before YEAR_START
# for year in range(YEAR_FIRST_GRID,YEAR_START):
#     # storage_market_share = storage_market_share.append(pd.Series(storage_market_share.loc[storage_market_share.last_valid_index()], name=year))
#     row = pd.DataFrame([storage_market_share.loc[storage_market_share.first_valid_index()]])
#     storage_market_share.loc[year] = row.iloc[0]
    
# storage_market_share = storage_market_share.sort_index(axis=0)

# #First we calculate the share of the inflow using only a few of the technologies in the storage market share
# #The selection represents only the batteries that are suitable for EV & mobile applications

# #normalize the selection of market shares, so that total market share is 1 again (taking the relative share in the selected battery techs)
# market_share_EVs = storage_market_share[EV_BATTERIES].div(storage_market_share[EV_BATTERIES].sum(axis=1), axis=0)


###################
# Battery Weights #
###################

###################
#%% Capacity #
###################


#%%% 2. Dynamic battery capacity #

# With a pre-determined battery capacity in 2018, we assume an increasing capacity (as an effect of an increased density) based on a fixed weight assumption
# BEV_dynamic_capacity = (weighted_average_density[2018] * BEV_CAPACITY_CURRENT) / weighted_average_density
# PHEV_dynamic_capacity = (weighted_average_density[2018] * PHEV_CAPACITY_CURRENT) / weighted_average_density

# Define the V2G settings over time (assuming slow adoption to the maximum usable capacity)
# year_v2g_start = 2025-1 # -1 leads to usable capacity in 2025
# year_full_capacity = 2040
# v2g_usable = pd.Series(index=list(range(YEAR_START,YEAR_OUT+1)), name='years')    #fraction of the cars ready and willing to use v2g
# for year in list(range(YEAR_START,YEAR_OUT+1)):
#     if (year <= year_v2g_start):
#         v2g_usable[year] = 0
#     elif (year >= year_full_capacity):
#         v2g_usable[year] = 1
# v2g_usable = v2g_usable.interpolate()

# #total capacity per car connected to v2g (in %)
# max_capacity_BEV = v2g_usable * capacity_usable_BEV  
# max_capacity_PHEV = v2g_usable * capacity_usable_PHEV   

# #usable capacity per car connected to v2g (in kWh)
# usable_capacity_BEV = max_capacity_BEV.mul(BEV_dynamic_capacity[:YEAR_OUT])     
# usable_capacity_PHEV = max_capacity_PHEV.mul(PHEV_dynamic_capacity[:YEAR_OUT])  




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

# # 2nd level of the MultiIndex are materials (1971,'Aluminium')
# ev_battery_materials = stor_materials_interpol.copy()
# ev_battery_materials.index.names = ["year", "material"]  # assign names
# ev_battery_materials = ev_battery_materials.reset_index(level=["year", "material"])
# # Pivot so we get (technology, material) as columns, years as index
# ev_battery_materials = ev_battery_materials.melt(id_vars=["year", "material"], var_name="technology", value_name="value")
# ev_battery_materials = ev_battery_materials.pivot_table(index="year", columns=["technology", "material"], values="value")
# # Ensure proper MultiIndex column names
# ev_battery_materials.columns = pd.MultiIndex.from_tuples(ev_battery_materials.columns, names=["technology", "material"])
# xr_ev_battery_materials = dataset_to_array(pandas_to_xarray(ev_battery_materials, unit_mapping), *(["Cohort"], ["Type", "material"],))
# xr_ev_battery_materials = xr_ev_battery_materials.sel(Type=EV_BATTERIES)
# xr_energy_density_interpol = dataset_to_array(pandas_to_xarray(energy_density_interpol, unit_mapping), *(["Cohort"], ["Type"],))
# xr_ev_density_interpol = xr_energy_density_interpol.sel(Type=EV_BATTERIES)
# oth_storage_materialintens = xr_ev_battery_materials * xr_energy_density_interpol



# ##########################################################################################################
# %% Prep_data
# ##########################################################################################################

# Conversion table for all coordinates, to be removed/adapted after input tables are fixed.
conversion_table = {
    "battery_shares": (["Cohort"], ["battery"],),
    "battery_weights": (["Cohort"], ["Type", "SubType"], {"Type": ["Type", "SubType"]}),
    "ev_battery_materials": (["Cohort"], ["battery", "material"],),
    "ev_energy_density": (["Cohort"], ["battery"],),
    "vhc_fraction_v2g": (["Time"], ["Type"]), # TODO: check if Time coord is ok here
    "capacity_fraction_v2g": (["Type"]),
}
## "gcap_materials_interpol": (["Cohort"], ["Type", "SubType", "material"], {"Type": ["Type", "SubType"]})

results_dict = {
        "battery_shares": market_share_EVs, # 1
        "battery_weights": battery_weights, #2
        "ev_battery_materials": xr_ev_battery_materials, #3
        "ev_energy_density": xr_ev_density_interpol, #4
        "vhc_fraction_v2g": xr_vhc_fraction_v2g, #5
        "capacity_fraction_v2g": xr_capacity_fraction_v2g, #6
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