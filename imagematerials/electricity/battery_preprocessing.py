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
    normalize_selected_techs,
    calculate_storage_market_shares
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
# VARIANT = "climate_policy_overshoot"
VARIANT = "baseline"
scen_folder = SCEN + "_" + VARIANT
# path_base = Path().resolve() # TODO absolute path of file "preprocessing.py" ? current solution can differ depending on IDE used (?) 
path_current = Path().resolve()
path_base = path_current.parent.parent # base path of the project -> image-materials
print(f"Base path: {path_base}")

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

# prism.unit_registry.load_definitions(path_base / "imagematerials" / "units.txt")
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
storage_materials_data = pd.read_csv(path_external_data_standard / 'storage_materials_dynamic.csv', usecols=lambda col: col != "unit")  # wt% of total battery weight for various materials, total battery weight is given by the density file above

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
# vehicle_list = [f"{super_type} - {sub_type}" for super_type, sub_type in product(typical_modes, drive_trains)]
vehicle_list = ['Cars - BEV', 'Cars - FCV', 'Cars - HEV', 'Cars - ICE', 'Cars - PHEV',
       'Cars - Trolley', 'Heavy Freight Trucks - BEV',
       'Heavy Freight Trucks - FCV', 'Heavy Freight Trucks - HEV',
       'Heavy Freight Trucks - ICE', 'Heavy Freight Trucks - PHEV',
       'Heavy Freight Trucks - Trolley', 'Light Commercial Vehicles - BEV',
       'Light Commercial Vehicles - FCV', 'Light Commercial Vehicles - HEV',
       'Light Commercial Vehicles - ICE', 'Light Commercial Vehicles - PHEV',
       'Light Commercial Vehicles - Trolley', 'Medium Freight Trucks - BEV',
       'Medium Freight Trucks - FCV', 'Medium Freight Trucks - HEV',
       'Medium Freight Trucks - ICE', 'Medium Freight Trucks - PHEV',
       'Medium Freight Trucks - Trolley', 'Midi Buses - BEV',
       'Midi Buses - FCV', 'Midi Buses - HEV', 'Midi Buses - ICE',
       'Midi Buses - PHEV', 'Midi Buses - Trolley', 'Regular Buses - BEV',
       'Regular Buses - FCV', 'Regular Buses - HEV', 'Regular Buses - ICE',
       'Regular Buses - PHEV', 'Regular Buses - Trolley']
vehicle_list_non_ev = ['Cars - ICE', 'Cars - HEV', 'Cars - FCV', 'Cars - Trolley', 
        'Regular Buses - ICE', 'Regular Buses - HEV', 'Regular Buses - FCV', 
        'Regular Buses - Trolley', 'Midi Buses - ICE', 'Midi Buses - HEV',
       'Midi Buses - FCV', 'Midi Buses - Trolley', 'Heavy Freight Trucks - ICE',
       'Heavy Freight Trucks - HEV', 'Heavy Freight Trucks - FCV',
       'Heavy Freight Trucks - Trolley', 'Medium Freight Trucks - ICE',
       'Medium Freight Trucks - HEV', 'Medium Freight Trucks - FCV',
       'Medium Freight Trucks - Trolley', 'Light Commercial Vehicles - ICE',
       'Light Commercial Vehicles - HEV', 'Light Commercial Vehicles - FCV',
       'Light Commercial Vehicles - Trolley']
vhc_knowledge_graph = create_vehicle_graph()


# 1. Market Shares -----------------------------------------------------------------------------

# 1.1 storage costs
years = storage_costs.index.astype(int) #.astype(int) to convert years from strings to integers
techs = storage_costs.columns
data_array = storage_costs.to_numpy()
xr_storage_costs = xr.DataArray(
    data_array,
    dims=('Cohort', 'BatteryType'),
    coords={
        'Cohort': years,
        'BatteryType': techs
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
    dims=('Cohort', 'BatteryType'),
    coords={
        'Cohort': years,
        'BatteryType': techs
    },
    name='StorageCostsCorrection'
)
xr_costs_correction = prism.Q_(xr_costs_correction, "dimensionless")

# 1.3 storage costs longterm decline factor
techs = cost_decline_longterm_correction.index.rename(None)
data_array = cost_decline_longterm_correction.to_numpy()
xr_cost_decline_longterm_correction = xr.DataArray(
    data_array,
    dims=('BatteryType',),
    coords={
        'BatteryType': techs
    },
    name='StorageCostsDeclineLongterm'
)
xr_cost_decline_longterm_correction = prism.Q_(xr_cost_decline_longterm_correction, "fraction")

# 2. battery weigths -----------------------------------------------------------------------

xr_battery_weights = (
    battery_weights_data
    .rename_axis(index={"time": "Cohort", "type": "Drivetrain"})
    .to_xarray()                        # Convert the pandas DataFrame to xarray with dims: Cohort, Drivetrain, Vehicle
    .to_array("Vehicle")                # Move the DataFrame columns into an explicit xarray dimension
    .rename("BatteryWeights")
)
# Combine Vehicle and Drivetrain into a single dimension
xr_battery_weights = xr_battery_weights.stack(Type=("Vehicle", "Drivetrain"))
# combine Type string labels: ('Cars', 'BEV') -> 'Cars - BEV'
new_type = [f"{v} - {d}" for v, d in xr_battery_weights.indexes["Type"]]

xr_battery_weights = (
    xr_battery_weights
    .drop_vars(["Vehicle", "Drivetrain"]) # Remove now-redundant level coordinates
    .assign_coords(Type=new_type) # Replace the stacked MultiIndex with the combined string labels
)
xr_battery_weights = prism.Q_(xr_battery_weights, "kg")
# rebroadcast so that the Type coordinates have the correct sequence
xr_battery_weights = vhc_knowledge_graph.rebroadcast_xarray(xr_battery_weights, output_coords=vehicle_list, dim="Type")

# 3. material intensities -----------------------------------------------------------------------

# Reshape the DataFrame long format: Each material column (steel, aluminium, etc.) is converted into rows,
# with 'material' indicating the material type and 'MaterialFractions' the corresponding value.
storage_materials = storage_materials_data.melt(id_vars=['Cohort', 'Type'], var_name='material', value_name='MaterialFractions')
# Make sure Cohort and Type are treated as categorical (to preserve order)
storage_materials['Cohort'] = storage_materials['Cohort'].astype(int)
storage_materials['Type'] = storage_materials['Type'].astype(str)
# Create xarray DataArray directly
xr_storage_materials = storage_materials.set_index(['material', 'Cohort', 'Type'])['MaterialFractions'].to_xarray()
# Ensure correct dimension order
xr_storage_materials = xr_storage_materials.transpose('material', 'Cohort', 'Type')
xr_storage_materials = xr_storage_materials.rename({'Type': 'BatteryType'})
xr_storage_materials = prism.Q_(xr_storage_materials, "fraction")
xr_battery_materials = xr_storage_materials.sel(BatteryType=EV_BATTERIES)


# 4. energy density -----------------------------------------------------------------------------
years = energy_density.index.astype(int)
energy_density.columns.name = None # remove header "kg/kWh" to avoid issues
techs = energy_density.columns
data_array = energy_density.to_numpy()
xr_energy_density = xr.DataArray(
    data_array,
    dims=('Cohort', 'BatteryType'),
    coords={
        'Cohort': years,
        'BatteryType': techs
    },
    name='EnergyDensity'
)
xr_energy_density = prism.Q_(xr_energy_density, "kg/kWh")
xr_energy_density = xr_energy_density.sel(BatteryType=EV_BATTERIES)


# 5. EV fraction available for V2G --------------------------------------------------------------------
# For this variable first the interpolations are done and then the conversion to xarray DataArray
x = ev_fraction_v2g_data.reindex(range(ev_fraction_v2g_data.index[0],ev_fraction_v2g_data.index[-1]+1)).interpolate(method='linear')
y = logistic(x, L=x.iloc[-1].values)
# y = quadratic(x)
ev_fraction_v2g = ev_fraction_v2g_data.reindex(range(YEAR_FIRST_GRID,YEAR_OUT+1)).interpolate(method='linear') # create dataframe with full index; values before first data points will be Nans, between data points interpolated linearly, after last data point will be last known value
ev_fraction_v2g.loc[:ev_fraction_v2g_data.index[0]] = 0 # set values before first data point to 0
ev_fraction_v2g.loc[ev_fraction_v2g_data.index[0]:ev_fraction_v2g_data.index[1]] = y # set values between (originally) first and last data point to quadratic/logistic interpolation

# vhc_fraction_v2g = ev_fraction_v2g.copy()
# Create an empty expanded df
# vhc_fraction_v2g = pd.DataFrame(index=ev_fraction_v2g.index)
# # transfer the values for the relevant vehicle types (for now: Cars - BEV and Cars - PHEV)
# for item in vehicle_list: # TODO: is this even necessary? If not all coordinates are defined, maybe they are just ignored in later calculations?
#     if item in ev_fraction_v2g.columns:
#         vhc_fraction_v2g[item] = ev_fraction_v2g[item]
#     else:
#         vhc_fraction_v2g[item] = 0 # other drive trains are not V2G capable
# Build xarray DataArray
years = ev_fraction_v2g.index.astype(int).rename(None)
techs = ev_fraction_v2g.columns
data_array = ev_fraction_v2g.to_numpy()
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

storage_market_share = calculate_storage_market_shares(
    xr_storage_costs,
    xr_costs_correction,
    xr_cost_decline_longterm_correction,
    mnlogit_param=-0.2,
    t_start_interpolation=YEAR_START,
    t_end_interpolation=2050
)


# # market shares ---
# # use the storage price development in the logit model to get market shares
# storage_market_share = MNLogit(xr_storage_costs_cor, -0.2) #assumes input of an ordered dataframe with rows as years and columns as technologies, values as prices. Logitpar is the calibrated Logit parameter (usually a nagetive number between 0 and 1)
# fix the market share of storage technologies before YEAR_START and after 2050
storage_market_share_interp = interpolate_xr(storage_market_share, YEAR_FIRST_GRID, YEAR_OUT)
# As not all storage technologies are suitable for EV & mobile applications, select only those technologies.
# normalize the selection of EV battery technologies, so that total market share is 1 again (taking the relative share in the selected battery techs)
market_share_EVs = normalize_selected_techs(storage_market_share_interp, EV_BATTERIES, dim_type="BatteryType") # TODO: this should be done differently as market shares of EV batteries probably differ from their market shares in total storage market


# TEST TEST TEST ==============================================
# Calculations test ------
test_allstor = storage_market_share_interp.copy()
storage_subtypes_categories = ["mechanical storage", "lithium ion battery", "lithium metal battery", "flow battery", "molten salt battery", "lead acid and nickel battery", "fuel cell"]
# test_allstor = elc_knowledge_graph.aggregate_sum(test_allstor, storage_subtypes_categories, dim="BatteryType")
tech1 = ['LMO', 'NMC', 'NCA', 'LFP'] #'lithium ion battery'
tech2 = ['Lithium Sulfur', 'Lithium Ceramic', 'Lithium-air'] #'lithium metal battery'
print(test_allstor.sel(Cohort=2100, BatteryType = tech1).sum(dim="BatteryType") / test_allstor.sel(Cohort=2100, BatteryType = tech2).sum(dim="BatteryType"))

test_evbat = market_share_EVs.copy()
storage_subtypes_categories = ["lithium ion battery", "lithium metal battery", "lead acid and nickel battery"]
# test_evbat = elc_knowledge_graph.aggregate_sum(test_evbat, storage_subtypes_categories, dim="BatteryType")
tech1 = ['LMO', 'NMC', 'NCA', 'LFP', 'LTO'] #'lithium ion battery'
tech2 = ['Lithium Sulfur', 'Lithium Ceramic', 'Lithium-air'] #'lithium metal battery'
print(test_evbat.sel(Cohort=2100, BatteryType = tech1).sum(dim="BatteryType") / test_evbat.sel(Cohort=2100, BatteryType = tech2).sum(dim="BatteryType"))

#------
test = storage_market_share_interp.copy()

fig, ax = plt.subplots(figsize=(8, 6))
battery_types = test.BatteryType.values
ax.stackplot(
    test.Cohort,
    [test.sel(BatteryType=bt) for bt in battery_types],
    labels=battery_types
)
ax.set_xlim(1990,2100)
ax.set_xlabel("Year")
ax.set_ylabel("Share of total battery stock")
ax.set_ylim(0, 1)
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
ax.set_title(f"Normalized battery stock by battery type (based on stock in Wh; SSP2_baseline)")
plt.show()

colormap_storage_subtypes_categories = {
    "mechanical storage":           "#4387F6", 
    "lithium ion battery":          "#F0BA3C", 
    "lithium metal battery":        "#EDCD76", 
    "flow battery":                 "#5C55D6", 
    "molten salt battery":          "#31942A", 
    "lead acid and nickel battery": "#A72C41", 
    "fuel cell":                    "#6A26D1"}

test2 = storage_market_share_interp.copy()

storage_subtypes_categories = ["mechanical storage", "lithium ion battery", "lithium metal battery", "flow battery", "molten salt battery", "lead acid and nickel battery", "fuel cell"]
elc_knowledge_graph = create_electricity_graph()
test2 = elc_knowledge_graph.aggregate_sum(test2, storage_subtypes_categories, dim="BatteryType")

fig, ax = plt.subplots(figsize=(8, 6))
battery_types = test2.BatteryType.values
ax.stackplot(
    test2.Cohort,
    [test2.sel(BatteryType=bt) for bt in battery_types],
    labels=battery_types,
    colors=[colormap_storage_subtypes_categories[bt] for bt in battery_types],
)
ax.set_xlim(1990,2100)
ax.set_xlabel("Year")
ax.set_ylabel("Share of total battery stock")
ax.set_ylim(0, 1)
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
ax.set_title(f"Normalized battery stock by battery type (based on stock in Wh; SSP2_baseline)")
plt.show()

# -----
test = market_share_EVs.copy()
fig, ax = plt.subplots(figsize=(8, 6))
battery_types = test.BatteryType.values
ax.stackplot(
    test.Cohort,
    [test.sel(BatteryType=bt) for bt in battery_types],
    labels=battery_types
)
ax.set_xlim(1990,2100)
ax.set_xlabel("Year")
ax.set_ylabel("Share of battery inflow")
ax.set_ylim(0, 1)
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
ax.set_title(f"Normalized EV battery inflow market shares (based on inflow in count, SSP2_baseline)")
plt.show()

#----
test2 = market_share_EVs.copy()
storage_subtypes_categories = ["lithium ion battery", "lithium metal battery", "lead acid and nickel battery"]
elc_knowledge_graph = create_electricity_graph()
test2 = elc_knowledge_graph.aggregate_sum(test2, storage_subtypes_categories, dim="BatteryType")

fig, ax = plt.subplots(figsize=(8, 6))
battery_types = test2.BatteryType.values
ax.stackplot(
    test2.Cohort,
    [test2.sel(BatteryType=bt) for bt in battery_types],
    labels=battery_types,
    colors=[colormap_storage_subtypes_categories[bt] for bt in battery_types],
)
ax.set_xlim(1990,2100)
ax.set_xlabel("Year")
ax.set_ylabel("Share of battery inflow")
ax.set_ylim(0, 1)
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
ax.set_title(f"Normalized EV battery inflow market shares (based on inflow in count, SSP2_baseline)")
plt.show()
# END TEST TEST ==============================================


market_share_EVs = market_share_EVs.expand_dims(Type=vehicle_list).copy() # add vehicle type dimension
market_share_EVs.loc[dict(Type=vehicle_list_non_ev)] = 0 # set non-EV vehicle types to zero

# 2. Battery Weights ----------------------------------------------------------------
xr_battery_weights_interp = interpolate_xr(xr_battery_weights, YEAR_FIRST_GRID, YEAR_OUT)

# 3. Material Intensities ----------------------------------------------------------------
xr_battery_materials_interp = interpolate_xr(xr_battery_materials, YEAR_FIRST_GRID, YEAR_OUT)

#  4. Energy Density ----------------------------------------------------------------
xr_energy_density_interp = interpolate_xr(xr_energy_density, YEAR_FIRST_GRID, YEAR_OUT)



#%% ##########################################################################################################
da = xr_battery_weights_interp

dim_type = "Type" 
# dim_type = "BatteryType" 
dim_time = "Time"
dim_time = "Cohort"

fig, ax = plt.subplots(figsize=(8, 4))
for t in ["Cars - BEV", "Cars - PHEV"]: #da[dim_type].values  , "Cars - PHEV"
    ax.plot(da[dim_time], da.sel({dim_type: t}), label=t)
# ax.plot(da.Time, da.sel(Type="Cars - BEV"), label="Cars - BEV")
ax.set_xlabel("Year")
ax.set_ylabel(str(da.name) + " " + str(prism.U_(da)))
# ax.set_ylim(-0.05, 1.05)
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
ax.set_title(str(da.name))

plt.tight_layout()
plt.show()






###########################################################################################################
#%% Prep_data File #


# bring preprocessing data into a generic format for the model
prep_data = {}
prep_data["shares"] = market_share_EVs
prep_data["battery_weights"] = xr_battery_weights_interp
prep_data["material_intensities"] = xr_battery_materials_interp
prep_data["energy_density"] = xr_energy_density_interp
prep_data["vhc_fraction_v2g"] = xr_vhc_fraction_v2g
prep_data["capacity_fraction_v2g"] = xr_capacity_fraction_v2g
prep_data["knowledge_graph"] = create_electricity_graph()

# prep_data["set_unit_flexible"] = prism.U_(prep_data["stocks"]) # prism.U_ gives the unit back


# time_start = 1970
# complete_timeline = prism.Timeline(time_start, YEAR_END, 1)
# simulation_timeline = prism.Timeline(YEAR_START, YEAR_END, 1) #1970

# sec_electr_gen = Sector("ev_batteries", prep_data)





#%% old 


# --battery weights--
# df = battery_weights_data.copy()
# # 1. Extract values
# times = sorted(battery_weights_data.index.get_level_values(0).unique())
# drivetrains = sorted(battery_weights_data.index.get_level_values(1).unique())
# vehicles = battery_weights_data.columns
# # 2. Create combined Type labels: 'Vehicle - Drivetrain'
# types = [f"{v} - {d}" for v in vehicles for d in drivetrains]
# # 3. Convert DataFrame to NumPy array
# df_sorted = battery_weights_data.sort_index(level=[0,1]) # First, ensure row order: time × drivetrain
# data = df_sorted.to_numpy()  # Flatten rows and columns into 2D array: shape: (len(times)*len(drivetrains), len(vehicles))
# # 4. Reorder / reshape to (time, Type)
# data_2d = np.hstack([data[i::len(drivetrains), :] for i in range(len(drivetrains))]) # For each time, concatenate all drivetrains in order
# # 5. Build xarray
# xr_battery_weights = xr.DataArray(
#     data_2d,
#     dims=("Cohort", "Type"),
#     coords={
#         "Cohort": times,
#         "Type": types
#     },
#     name="BatteryWeights"
# )

# --storage_materials --
# storage_materials.columns = storage_materials_data.columns.rename([None, None]) # remove column MultiIndex name "g/MW" as it causes issues when converting to xarray
# years = sorted(storage_materials.columns.get_level_values(0).unique())
# techs = storage_materials.columns.get_level_values(1).unique()
# materials = storage_materials.index
# # Convert to 3D array: (Material, Year, Tech)
# data_array = storage_materials.to_numpy().reshape(len(materials), len(years), len(techs))
# # Build xarray DataArray
# xr_storage_materials = xr.DataArray(
#     data_array,
#     dims=('material', 'Cohort', 'BatteryType'),
#     coords={
#         'material': materials,
#         'Cohort': years,
#         'BatteryType': techs
#     },
#     name='MaterialFractions'
# )