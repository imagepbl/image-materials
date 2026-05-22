#%% Import modules and constants
import pandas as pd
import numpy as np
from pathlib import Path
import pint
import xarray as xr
from importlib.resources import files
import matplotlib.pyplot as plt

import prism
from imagematerials.read_mym import read_mym_df
from imagematerials.util import dataset_to_array, pandas_to_xarray
from imagematerials.concepts import create_electricity_graph, create_region_graph
from imagematerials.electricity.utils import (
    MNLogit, 
    stock_tail, 
    create_prep_data, 
    logistic, 
    quadratic,
    interpolate_xr, 
    add_historic_stock, 
    normalize_selected_techs,
    calculate_storage_market_shares
)

from imagematerials.constants import IMAGE_REGIONS

from imagematerials.electricity.constants import (
    STANDARD_SCEN_EXTERNAL_DATA,
    YEAR_FIRST_GRID,
    SENS_ANALYSIS,
    unit_mapping
)

# for ttesting, remove later:
year_start = 1971
year_out = 2100
END_YEAR = 2100
INTERMEDIATE_YEAR = 2080

units_file = files("imagematerials") / "units.txt"
prism.unit_registry.load_definitions(units_file)


path_current = Path().resolve()
path_base = path_current.parent.parent #.parent # base path of the project -> image-materials
path_base = Path(path_base, "data", "raw")

scenario = "SSP2_baseline"
path_external_data_standard = Path(path_base, "electricity", "standard_data")
path_external_data_scenario = Path(path_base, "electricity", scenario) #test
path_image = Path(path_base, "image", scenario)

# test if path_external_data_scenario exists and if not set to standard scenario
if not path_external_data_scenario.exists():
    path_external_data_scenario = Path(path_base, "electricity", STANDARD_SCEN_EXTERNAL_DATA)

idx = pd.IndexSlice   


def plot_market_shares(shares, name, save=False):
    fig, ax = plt.subplots()
    colors = plt.get_cmap('tab20').colors[:17]
    for i, col in enumerate(shares.columns):
        ax.plot(shares.index, shares[col], color=colors[i], label=col)

    ax.set_xlim(1971,2070)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.grid(alpha=0.3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.set_xlabel("Time", fontsize=14)
    ax.set_ylabel(f"shares", fontsize=14)
    ax.set_title(name, fontweight="bold")
    plt.show()


###########################################################################################################
#%% Data
###########################################################################################################
#----------------------------------------------------------------------------------------------------------


# 1. External Data ======================================================================================== 

# # read in the storage costs according to IRENA storage report & other sources in the SI
# storage_costs = pd.read_csv(path_external_data_standard / 'storage_cost.csv', index_col=0).transpose()

# # read in the assumed malus & bonus of storage costs (malus for advanced technologies, still under development; bonus for batteries currently used in EVs, we assume that a large volume of used EV batteries will be available and used for dedicated electricity storage, thus lowering costs), only the bonus remains by 2030
# storage_malus = pd.read_csv(path_external_data_standard / 'storage_malus.csv', index_col=0).transpose()

# #read in the assumptions on the long-term price decline after 2050. Prices are in $ct / kWh electricity cycled (the fraction of the annual growth rate (determined based on 2018-2030) that will be applied after 2030, ranging from 0.25 to 1 - 0.25 means the price decline is not expected to continue strongly, while 1 means that the same (2018-2030) annual price decline is also applied between 2030 and 2050)
# storage_ltdecline = pd.Series(pd.read_csv(path_external_data_standard / 'storage_ltdecline.csv',index_col=0,  header=None).transpose().iloc[0])

# #read in the energy density assumptions (kg/kWh storage capacity - mass required to store one unit of energy — more mass per energy = worse performance)
# storage_density = pd.read_csv(path_external_data_standard / 'storage_density_kg_per_kwh.csv',index_col=0).transpose()

# # material compositions (storage) in wt%
# storage_materials = pd.read_csv(path_external_data_standard / 'storage_materials_dynamic.csv',index_col=[0,1],usecols=lambda col: col != "unit").transpose()  # wt% of total battery weight for various materials, total battery weight is given by the density file above
material_intensities = pd.read_csv(path_external_data_standard / 'storage_and_EV_batteries_material_intensities.csv', usecols=["Time", "sub_technology", "material", "value"])# .transpose()
market_shares = pd.read_csv(path_external_data_standard / 'storage_stationary_market_shares.csv', usecols=["Time", "sub_technology", "value"]) #index_col=[0,1],usecols=lambda col: col != "unit"

TECH_STATIONARY_STORAGE = ['LFP', 'NMC333', 'NMC532', 'NMC622', 'NMC811', 'NMC955', 'Na-ion', 
                           'flow-ZnBr', 'flow-vanadium', 'lead-acid']

# Build the xarray DataArray
# Pivot to get a multi-index, then convert to xarray
da_material_intensities = (
    material_intensities.set_index(["Time", "sub_technology", "material"])["value"]
    .to_xarray()
    .rename({"sub_technology": "Type"})
)
da_material_intensities = prism.Q_(da_material_intensities, "kg/kWh")

da_market_shares = (
    market_shares.set_index(["Time", "sub_technology"])["value"]
    .to_xarray()
    .rename({"sub_technology": "Type"})
)
da_market_shares = prism.Q_(da_market_shares, "dimensionless")

year_start = YEAR_FIRST_GRID
year_end = 2100
# if material intensities only have one value (e.g., for 2020), then we can simply broadcast this 
# value across all years, otherwise we need to interpolate between the given values and then 
# extrapolate the interpolated values to the full time range
if da_material_intensities.sizes["Time"] == 1:
    # Broadcast the single value across all new time coordinates
    da_material_intensities = da_material_intensities.reindex(Time=np.arange(year_start, year_end), method="nearest")
else:
    da_material_intensities = interpolate_xr(da_material_intensities, year_start, year_end)

da_market_shares = interpolate_xr(da_market_shares, year_start, year_end)

fig, ax = plt.subplots()
for t in da_market_shares.Type.values:
    ax.plot(da_market_shares.Time, da_market_shares.sel(Type=t), label=t)
ax.set_xlim(1980, 2070)
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
plt.title("Market shares of stationary storage technologies", fontweight="bold")

fig, ax = plt.subplots()
for t in da_material_intensities.material.values:
    ax.plot(da_material_intensities.Time, da_material_intensities.sel(material=t, Type="NMC333"), label=t)
ax.set_xlim(1980, 2070)
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
plt.title("Material intensities of stationary storage technologies", fontweight="bold")


# 2. IMAGE/TIMER files ====================================================================================

# read TIMER installed storage capacity (MWh, reservoir)
storage = read_mym_df(Path(path_image, "EnergyServices", "StorResTot.out"))   #storage capacity in MWh (reservoir, so energy capacity, not power capacity, the latter is used later on in the pumped hydro storage calculations)
# storage = read_mym_df(climate_policy_config["config_file_path"] / climate_policy_config["data_files"]['StorResTot'])

#storage capacity in MW (power capacity), to compare it to Pumped hydro storage projections (also given in MW, power capacity)
storage_power = read_mym_df(Path(path_image, "EnergyServices", "StorCapTot.out"))
# storage_power = read_mym_df(climate_policy_config["config_file_path"] / climate_policy_config["data_files"]['StorCapTot'])


####################################################################################################
#%% To extract Sebastiaans material intensities in kg/kWh 
 
# Read both CSVs
materials_df = pd.read_csv(path_external_data_standard / 'storage_materials_dynamic.csv', usecols=lambda col: col != "unit") #pd.read_csv(path_external_data_standard / 'storage_materials_dynamic.csv')
energy_density = pd.read_csv(path_external_data_standard / "storage_density_kg_per_kwh.csv",index_col=0).transpose() #pd.read_csv(path_external_data_standard / 'storage_density_kg_per_kwh.csv')


years = energy_density.index.astype(int)
energy_density.columns.name = None # remove header "kg/kWh" to avoid issues
techs = energy_density.columns
data_array = energy_density.to_numpy()
xr_energy_density = xr.DataArray(
    data_array,
    dims=("Cohort", "BatteryType"),
    coords={
        "Cohort": years,
        "BatteryType": techs
    },
    name="EnergyDensity"
)
xr_energy_density = prism.Q_(xr_energy_density, "kg/kWh")

storage_materials = materials_df.melt(id_vars=['Cohort', 'Type'], var_name='material', value_name='MaterialFractions')
# Make sure Cohort and Type are treated as categorical (to preserve order)
storage_materials['Cohort'] = storage_materials['Cohort'].astype(int)
storage_materials['Type'] = storage_materials['Type'].astype(str)
# Create xarray DataArray directly
xr_storage_materials = storage_materials.set_index(['material', 'Cohort', 'Type'])['MaterialFractions'].to_xarray()
# Ensure correct dimension order
xr_storage_materials = xr_storage_materials.transpose('material', 'Cohort', 'Type')
xr_storage_materials = xr_storage_materials.rename({'Type': 'BatteryType'})
xr_storage_materials = prism.Q_(xr_storage_materials, "fraction")

xr_materials_interp = interpolate_xr(xr_storage_materials, 2000, 2050)
xr_energy_density_interp = interpolate_xr(xr_energy_density, 2000, 2050)

xr_material_intensity_interp = xr_materials_interp * xr_energy_density_interp

print(xr_material_intensity_interp.sel(Cohort=2020).to_pandas())
xr_material_intensity_interp.sel(Cohort=2020).to_pandas().round(3).to_csv(path_external_data_standard / "storage_materials_kg_per_kwh.csv", index=True)


# ----------------------------------------------------------------------------------------------------------
# ##########################################################################################################
# %% prepare storage interpol
# ##########################################################################################################
# ----------------------------------------------------------------------------------------------------------


storage = storage.iloc[:, :26]    # drop global total column and empty (27) column

# J: in high storage scenario the storage demand linearly increases between 2021 and 2050 compared to its original value until it is double by 2050, and then remains constant
if SENS_ANALYSIS == 'high_stor':
    storage_multiplier = storage
    for year in range(2021,2051):
        storage_multiplier.loc[year] = storage.loc[year] * (1 + (1/30*(year-2020)))
    for year in range(2051,year_end+1):
        storage_multiplier.loc[year] = storage.loc[year] * 2


# turn index to integer for sorting during the next step
storage_costs.index = storage_costs.index.astype('int64')
storage_malus.index = storage_malus.index.astype('int64')

# to interpolate between 2018 and 2030, first create empty rows (NaN values) 
storage_start = storage_costs.first_valid_index()
storage_end =   storage_costs.last_valid_index()
for i in range(storage_start+1,storage_end):
    storage_costs = pd.concat([storage_costs, pd.DataFrame(index=[i])]) #, ignore_index=True
    storage_malus = pd.concat([storage_malus, pd.DataFrame(index=[i])])         # mind: the malus needs to be defined for the same years as the cost indications
    
# then, do the actual interpolation on the sorted dataframes                                                    
storage_costs_interpol = storage_costs.sort_index(axis=0).interpolate(axis=0)#.index.astype('int64')
storage_malus_interpol = storage_malus.sort_index(axis=0).interpolate(axis=0)



# ----------------------------------------------------------------------------------------------------------
# ##########################################################################################################
# %% prepare storage interpol
# ##########################################################################################################
# ----------------------------------------------------------------------------------------------------------


# Determine MARKET SHARE of the storage capacity using a multi-nomial logit function

# storage costs ---
# determine the annual % decline of the costs based on the 2018-2030 data (original, before applying the malus)
decline = ((storage_costs_interpol.loc[storage_start,:]-storage_costs_interpol.loc[storage_end,:])/(storage_end-storage_start))/storage_costs_interpol.loc[storage_start,:]
decline_used = decline*storage_ltdecline #TODO: what is happening here? Why?
# storage_ltdecline is a single number and should describe the long-term decline after 2030 relative to the 2018-2030 decline

storage_costs_new = storage_costs_interpol.copy() * storage_malus_interpol
# calculate the development from 2030 to 2050 (using annual price decline)
for year in range(storage_end+1,2050+1):
    # print(year)
    # storage_costs_new = storage_costs_new.append(pd.Series(storage_costs_new.loc[storage_costs_new.last_valid_index()]*(1-decline_used), name=year))
    # storage_costs_new = pd.concat([storage_costs_new, pd.Series(storage_costs_new.loc[storage_costs_new.last_valid_index()] * (1 - decline_used), name=year)])
    row = pd.DataFrame([storage_costs_new.loc[storage_costs_new.last_valid_index()] * (1 - decline_used)])
    storage_costs_new.loc[year] = row.iloc[0]
# for historic price development, assume 2x AVERAGE annual price decline on all technologies, except lead-acid (so that lead-acid gets a relative price advantage from 1970-2018)
for year in reversed(range(year_start,storage_start)):
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
for year in range(2050+1,year_out+1):
    # storage_market_share = storage_market_share.append(pd.Series(storage_market_share.loc[storage_market_share.last_valid_index()], name=year))
    row = pd.DataFrame([storage_market_share.loc[storage_market_share.last_valid_index()]])
    storage_market_share.loc[year] = row.iloc[0]
# fix the market share of storage technologies before YEAR_START
for year in range(YEAR_FIRST_GRID,year_start):
    # storage_market_share = storage_market_share.append(pd.Series(storage_market_share.loc[storage_market_share.last_valid_index()], name=year))
    row = pd.DataFrame([storage_market_share.loc[storage_market_share.first_valid_index()]])
    storage_market_share.loc[year] = row.iloc[0]
    
storage_market_share = storage_market_share.sort_index(axis=0)

shares_old = storage_market_share.copy()

plot_market_shares(shares_old, "shares old")


# ----------------------------------------------------------------------------------------------------------
# ##########################################################################################################
# %% Using costs directly
# ##########################################################################################################
# ----------------------------------------------------------------------------------------------------------

storage_costs_new = storage_costs_interpol.copy()

for year in range(storage_end+1,2050+1):
    # print(year)
    # storage_costs_new = storage_costs_new.append(pd.Series(storage_costs_new.loc[storage_costs_new.last_valid_index()]*(1-decline_used), name=year))
    # storage_costs_new = pd.concat([storage_costs_new, pd.Series(storage_costs_new.loc[storage_costs_new.last_valid_index()] * (1 - decline_used), name=year)])
    row = pd.DataFrame([storage_costs_new.loc[storage_costs_new.last_valid_index()]])
    storage_costs_new.loc[year] = row.iloc[0]
# for historic price development, assume 2x AVERAGE annual price decline on all technologies, except lead-acid (so that lead-acid gets a relative price advantage from 1970-2018)
for year in reversed(range(year_start,storage_start)):
    # storage_costs_new = storage_costs_new.append(pd.Series(storage_costs_new.loc[storage_costs_new.first_valid_index()]*(1+(2*decline_used.mean())), name=year)).sort_index(axis=0)
    row = pd.DataFrame([storage_costs_new.loc[storage_costs_new.first_valid_index()]])
    storage_costs_new.loc[year] = row.iloc[0]
    storage_costs_new.sort_index(axis=0, inplace=True) 

storage_costs_new.sort_index(axis=0, inplace=True) 

# market shares ---
# use the storage price development in the logit model to get market shares
storage_market_share = MNLogit(storage_costs_new, -0.2) #assumes input of an ordered dataframe with rows as years and columns as technologies, values as prices. Logitpar is the calibrated Logit parameter (usually a nagetive number between 0 and 1)

# fix the market share of storage technologies after 2050
for year in range(2050+1,year_out+1):
    # storage_market_share = storage_market_share.append(pd.Series(storage_market_share.loc[storage_market_share.last_valid_index()], name=year))
    row = pd.DataFrame([storage_market_share.loc[storage_market_share.last_valid_index()]])
    storage_market_share.loc[year] = row.iloc[0]
# fix the market share of storage technologies before YEAR_START
for year in range(YEAR_FIRST_GRID,year_start):
    # storage_market_share = storage_market_share.append(pd.Series(storage_market_share.loc[storage_market_share.last_valid_index()], name=year))
    row = pd.DataFrame([storage_market_share.loc[storage_market_share.first_valid_index()]])
    storage_market_share.loc[year] = row.iloc[0]
    
storage_market_share = storage_market_share.sort_index(axis=0)

shares_no_changes = storage_market_share.copy()

plot_market_shares(shares_no_changes, "shares no changes")