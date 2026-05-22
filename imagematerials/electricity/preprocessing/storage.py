#%% Import modules and constants
import pandas as pd
import numpy as np
from pathlib import Path
import pint
import xarray as xr
from importlib.resources import files

import prism
from imagematerials.read_mym import read_mym_df
from imagematerials.util import dataset_to_array, pandas_to_xarray, convert_lifetime
from imagematerials.concepts import create_electricity_graph, create_image_region_graph, create_region_graph
from imagematerials.electricity.utils import (
    MNLogit, 
    stock_tail, 
    create_prep_data,
    derive_phs_installed_capacity,
    derive_btm_installed_capacity,
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
    STD_LIFETIMES_ELECTR,
    EPG_TECHNOLOGIES,
    TECH_STATIONARY_STORAGE,
    TECH_STATIONARY_STORAGE_ALL,
    unit_mapping,
    mean_discharge_duration,
    factor_phs_growth_rel_demand,
    flag_phs_scenario,
    ratio_btm_to_solar
)

# for ttesting, remove later:
END_YEAR = 2100
INTERMEDIATE_YEAR = 2080

# delete again ------------------
scenario = "SSP2_baseline"
path_current = Path().resolve()
path_base = path_current.parent.parent#.parent # base path of the project -> image-materials
path_image_output = Path(path_base, "data", "raw", "image", scenario, "EnergyServices")
path_base = Path(path_base, "data", "raw")
year_start = 1971
year_end = 2100

# TECH_STATIONARY_STORAGE = ['LFP', 'NMC333', 'NMC532', 'NMC622', 'NMC811', 'NMC955', 'Na-ion', 
#                            'flow-ZnBr', 'flow-vanadium', 'lead-acid']
# ----------------------------------


#############################################################################################################
#############################################################################################################

# def get_preprocessing_data_stor(path_base: str, climate_policy_config: dict, circular_economy_config: dict, scenario: str, year_start: int, year_end: int, year_out: int):
"""Preprocess electricity storage data and creates "prep_data" suitable for stock modelling using IMAGE-Materials.

It does so for the two sub-sectors:
1) Pumped Hydro Storage (PHS)
2) Other Electricity Storage Technologies (e.g., batteries, compressed air, flywheels)
(in the future also vehicle batteries will be added here)


The function performs the following steps:
1. Reads input data on storage stocks scenario projections (IMAGE/TIMER) and storage technologies, including costs, lifetimes, energy densities, and material compositions (different sources, e.g. IRENA, IHA)
2. Interpolates data over time to fill missing years.
3. Calculates market shares of different storage technologies using a multinomial logit model.
4. Infers pumped hydro storage demand from IHA projections and IMAGE hydro power projections.
5. Computes shares of total storage capacity from PHS, (electric vehicles - not yet), and other storage systems.

Parameters
----------
path_base : str
    Base directory of the project where the data structure is located (image-materials/data/raw/).
scenario : str
    Scenario name (e.g., "SSP2_baseline").
climate_policy_config : dict
    Dictionary created from a scenario-specific config.toml file.
    Contains all TOML entries ("data_files" mapping) plus a path:
    - "config_file_path": pathlib.Path to the scenario directory (e.g. SSP2_baseline) containing config.toml and all referenced
    output subfolders. 
    Used to construct full paths to scenario output files, e.g. read_mym_df(climate_policy_config["config_file_path"]/climate_policy_config["data_files"]["variable_x"]).
year_start : int
    First simulation year for data processing (e.g., 1971).
year_end : int
    Last year for data interpolation (used for intermediate computations).
year_out : int
    Final output year for the results (e.g., 2100).

Returns
-------
prep_data_phs: 
    Prepared dataset for pumped hydro storage, including:
        * "stocks": Time series of storage capacity (MWh)
        * "material_intensities": Material intensities (kg/MWh)
        * "lifetimes": Lifetime mean and standard deviation
        * "set_unit_flexible": Unit mapping for Prism integration
prep_data_oth_storage: 
    Prepared dataset for other storage technologies, including:
        * "stocks": Time series of storage capacity (MWh)
        * "material_intensities": Material intensities (kg/kWh)
        * "lifetimes": Lifetime mean and standard deviation
        * "shares": Market shares of storage technologies
        * "set_unit_flexible": Unit mapping for Prism integration


Notes
-----
- Assumes that the IMAGE/TIMER model outputs and other technology reference data are available in the expected
    directory structure under `path_base/data/raw/`.
- Sensitivity variants (e.g., `high_stor`) affect storage demand scaling and pumped hydro projections. # TODO: move to parameters?

"""

path_external_data_standard = Path(path_base, "electricity", "standard_data")
path_external_data_scenario = Path(path_base, "electricity", scenario) #test

# test if path_external_data_scenario exists and if not set to standard scenario
if not path_external_data_scenario.exists():
    path_external_data_scenario = Path(path_base, "electricity", STANDARD_SCEN_EXTERNAL_DATA)

# print(f"Path to image output: {path_image_output}")
# assert path_image_output.is_dir()
# assert path_external_data_standard.is_dir()
# assert path_external_data_scenario.is_dir()

idx = pd.IndexSlice   

#----------------------------------------------------------------------------------------------------------
###########################################################################################################
#%%% 2.1) Read in files
###########################################################################################################
#----------------------------------------------------------------------------------------------------------


# 1. External Data ======================================================================================== 

#read in the lifetime of storage technologies (in yrs). The lifetime is assumed to be 1.5* the number of cycles divided by the number of days in a year (assuming diurnal use, and 50% extra cycles before replacement, representing continued use below 80% remaining capacity) OR the maximum lifetime in years, which-ever comes first 
lifetimes = pd.read_csv(path_external_data_standard / 'storage_and_EV_batteries_lifetimes.csv', usecols=["Time", "sub_technology", "value"])

# material intensities in kg/kWh
material_intensities = pd.read_csv(path_external_data_standard / 'storage_and_EV_batteries_material_intensities.csv', usecols=["Time", "sub_technology", "material", "value"])# .transpose()

# market shares in % of total storage capacity (excluding pumped hydropower storage; values: 0-1)
market_shares = pd.read_csv(path_external_data_standard / 'storage_stationary_market_shares.csv', usecols=["Time", "sub_technology", "value"]) #index_col=[0,1],usecols=lambda col: col != "unit"


# Data for Pumped Hydropower Storage (PHS) ---------------------------------------------------------
# Hydro-dam power capacity (also MW) within 5 regions reported by the IHA (international Hydropwer Association)
# phs_projections = pd.read_csv(path_external_data_standard / 'PHS.csv', index_col='t')   # pumped hydro storage capacity (MW)
# Dataset 1: IHA data set of individual PHS sites, with commissioning year, capacity, energy stored,
# operating status, and hydro type
df_data1 =  pd.read_csv(Path(path_external_data_standard,"phs_data1_iha_capacity_by_individual_facility.csv"), 
                    usecols=["Operational Status", "Country", "Commisioning Year", "Hydro Type", 
                            "Generating Capacity", "Energy stored (GWh)"])

# Dataset 2: IHA World Hydropower Outlook dataset of aggregated PHS power capacity (MW) by country for 3 years (2014, 2019, 2024) 
#           + IEA global estimates
df_data2 =  pd.read_csv(Path(path_external_data_standard,"phs_data2_iha_world_hydropower_outlook_historic_stocks.csv"),
                        usecols=["Time", "unit", "Region", "value"])

# Dataset 3: IHA data on PHS projects under contruction, planned and announced per IHA region
df_data3 =  pd.read_csv(Path(path_external_data_standard,"phs_data3_iha_future_planned_capacity_mw_per_iha_region.csv"),
                        usecols=["status", "unit", "Region", "value"])

# Manual adjustments to future shares based on literature insights, but authors estimation (unit: shares)
df_shares_adjustment_2030 =  pd.read_csv(Path(path_external_data_standard,"phs_regional_shares_per_iha_region_in_2030.csv"),
                        usecols=["time", "Region", "value"])

# Data for Behind-the-Meter Storage (BTM) ----------------------------------------------------------
# share of solar PV capacity paired with BTM storage per IMAGE region (values: 0-1)
ratio_btm_deployment_data = pd.read_csv(Path(path_external_data_standard, "behind_the_meter_battery_deployment_ratio.csv"), usecols=["Time", "Region", "Value"])



# 2. IMAGE/TIMER files ====================================================================================

# IMAGE-energy: storage energy capacity demand (MWh, reservoir)
storage_energy = read_mym_df(path_image_output.joinpath("StorResTot.out"))   #storage capacity in MWh (reservoir, so energy capacity, not power capacity, the latter is used later on in the pumped hydro storage calculations)
# storage_energy = read_mym_df(climate_policy_config["config_file_path"] / climate_policy_config["data_files"]['StorResTot'])

# IMAGE-energy: storage power capacity demand (MW)
storage_power = read_mym_df(path_image_output / 'StorCapTot.out')  
# storage_power = read_mym_df(climate_policy_config["config_file_path"] / climate_policy_config["data_files"]['StorCapTot'])

# Generation capacity (stock & inflow/new) in MW peak capacity, FILES from TIMER
gcap_data = read_mym_df(path_image_output / 'GCap.out')  
# gcap_data = read_mym_df(climate_policy_config["config_file_path"] / climate_policy_config["data_files"]['GCap']) # needed to get hydro power for storage
# gcap_data = gcap_data.iloc[:, :26]


# ----------------------------------------------------------------------------------------------------------
# ##########################################################################################################
# %% 2.2) Prepare general variables
# ##########################################################################################################
# ----------------------------------------------------------------------------------------------------------


# ##########################################################################################################
# %%% To xarray

knowledge_graph_region = create_image_region_graph()
knowledge_graph_electr = create_electricity_graph()

# storage energy capacity (MWh) --------------------------------------------------------------------
storage_energy = storage_energy.iloc[:, :26]
storage_energy.index.name = "Time"
storage_energy.columns.name = "Region"
storage_energy_da = xr.DataArray(
    storage_energy.values,
    dims=["Time", "Region"],
    coords={
        "Time": storage_energy.index,
        "Region": [("region_" + str(r)) for r in storage_energy.columns]
    }
)
storage_energy_da = prism.Q_(storage_energy_da, unit="MWh")
storage_energy_da = knowledge_graph_region.rebroadcast_xarray(storage_energy_da, output_coords=IMAGE_REGIONS, dim="Region")

# storage power capacity (MW) ----------------------------------------------------------------------
storage_power = storage_power.iloc[:, :26]
storage_power.index.name = "Time"
storage_power.columns.name = "Region"
storage_power_da = xr.DataArray(
    storage_power.values,
    dims=["Time", "Region"],
    coords={
        "Time": storage_power.index,
        "Region": [("region_" + str(r)) for r in storage_power.columns]
    }
)
storage_power_da = prism.Q_(storage_power_da, unit="MW")
storage_power_da = knowledge_graph_region.rebroadcast_xarray(storage_power_da, output_coords=IMAGE_REGIONS, dim="Region")

# power generation capacity (MW) -------------------------------------------------------------------
gcap_data = gcap_data.loc[~gcap_data['DIM_1'].isin([27,28])]  # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies
gcap_data = gcap_data.loc[gcap_data['time'].isin(range(year_start, year_end + 1)), ['time', 'DIM_1', *range(1, len(EPG_TECHNOLOGIES) + 1)]]  # only keep relevant years and technology columns
# Extract coordinate labels
years = sorted(gcap_data['time'].unique())
regions = sorted(gcap_data['DIM_1'].unique())
techs = list(range(1, len(EPG_TECHNOLOGIES)+1))
# Convert to 3D array: (Year, Region, Tech)
data_array = gcap_data[techs].to_numpy().reshape(len(years), len(regions), len(techs))
# Build xarray DataArray
gcap_da = xr.DataArray(
    data_array,
    dims=('Time', 'Region', 'Type'),
    coords={
        'Time': years,
        'Region': [("region_" + str(r)) for r in regions],
        'Type': [str(r) for r in techs]
    },
    name='GenerationCapacity'
)
gcap_da = prism.Q_(gcap_da, "MW")
gcap_da = knowledge_graph_region.rebroadcast_xarray(gcap_da, output_coords=IMAGE_REGIONS, dim="Region") 
gcap_da = knowledge_graph_electr.rebroadcast_xarray(gcap_da, output_coords=EPG_TECHNOLOGIES, dim="Type")

# material intensities (kg/kWh) --------------------------------------------------------------------
material_intensities_da = (
    material_intensities.set_index(["Time", "sub_technology", "material"])["value"]
    .to_xarray()
    .rename({"sub_technology": "Type"})
)
material_intensities_da = prism.Q_(material_intensities_da, "kg/kWh")

# market shares (%) --------------------------------------------------------------------------------
market_shares_da = (
    market_shares.set_index(["Time", "sub_technology"])["value"]
    .to_xarray()
    .rename({"sub_technology": "Type"})
)
market_shares_da = prism.Q_(market_shares_da, "dimensionless")

# lifetimes (yr) -----------------------------------------------------------------------------------
lifetimes_da = (
    lifetimes.set_index(["Time", "sub_technology"])["value"]
    .to_xarray()
    .rename({"sub_technology": "Type"})
)
lifetimes_da = lifetimes_da.expand_dims({"DistributionParams": ["mean", "stdev"]})
lifetimes_da = lifetimes_da.sel(Type=[t for t in TECH_STATIONARY_STORAGE if t in lifetimes_da.coords["Type"].values])
lifetimes_da = prism.Q_(lifetimes_da, "years")

# Interpolations ===================================================================================

# TIMER data only start in 1971, so we add a historic tail back to YEAR_FIRST_GRID=1921
gcap_da_interp = add_historic_stock(gcap_da, YEAR_FIRST_GRID)

year_start = YEAR_FIRST_GRID
year_end = 2100
# if material intensities only have one value (e.g., for 2020), then we can simply broadcast this 
# value across all years, otherwise we need to interpolate between the given values and then 
# extrapolate the interpolated values to the full time range
if material_intensities_da.sizes["Time"] == 1:
    # Broadcast the single value across all new time coordinates
    material_intensities_da = material_intensities_da.reindex(Time=np.arange(year_start, year_end), method="nearest")
else:
    material_intensities_da = interpolate_xr(material_intensities_da, year_start, year_end)

market_shares_da = interpolate_xr(market_shares_da, year_start, year_end)


####################################################################################################
#%%% 1. Pumped Hydropower Storage

data_phs = [df_data1, df_data2, df_data3, df_shares_adjustment_2030, storage_energy_da]
phs_power, phs_energy = derive_phs_installed_capacity(data_phs, factor_phs_growth_rel_demand, mean_discharge_duration, flag_phs_scenario) #mean_discharge_duration


####################################################################################################
#%%% 2. Behind-the-meter storage

# btm = derive_btm_installed_capacity(gcap_da_interp, ratio_btm_deployment_data, ratio_btm_to_solar)



####################################################################################################
#%%% 3. Grid-scale storage

path_output = Path(path_base.parent.parent.parent, "elc-analysis", "output", "model_set_up")

storage_energy_remaining = (storage_energy_da - phs_energy).clip(min=0)

floor_fraction = xr.zeros_like(storage_energy_remaining)

# Linearly interpolate floor fraction from 0.05 (2010) to 0.2 (2020), capped at 0.2 after
floor_fraction = xr.where(storage_energy_remaining.Time >= 2010,
                    0.05 + (0.2 - 0.05) * (storage_energy_remaining.Time - 2010) / (2020 - 2010),
                    floor_fraction)
floor_fraction = xr.where(storage_energy_remaining.Time >= 2020, 0.2, floor_fraction)

# Apply floor
min_value = floor_fraction * storage_energy_da
storage_energy_remaining = xr.where(
    (storage_energy_remaining.Time >= 2000) & (storage_energy_remaining < min_value),
    min_value,
    storage_energy_remaining
)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
test = floor_fraction.sel(Region="NAF").plot(ax=ax, label="Floor Fraction")

storage_energy_remaining0  = (storage_energy_da - phs_energy).clip(min=0)

reg = "WEU"
fig, ax = plt.subplots()
storage_energy_da.sel(Region=reg).plot(ax=ax, label="Total storage demand")
storage_energy_remaining0.sel(Region=reg).plot(ax=ax, label="(before floor)")
storage_energy_remaining.sel(Region=reg).plot(ax=ax, label="(after floor)")
plt.legend()


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
prep_data_phs["set_unit_flexible"] = str(prism.U_(prep_data_phs["stocks"])) # prism.U_ gives the unit back


# Other storage--------------------------------------------------------------------------------------------

oth_storage_stock = oth_storage.copy()

# Bring dataframes into correct shape for the results_dict

# stocks: (years, regions) index and technologies as columns -> years as index and (technology, region) as columns
# Current columns are regions
regions = oth_storage.columns.tolist()
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
prep_data_oth_storage["set_unit_flexible"] = str(prism.U_(prep_data_oth_storage["stocks"])) # prism.U_ gives the unit back
prep_data_oth_storage["knowledge_graph"] = create_electricity_graph()


# change region names to IMAGE_REGIONS # TODO: this should be done in hte beginning of preprocessing
knowledge_graph_region =            create_region_graph()
prep_data_phs["stocks"] =           knowledge_graph_region.rebroadcast_xarray(prep_data_phs["stocks"], output_coords=IMAGE_REGIONS, dim="Region")
prep_data_oth_storage["stocks"] =   knowledge_graph_region.rebroadcast_xarray(prep_data_oth_storage["stocks"], output_coords=IMAGE_REGIONS, dim="Region")

# Have both stocks and stocks_non_phs in the prep_data_oth_storage. In case vehicle-to-grid (V2G) is considered and the ev battery + Link module is added
# to the joined model run, stocks_non_phs is used and stocks is replaced with the remaining storage demand after subtracting the EV battery storage. In 
# case V2G is not considered, stocks is used directly and represents the storage demand fulfilled by dedicated grid storage technologies (non-PHS).
prep_data_oth_storage["stocks_non_phs"] = prep_data_oth_storage["stocks"].copy()

# return prep_data_phs, prep_data_oth_storage

