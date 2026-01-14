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

# YEAR_START = 1971
# YEAR_END = 2100
# YEAR_OUT = 2100

# SCEN = "SSP2"
# # VARIANT = "VLHO"
# VARIANT = "M_CP"
# scen_folder = SCEN + "_" + VARIANT
# # path_base = Path().resolve() # TODO absolute path of file "preprocessing.py" ? current solution can differ depending on IDE used (?) 
# path_current = Path().resolve()
# path_base = path_current.parent.parent # base path of the project -> image-materials
# print(f"Base path: {path_base}")

# path_image_output = Path(path_base, "data", "raw", "image", scen_folder, "EnergyServices")

# path_external_data_standard = Path(path_base, "data", "raw", "electricity", "standard_data")
# path_external_data_scenario = Path(path_base, "data", "raw", "electricity", scen_folder)
# # test if path_external_data_scenario exists and if not set to standard scenario
# if not path_external_data_scenario.exists():
#     path_external_data_scenario = Path(path_base, "data", "raw", "electricity", STANDARD_SCEN_EXTERNAL_DATA)
# print(f"Path to image output: {path_image_output}")

# assert path_image_output.is_dir()
# assert path_external_data_standard.is_dir()
# assert path_external_data_scenario.is_dir()

# # create the folder out_test if it does not exist
# if not (path_base / 'imagematerials' / 'electricity' / 'out_test').is_dir():
#     (path_base / 'imagematerials' / 'electricity' / 'out_test').mkdir(parents=True)

# prism.unit_registry.load_definitions(path_base / "imagematerials" / "units.txt")
# C:\Users\Judit\PhD\Coding\image-materials\imagematerials\units.txt
idx = pd.IndexSlice   


#----------------------------------------------------------------------------------------------------------
###########################################################################################################
# %% As function
###########################################################################################################
#----------------------------------------------------------------------------------------------------------

def get_preprocessing_data_evbattery(path_base: str, scenario, year_start, year_out):
    

    path_image_output = Path(path_base, "data", "raw", "image", scenario, "EnergyServices")
    path_external_data_standard = Path(path_base, "data", "raw", "electricity", "standard_data")
    path_external_data_scenario = Path(path_base, "data", "raw", "electricity", scenario)
    # test if path_external_data_scenario exists and if not set to standard scenario
    if not path_external_data_scenario.exists():
        path_external_data_scenario = Path(path_base, "data", "raw", "electricity", STANDARD_SCEN_EXTERNAL_DATA)
    print(f"Path to image output: {path_image_output}")
    assert path_image_output.is_dir()
    assert path_external_data_standard.is_dir()
    assert path_external_data_scenario.is_dir()

    prism.unit_registry.load_definitions(path_base / "imagematerials" / "units.txt")


    ###########################################################################################################
    # Read in files #
    
    # 1. External Data ======================================================================================== 

    # storage costs according to IRENA storage report & other sources in the SI ($ct/kWh electricity cycled)
    storage_costs = pd.read_csv(path_external_data_standard / "storage_cost.csv", index_col=0).transpose()

    # assumed malus & bonus of storage costs (malus for advanced technologies, still under development; bonus for batteries currently used in EVs, we assume that a large volume of used EV batteries will be available and used for dedicated electricity storage, thus lowering costs), only the bonus remains by 2030
    costs_correction = pd.read_csv(path_external_data_standard / "storage_malus.csv", index_col=0).transpose()

    # assumptions on the long-term price decline after 2050. the fraction of the annual growth rate (determined based on 2018-2030) that will be applied after 2030, ranging from 0.25 to 1 - 0.25 means the price decline is not expected to continue strongly, while 1 means that the same (2018-2030) annual price decline is also applied between 2030 and 2050)
    cost_decline_longterm_correction = pd.Series(pd.read_csv(path_external_data_standard / "storage_ltdecline.csv",index_col=0,  header=None).transpose().iloc[0]) 

    # INVERS energy density (note that the unit is kg/kWh, the invers of energy density: storage capacity - mass required to store one unit of energy —> more mass per energy = worse performance)
    # TODO: read in directly IRENA data?
    energy_density = pd.read_csv(path_external_data_standard / "storage_density_kg_per_kwh.csv",index_col=0).transpose()
    # lifetime of storage technologies (in yrs). The lifetime is assumed to be 1.5* the number of cycles divided by the number of days in a year (assuming diurnal use, and 50% extra cycles before replacement, representing continued use below 80% remaining capacity) OR the maximum lifetime in years, which-ever comes first 
    storage_lifetime = pd.read_csv(path_external_data_standard / "storage_lifetime.csv",index_col=0).transpose()

    # material compositions (storage) in wt%
    storage_materials = pd.read_csv(path_external_data_standard / "storage_materials_dynamic.csv",index_col=[0,1]).transpose()  # wt% of total battery weight for various materials, total battery weight is given by the density file above

    # Using the 250 Wh/kg on the kWh of the various batteries a weight (in kg) of the battery per vehicle category is determined
    # TODO: where is this data from? kWh per battery type?
    battery_weights_data = pd.read_csv(path_external_data_standard / "battery_weights_kg.csv", index_col=[0,1])

    # usable capacity of EV batteries for V2G applications (relative: fraction of the total battery capacity that can be used for V2G)
    if SENS_ANALYSIS == 'high_stor':
    # pessimistic sensitivity variant (meaning more additional storage is needed) -> smaller fraction of the EV capacities is usable as storage compared to the normal case
    #    capacity_usable_PHEV = 0.025   # 2.5% of capacity of PHEV is usable as storage (in the pessimistic sensitivity variant)
    #    capacity_usable_BEV  = 0.05    # 5  % of capacity of BEVs is usable as storage (in the pessimistic sensitivity variant)
        ev_capacity_fraction_v2g = pd.read_csv(path_external_data_standard / "ev_battery_capacity_usable_for_v2g_variant_high_storage.csv")
    else:
        ev_capacity_fraction_v2g = pd.read_csv(path_external_data_standard / "ev_battery_capacity_usable_for_v2g.csv")

    # fraction of EVs available for V2G (considering that not all EVs are capable of bi-directional loading, economic incentives are still missing, and not all owners are willing to provide V2G services)
    ev_fraction_v2g_data = pd.read_csv(path_external_data_standard / "ev_fraction_available_for_v2g.csv", index_col=[0])


    ###########################################################################################################
    # Transform to xarray #

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
    vhc_knowledge_graph = create_vehicle_graph()

    # 1. Market Shares -----------------------------------------------------------------------------

    # 1.1 storage costs
    years = storage_costs.index.astype(int) #.astype(int) to convert years from strings to integers
    techs = storage_costs.columns
    data_array = storage_costs.to_numpy()
    xr_storage_costs = xr.DataArray(
        data_array,
        dims=("Cohort", "BatteryType"),
        coords={
            "Cohort": years,
            "BatteryType": techs
        },
        name="StorageCosts"
    )
    xr_storage_costs = prism.Q_(xr_storage_costs, "USD_cent/kWh")

    # 1.2 storage costs correction (malus/bonus multiplicative factor) 
    years = costs_correction.index.astype(int)
    techs = costs_correction.columns
    data_array = costs_correction.to_numpy()
    xr_costs_correction = xr.DataArray(
        data_array,
        dims=("Cohort", "BatteryType"),
        coords={
            "Cohort": years,
            "BatteryType": techs
        },
        name="StorageCostsCorrection"
    )
    xr_costs_correction = prism.Q_(xr_costs_correction, "dimensionless")

    # 1.3 storage costs longterm decline factor
    techs = cost_decline_longterm_correction.index.rename(None)
    data_array = cost_decline_longterm_correction.to_numpy()
    xr_cost_decline_longterm_correction = xr.DataArray(
        data_array,
        dims=("BatteryType",),
        coords={
            "BatteryType": techs
        },
        name="StorageCostsDeclineLongterm"
    )
    xr_cost_decline_longterm_correction = prism.Q_(xr_cost_decline_longterm_correction, "fraction")

    # 2. battery weigths -----------------------------------------------------------------------

    df = battery_weights_data.copy()
    # 1. Extract values
    times = sorted(battery_weights_data.index.get_level_values(0).unique())
    drivetrains = sorted(battery_weights_data.index.get_level_values(1).unique())
    vehicles = battery_weights_data.columns
    # 2. Create combined Type labels: 'Vehicle - Drivetrain'
    types = [f"{v} - {d}" for v in vehicles for d in drivetrains]
    # 3. Convert DataFrame to NumPy array
    df_sorted = battery_weights_data.sort_index(level=[0,1]) # First, ensure row order: time × drivetrain
    data = df_sorted.to_numpy()  # Flatten rows and columns into 2D array: shape: (len(times)*len(drivetrains), len(vehicles))
    # 4. Reorder / reshape to (time, Type)
    data_2d = np.hstack([data[i::len(drivetrains), :] for i in range(len(drivetrains))]) # For each time, concatenate all drivetrains in order
    # 5. Build xarray
    xr_battery_weights = xr.DataArray(
        data_2d,
        dims=("Cohort", "Type"),
        coords={
            "Cohort": times,
            "Type": types
        },
        name="BatteryWeights"
    )
    xr_battery_weights = prism.Q_(xr_battery_weights, "kg")
    xr_battery_weights = vhc_knowledge_graph.rebroadcast_xarray(xr_battery_weights, output_coords=vehicle_list, dim="Type")

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
        dims=("material", "Cohort", "BatteryType"),
        coords={
            "material": materials,
            "Cohort": years,
            "BatteryType": techs
        },
        name="MaterialFractions"
    )
    xr_storage_materials = prism.Q_(xr_storage_materials, "fraction")
    xr_battery_materials = xr_storage_materials.sel(BatteryType=EV_BATTERIES) # Select only those technologies from the storage technologies that are suitable for EV & mobile applications


    # 4. energy density -----------------------------------------------------------------------------
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
    xr_energy_density = xr_energy_density.sel(BatteryType=EV_BATTERIES) # Select only those technologies from the storage technologies that are suitable for EV & mobile applications


    # 5. EV fraction available for V2G --------------------------------------------------------------------
    # For this variable first the interpolations are done and then the conversion to xarray DataArray
    x = ev_fraction_v2g_data.reindex(range(ev_fraction_v2g_data.index[0],ev_fraction_v2g_data.index[-1]+1)).interpolate(method="linear")
    y = logistic(x, L=x.iloc[-1].values)
    # y = quadratic(x)
    ev_fraction_v2g = ev_fraction_v2g_data.reindex(range(YEAR_FIRST_GRID,year_out+1)).interpolate(method="linear") # create dataframe with full index; values before first data points will be Nans, between data points interpolated linearly, after last data point will be last known value
    ev_fraction_v2g.loc[:ev_fraction_v2g_data.index[0]] = 0 # set values before first data point to 0
    ev_fraction_v2g.loc[ev_fraction_v2g_data.index[0]:ev_fraction_v2g_data.index[1]] = y # set values between (originally) first and last data point to quadratic/logistic interpolation
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
    data_array = vhc_fraction_v2g.to_numpy()    # shape (Time, Type)
    data_array = data_array[:, :, np.newaxis]   # shape (Time, Type, 1)
    data_array = np.broadcast_to(data_array, data_array.shape[:2] + (len(IMAGE_REGIONS),)) # (Time, Type, Region) - add region dimension, though all regions have the same values for now
    xr_vhc_fraction_v2g = xr.DataArray(
        data_array,
        dims=("Time", "Type", "Region"),
        coords={
            "Time": years,
            "Type": techs,
            "Region": IMAGE_REGIONS
        },
        name="VehicleFractionV2G"
    )
    xr_vhc_fraction_v2g = prism.Q_(xr_vhc_fraction_v2g, "fraction")
    xr_vhc_fraction_v2g = vhc_knowledge_graph.rebroadcast_xarray(xr_vhc_fraction_v2g, output_coords=vehicle_list, dim="Type")

    # 6. capacity used for V2G -----------------------------------------------------------------------------
    # Build xarray DataArray
    techs = ev_capacity_fraction_v2g.columns
    data_array = ev_capacity_fraction_v2g.to_numpy().ravel()  # flatten to 1D
    xr_capacity_fraction_v2g = xr.DataArray(
        data_array,
        dims=("Type"),
        coords={
            "Type": techs
        },
        name="CapacityFractionV2g"
    )
    xr_capacity_fraction_v2g = prism.Q_(xr_capacity_fraction_v2g, "fraction")


    ###########################################################################################################
    # Calculations #

    # 1. Market Shares -----------------------------------------------------------------------------

    # calc. storage market shares based on cost developments & multinomial logit model for the years 1970-2050
    storage_market_share = calculate_storage_market_shares(
        xr_storage_costs,
        xr_costs_correction,
        xr_cost_decline_longterm_correction,
        mnlogit_param=-0.2,
        t_start_interpolation=year_start,
        t_end_interpolation=2050
    )

    # fix the market share of storage technologies before year_start and after 2050
    storage_market_share_interp = interpolate_xr(storage_market_share, YEAR_FIRST_GRID, year_out)
    # Select only those technologies from the storage technologies that are suitable for EV & mobile applications
    # normalize the selection of EV battery technologies, so that total market share is 1 again (taking the relative share in the selected battery techs)
    market_share_EVs = normalize_selected_techs(storage_market_share_interp, EV_BATTERIES, dim_type="BatteryType") # TODO: this should be done differently as market shares of EV batteries probably differ from their market shares in total storage market

    # 2. Battery Weights ----------------------------------------------------------------
    xr_battery_weights_interp = interpolate_xr(xr_battery_weights, YEAR_FIRST_GRID, year_out)

    # 3. Material Intensities ----------------------------------------------------------------
    xr_battery_materials_interp = interpolate_xr(xr_battery_materials, YEAR_FIRST_GRID, year_out)

    #  4. Energy Density ----------------------------------------------------------------
    xr_energy_density_interp = interpolate_xr(xr_energy_density, YEAR_FIRST_GRID, year_out)


    ###########################################################################################################
    # Prep_data File #

    # bring preprocessing data into a generic format for the model
    prep_data = {}
    prep_data["shares"] = market_share_EVs
    prep_data["weights"] = xr_battery_weights_interp
    prep_data["material_fractions"] = xr_battery_materials_interp
    prep_data["energy_density"] = xr_energy_density_interp
    prep_data["vhc_fraction_v2g"] = xr_vhc_fraction_v2g
    prep_data["capacity_fraction_v2g"] = xr_capacity_fraction_v2g
    prep_data["knowledge_graph"] = create_electricity_graph()

    return prep_data



#%%% create sector %%%

# prep_data = get_preprocessing_data_evbattery(path_base=path_base, scenario = "SSP2_M_CP", year_start=YEAR_START, year_out=YEAR_OUT)

# time_start = 1970
# complete_timeline = prism.Timeline(time_start, YEAR_END, 1)
# simulation_timeline = prism.Timeline(YEAR_START, YEAR_END, 1) #1970

# sec_evbattery = Sector("ev_batteries", prep_data, check_coordinates=False)







"""
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

df = battery_weights_data.copy()
# 1. Extract values
times = sorted(battery_weights_data.index.get_level_values(0).unique())
drivetrains = sorted(battery_weights_data.index.get_level_values(1).unique())
vehicles = battery_weights_data.columns
# 2. Create combined Type labels: 'Vehicle - Drivetrain'
types = [f"{v} - {d}" for v in vehicles for d in drivetrains]
# 3. Convert DataFrame to NumPy array
df_sorted = battery_weights_data.sort_index(level=[0,1]) # First, ensure row order: time × drivetrain
data = df_sorted.to_numpy()  # Flatten rows and columns into 2D array: shape: (len(times)*len(drivetrains), len(vehicles))
# 4. Reorder / reshape to (time, Type)
data_2d = np.hstack([data[i::len(drivetrains), :] for i in range(len(drivetrains))]) # For each time, concatenate all drivetrains in order
# 5. Build xarray
xr_battery_weights = xr.DataArray(
    data_2d,
    dims=("Cohort", "Type"),
    coords={
        "Cohort": times,
        "Type": types
    },
    name="BatteryWeights"
)
xr_battery_weights = prism.Q_(xr_battery_weights, "kg")
xr_battery_weights = vhc_knowledge_graph.rebroadcast_xarray(xr_battery_weights, output_coords=vehicle_list, dim="Type")

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
    dims=('material', 'Cohort', 'BatteryType'),
    coords={
        'material': materials,
        'Cohort': years,
        'BatteryType': techs
    },
    name='MaterialFractions'
)
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

storage_market_share = calculate_storage_market_shares(
    xr_storage_costs,
    xr_costs_correction,
    xr_cost_decline_longterm_correction,
    mnlogit_param=-0.2,
    t_start_interpolation=YEAR_START,
    t_end_interpolation=2050
)


# t_start = xr_storage_costs.Cohort.values[0]
# t_end   = xr_storage_costs.Cohort.values[-1]

# # interpolate from first to last vailable year within the data, then extend to  YEAR_START and 2050 (keep values constant before first and after last year)
# xr_storage_costs      = interpolate_xr(xr_storage_costs, YEAR_START, 2050)
# xr_costs_correction   = interpolate_xr(xr_costs_correction, YEAR_START, 2050)

# # determine the annual % decline of the costs based on the 2018-2030 data (original, before applying the malus)
# xr_cost_decline = (((xr_storage_costs.loc[t_start,:]-xr_storage_costs.loc[t_end,:])/(t_end-t_start))/xr_storage_costs.loc[t_start,:]).drop_vars('Cohort')
# xr_cost_decline_longterm = xr_cost_decline*xr_cost_decline_longterm_correction # cost decline after 2030 = cost decline 2018-2030 * correction factor
# # cost_decline_longterm_correction is a single number and should describe the long-term decline after 2030 relative to the 2018-2030 decline

# # storage_costs_new = storage_costs_interpol * costs_correction_interpol
# xr_storage_costs_cor = xr_storage_costs * xr_costs_correction

# # ---------- future development ----------
# # calculate the development from 2030 to 2050 (using annual price decline)
# # vectorized approach for "for t in range(t_end, 2050+1): ..."
# years_fwd = xr_storage_costs_cor.Cohort.where(
#     xr_storage_costs_cor.Cohort > t_end, drop=True
# )
# n_fwd = years_fwd - t_end
# factors_fwd = (1 - xr_cost_decline_longterm) ** n_fwd

# xr_storage_costs_cor.loc[dict(Cohort=years_fwd)] = (
#     xr_storage_costs_cor.sel(Cohort=t_end) * factors_fwd
# )

# # ---------- past development ----------
# # for historic price development, assume 2x AVERAGE annual price decline on all technologies
# years_bwd = xr_storage_costs_cor.Cohort.where(
#     xr_storage_costs_cor.Cohort < t_start, drop=True
# )
# n_bwd = t_start - years_bwd
# factors_bwd = (1 + 2 * xr_cost_decline_longterm.mean()) ** n_bwd

# xr_storage_costs_cor.loc[dict(Cohort=years_bwd)] = (
#     xr_storage_costs_cor.sel(Cohort=t_start) * factors_bwd
# )
# # restore values for lead-acid (set to constant 2018 values) -> exception: so that lead-acid gets a relative price advantage from 1970-2018
# xr_storage_costs_cor.loc[dict(Cohort=years_bwd, Type="Deep-cycle Lead-Acid")] = (
#     xr_storage_costs_cor.sel(Cohort=t_start, Type="Deep-cycle Lead-Acid")
# )

# # plt.figure()
# # xr_storage_costs_cor.sel(
# #     Type=['Flywheel'] #, 'Compressed Air'
# # ).plot.scatter(x='Cohort', label='Flywheel', color='black')
# # xr_storage_costs_cor.sel(
# #     Type=['Deep-cycle Lead-Acid']
# # ).plot.scatter(x='Cohort', label='Deep-cycle Lead-Acid', color='red')
# # plt.legend()
# # plt.show()


# # market shares ---
# # use the storage price development in the logit model to get market shares
# storage_market_share = MNLogit(xr_storage_costs_cor, -0.2) #assumes input of an ordered dataframe with rows as years and columns as technologies, values as prices. Logitpar is the calibrated Logit parameter (usually a nagetive number between 0 and 1)
# fix the market share of storage technologies before YEAR_START and after 2050
storage_market_share_interp = interpolate_xr(storage_market_share, YEAR_FIRST_GRID, YEAR_OUT)
# As not all storage technologies are suitable for EV & mobile applications, select only those technologies.
# normalize the selection of EV battery technologies, so that total market share is 1 again (taking the relative share in the selected battery techs)
market_share_EVs = normalize_selected_techs(storage_market_share_interp, EV_BATTERIES, dim_type="BatteryType") # TODO: this should be done differently as market shares of EV batteries probably differ from their market shares in total storage market


# 2. Battery Weights ----------------------------------------------------------------
xr_battery_weights_interp = interpolate_xr(xr_battery_weights, YEAR_FIRST_GRID, YEAR_OUT)

# 3. Material Intensities ----------------------------------------------------------------
xr_battery_materials_interp = interpolate_xr(xr_battery_materials, YEAR_FIRST_GRID, YEAR_OUT)

#  4. Energy Density ----------------------------------------------------------------
xr_energy_density_interp = interpolate_xr(xr_energy_density, YEAR_FIRST_GRID, YEAR_OUT)




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
    """
