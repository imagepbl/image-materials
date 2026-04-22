#%%
import pandas as pd
import numpy as np
from pathlib import Path
import pint
import xarray as xr
import matplotlib.pyplot as plt
from importlib.resources import files
import warnings
from pint.errors import UnitStrippedWarning

import prism
from imagematerials.read_mym import read_mym_df
from imagematerials.constants import IMAGE_REGIONS
from imagematerials.concepts import create_image_region_graph, create_electricity_graph, create_region_graph
from imagematerials.electricity.constants import (
    COLORS_IMAGE_REGIONS,
    STANDARD_SCEN_EXTERNAL_DATA,
    YEAR_FIRST_GRID,
    EPG_TECHNOLOGIES,
    STD_LIFETIMES_ELECTR,
)
from imagematerials.electricity.utils import add_historic_stock, interpolate_xr

path_current = Path().resolve()
path_base = path_current.parent.parent # base path of the project -> image-materials
path_data = Path(path_base, "data", "raw")

units_file = files("imagematerials") / "units.txt"
prism.unit_registry.load_definitions(units_file)

year_start = 1970
year_end = 2100

####################################################################################################
#%% Load data & Preprocess

# Read in data -----------------------------------------
# Generation capacity (stock demand per generation technology) in MW peak capacity
gcap_data = read_mym_df(Path(path_data, "image", "SSP2_baseline", "EnergyServices", "GCap.out"))
#storage capacity in MW (power capacity), to compare it to Pumped hydro storage projections (also given in MW, power capacity)
storage_power = read_mym_df(Path(path_data, "image", "SSP2_baseline", "EnergyServices", "StorCapTot.out"))

ratio_btm_deployment_data = pd.read_csv(Path(path_data, "electricity",'behind_the_meter_battery_deployment_ratio.txt'), usecols=["Time", "Region", "Value"])
ratio_btm_deployment_data["Value"] = ratio_btm_deployment_data["Value"] / 100 # convert percentage to fraction
ratio_btm_to_solar = 0.5 #0.5 MW storage per 1 MW Solar PV


# Transform to xarray -----------------------------------------
knowledge_graph_region = create_image_region_graph() #create_region_graph()
knowledge_graph_electr = create_electricity_graph()

# Gcap ------
gcap_data = gcap_data.loc[~gcap_data['DIM_1'].isin([27,28])]  # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies
gcap_data = gcap_data.loc[gcap_data['time'].isin(range(year_start, year_end + 1)), ['time', 'DIM_1', *range(1, len(EPG_TECHNOLOGIES) + 1)]]  # only keep relevant years and technology columns
# Extract coordinate labels
years = sorted(gcap_data['time'].unique())
regions = sorted(gcap_data['DIM_1'].unique())
techs = list(range(1, len(EPG_TECHNOLOGIES)+1))
# Convert to 3D array: (Year, Region, Tech)
data_array = gcap_data[techs].to_numpy().reshape(len(years), len(regions), len(techs))
# Build xarray DataArray
gcap_xr = xr.DataArray(
    data_array,
    dims=('Time', 'Region', 'Type'),
    coords={
        'Time': years,
        'Region': [("region_" + str(r)) for r in regions],
        'Type': [str(r) for r in techs]
    },
    name='GenerationCapacity'
)
gcap_xr = prism.Q_(gcap_xr, "MW")
gcap_xr = knowledge_graph_region.rebroadcast_xarray(gcap_xr, output_coords=IMAGE_REGIONS, dim="Region") 
gcap_xr = knowledge_graph_electr.rebroadcast_xarray(gcap_xr, output_coords=EPG_TECHNOLOGIES, dim="Type")

# deployment ratio ------

ratio_btm_deployment = ratio_btm_deployment_data.groupby(["Region", "Time"])[["Value"]].sum()
ratio_btm_deployment = xr.DataArray.from_series(ratio_btm_deployment["Value"])
ratio_btm_deployment = prism.Q_(ratio_btm_deployment, "fraction")

# Storage capacity ------
storage_power = storage_power.iloc[:, :26]
storage_power.index.name = "Time"
storage_power.columns.name = "Region"
storage_xr = xr.DataArray(
    storage_power.values,
    dims=["Time", "Region"],
    coords={
        "Time": storage_power.index,
        "Region": [("region_" + str(r)) for r in storage_power.columns]
    }
)
storage_xr = knowledge_graph_region.rebroadcast_xarray(storage_xr, output_coords=IMAGE_REGIONS, dim="Region") 


# Interpolate -----------------------------------------
# TIMER data only start in 1971, so we add a historic tail back to YEAR_FIRST_GRID=1921
gcap_xr_interp = add_historic_stock(gcap_xr, YEAR_FIRST_GRID)

ratio_btm_deployment_xr_interp = interpolate_xr(ratio_btm_deployment, t_start = 2000, t_end = 2100)


fig, ax = plt.subplots(figsize=(14, 6))
for i, region in enumerate(test.Region.values):
    ax.plot(test.Time.values, test.sel(Region=region).values, linewidth=1.5, color=COLORS_IMAGE_REGIONS[i], label=region)
# ax.axvline(x=2014, color='gray', linestyle='--', alpha=0.5)
# ax.axvline(x=2019, color='gray', linestyle='--', alpha=0.5)
# ax.axvline(x=2024, color='gray', linestyle='--', alpha=0.5)
plt.xlabel("Year")
plt.ylabel("behind-the-meter battery deployment ratio")
plt.title("Interpolated behind-the-meter battery deployment ratios by IMAGE region")
plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8, ncol=2)
plt.tight_layout()


####################################################################################################
#%%

btm = ratio_btm_deployment_xr_interp * gcap_xr.sel(Type="SPVR").drop_vars("Type") * ratio_btm_to_solar
btm_global = btm.sum(dim="Region")

fig, ax = plt.subplots(figsize=(14, 6))
for i, region in enumerate(btm.Region.values):
    ax.plot(btm.Time.values, btm.sel(Region=region).values, linewidth=1.5, color=COLORS_IMAGE_REGIONS[i], label=region)
ax.plot(btm_global.Time.values, btm_global.values, linewidth=2, color='black', label="Global")
plt.xlabel("Year")
plt.ylabel("behind-the-meter battery capacity (MW)")
plt.title("behind-the-meter battery deployment capacities by IMAGE region")
plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8, ncol=2)
plt.tight_layout()

test = storage_xr.sum("Region")

fig, ax = plt.subplots(figsize=(14, 6))
for i, region in enumerate(btm.Region.values):
    ax.plot(btm.sel(Time=slice(2015, 2040)).Time.values, btm.sel(Region=region,Time=slice(2015, 2040)).values, linewidth=1.5, color=COLORS_IMAGE_REGIONS[i], label=region)
ax.plot(btm_global.sel(Time=slice(2015, 2040)).Time.values, btm_global.sel(Time=slice(2015, 2040)).values, linewidth=2, color='black', label="Global")
ax.plot(test.sel(Time=slice(2015, 2040)).Time.values, test.sel(Time=slice(2015, 2040)).values, linewidth=2, color='red', label="TIMER storage demand", linestyle='--')
plt.xlabel("Year")
plt.ylabel("behind-the-meter battery capacity (MW)")
plt.title("behind-the-meter battery deployment capacities by IMAGE region")
plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8, ncol=2)
plt.tight_layout()





