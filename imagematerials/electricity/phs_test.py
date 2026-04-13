#%%
import pandas as pd
import numpy as np
from pathlib import Path
import pint
import xarray as xr
import matplotlib.pyplot as plt 

from imagematerials.constants import IMAGE_REGIONS
from imagematerials.concepts import create_image_region_graph
from imagematerials.electricity.constants import COLORS_IMAGE_REGIONS

path_current = Path().resolve()
path_base = path_current.parent.parent # base path of the project -> image-materials
path_data = Path(path_base, "data", "raw")

df_phs =  pd.read_csv(Path(path_data,"electricity","IHA_PSH_Capacity_data.csv"), 
                      usecols=["Operational Status", "Country", "Commisioning Year", "Hydro Type", 
                               "Generating Capacity", "Energy stored (GWh)"])
# a = df_phs['Hydro Type'].unique()
df_phs["Commisioning Year"] = pd.to_numeric(df_phs["Commisioning Year"], errors="coerce").astype("Int64")

df_sum = df_phs.groupby(["Country", "Commisioning Year"])[
    ["Generating Capacity", "Energy stored (GWh)"]
].sum()

ds_sum = df_sum.to_xarray()
ds_sum = ds_sum.rename({'Country': 'Region', 'Commisioning Year': 'Commissioning Year'})



####################################################################################################
#%% Aggregate to IMAGE regions

knowledge_graph_region = create_image_region_graph()
ds_sum_region = ds_sum.copy()

ds_sum_region = knowledge_graph_region.aggregate_sum(ds_sum_region, output_coords=IMAGE_REGIONS, dim="Region", require_relation=False)

####################################################################################################
#%% Aggregate years to decades and calculate shares

ds = ds_sum_region.copy()
# Define 10-year bin edges — extend as needed to cover your full range
bin_edges = range(1905, 2045, 10)  # 1905, 1915, 1925, ...
labels = [f"{y}-{y+9}" for y in range(1905, 2035, 10)]

years = ds["Commissioning Year"].values

# Assign each year to a bin label
bin_labels = pd.cut(years, bins=list(bin_edges), right=False, labels=labels)
year_to_bin = dict(zip(years, bin_labels))

# Add a bin coordinate to the dataset
ds = ds.assign_coords(bin_years=("Commissioning Year",
                           [year_to_bin[y] for y in years]))

# Sum over each bin per region
ds_binned = ds.groupby("bin_years").sum("Commissioning Year")
# Result: dims (Region, bin_years)

# Compute regional shares per bin (each region's fraction of the bin total)
bin_totals = ds_binned.sum("Region")
ds_shares = ds_binned / bin_totals # ds_shares has the same dims; each value is that region's fraction [0–1]

# Inspect
print(ds_binned)
print(ds_shares)


####################################################################################################
#%% Plots

#---------------------------------------------------------------------------------------------------
#%%% 
test = ds_sum.sum(["Country"])

fig, ax = plt.subplots()
test["Generating Capacity"].plot(ax=ax, label="Annual")    
test["Generating Capacity"].cumsum(dim="Commisioning Year").plot(
    ax=ax, label="Cumulative"
)
plt.xlabel("Commisioning Year")
plt.ylabel("Generating Capacity (MW)")
plt.legend()


fig, ax = plt.subplots()
test["Energy stored (GWh)"].plot(ax=ax, label="Annual")    
test["Energy stored (GWh)"].cumsum(dim="Commisioning Year").plot(
    ax=ax, label="Cumulative"
)
plt.xlabel("Commisioning Year")
plt.ylabel("Energy stored (GWh)")
plt.legend()


#---------------------------------------------------------------------------------------------------
#%%% per region

test = ds_sum_region_tests.sel(Region="MEX")

fig, ax = plt.subplots()
test["Generating Capacity"].plot(ax=ax, label="Annual")    
test["Generating Capacity"].cumsum(dim="Commisioning Year").plot(
    ax=ax, label="Cumulative"
)
plt.xlabel("Commisioning Year")
plt.ylabel("Generating Capacity (MW)")
plt.legend()


#---------------------------------------------------------------------------------------------------
#%%% shares per region

# Extract data as a 2D numpy array: shape (13 bins, 26 regions)
data = ds_shares["Generating Capacity"].values  # (bin_years, Region)
regions = ds_shares["Region"].values
bin_labels = ds_shares["bin_years"].values

x = np.arange(len(bin_labels))

fig, ax = plt.subplots(figsize=(14, 6))

for i, region in enumerate(regions):
    ax.plot(x, data[:, i], marker="o", linewidth=1.5, color=COLORS_IMAGE_REGIONS[i], label=region)

ax.set_xticks(x)
ax.set_xticklabels(bin_labels, rotation=45, ha="right")
ax.set_ylabel("Share of Generating Capacity (%)")
ax.set_xlabel("Period")
ax.set_title("Regional shares of newly installed Generating Capacity by decade")
ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8, ncol=2)
plt.tight_layout()
plt.show()