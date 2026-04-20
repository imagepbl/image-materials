#%%
import pandas as pd
import numpy as np
from pathlib import Path
import pint
import xarray as xr
import matplotlib.pyplot as plt 

import prism
from imagematerials.constants import IMAGE_REGIONS
from imagematerials.concepts import create_image_region_graph
from imagematerials.electricity.constants import COLORS_IMAGE_REGIONS

path_current = Path().resolve()
path_base = path_current.parent.parent # base path of the project -> image-materials
path_data = Path(path_base, "data", "raw")

####################################################################################################
#%% Load data

# Dataset 1: IHA data set of individual PHS sites, with commissioning year, capacity, energy stored, operating status, and hydro type
df_data1 =  pd.read_csv(Path(path_data,"electricity","IHA_PSH_Capacity_data.csv"), 
                      usecols=["Operational Status", "Country", "Commisioning Year", "Hydro Type", 
                               "Generating Capacity", "Energy stored (GWh)"])
df_data1 = df_data1.loc[df_data1["Operational Status"] == "Operational"]
df_data1 = df_data1.rename(columns={'Country': 'Region', 'Commisioning Year': 'Cohort'})
df_data1["Cohort"] = pd.to_numeric(df_data1["Cohort"], errors="coerce").astype("Int64")

df_data1_country = df_data1.groupby(["Region", "Cohort"])[
    ["Generating Capacity", "Energy stored (GWh)"]
].sum() # rows with NaN in Region or Cohort are dropped during this step

ds_data1_country = df_data1_country.to_xarray()


# Dataset 2: IHA World Hydropower Outlook dataset of aggregated PHS power capacity (MW) by country for 3 years (2014, 2019, 2024) 
#           + IEA global estimates

df_data2 =  pd.read_csv(Path(path_data,"electricity","pumped_hydropower_storage_historic_stocks.csv"),
                        usecols=["Time", "Region", "value"])
df_data2 = df_data2[df_data2["Region"] != "World"] # drop global estimates, only keep World Hydropower Outlook
df_data2_country = df_data2.groupby(["Region", "Time"])[["value"]].sum()
da_data2_country = xr.DataArray.from_series(df_data2_country["value"]).rename("PumpedHydropowerStorageCapacity") # create Dataarray
da_data2_country = prism.Q_(da_data2_country, "MW")

#%%% tests

# countries = [
#     "Burundi", "Comoros", "Ethiopia", "Eritrea", "Djibouti",
#     "Kenya", "Madagascar", "Mauritius", "Reunion", "Rwanda",
#     "Seychelles", "Somalia", "Sudan", "Uganda"
# ]
# countries = [
#     "Afghanistan", "Bangladesh", "Bhutan", "Sri Lanka", "Maldives",
#     "Nepal", "Pakistan"
# ]

# test = df_data1.loc[df_data1["Region"].isin(countries)]
# test

####################################################################################################
#%% Aggregate to IMAGE regions

knowledge_graph_region = create_image_region_graph()

ds_data1_image_region = ds_data1_country.copy()
ds_data1_image_region = knowledge_graph_region.aggregate_sum(ds_data1_image_region, output_coords=IMAGE_REGIONS, dim="Region", require_relation=False)

ds_data2_image_region = ds_data2_country.copy()
ds_data2_image_region = knowledge_graph_region.aggregate_sum(ds_data2_image_region, output_coords=IMAGE_REGIONS, dim="Region", require_relation=False)

####################################################################################################
#%% Aggregate years to decades and calculate shares

ds = ds_data1_image_region.copy()
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
#%% Find starting year per region

# Mask where Generating Capacity != 0, then find the first True along Cohort axis
mask = ds_data1_image_region["Generating Capacity"] != 0  # shape: (Region, Cohort)

# Get index position of first nonzero per region (-1 if none found)
first_idx = mask.argmax(dim="Cohort")

# Map index positions back to actual Cohort coordinate values
earliest_cohort = ds_data1_image_region.Cohort.isel(Cohort=first_idx).astype(float)

# For regions where ALL values are 0, argmax returns 0 (misleading) — mask those out
has_nonzero = mask.any(dim="Cohort")
earliest_cohort = earliest_cohort.where(has_nonzero)

# As a dict
result = {
    region.item(): int(val.item()) if not np.isnan(val.item()) else 0 #2030
    for region, val in zip(earliest_cohort.Region, earliest_cohort)
}
regions_no_cohort = has_nonzero.Region[~has_nonzero].values.tolist()


#

zero_regions = ds_data2_image_region["value"].where(ds_data2_image_region["value"] == 0).isnull().all(dim="Time")

zero_regions = (ds_data2_image_region["value"] == 0).all(dim="Time")
zero_regions.Region[zero_regions].values

zero_regions = (ds_data2_image_region["value"].sel(Time=2014) == 0)
zero_regions.Region[zero_regions].values

#%% intra and extrapolate phs data

def interpolate_xr_test(data_array, t_start, t_end, interp_method='linear'):
    """
    Interpolate an xarray.DataArray over a continuous time range and
    extend its boundary values beyond the available data to span t_start - t_end.
    The function performs (linear) interpolation between all existing time
    coordinates in the input DataArray and fills values outside the
    original time range with the first and last available data, respectively.
    If t_start is a dict, different start years can be specified per Region,
    with values linearly interpolated to 0 at the region-specific start year,
    and zero-filled between the global t_start and the region-specific start year.

    Parameters
    ----------
    data_array : xarray.DataArray
        Input DataArray with a 'Time' coordinate containing numeric values (e.g. 2020, 2050).
    t_start : int, float, or dict
        Start year for the interpolation range. If a dict, keys are Region names
        and values are the year at which that region's data reaches 0. Regions not
        in the dict fall back to the minimum of the dict values.
    t_end : int or float
        End year for the interpolation range.
    interp_method : str, optional
        Interpolation method to use (default is 'linear'). See xarray documentation
        for available methods.

    Returns
    -------
    xarray.DataArray
        DataArray interpolated across the full range from global t_start to t_end.

    Note:
    Units are temporarily stripped during interpolation but are reattached
    before returning the result. The corresponding warning is suppressed.
    """
    dim = 'Time' if 'Time' in data_array.dims else 'Cohort'
    unit = prism.U_(data_array)

    coord_values = data_array[dim].values

    # --- Determine global t_start ---
    if isinstance(t_start, dict):
        global_t_start = min(t_start.values())
    else:
        global_t_start = t_start

    new_range = np.arange(global_t_start, t_end + 1)

    # --- Interpolate over full time range ---
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UnitStrippedWarning)
        da_interp = data_array.interp({dim: new_range}, method=interp_method)

    # --- Fill beyond original data range (constant extrapolation) ---
    da_interp.loc[{dim: slice(None, coord_values.min())}] = da_interp.sel({dim: coord_values.min()})
    da_interp.loc[{dim: slice(coord_values.max(), None)}] = da_interp.sel({dim: coord_values.max()})

    # --- Handle region-specific start years ---
    if isinstance(t_start, dict) and 'Region' in data_array.dims:
        regions = data_array.Region.values
        # fallback year for regions not in dict
        fallback_year = min(t_start.values())

        for region in regions:
            region_start = t_start.get(region, fallback_year)

            if region_start <= global_t_start:
                continue  # nothing to do

            # get the first actual data value for this region
            first_value = da_interp.sel(Region=region, **{dim: coord_values.min()}).values

            # linearly interpolate from 0 at region_start to first_value at coord_values.min()
            ramp_times = np.arange(region_start, coord_values.min() + 1)
            ramp_values = np.linspace(0, float(first_value), len(ramp_times))
            da_interp.loc[{dim: ramp_times, 'Region': region}] = ramp_values

            # zero-fill from global_t_start up to (but not including) region_start
            zero_times = np.arange(global_t_start, region_start)
            da_interp.loc[{dim: zero_times, 'Region': region}] = 0.0

    # --- Reattach unit ---
    if unit != prism.Unit('dimensionless'):
        da_interp = prism.Q_(da_interp, unit)

    return da_interp

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