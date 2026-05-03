#%%
import pandas as pd
import numpy as np
from pathlib import Path
import pint
import xarray as xr
import matplotlib.pyplot as plt
import warnings
from pint.errors import UnitStrippedWarning

import prism
from imagematerials.constants import IMAGE_REGIONS
from imagematerials.concepts import KnowledgeGraph, Node, create_image_region_graph
from imagematerials.electricity.constants import COLORS_IMAGE_REGIONS, IHA_REGIONS, iha_region_map, create_iha_region_graph
from imagematerials.electricity.utils import interpolate_xr
from imagematerials.read_mym import read_mym_df

path_current = Path().resolve()
path_base = path_current.parent.parent # base path of the project -> image-materials
path_data = Path(path_base, "data", "raw")

flag_phs = "phs_high"

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

# Dataset 3: IHA data on PHS projects under contruction, planned and announced per IHA region
df_data3 =  pd.read_csv(Path(path_data,"electricity","IHA_future_planned_PHS_capacity_MW_per_world_region.txt"),
                        usecols=["status", "Region", "value"])
unit_data3 = pd.read_csv(Path(path_data,"electricity","IHA_future_planned_PHS_capacity_MW_per_world_region.txt"),
                        usecols=["unit"])["unit"].iloc[0] # extract unit from the file (assumes it's the same for all rows)


# storage energy capacity (MWh, reservoir)
storage_energy = read_mym_df(Path(path_data, "image", "SSP2_baseline", "EnergyServices", "StorResTot.out"))
storage_energy = storage_energy.iloc[:, :26]
storage_energy.index.name = "Time"
storage_energy.columns.name = "Region"
storage_energy_xr = xr.DataArray(
    storage_energy.values,
    dims=["Time", "Region"],
    coords={
        "Time": storage_energy.index,
        "Region": [("region_" + str(r)) for r in storage_energy.columns]
    }
)
storage_energy_xr = knowledge_graph_region.rebroadcast_xarray(storage_energy_xr, output_coords=IMAGE_REGIONS, dim="Region") 



####################################################################################################
#%% Aggregate to IMAGE regions

knowledge_graph_region = create_image_region_graph()

ds_data1_image_region = ds_data1_country.copy()
ds_data1_image_region = knowledge_graph_region.aggregate_sum(ds_data1_image_region, output_coords=IMAGE_REGIONS, dim="Region", require_relation=False)

da_data2_image_region = da_data2_country.copy()
da_data2_image_region = knowledge_graph_region.aggregate_sum(da_data2_image_region, output_coords=IMAGE_REGIONS, dim="Region", require_relation=False)

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
dict_regional_start_years = {
    region.item(): int(val.item()) if not np.isnan(val.item()) else 2024
    for region, val in zip(earliest_cohort.Region, earliest_cohort)
}
regions_no_cohort = has_nonzero.Region[~has_nonzero].values.tolist()


#
# check which regions are 0 in all time steps
zero_regions = (da_data2_image_region == 0).all(dim="Time")
zero_regions.Region[zero_regions].values

# check only if 0 in 2014
zero_regions = (da_data2_image_region.sel(Time=2014) == 0)
zero_regions.Region[zero_regions].values


####################################################################################################
#%% intra and extrapolate phs data

#---------------------------------------------------------------------------------------------------
#%%% extrapolating historic data + constant extrapolation after last known data point (2024) until 2060
phs = interpolate_xr(da_data2_image_region, dict_regional_start_years, 2060, interp_method='linear', extrap_before='zero', extrap_before_method='logistic')

test = phs
regions= ["WAF", "TUR"]
regions = test.Region.values
fig, ax = plt.subplots(figsize=(14, 6))
for i, region in enumerate(regions): #test.Region.values
    ax.plot(test.Time.values, test.sel(Region=region).values, linewidth=1.5, color=COLORS_IMAGE_REGIONS[i], label=region)
ax.axvline(x=2014, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=2019, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=2024, color='gray', linestyle='--', alpha=0.5)
plt.xlabel("Year")
plt.ylabel("Pumped Hydropower Storage Capacity (MW)")
plt.title("Interpolated PHS capacity trajectories by IMAGE region")
plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8, ncol=2)
plt.tight_layout()

#---------------------------------------------------------------------------------------------------
#%%% calculate region shares

# select time
da = test.sel(Time=[2024])

# mapping

# create a coordinate for superregions
superregion = xr.DataArray(
    [iha_region_map[r] for r in da.Region.values],
    coords={"Region": da.Region},
    dims="Region"
)

# attach it
da = da.assign_coords(Superregion=superregion)

# compute shares within each superregion
shares = da.groupby("Superregion").map(lambda x: x / x.sum())
shares = shares.pint.dequantify()

years = np.arange(2024, 2061)
shares = shares.reindex(Time=years, method="ffill") # forward fill shares to future years (assumes shares remain constant after 2024)

# Adjust shares based on literature insights
override_regions = ["JAP", "INDO", "CHN", "OCE", "KOR", "SEAS"]
override_2030 = {
    "JAP": 0.12,
    "INDO": 0.05,
    "CHN": 0.7,
    "OCE": 0.05,
    "KOR": 0.03,
    "SEAS": 0.05
}
da_override = xr.full_like(
    shares.sel(Time=[2024, 2030]),
    fill_value=np.nan
)
for r in override_regions:
    da_override.loc[dict(Region=r, Time=2030)] = override_2030[r]
    da_override.loc[dict(Region=r, Time=2024)] = shares.sel(Time=2024, Region=r).values
# Interpolate only between 2024–2030
da_override = interpolate_xr(da_override, t_start=2024, t_end=2060, interp_method="linear")

# update shares
shares.loc[dict(Time=slice(2024, None), Region=override_regions)] = da_override.sel(Time=slice(2024, None), Region=override_regions)

df = shares.to_dataframe(name="value").reset_index()

#---------------------------------------------------------------------------------------------------
#%%% extrapolating future data (2024-2060)

def build_status_timeseries(df: pd.DataFrame, status: str, t_start: int, t_end: int, shares: xr.DataArray, unit: str) -> xr.DataArray:
    """ Build a region-level time series DataArray for a given PHS planning status.

    This function extracts values for a specific planning status from a dataframe, and constructs a 
    time-expanded xarray DataArray. The values are initialized at the start and end time points,
    interpolated over the full time range, and finally rebroadcast to a target
    regional classification using a region graph and weighting shares.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing at least the columns:
        - "status" (str): project status category
        - "Region" (str): region identifier (IHA region names)
        - "value" (numeric): associated value for each region-status pair

    status : str
        Status category to filter the dataframe ("under construction", "planned regulator approved",
        "planned pending approval" or "announced").

    t_start : int
        Start year of the time dimension.

    t_end : int
        End year of the time dimension.

    shares : xr.DataArray
        Regional share weights used for rebroadcasting values across the
        target regional classification (IMAGE_REGIONS).

    unit : str
        Unit of the values in the dataframe.

    Returns
    -------
    xr.DataArray
        A (Time × Region) DataArray representing the interpolated and
        regionally rebroadcast time series for the given status.
    """
    if status not in ["under construction", "planned regulator approved", "planned pending approval",
                       "announced"]:
        raise ValueError(f"Invalid status: {status}. Must be one of 'under construction', 'planned regulator approved', 'planned pending approval' or 'announced'.")
    
    # select the relevant row based on status
    df_status = df[df["status"] == status]
    values = (
        df_status.set_index("Region")
            .reindex(IHA_REGIONS)["value"]
            .fillna(0)
            .values
    )
    
    # create a new DataArray with Time dimension from t_start to t_end
    xr_status = xr.DataArray(
        data=np.vstack([np.zeros(len(IHA_REGIONS)), values]),
        coords={
            "Time": [t_start, t_end],
            "Region": IHA_REGIONS
        },
        dims=["Time", "Region"],
    )
    xr_status = interpolate_xr(xr_status, t_start, t_end)

    region_graph = create_iha_region_graph()
    xr_status = region_graph.rebroadcast_xarray(xr_status, output_coords=IMAGE_REGIONS, dim="Region", shares=shares)
    
    xr_status = prism.Q_(xr_status, unit)

    return xr_status

# build time series for each status category
xr_under_construction = build_status_timeseries(df_data3, "under construction", 2024, 2035, shares, unit_data3)
xr_planned_regulator_approved = build_status_timeseries(df_data3, "planned regulator approved", 2035, 2060, shares, unit_data3)
xr_planned_pending_approval = build_status_timeseries(df_data3, "planned pending approval", 2035, 2060, shares, unit_data3)
xr_announced = build_status_timeseries(df_data3, "announced", 2035, 2060, shares, unit_data3)

# flag_phs = "phs_high"
time = np.arange(phs.Time.min().item(), 2061)
if flag_phs == "phs_low":
    uc = xr_under_construction.reindex(Time=time, method="ffill").fillna(0)
    pra = xr_planned_regulator_approved.reindex(Time=time).fillna(0)
    phs_test = phs + uc + pra
elif flag_phs == "phs_high":
    uc = xr_under_construction.reindex(Time=time, method="ffill").fillna(0)
    pra = xr_planned_regulator_approved.reindex(Time=time).fillna(0)
    ppa = xr_planned_pending_approval.reindex(Time=time).fillna(0)
    ann = xr_announced.reindex(Time=time).fillna(0)
    phs_test = phs + uc + pra + ppa + ann

test = phs_test
regions= ["WAF", "TUR"]
regions = test.Region.values
fig, ax = plt.subplots(figsize=(14, 6))
for i, region in enumerate(regions): #test.Region.values
    ax.plot(test.Time.values, test.sel(Region=region).values, linewidth=1.5, color=COLORS_IMAGE_REGIONS[i], label=region)
ax.axvline(x=2014, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=2019, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=2024, color='gray', linestyle='--', alpha=0.5)
plt.xlabel("Year")
plt.ylabel("Pumped Hydropower Storage Capacity (MW)")
plt.title("Interpolated PHS capacity trajectories by IMAGE region")
plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8, ncol=2)
plt.tight_layout()

#---------------------------------------------------------------------------------------------------
#%% extrapolating future data (2060-2100)

# TEST TEST: 
# for the high PHS scenario, we assume that after 2060, the capacity continues to grow with the same rate as the total storage energy capacity demand. We assume no negative growth, in case the storage energy capacity demand decreases in the future, due to the high lifetime and upfront investment cost of PHS which make it unlikely to be decommissioned once built.
growth_rate = (
    (storage_energy_xr.sel(Time=slice(2060, 2100))
    .shift(Time=-1) / storage_energy_xr.sel(Time=slice(2060, 2100))) - 1
) # growth rate = f(t+1)/f(t) - 1
growth_rate = growth_rate.clip(min=0) # set negative growth rates to 0, assuming storage energy capacity does not decrease over time
growth_rate = growth_rate.shift(Time=1) # align growth rate with the correct time steps for the following multiplication (the growth rate calculated above is the growth from t to t+1, so if multiplicated with f(t) it should give f(t+1) with correct time dimension t+1)
# cumulative product of (1 + growth)
factor = (1 + growth_rate).cumulative("Time").prod().sel(Time=slice(2061, 2100))
# apply growth
phs_test_2061_to_2100 = phs_test.sel(Time=2060, drop=True) * factor
phs_test2 = xr.concat([phs_test.sel(Time=slice(None, 2060)), phs_test_2061_to_2100], dim="Time")


a = phs_test.sel(Time=slice(2020, 2060)) / storage_energy_xr.sel(Time=slice(2020, 2060))

test = a
# test = storage_energy_xr.sel(Time=slice(2015, 2100))
regions= test.Region.values
# regions = [r for r in test.Region.values if r not in ["WAF", "EAF","NAF", "BRA"]]
fig, ax = plt.subplots(figsize=(14, 6))
for i, region in enumerate(regions): #test.Region.values
    ax.plot(test.Time.values, test.sel(Region=region).values, linewidth=1.5, color=COLORS_IMAGE_REGIONS[i], label=region)
plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8, ncol=2)



#-----------------------------
flag_phs = "phs_low"
time = np.arange(phs.Time.min().item(), 2101)

if flag_phs == "phs_low":
    # for the low PHS scenario, we assume that after 2060, the capacity remains constant at the 2060 
    # level (i.e., no further growth after 2060)
    phs_test2 = phs_test.reindex(Time=time, method="ffill").fillna(0)
elif flag_phs == "phs_high":
    # for now the same as in the low scenario, but could be updated with a growth assumption
    phs_test2 = phs_test.reindex(Time=time, method="ffill").fillna(0)

test = phs_test2.sel(Time=slice(2015, 2100))
# test = storage_energy_xr.sel(Time=slice(2015, 2100))
regions= test.Region.values
# regions = [r for r in test.Region.values if r not in ["WAF", "EAF","NAF", "BRA"]]
fig, ax = plt.subplots(figsize=(14, 6))
for i, region in enumerate(regions): #test.Region.values
    ax.plot(test.Time.values, test.sel(Region=region).values, linewidth=1.5, color=COLORS_IMAGE_REGIONS[i], label=region)
plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8, ncol=2)

######
test = phs_test2.sel(Time=slice(2015, 2100))
test2 = storage_energy_xr.sel(Time=slice(2015, 2100))
regions= test.Region.values
# regions = [r for r in test.Region.values if r not in ["WAF", "EAF","NAF", "BRA"]]
fig, ax = plt.subplots(figsize=(14, 6))
for i, region in enumerate(["RUS"]): #test.Region.values
    ax.plot(test.Time.values, test.sel(Region=region).values, linewidth=1.5, color=COLORS_IMAGE_REGIONS[i], label=region)
    ax.plot(test2.Time.values, test2.sel(Region=region).values, linewidth=1.5, color=COLORS_IMAGE_REGIONS[i], label=region)
plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8, ncol=2)






















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