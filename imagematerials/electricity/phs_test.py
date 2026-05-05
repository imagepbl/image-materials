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



def _build_status_timeseries_phs_data(df: pd.DataFrame, status: str, t_start: int, t_end: int, shares: xr.DataArray, unit: str) -> xr.DataArray:
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


####################################################################################################
#%% Load data

# Dataset 1: IHA data set of individual PHS sites, with commissioning year, capacity, energy stored,
# operating status, and hydro type
df_data1 =  pd.read_csv(Path(path_data,"electricity","iha_phs_capacity_data.csv"), 
                      usecols=["Operational Status", "Country", "Commisioning Year", "Hydro Type", 
                               "Generating Capacity", "Energy stored (GWh)"])

# Dataset 2: IHA World Hydropower Outlook dataset of aggregated PHS power capacity (MW) by country for 3 years (2014, 2019, 2024) 
#           + IEA global estimates
df_data2 =  pd.read_csv(Path(path_data,"electricity","iha_world_hydropower_outlook_phs_historic_stocks.csv"),
                        usecols=["Time", "unit", "Region", "value"])

# Dataset 3: IHA data on PHS projects under contruction, planned and announced per IHA region
df_data3 =  pd.read_csv(Path(path_data,"electricity","iha_future_planned_phs_capacity_mw_per_world_region.txt"),
                        usecols=["status", "unit", "Region", "value"])

# Manual adjustments to future shares based on literature insights, but authors estimation (unit: shares)
df_shares_adjustment_2030 =  pd.read_csv(Path(path_data,"electricity","phs_regional_shares_per_iha_region_in_2030.txt"),
                        usecols=["time", "Region", "value"])

# IMAGE-energy: storage energy capacity (MWh, reservoir)
storage_energy = read_mym_df(Path(path_data, "image", "SSP2_baseline", "EnergyServices", "StorResTot.out"))

knowledge_graph_region = create_image_region_graph()
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


data_phs = [df_data1, df_data2, df_data3, df_shares_adjustment_2030, storage_energy_xr]


#storage power capacity in MW
storage_power = read_mym_df(Path(path_data, "image", "SSP2_baseline", "EnergyServices", "StorCapTot.out"))

storage_power = storage_power.iloc[:, :26]
storage_power.index.name = "Time"
storage_power.columns.name = "Region"
storage_power_xr = xr.DataArray(
    storage_power.values,
    dims=["Time", "Region"],
    coords={
        "Time": storage_power.index,
        "Region": [("region_" + str(r)) for r in storage_power.columns]
    }
)
storage_power_xr = knowledge_graph_region.rebroadcast_xarray(storage_power_xr, output_coords=IMAGE_REGIONS, dim="Region") 

# Constants ----------------------------------------------------------------------------------------
factor_phs_growth_rel_demand = 0.5
# Assumption on average discharge duration of PHS plants in hours. The discharge duration can vary 
# widely in reality, typically ranging 6-24 h. This simplifying assumption is needed due to the lack
# of data on installed storage energy capacity of PHS plants (MWh). Since the stock model is done in 
# terms of energy capacity, we need to convert the available power capacity data (MW) to energy capacity.
# Note: important for us is not the total energy produced by the PHS plant/energy cycled in a year, 
# but the energy capacity of the reservoir, which determines how much energy can be stored at a given 
# time as well as the materials needed for the reservoir construction (size of the reservoir).
mean_discharge_duration = 10

####################################################################################################
####################################################################################################
#%% Run as script

################################################################################################
# Pretreat data

df_data1 = df_data1.loc[df_data1["Operational Status"] == "Operational"]
df_data1 = df_data1.rename(columns={'Country': 'Region', 'Commisioning Year': 'Cohort'})
df_data1["Cohort"] = pd.to_numeric(df_data1["Cohort"], errors="coerce").astype("Int64")
df_data1_country = df_data1.groupby(["Region", "Cohort"])[
    ["Generating Capacity", "Energy stored (GWh)"]
].sum() # rows with NaN in Region or Cohort are dropped during this step
ds_data1_country = df_data1_country.to_xarray()

unit_data2 = df_data2["unit"].iloc[0] # extract unit from the file (assumes it's the same for all rows)
df_data2 = df_data2.drop(columns="unit")
df_data2_country = df_data2.groupby(["Region", "Time"])[["value"]].sum()
da_data2_country = xr.DataArray.from_series(df_data2_country["value"]).rename("PumpedHydropowerStorageCapacity") # create Dataarray
da_data2_country = prism.Q_(da_data2_country, unit_data2)

unit_data3 = df_data3["unit"].iloc[0] # extract unit from the file (assumes it's the same for all rows)
df_data3 = df_data3.drop(columns="unit")
mean_discharge_duration = prism.Q_(mean_discharge_duration, "hour")

####################################################################################################
# Aggregate to IMAGE regions

ds_data1_image_region = ds_data1_country.copy()
ds_data1_image_region = knowledge_graph_region.aggregate_sum(ds_data1_image_region, output_coords=IMAGE_REGIONS, dim="Region", require_relation=False)

da_data2_image_region = da_data2_country.copy()
da_data2_image_region = knowledge_graph_region.aggregate_sum(da_data2_image_region, output_coords=IMAGE_REGIONS, dim="Region", require_relation=False)

####################################################################################################
# Find starting year per region

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

####################################################################################################
# intra and extrapolate phs data

#---------------------------------------------------------------------------------------------------
# extrapolating historic data (+ constant extrapolation after last known data point (2024) until 2060, overwritten later)
phs = interpolate_xr(da_data2_image_region, dict_regional_start_years, 2060, interp_method='linear', extrap_before='zero', extrap_before_method='logistic')


#---------------------------------------------------------------------------------------------------
# calculate region shares

# select time
da = phs.sel(Time=[2024])
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
shares = shares.reindex(Time=np.arange(2024, 2061), method="ffill") # forward fill shares to future years (assumes shares remain constant after 2024)

# Adjust shares based on literature insights
override_regions = df_shares_adjustment_2030["Region"].values
time_override = 2030
da_override = xr.full_like(
    shares.sel(Time=[2024, 2030]),
    fill_value=np.nan
)
for r in override_regions:
    da_override.loc[dict(Region=r, Time=time_override)] = df_shares_adjustment_2030.loc[(df_shares_adjustment_2030["time"] == time_override) & (df_shares_adjustment_2030["Region"] == r), "value"].values[0] #override_2030[r]
    da_override.loc[dict(Region=r, Time=2024)] = shares.sel(Time=2024, Region=r).values
# Interpolate shares (linearly between 2024–2030, constant after 2030)
da_override = interpolate_xr(da_override, t_start=2024, t_end=2060, interp_method="linear")

# update shares
shares.loc[dict(Time=slice(2024, None), Region=override_regions)] = da_override.sel(Time=slice(2024, None), Region=override_regions)

#---------------------------------------------------------------------------------------------------
# extrapolating future data (2024-2060)

# build time series for each operational status category
xr_under_construction = _build_status_timeseries_phs_data(df_data3, "under construction", 2024, 2035, shares, unit_data3)
xr_planned_regulator_approved = _build_status_timeseries_phs_data(df_data3, "planned regulator approved", 2035, 2060, shares, unit_data3)
xr_planned_pending_approval = _build_status_timeseries_phs_data(df_data3, "planned pending approval", 2035, 2060, shares, unit_data3)
xr_announced = _build_status_timeseries_phs_data(df_data3, "announced", 2035, 2060, shares, unit_data3)

# flag_phs = "phs_high"
time = np.arange(phs.Time.min().item(), 2061)
if flag_phs == "phs_low":
    uc = xr_under_construction.reindex(Time=time, method="ffill").fillna(0)
    pra = xr_planned_regulator_approved.reindex(Time=time).fillna(0)
    phs_2060 = phs + uc + pra
elif flag_phs == "phs_high":
    uc = xr_under_construction.reindex(Time=time, method="ffill").fillna(0)
    pra = xr_planned_regulator_approved.reindex(Time=time).fillna(0)
    ppa = xr_planned_pending_approval.reindex(Time=time).fillna(0)
    ann = xr_announced.reindex(Time=time).fillna(0)
    phs_2060 = phs + uc + pra + ppa + ann

#---------------------------------------------------------------------------------------------------
# extrapolating future data (2060-2100)

if flag_phs == "phs_low":
    # for the low PHS scenario, we assume that after 2060, the capacity remains constant at the 2060 
    # level (i.e., no further growth after 2060)
    phs_power = phs_2060.reindex(Time=np.arange(phs.Time.min().item(), 2101), method="ffill").fillna(0)
elif flag_phs == "phs_high":
    # --- select relevant time range ---
    # this factor determines how strongly the growth of PHS capacity is linked to the growth of storage 
    # energy demand; a value of 0.5 means that if the storage energy demand grows by 10% from one year 
    # to the next, the PHS capacity will grow by 5% in that year, if the condition below is met. This is 
    # a simplifying assumption and can be adjusted based on literature insights or sensitivity analysis. 
    phs_temporary = phs_2060.sel(Time=[2060]).reindex(Time=np.arange(2060, 2101), method="ffill") # forward fill from 2060 to 2100 with constant values (will be updated with growth assumption in the next steps)
    demand = storage_energy_xr.sel(Time=slice(2060, 2100))
    # align sequence of dimensions
    phs_temporary = phs_temporary.transpose("Time", "Region")
    demand = demand.transpose("Time", "Region")

    # --- compute base growth rates (NumPy) ---
    # growth rate = f(t+1)/f(t) - 1, by rolling -1 values for year 2061 are now saved under year 2060, 
    # so when we divide by demand_vals which are still aligned with the original years, we get the 
    # growth rate from t to t+1 aligned with year t. 
    demand_vals = demand.values
    growth_rate = (np.roll(demand_vals, -1, axis=0) / demand_vals) - 1 
    # last timestep has no forward value → set to 0
    growth_rate[-1, ...] = 0
    # clip negative growth
    growth_rate = np.clip(growth_rate, 0, None)
    # shift to align with t+1
    # growth_rate = np.roll(growth_rate, 1, axis=0)
    growth_rate[0, ...] = 0  # first year has no previous growth

    # --- initialize result array ---
    phs_vals = phs_temporary.values.copy()

    # --- recursive loop over time ---
    for t in range(1, phs_vals.shape[0]):
        prev = phs_vals[t - 1]
        gr = growth_rate[t - 1]
        # condition: grow only if previous PHS <= 0.8 * demand at t-1
        mask = prev <= 0.8 * demand_vals[t - 1]
        # apply growth selectively
        phs_vals[t] = np.where(mask, prev * (1 + factor_phs_growth_rel_demand * gr), prev)

    # --- convert back to xarray ---
    phs_updated = xr.DataArray(
        phs_vals,
        coords=phs_temporary.coords,
        dims=phs_temporary.dims,
        name=phs_temporary.name
    )
    phs_updated = prism.Q_(phs_updated, "MW")
    # --- merge with original before 2060 ---
    phs_power = xr.concat(
        [phs_2060.sel(Time=slice(None, 2059)), phs_updated],
        dim="Time"
    )


#---------------------------------------------------------------------------------------------------
# convert power capacity (MW) to energy capacity (MWh)

phs_energy = phs_power * mean_discharge_duration

# phs_test2_energy_high = phs_test2_energy.copy()




####################################################################################################
####################################################################################################
#%% As function

def derive_phs_installed_capacity(data: list,
                                  factor_phs_growth_rel_demand: int = 0.5,
                                  mean_discharge_duration: int = 10,
                                  flag_phs: str = "phs_low"):
    
    df_data1 = data[0]
    df_data2 = data[1]
    df_data3 = data[2]
    df_shares_adjustment_2030 = data[3]
    storage_energy_xr = data[4]

    ################################################################################################
    # Pretreat data 
    
    df_data1 = df_data1.loc[df_data1["Operational Status"] == "Operational"]
    df_data1 = df_data1.rename(columns={'Country': 'Region', 'Commisioning Year': 'Cohort'})
    df_data1["Cohort"] = pd.to_numeric(df_data1["Cohort"], errors="coerce").astype("Int64")
    df_data1_country = df_data1.groupby(["Region", "Cohort"])[
        ["Generating Capacity", "Energy stored (GWh)"]
    ].sum() # rows with NaN in Region or Cohort are dropped during this step
    ds_data1_country = df_data1_country.to_xarray()

    unit_data2 = df_data2["unit"].iloc[0] # extract unit from the file (assumes it's the same for all rows)
    df_data2 = df_data2.drop(columns="unit")
    df_data2_country = df_data2.groupby(["Region", "Time"])[["value"]].sum()
    da_data2_country = xr.DataArray.from_series(df_data2_country["value"]).rename("PumpedHydropowerStorageCapacity") # create Dataarray
    da_data2_country = prism.Q_(da_data2_country, unit_data2)
    
    unit_data3 = df_data3["unit"].iloc[0] # extract unit from the file (assumes it's the same for all rows)
    df_data3 = df_data3.drop(columns="unit")

    mean_discharge_duration = prism.Q_(mean_discharge_duration, "hour")
    
    ################################################################################################
    # Aggregate to IMAGE regions

    ds_data1_image_region = ds_data1_country.copy()
    ds_data1_image_region = knowledge_graph_region.aggregate_sum(ds_data1_image_region, output_coords=IMAGE_REGIONS, dim="Region", require_relation=False)

    da_data2_image_region = da_data2_country.copy()
    da_data2_image_region = knowledge_graph_region.aggregate_sum(da_data2_image_region, output_coords=IMAGE_REGIONS, dim="Region", require_relation=False)

    ################################################################################################
    # Find starting year per region

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

    ####################################################################################################
    # intra and extrapolate phs data

    #---------------------------------------------------------------------------------------------------
    # extrapolating historic data (+ constant extrapolation after last known data point (2024) until 2060, overwritten later)
    phs = interpolate_xr(da_data2_image_region, dict_regional_start_years, 2060, interp_method='linear', extrap_before='zero', extrap_before_method='logistic')

    #---------------------------------------------------------------------------------------------------
    # calculate region shares

    # select time
    da = phs.sel(Time=[2024])
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
    shares = shares.reindex(Time=np.arange(2024, 2061), method="ffill") # forward fill shares to future years (assumes shares remain constant after 2024)

    # Adjust shares based on literature insights
    override_regions = df_shares_adjustment_2030["Region"].values
    time_override = 2030
    da_override = xr.full_like(
        shares.sel(Time=[2024, 2030]),
        fill_value=np.nan
    )
    for r in override_regions:
        da_override.loc[dict(Region=r, Time=time_override)] = df_shares_adjustment_2030.loc[(df_shares_adjustment_2030["time"] == time_override) & (df_shares_adjustment_2030["Region"] == r), "value"].values[0] #override_2030[r]
        da_override.loc[dict(Region=r, Time=2024)] = shares.sel(Time=2024, Region=r).values
    # Interpolate shares (linearly between 2024–2030, constant after 2030)
    da_override = interpolate_xr(da_override, t_start=2024, t_end=2060, interp_method="linear")

    # update shares
    shares.loc[dict(Time=slice(2024, None), Region=override_regions)] = da_override.sel(Time=slice(2024, None), Region=override_regions)

    #---------------------------------------------------------------------------------------------------
    # extrapolating future data (2024-2060)

    # build time series for each operational status category
    xr_under_construction = _build_status_timeseries_phs_data(df_data3, "under construction", 2024, 2035, shares, unit_data3)
    xr_planned_regulator_approved = _build_status_timeseries_phs_data(df_data3, "planned regulator approved", 2035, 2060, shares, unit_data3)
    xr_planned_pending_approval = _build_status_timeseries_phs_data(df_data3, "planned pending approval", 2035, 2060, shares, unit_data3)
    xr_announced = _build_status_timeseries_phs_data(df_data3, "announced", 2035, 2060, shares, unit_data3)

    time = np.arange(phs.Time.min().item(), 2061)
    if flag_phs == "phs_low":
        uc = xr_under_construction.reindex(Time=time, method="ffill").fillna(0)
        pra = xr_planned_regulator_approved.reindex(Time=time).fillna(0)
        phs_2060 = phs + uc + pra
    elif flag_phs == "phs_high":
        uc = xr_under_construction.reindex(Time=time, method="ffill").fillna(0)
        pra = xr_planned_regulator_approved.reindex(Time=time).fillna(0)
        ppa = xr_planned_pending_approval.reindex(Time=time).fillna(0)
        ann = xr_announced.reindex(Time=time).fillna(0)
        phs_2060 = phs + uc + pra + ppa + ann

    #---------------------------------------------------------------------------------------------------
    # extrapolating future data (2060-2100)

    if flag_phs == "phs_low":
        # for the low PHS scenario, we assume that after 2060, the capacity remains constant at the 2060 
        # level (i.e., no further growth after 2060)
        phs_power = phs_2060.reindex(Time=np.arange(phs.Time.min().item(), 2101), method="ffill").fillna(0)
    elif flag_phs == "phs_high":
        # --- select relevant time range ---
        # this factor determines how strongly the growth of PHS capacity is linked to the growth of storage 
        # energy demand; a value of 0.5 means that if the storage energy demand grows by 10% from one year 
        # to the next, the PHS capacity will grow by 5% in that year, if the condition below is met. This is 
        # a simplifying assumption and can be adjusted based on literature insights or sensitivity analysis. 
        phs_temporary = phs_2060.sel(Time=[2060]).reindex(Time=np.arange(2060, 2101), method="ffill") # forward fill from 2060 to 2100 with constant values (will be updated with growth assumption in the next steps)
        demand = storage_energy_xr.sel(Time=slice(2060, 2100))
        # align sequence of dimensions
        phs_temporary = phs_temporary.transpose("Time", "Region")
        demand = demand.transpose("Time", "Region")

        # --- compute base growth rates (NumPy) ---
        # growth rate = f(t+1)/f(t) - 1, by rolling -1 values for year 2061 are now saved under year 2060, 
        # so when we divide by demand_vals which are still aligned with the original years, we get the 
        # growth rate from t to t+1 aligned with year t. 
        demand_vals = demand.values
        growth_rate = (np.roll(demand_vals, -1, axis=0) / demand_vals) - 1 
        # last timestep has no forward value → set to 0
        growth_rate[-1, ...] = 0
        # clip negative growth
        growth_rate = np.clip(growth_rate, 0, None)
        # shift to align with t+1
        # growth_rate = np.roll(growth_rate, 1, axis=0)
        growth_rate[0, ...] = 0  # first year has no previous growth

        # --- initialize result array ---
        phs_vals = phs_temporary.values.copy()

        # --- recursive loop over time ---
        for t in range(1, phs_vals.shape[0]):
            prev = phs_vals[t - 1]
            gr = growth_rate[t - 1]
            # condition: grow only if previous PHS <= 0.8 * demand at t-1
            mask = prev <= 0.8 * demand_vals[t - 1]
            # apply growth selectively
            phs_vals[t] = np.where(mask, prev * (1 + factor_phs_growth_rel_demand * gr), prev)

        # --- convert back to xarray ---
        phs_updated = xr.DataArray(
            phs_vals,
            coords=phs_temporary.coords,
            dims=phs_temporary.dims,
            name=phs_temporary.name
        )
        phs_updated = prism.Q_(phs_updated, "MW")
        # --- merge with original before 2060 ---
        phs_power = xr.concat(
            [phs_2060.sel(Time=slice(None, 2059)), phs_updated],
            dim="Time"
        )

    #---------------------------------------------------------------------------------------------------
    # convert power capacity (MW) to energy capacity (MWh)

    phs_energy = phs_power * mean_discharge_duration

    return phs_power, phs_energy




phs_power_low, phs_energy_low = derive_phs_installed_capacity(data_phs, factor_phs_growth_rel_demand, 20, "phs_low") #mean_discharge_duration
phs_power_high, phs_energy_high = derive_phs_installed_capacity(data_phs, factor_phs_growth_rel_demand, 20, "phs_high")


####################################################################################################
#%% Plots




#%%%% COMPARISON - GWh -----------------------------------------------------------------------------

phs_low_world = phs_energy_low.sel(Time=slice(2000, 2100)).sum(dim="Region").pint.to("GWh")
phs_high_world = phs_energy_high.sel(Time=slice(2000, 2100)).sum(dim="Region").pint.to("GWh")
timer = storage_energy_xr.sel(Time=slice(2000, 2100)).sum(dim="Region")
timer = prism.Q_(timer, "MWh")
timer = timer.pint.to("GWh")

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(phs_low_world.Time.values, phs_low_world.values, linewidth=1.5, color="black", label="PHS - low scenario")
ax.plot(phs_high_world.Time.values, phs_high_world.values, linewidth=1.5, color="grey", label="PHS - high scenario")
ax.plot(timer.Time.values, timer.values, linewidth=1.5, color="black", linestyle="--", label="TIMER")
ax.scatter(2017, 4508, marker="v",s=100, color="#06DE8B", label="IRENA")
ax.scatter(2030, 6068, marker="v",s=100, color="#A0F3D3", label="IRENA-REmap_Doubling_min")
ax.scatter(2030, 6848, marker="v",s=100, color="#508C75", label="IRENA-REmap_Doubling_max")

plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8, ncol=2)
ax.set_xlabel("Time", fontsize=14)
ax.set_ylabel("Installed energy capacity (GWh)", fontsize=14)
ax.set_title("Global PHS Energy Capacity (GWh)", fontweight="bold")
ax.legend(title="Source")
ax.grid(True, linestyle="--", alpha=0.3)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()


#%%%% COMPARISON - GW ------------------------------------------------------------------------------

iha_world = da_data2_image_region.sum("Region").pint.to("GW")
timer = storage_power_xr.sel(Time=slice(2000, 2100)).sum(dim="Region")
timer = prism.Q_(timer, "MW")
timer = timer.pint.to("GW")
phs_low_world = phs_power_low.sel(Time=slice(2000, 2100)).sum(dim="Region").pint.to("GW")
phs_high_world = phs_power_high.sel(Time=slice(2000, 2100)).sum(dim="Region").pint.to("GW")

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(phs_low_world.Time.values, phs_low_world.values, linewidth=1.5, color="black", label="PHS - low scenario")
ax.plot(phs_high_world.Time.values, phs_high_world.values, linewidth=1.5, color="grey", label="PHS - high scenario")
# ax.plot(timer.Time.values, timer.values, linewidth=1.5, color="black", linestyle="--", label="TIMER")
ax.scatter([2021,2023], [160,181], marker="v",s=100, color="#BD09CA", label="IEA")
ax.scatter([2030], [249], marker="v",s=100, color="#F18BF9", label="IEA-STEPS")
ax.scatter([2014], [142], marker="v",s=100, color="#06DE8B", label="IRENA")
ax.scatter(iha_world.Time.values, iha_world.values, marker="v",s=100, color="#3B7EFA", label="IHA")

plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8, ncol=2)
ax.set_xlabel("Time", fontsize=14)
ax.set_ylabel("Installed capacity (GW)", fontsize=14)
ax.set_title("Global PHS Power Capacity (GW)", fontweight="bold")
ax.legend(title="Source")
ax.grid(True, linestyle="--", alpha=0.3)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()



#%%%%
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


#%%%%

test = phs_test2_energy.sel(Time=slice(2000, 2100))
# test = storage_energy_xr.sel(Time=slice(2015, 2100))
regions= test.Region.values
# regions = [r for r in test.Region.values if r not in ["WAF", "EAF","NAF", "BRA"]]
fig, ax = plt.subplots(figsize=(14, 6))
for i, region in enumerate(regions): #test.Region.values
    ax.plot(test.Time.values, test.sel(Region=region).values, linewidth=1.5, color=COLORS_IMAGE_REGIONS[i], label=region)
plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8, ncol=2)

#%%%%
test = phs_test2_energy.sel(Time=slice(2015, 2100))
test2 = storage_energy_xr.sel(Time=slice(2015, 2100))
regions= test.Region.values
# regions = [r for r in test.Region.values if r not in ["WAF", "EAF","NAF", "BRA"]]
fig, ax = plt.subplots(figsize=(14, 6))
for i, region in enumerate(["CHN"]): #test.Region.values
    ax.plot(test.Time.values, test.sel(Region=region).values, linewidth=1.5, color=COLORS_IMAGE_REGIONS[i], label=region+" PHS capacity")
    ax.plot(test2.Time.values, test2.sel(Region=region).values, linewidth=1.5, color=COLORS_IMAGE_REGIONS[i], label=region+" storage energy demand", linestyle="--")
plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8, ncol=2)

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