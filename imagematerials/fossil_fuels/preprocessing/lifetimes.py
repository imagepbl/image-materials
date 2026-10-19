#Outline (split into each stage (extraction, processing, transport, pipelines) for each fuel (coal, oil, gas)):
# 1) Import general modules and constants
# 2) read in necessary data files for lifetimes 
# 3) create xarray

#%% Import general modules and constants
import pint
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path

import prism

from imagematerials.concepts import create_fossil_fuel_graph, create_region_graph
from imagematerials.fossil_fuels.constants import SD_LIFETIME

year_start = 1880
year_end = 2100
year_out = 2100

knowledge_graph_region = create_region_graph()
fossil_fuel_knowledge_graph = create_fossil_fuel_graph()

####################################################################################################
# Extraction stage (coal, oil, gas)
####################################################################################################

def compute_extraction_lifetimes(extraction_lifetime_data):
    """Build the lifetime xarray (mean and stdev per cohort and technology).
    
    extraction_lifetime_data: lifetimes in years, index (Year, Tech Type),
    column "Lifetime" (= inputs["lifetimes"]["extraction"] from inputs.py)

    """
    types = extraction_lifetime_data.index.get_level_values("Tech Type").unique().astype(str).tolist()
    values = extraction_lifetime_data["Lifetime"].unstack().reindex(columns=types).to_numpy(dtype=float)
    times = extraction_lifetime_data.index.get_level_values("Year").unique().astype(int).tolist()

    scipy_params = ["mean", "stdev"]

    # Build xarray with shape (ScipyParam, Time, Type)
    data_array = np.stack([values, values * SD_LIFETIME], axis=0)
    # data_array = np.stack([values, np.full_like(values, np.nan)], axis=0)
    #create xarray with dimensions and coordinates
    extraction_lifetime_xr = xr.DataArray(
        data_array,
        dims=["DistributionParams", "Cohort", "Type"],
        coords={
            "DistributionParams": scipy_params,
            "Cohort": times,
            "Type": types
        },
        name="ExtractionLifetime"
    )

    # Expand 2020 across 1880-2100
    year_range = np.arange(1880, 2101)
    value_2020 = extraction_lifetime_xr.sel(Cohort=2020)
    extraction_lifetime_xr = value_2020.expand_dims(Cohort=year_range).transpose("DistributionParams", "Cohort", "Type")

    #Add units 
    extraction_lifetime_xr = prism.Q_(extraction_lifetime_xr, "year")
    #Set order of technology coordinate to match the order in the lifetime and material intensity data, so that we can easily combine the data in the model (since they all have to be in the same order of technologies)
    extraction_order = [
        'Coal Opencast',
        'Coal Underground',
        'Gas Offshore',
        'Gas Onshore',
        'Oil Offshore',
        'Oil Onshore'
    ]
    extraction_lifetime_xr = extraction_lifetime_xr.reindex(Type=extraction_order)

    return extraction_lifetime_xr

####################################################################################################
#%% Processing stage
####################################################################################################

# COAL preparation and OIL refinery) ---------------------------------------------------------------
def compute_processing_lifetimes(processing_lifetime_data):
    """Build the lifetime xarray (mean and stdev per cohort and technology).

    processing_lifetime_data: lifetimes in years, index (Year, Tech Type),
    column "Lifetime" (= inputs["lifetimes"]["processing"] from inputs.py)

    """
    values = processing_lifetime_data["Lifetime"].unstack().to_numpy(dtype=float)
    #Create coordinates
    times = processing_lifetime_data.index.get_level_values("Year").unique().astype(int).tolist()
    types = processing_lifetime_data.index.get_level_values("Tech Type").unique().astype(str).tolist()
    scipy_params = ["mean", "stdev"]
    # Build full array: shape (ScipyParam, Time, Type)
    data_array = np.stack([values, values * SD_LIFETIME], axis=0)
    # Create DataArray
    processing_lifetime_xr = xr.DataArray(
            data_array,
            dims=["DistributionParams", "Cohort", "Type"],
            coords={
                "DistributionParams": scipy_params,
                "Cohort": times,
                "Type": types
            },
            name="ProcessingLifetime"
        )
    # Expand 2020 across 1880-2100
    year_range = np.arange(1880, 2101)
    value_2020 = processing_lifetime_xr.sel(Cohort=2020)
    processing_lifetime_xr = value_2020.expand_dims(Cohort=year_range).transpose("DistributionParams", "Cohort", "Type")
    processing_lifetime_xr = prism.Q_(processing_lifetime_xr, "year")
 
    return processing_lifetime_xr

# OIL storage and GAS processing) ------------------------------------------------------------------

def compute_storage_lifetimes(storage_lifetime_data):
    """Build the lifetime xarray (mean and stdev per cohort and technology).
    
    storage_lifetime_data: lifetimes in years, index (Year, Tech Type),
    column "Lifetime" (= inputs["lifetimes"]["storage"] from inputs.py)

    """
    values = storage_lifetime_data["Lifetime"].unstack().to_numpy(dtype=float)
    #Create coordinates
    times = storage_lifetime_data.index.get_level_values("Year").unique().astype(int).tolist()
    types = storage_lifetime_data.index.get_level_values("Tech Type").unique().astype(str).tolist()
    scipy_params = ["mean", "stdev"]
        # Build full array: shape (ScipyParam, Time, Type)
    data_array = np.stack([values, values * SD_LIFETIME], axis=0)
    # data_array = np.stack([values, np.full_like(values, np.nan)], axis=0)
        # Create DataArray
    storage_lifetime_xr = xr.DataArray(
            data_array,
            dims=["DistributionParams", "Cohort", "Type"],
            coords={
                "DistributionParams": scipy_params,
                "Cohort": times,
                "Type": types
            },
            name="ProcessingStorageLifetime"
        )
    # Expand 2020 across 1880-2100
    year_range = np.arange(1880, 2101)
    value_2020 = storage_lifetime_xr.sel(Cohort=2020)
    storage_lifetime_xr = value_2020.expand_dims(Cohort=year_range).transpose("DistributionParams", "Cohort", "Type")
    storage_lifetime_xr = prism.Q_(storage_lifetime_xr, "year")

    return storage_lifetime_xr

####################################################################################################
#%% Transport/Vehicles stage (coal, oil, gas)
####################################################################################################

def compute_transport_lifetimes(transport_lifetime_data):
    """Build the lifetime xarray (mean and stdev per cohort and technology).

    transport_lifetime_data: lifetimes in years, index (Year, Tech Type),
    column "Lifetime" (= inputs["lifetimes"]["transport"] from inputs.py)

    """
    values = transport_lifetime_data["Lifetime"].unstack().to_numpy(dtype=float)
    #Create coordinates
    times = transport_lifetime_data.index.get_level_values("Year").unique().astype(int).tolist()
    types = transport_lifetime_data.index.get_level_values("Tech Type").unique().astype(str).tolist()
    scipy_params = ["mean", "stdev"]
    # Build full array: shape (ScipyParam, Time, Type)
    data_array = np.stack([values, values * SD_LIFETIME], axis=0)
    # Create DataArray
    transport_lifetime_xr = xr.DataArray(
            data_array,
            dims=["DistributionParams", "Cohort", "Type"],
            coords={
                "DistributionParams": scipy_params,
                "Cohort": times,
                "Type": types
            },
            name="TransportLifetime"
        )

    # Expand 2020 across 1880-2100
    year_range = np.arange(1880, 2101)
    value_2020 = transport_lifetime_xr.sel(Cohort=2020)
    transport_lifetime_xr = value_2020.expand_dims(Cohort=year_range).transpose("DistributionParams", "Cohort", "Type")
    transport_lifetime_xr = prism.Q_(transport_lifetime_xr, "year")

    return transport_lifetime_xr

####################################################################################################
#%% Pipelines stage (oil, gas) 
####################################################################################################

def compute_pipelines_lifetimes(pipelines_lifetime_data):
    """Build the lifetime xarray (mean and stdev per cohort and technology).

    pipelines_lifetime_data: lifetimes in years, index (Year, Tech Type),
    column "Lifetime" (= inputs["lifetimes"]["pipelines"] from inputs.py)

    """
    types = pipelines_lifetime_data.index.get_level_values("Tech Type").unique().astype(str).tolist()
    values = pipelines_lifetime_data["Lifetime"].unstack().reindex(columns=types).to_numpy(dtype=float)
    times = pipelines_lifetime_data.index.get_level_values("Year").unique().astype(int).tolist()

    scipy_params = ["mean", "stdev"]
    # Build full array: shape (ScipyParam, Time, Type)
    data_array = np.stack([values, values * SD_LIFETIME], axis=0)
    # Create DataArray
    pipelines_lifetime_xr = xr.DataArray(
            data_array,
            dims=["DistributionParams", "Cohort", "Type"],
            coords={
                "DistributionParams": scipy_params,
                "Cohort": times,
                "Type": types
            },
            name="PipelinesLifetime"
        )

    # Expand 2020 across 1880-2100
    year_range = np.arange(1880, 2101)
    value_2020 = pipelines_lifetime_xr.sel(Cohort=2020)
    pipelines_lifetime_xr = value_2020.expand_dims(Cohort=year_range).transpose("DistributionParams", "Cohort", "Type")
    pipelines_lifetime_xr = prism.Q_(pipelines_lifetime_xr, "year")

    pipeline_order = [
        'Gas Distribution Pipeline',
        'Gas Transmission Pipeline',
        'Oil Offshore Crude Pipeline',
        'Oil Onshore Crude Pipeline',
        'Oil Offshore Product Pipeline',
        'Oil Onshore Product Pipeline'
    ]

    #Reorder technology coordinate in all three datasets (capacity, lifetime, material intensities) to match the same order of technologies (coal opencast, coal underground, gas offshore, gas onshore, oil offshore, oil onshore), so that we can easily combine the data in the model (since they all have to be in the same order of technologies)
    pipelines_lifetime_xr = pipelines_lifetime_xr.reindex(Type=pipeline_order)

    return pipelines_lifetime_xr