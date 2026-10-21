"""
Process lifetime data and transform to xarray DataArray for all processing stages separately.
"""

import pint
import xarray as xr
import numpy as np
import pandas as pd

import prism

from imagematerials.electricity.utils import interpolate_xr
from imagematerials.fossil_fuels.constants import (
    SD_LIFETIME,
    EXTRACTION_TYPES,
    PIPELINE_TYPES,
    YEAR_FIRST_FF,
    END_YEAR
)


####################################################################################################
# Extraction stage (coal, oil, gas)
####################################################################################################

def compute_extraction_lifetimes(extraction_lifetime_data: pd.DataFrame) -> xr.DataArray:
    """Build the lifetime xarray of the extraction technologies.

    The lifetime of each technology is described by two parameters: the mean and the
    standard deviation. The mean comes from the input table, and the standard
    deviation is the mean multiplied by SD_LIFETIME (constants.py). The result is
    given per cohort (the year in which the infrastructure was built) and technology.

    Parameters
    ----------
    extraction_lifetime_data: Table with the columns "Year", "Tech Type" and
        "Lifetime" (mean lifetime in years), one row per year and technology, in
        any order (= inputs["lifetimes"]["extraction"] from inputs.py).

    Returns
    -------
    Lifetimes in years, with dimensions (DistributionParams, Cohort, Type):
        - DistributionParams: "mean" and "stdev"
        - Cohort: the years from YEAR_FIRST_FF to END_YEAR, interpolated from the
            years in the input table by interpolate_xr()
        - Type: the extraction technologies in the fixed order EXTRACTION_TYPES,
            which matches the capacity and material intensity data

    Notes
    -----
    - A technology name in the table but not in EXTRACTION_TYPES is dropped, and one
    in EXTRACTION_TYPES but missing from the table becomes NaN (effects of reindex).

    """
    extraction_lifetime_xr = (
        extraction_lifetime_data.set_index(["Year", "Tech Type"])["Lifetime"]
        .to_xarray()
        .rename({"Year": "Cohort", "Tech Type": "Type"})
    )
    extraction_lifetime_xr.name = "ExtractionLifetime"
    extraction_lifetime_xr = prism.Q_(extraction_lifetime_xr, "year") # add units

    # input data contain mean lifetimes, add standard deviation by multiplying mean with relative std
    extraction_lifetime_xr = extraction_lifetime_xr.expand_dims({"DistributionParams": ["mean", "stdev"]}).copy() # copy() needed as expand_dims only returns a read-only, broadcast view of the original array
    extraction_lifetime_xr.loc[dict(DistributionParams="stdev")] = extraction_lifetime_xr.loc[dict(DistributionParams="mean")] * SD_LIFETIME
    # interpolate over whole periode
    extraction_lifetime_xr = interpolate_xr(extraction_lifetime_xr, YEAR_FIRST_FF,END_YEAR)
    # ensure order of types is in line with material intensities
    extraction_lifetime_xr = extraction_lifetime_xr.reindex(Type=EXTRACTION_TYPES)

    return extraction_lifetime_xr


####################################################################################################
#%% Processing stage
####################################################################################################

def compute_processing_lifetimes(processing_lifetime_data: pd.DataFrame) -> xr.DataArray:
    """Build the lifetime xarray of the processing technologies (coal preparation, gas processing, oil refining).

    The lifetime of each technology is described by two parameters: the mean and the
    standard deviation. The mean comes from the input table, and the standard
    deviation is the mean multiplied by SD_LIFETIME (constants.py). The result is
    given per cohort (the year in which the infrastructure was built) and technology.

    Parameters
    ----------
    processing_lifetime_data: Table with the columns "Year", "Tech Type" and
        "Lifetime" (mean lifetime in years), one row per year and technology, in
        any order (= inputs["lifetimes"]["extraction"] from inputs.py).

    Returns
    -------
    Lifetimes in years, with dimensions (DistributionParams, Cohort, Type):
        - DistributionParams: "mean" and "stdev"
        - Cohort: the years from YEAR_FIRST_FF to END_YEAR, interpolated from the
            years in the input table by interpolate_xr()
        - Type: the processing types/technologies

    """
    processing_lifetime_xr = (
        processing_lifetime_data.set_index(["Year", "Tech Type"])["Lifetime"]
        .to_xarray()
        .rename({"Year": "Cohort", "Tech Type": "Type"})
    )
    processing_lifetime_xr.name = "ProcessingLifetime"
    processing_lifetime_xr = prism.Q_(processing_lifetime_xr, "year") # add units

    # input data contain mean lifetimes, add standard deviation by multiplying mean with relative std
    processing_lifetime_xr = processing_lifetime_xr.expand_dims({"DistributionParams": ["mean", "stdev"]}).copy() # copy() needed as expand_dims only returns a read-only, broadcast view of the original array
    processing_lifetime_xr.loc[dict(DistributionParams="stdev")] = processing_lifetime_xr.loc[dict(DistributionParams="mean")] * SD_LIFETIME
    # interpolate over whole periode
    processing_lifetime_xr = interpolate_xr(processing_lifetime_xr, YEAR_FIRST_FF,END_YEAR)
 
    return processing_lifetime_xr


####################################################################################################
#%% Storage stage
####################################################################################################

def compute_storage_lifetimes(storage_lifetime_data: pd.DataFrame) -> xr.DataArray:
    """Build the lifetime xarray of the storage technologies (oil storage).

    The lifetime of each technology is described by two parameters: the mean and the
    standard deviation. The mean comes from the input table, and the standard
    deviation is the mean multiplied by SD_LIFETIME (constants.py). The result is
    given per cohort (the year in which the infrastructure was built) and technology.

    Parameters
    ----------
    processing_lifetime_data: Table with the columns "Year", "Tech Type" and
        "Lifetime" (mean lifetime in years), one row per year and technology, in
        any order (= inputs["lifetimes"]["extraction"] from inputs.py).

    Returns
    -------
    Lifetimes in years, with dimensions (DistributionParams, Cohort, Type):
        - DistributionParams: "mean" and "stdev"
        - Cohort: the years from YEAR_FIRST_FF to END_YEAR, interpolated from the
            years in the input table by interpolate_xr()
        - Type: the storage type/tech

    """
    storage_lifetime_xr = (
        storage_lifetime_data.set_index(["Year", "Tech Type"])["Lifetime"]
        .to_xarray()
        .rename({"Year": "Cohort", "Tech Type": "Type"})
    )
    storage_lifetime_xr.name = "ProcessingLifetime"
    storage_lifetime_xr = prism.Q_(storage_lifetime_xr, "year") # add units

    # input data contain mean lifetimes, add standard deviation by multiplying mean with relative std
    storage_lifetime_xr = storage_lifetime_xr.expand_dims({"DistributionParams": ["mean", "stdev"]}).copy() # copy() needed as expand_dims only returns a read-only, broadcast view of the original array
    storage_lifetime_xr.loc[dict(DistributionParams="stdev")] = storage_lifetime_xr.loc[dict(DistributionParams="mean")] * SD_LIFETIME
    # interpolate over whole periode
    storage_lifetime_xr = interpolate_xr(storage_lifetime_xr, YEAR_FIRST_FF,END_YEAR)

    return storage_lifetime_xr


####################################################################################################
#%% Transport/Vehicles stage (coal, oil, gas)
####################################################################################################

def compute_transport_lifetimes(transport_lifetime_data: pd.DataFrame) -> xr.DataArray:
    """Build the lifetime xarray of the transport technologies (rail, ship, truck for coal, oil, gas).

    The lifetime of each technology is described by two parameters: the mean and the
    standard deviation. The mean comes from the input table, and the standard
    deviation is the mean multiplied by SD_LIFETIME (constants.py). The result is
    given per cohort (the year in which the infrastructure was built) and technology.

    Parameters
    ----------
    processing_lifetime_data: Table with the columns "Year", "Tech Type" and
        "Lifetime" (mean lifetime in years), one row per year and technology, in
        any order (= inputs["lifetimes"]["extraction"] from inputs.py).

    Returns
    -------
    Lifetimes in years, with dimensions (DistributionParams, Cohort, Type):
        - DistributionParams: "mean" and "stdev"
        - Cohort: the years from YEAR_FIRST_FF to END_YEAR, interpolated from the
            years in the input table by interpolate_xr()
        - Type: the transport types/techs

    """
    transport_lifetime_xr = (
        transport_lifetime_data.set_index(["Year", "Tech Type"])["Lifetime"]
        .to_xarray()
        .rename({"Year": "Cohort", "Tech Type": "Type"})
    )
    transport_lifetime_xr.name = "ProcessingLifetime"
    transport_lifetime_xr = prism.Q_(transport_lifetime_xr, "year") # add units

    # input data contain mean lifetimes, add standard deviation by multiplying mean with relative std
    transport_lifetime_xr = transport_lifetime_xr.expand_dims({"DistributionParams": ["mean", "stdev"]}).copy() # copy() needed as expand_dims only returns a read-only, broadcast view of the original array
    transport_lifetime_xr.loc[dict(DistributionParams="stdev")] = transport_lifetime_xr.loc[dict(DistributionParams="mean")] * SD_LIFETIME
    # interpolate over whole periode
    transport_lifetime_xr = interpolate_xr(transport_lifetime_xr, YEAR_FIRST_FF,END_YEAR)

    return transport_lifetime_xr


####################################################################################################
#%% Pipelines stage (oil, gas) 
####################################################################################################

def compute_pipelines_lifetimes(pipelines_lifetime_data: pd.DataFrame) -> xr.DataArray:
    """Build the lifetime xarray of pipelines (oil, gas).

    The lifetime of each technology is described by two parameters: the mean and the
    standard deviation. The mean comes from the input table, and the standard
    deviation is the mean multiplied by SD_LIFETIME (constants.py). The result is
    given per cohort (the year in which the infrastructure was built) and technology.

    Parameters
    ----------
    processing_lifetime_data: Table with the columns "Year", "Tech Type" and
        "Lifetime" (mean lifetime in years), one row per year and technology, in
        any order (= inputs["lifetimes"]["extraction"] from inputs.py).

    Returns
    -------
    Lifetimes in years, with dimensions (DistributionParams, Cohort, Type):
        - DistributionParams: "mean" and "stdev"
        - Cohort: the years from YEAR_FIRST_FF to END_YEAR, interpolated from the
            years in the input table by interpolate_xr()
        - Type: the pipeline types

    Notes
    -----
    - A technology name in the table but not in PIPELINE_TYPES is dropped, and one
    in PIPELINE_TYPES but missing from the table becomes NaN (effects of reindex).
    

    """
    pipelines_lifetime_xr = (
        pipelines_lifetime_data.set_index(["Year", "Tech Type"])["Lifetime"]
        .to_xarray()
        .rename({"Year": "Cohort", "Tech Type": "Type"})
    )
    pipelines_lifetime_xr.name = "ProcessingLifetime"
    pipelines_lifetime_xr = prism.Q_(pipelines_lifetime_xr, "year") # add units

    # input data contain mean lifetimes, add standard deviation by multiplying mean with relative std
    pipelines_lifetime_xr = pipelines_lifetime_xr.expand_dims({"DistributionParams": ["mean", "stdev"]}).copy() # copy() needed as expand_dims only returns a read-only, broadcast view of the original array
    pipelines_lifetime_xr.loc[dict(DistributionParams="stdev")] = pipelines_lifetime_xr.loc[dict(DistributionParams="mean")] * SD_LIFETIME
    # interpolate over whole periode
    pipelines_lifetime_xr = interpolate_xr(pipelines_lifetime_xr, YEAR_FIRST_FF,END_YEAR)
    
    #Reorder technology coordinate in all three datasets (capacity, lifetime, material intensities) 
    # to match the same order of technologies, so that we can easily combine the data in the model 
    # (since they all have to be in the same order of technologies)
    pipelines_lifetime_xr = pipelines_lifetime_xr.reindex(Type=PIPELINE_TYPES)

    return pipelines_lifetime_xr