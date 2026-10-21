"""
Process material intensity data and transform to xarray DataArray for all processing stages separately.
"""

import xarray as xr
import pandas as pd

import prism

from imagematerials.electricity.utils import interpolate_xr
from imagematerials.fossil_fuels.constants import (
    EXTRACTION_TYPES,
    PIPELINE_TYPES,
    YEAR_FIRST_FF,
    END_YEAR
)

# TODO: since these functions are doing the same they should be replaced by one generalized function
# that takes the parameters as input that are individual (unit, name of array)

####################################################################################################
#%% Extraction stage (COAL, OIL, GAS)
####################################################################################################

def compute_extraction_materials(extraction_materials_data: pd.DataFrame) -> xr.DataArray:
    """Build the material intensity xarray for extraction types/technologies (Cohort x Type x material).

    For coal, oil and gas.
    
    Parameters
    ----------
    extraction_materials_data: material intensities, index (Year, Tech Type), one column
        per material (= inputs["materials"]["extraction"] from inputs.py)

    Returns
    -------
    extraction_materials_xr: material intensity in kg/kg, with dimensions ("Cohort", "Type", "material"):

    """
    # Convert to 3D array: (Material, Year, Tech)
    extraction_materials_xr = (
        extraction_materials_data
            .to_xarray()            
            .to_array("material") 
            .rename({"Year": "Cohort", "Tech Type": "Type"})
    ).transpose("Cohort", "Type", "material") # change order of dimensions

    extraction_materials_xr.name = "ExtractionMaterialIntensities"
    extraction_materials_xr = prism.Q_(extraction_materials_xr, "kg/kg") # add units
    extraction_materials_xr = interpolate_xr(extraction_materials_xr, YEAR_FIRST_FF, END_YEAR)
    # ensure order of types is in line with lifetimes and capacity
    extraction_materials_xr = extraction_materials_xr.reindex(Type=EXTRACTION_TYPES)

    return extraction_materials_xr


####################################################################################################
#%% Processing stage
####################################################################################################

def compute_processing_materials(processing_materials_data: pd.DataFrame) -> xr.DataArray:
    """Build the material intensity xarray for processing types/technologies (Cohort x Type x material).

    For oil, coal and gas processing/refinery.

    Parameters
    ----------
    processing_materials_data: material intensities, index (Year, Tech Type), one column
        per material (= inputs["materials"]["processing"] from inputs.py)

    Returns
    -------
    processing_materials_xr: material intensity in kg/kg, with dimensions ("Cohort", "Type", "material"):

    """
    # Convert to 3D array: (Material, Year, Tech)
    processing_materials_xr = (
        processing_materials_data
            .to_xarray()           # keeps Year & Tech Type as coords
            .to_array("material")  # converts columns into a 'Material' dimension
            .rename({"Year": "Cohort", "Tech Type": "Type"})
    ).transpose("Cohort", "Type", "material") # change order of dimensions

    processing_materials_xr.name = "ProcessingMaterialIntensities"
    processing_materials_xr = prism.Q_(processing_materials_xr, "kg/kg")
    processing_materials_xr = interpolate_xr(processing_materials_xr, YEAR_FIRST_FF, END_YEAR)

    return processing_materials_xr


####################################################################################################
#%% Storage stage
####################################################################################################

def compute_storage_materials(storage_materials_data: pd.DataFrame) -> xr.DataArray:
    """Build the material intensity xarray for storage types/technologies (Cohort x Type x material).

    For oil storage.

    Parameters
    ----------
    storage_materials_data: material intensities, index (Year, Tech Type), one column
        per material (= inputs["materials"]["storage"] from inputs.py)

    Returns
    -------
    storage_materials_xr: material intensity in kg/m3, with dimensions ("Cohort", "Type", "material"):
    
    """
    # Convert to 3D array: (Material, Year, Tech)
    storage_materials_xr = (
        storage_materials_data
            .to_xarray()           # keeps Year & Tech Type as coords
            .to_array("material")  # converts columns into a 'Material' dimension
            .rename({"Year": "Cohort", "Tech Type": "Type"})
    ).transpose("Cohort", "Type", "material") # change order of dimensions

    storage_materials_xr.name = "StorageMaterialIntensities"
    storage_materials_xr = prism.Q_(storage_materials_xr, "kg/meter**3") # add units
    storage_materials_xr = interpolate_xr(storage_materials_xr, YEAR_FIRST_FF, END_YEAR)

    return storage_materials_xr


####################################################################################################
# #%% Transport/Vehicles stage (coal, oil, gas)
####################################################################################################

def compute_transport_materials(transport_materials_data: pd.DataFrame) -> xr.DataArray:
    """Build the material intensity xarray for transport types/technologies (Cohort x Type x material).

    For oil, gas and coal: shipping, rail and trucks.

    Parameters
    ----------
    transport_materials_data: material intensities, index (Year, Tech Type), one column
        per material (= inputs["materials"]["transport"] from inputs.py)

    Returns
    -------
    transport_materials_xr: material intensity in kg/kg, with dimensions ("Cohort", "Type", "material"):
    
    """
    # Convert to 3D array: (Material, Year, Tech)
    transport_materials_xr = (
        transport_materials_data
            .to_xarray()           
            .to_array("material")  
            .rename({"Year": "Cohort", "Tech Type": "Type"})
    ).transpose("Cohort", "Type", "material") # change order of dimensions

    transport_materials_xr.name = "TransportMaterialIntensities"
    transport_materials_xr = prism.Q_(transport_materials_xr, "kg/kg")
    transport_materials_xr = interpolate_xr(transport_materials_xr, YEAR_FIRST_FF, END_YEAR)   

    return transport_materials_xr


####################################################################################################
#%% Pipelines stage (OIL, GAS)
####################################################################################################

def compute_pipelines_materials(pipelines_materials_data: pd.DataFrame) -> xr.DataArray:
    """Build the material intensity xarray for pipeline types/technologies (Cohort x Type x material).

    For oil and gas pipelines.

    Parameters
    ----------
    pipelines_materials_data: material intensities, index (Year, Tech Type), one column
        per material (= inputs["materials"]["pipelines"] from inputs.py)

    Returns
    -------
    pipelines_materials_xr: material intensity in kg/km, with dimensions ("Cohort", "Type", "material"):
    
    """
    # Convert to 3D array: (Material, Year, Tech)
    pipelines_materials_xr = (
        pipelines_materials_data
            .to_xarray()           
            .to_array("material")  
            .rename({"Year": "Cohort", "Tech Type": "Type"})
    ).transpose("Cohort", "Type", "material") # change order of dimensions

    pipelines_materials_xr.name = "PipelinesMaterialIntensities"
    pipelines_materials_xr = prism.Q_(pipelines_materials_xr, "kg/km")
    pipelines_materials_xr = interpolate_xr(pipelines_materials_xr, YEAR_FIRST_FF, END_YEAR)

    #Reorder technology coordinate in all three datasets (capacity, lifetime, material intensities) to match the same order of technologies (coal opencast, coal underground, gas offshore, gas onshore, oil offshore, oil onshore), so that we can easily combine the data in the model (since they all have to be in the same order of technologies)
    pipelines_materials_xr = pipelines_materials_xr.reindex(Type=PIPELINE_TYPES)

    return pipelines_materials_xr