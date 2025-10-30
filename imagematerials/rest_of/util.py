import xarray as xr
import numpy as np

def sum_inflows_for_all_sectors(model, get_mfa_data: str):
    """
    Aggregate sector inflows into a single DataArray while preserving material and Type coordinates.

    Parameters
    ----------
    model :
        ImageMaterials model instance providing sector attributes (buildings, vehicles, generation, grid, ...)
        each exposing `.get(get_mfa_data).to_array()` returning an xarray.DataArray.
    get_mfa_data : str
        Key/name of the MFA data to extract from each sector (e.g. "inflow_materials").

    Returns
    -------
    xarray.DataArray
        Elementwise sum of the sector inflows. The 'material' and 'Type' coordinates are unioned
        across all sectors and missing combinations are filled with zeros. If a sector lacks a 'Type'
        coordinate its values are placed under a "_missing" Type entry.

    Notes
    -----
    - This function does not modify or align the Time coordinate; if sectors use different Time coords,
      align or reindex Time externally (or extend the function to include Time union/reindex).
    """
    
    # get the inflow per sector
    inflow_buildings = model.buildings.get(get_mfa_data).to_array()
    inflow_vehicles = model.vehicles.get(get_mfa_data).to_array()
    inflow_generation = model.generation.get(get_mfa_data).to_array()
    inflow_grid = model.grid.get(get_mfa_data).to_array()
    inflow_grid_additional = model.grid_additional.get(get_mfa_data).to_array()
    inflow_storage_pumped_hydropower = model.storage_pumped_hydropower.get(get_mfa_data).to_array()
    inflow_storage_other = model.storage_other.get(get_mfa_data).to_array()

    arrays = [inflow_buildings, inflow_vehicles, inflow_generation,
              inflow_grid, inflow_grid_additional,
              inflow_storage_pumped_hydropower, inflow_storage_other]

    # compute union coords
    all_materials = np.unique(np.concatenate([a.coords['material'].values for a in arrays]))
    all_types = np.unique(np.concatenate([
        a.coords['Type'].values if 'Type' in a.coords else np.array(['_missing']) for a in arrays
    ]))

    aligned = []
    for a in arrays:
        da = a
        # ensure dims exist
        if 'material' not in da.dims:
            da = da.expand_dims({'material': all_materials})
        if 'Type' not in da.dims:
            da = da.expand_dims({'Type': ['_missing']})
        # reindex to full union, fill missing with zeros
        da = da.reindex(material=all_materials, Type=all_types, fill_value=0)
        aligned.append(da)

    # elementwise sum of list (preserves coords)
    total_inflow = sum(aligned)

    return total_inflow