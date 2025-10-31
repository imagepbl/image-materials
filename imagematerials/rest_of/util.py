import xarray as xr
import numpy as np

from imagematerials.concepts import KnowledgeGraph, Node

def sum_inflows_for_all_sectors(model, get_mfa_data: str, list_sum_sctors: list):
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
    arrays = []
    # get the inflow per sector
    if "buildings" in list_sum_sctors:
        inflow_buildings = model.buildings.get(get_mfa_data).to_array()
        arrays.append(inflow_buildings)
    if "vehicles" in list_sum_sctors:
        inflow_vehicles = model.vehicles.get(get_mfa_data).to_array()
        arrays.append(inflow_vehicles)
    if "generation" in list_sum_sctors:
        inflow_generation = model.generation.get(get_mfa_data).to_array()
        arrays.append(inflow_generation)
    if "grid" in list_sum_sctors:
        inflow_grid = model.grid.get(get_mfa_data).to_array()
        arrays.append(inflow_grid)
    if "grid_additional" in list_sum_sctors:
        inflow_grid_additional = model.grid_additional.get(get_mfa_data).to_array()
        arrays.append(inflow_grid_additional)
    if "storage_pumped_hydropower" in list_sum_sctors:
        inflow_storage_pumped_hydropower = model.storage_pumped_hydropower.get(get_mfa_data).to_array()
        arrays.append(inflow_storage_pumped_hydropower)
    if "storage_other" in list_sum_sctors:
        inflow_storage_other = model.storage_other.get(get_mfa_data).to_array()
        arrays.append(inflow_storage_other)

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



def save_sum_as_csv(total_inflow: xr.DataArray, material_name: str):
    # TODO: for now still use class_ 1, later change to country names
    # save steel
    inflow_steel = total_inflow.sel(material=material_name).sum('Type')
    # convert to tons with pint (this is what the fitting uses)
    inflow_steel = inflow_steel.pint.to('ton')
    # save with years as rows and regions as columns
    inflow_steel.name = f"inflow_{material_name}"

    # detect region dim name
    region_dim = "Region" if "Region" in inflow_steel.coords else "region"
    time_dim = "Time" if "Time" in inflow_steel.coords else "time"

    # convert to tidy DataFrame then pivot so index = time and columns = class_ 1..class_26
    df_tidy = inflow_steel.to_dataframe(name="value").reset_index()
    df_pivot = df_tidy.pivot(index=time_dim, columns=region_dim, values="value")

    # ensure exact column order and fill missing classes with 0
    desired = [f"class_ {i}" for i in range(1, 27)]
    df_pivot = df_pivot.reindex(columns=desired, fill_value=0)

    # name the index column "time" in the CSV header
    df_pivot.index.name = "time"
    
    assert int(df_pivot.sum().sum()) == int(inflow_steel.sum().sum())

    df_pivot.to_csv(f"../data/raw/rest-of/metals/image_materials_{material_name}.csv")