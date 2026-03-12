import xarray as xr
import numpy as np
from pathlib import Path
from imagematerials.concepts import KnowledgeGraph, Node

from imagematerials.rest_of.const import numeric_region_map

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
    if "elc_gen" in list_sum_sctors:
        inflow_elc_gen = model.elc_gen.get(get_mfa_data).to_array()
        arrays.append(inflow_elc_gen)
    if "elc_grid_lines" in list_sum_sctors:
        inflow_elc_grid_lines = model.elc_grid_lines.get(get_mfa_data).to_array()
        arrays.append(inflow_elc_grid_lines)
    if "elc_grid_add" in list_sum_sctors:
        inflow_elc_grid_add = model.elc_grid_add.get(get_mfa_data).to_array()
        arrays.append(inflow_elc_grid_add)
    if "elc_stor_phs" in list_sum_sctors:
        inflow_elc_stor_phs = model.elc_stor_phs.get(get_mfa_data).to_array()
        arrays.append(inflow_elc_stor_phs)
    if "elc_stor_other" in list_sum_sctors:
        inflow_elc_stor_other = model.elc_stor_other.get(get_mfa_data).to_array()
        arrays.append(inflow_elc_stor_other)

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


def calculate_cement_equivalent(total_inflow: xr.DataArray, include_rest_of = True):
        from imagematerials.rest_of.const import cement_in_concrete_factor
        concrete = total_inflow.sel(material = 'concrete')
        if include_rest_of is True:
            cement_rest_of = total_inflow.sel(material = 'cement')
        # convert concrete to cement equivalent
        cement_in_concrete = concrete * cement_in_concrete_factor
        # change coordinate of material to 'cement'
        cement_in_concrete = cement_in_concrete.assign_coords(material = 'cement')
        if include_rest_of is True:
            inflow_material = cement_in_concrete + cement_rest_of
        else:
            inflow_material = cement_in_concrete
        print("summed cement and cement in concrete.")
        return inflow_material


def sand_gravel_crushed_rock_equivalent(total_inflow: xr.DataArray, rename= False, sand_available = False, include_rest_bool = True):
 
        from imagematerials.rest_of.const import sand_in_cement_conversion, sand_in_glass_conversion
        inflow_sand_in_cement = calculate_cement_equivalent(total_inflow, include_rest_of=include_rest_bool)*sand_in_cement_conversion
        inflow_sand_in_glass = total_inflow.sel(material = 'glass') * sand_in_glass_conversion
        # rename coord material to 'sand'
        if rename == False:
            inflow_sand_in_cement = inflow_sand_in_cement.assign_coords(material = 'sand_gravel_crushed_rock')
            inflow_sand_in_glass = inflow_sand_in_glass.assign_coords(material = 'sand_gravel_crushed_rock')
        if rename == True:
            inflow_sand_in_cement = inflow_sand_in_cement.assign_coords(material = 'sand')
            inflow_sand_in_glass = inflow_sand_in_glass.assign_coords(material = 'sand')
        
        if sand_available == True:
            sand = total_inflow.sel(material = 'sand')
            inflow_material = inflow_sand_in_cement + inflow_sand_in_glass + sand
        else:
            inflow_material = inflow_sand_in_cement + inflow_sand_in_glass
        print("summed sand in cement and sand in glass.")
        return inflow_material


def save_sum_as_csv(total_inflow: xr.DataArray, material_name: str, 
                    save_path: Path):
    # detect Region coord name (case variations) and apply mapping
    dim = "Region" if "Region" in total_inflow.coords else "region"
    old_regions = [str(r) for r in total_inflow.coords[dim].values]

    # ensure mapping keys are strings
    map_str = {str(k): v for k, v in numeric_region_map.items()}
    new_labels = [map_str.get(r, r) for r in old_regions]

    # assign new labels, aggregate duplicates, then ensure class_ 1..class_26 exist
    total_inflow = total_inflow.assign_coords({dim: ("Region", new_labels)})  # keep dim name consistent
    total_inflow = total_inflow.groupby(dim).sum()

    desired = [f"class_ {i}" for i in range(1, 27)]
    total_inflow = total_inflow.reindex({dim: desired}, fill_value=0)

    # TODO: for now still use class_ 1, later change to country names
    # save material

    if material_name == "cement":
        inflow_material = calculate_cement_equivalent(total_inflow)

    if material_name == "sand_gravel_crushed_rock":
        inflow_material = sand_gravel_crushed_rock_equivalent(total_inflow)
    
    else:
        inflow_material = total_inflow.sel(material=material_name)
    
    # reduce Type dimension by summing over it
    inflow_material = inflow_material.sum('Type')
    inflow_compare = inflow_material.copy()
    inflow_compare = inflow_compare.pint.to('ton')
    # convert to tons with pint (this is what the fitting uses)
    inflow_material = inflow_material.pint.to('ton')
    # save with years as rows and regions as columns
    inflow_material.name = f"inflow_{material_name}"

    # detect region dim name
    region_dim = "Region" if "Region" in inflow_material.coords else "region"
    time_dim = "Time" if "Time" in inflow_material.coords else "time"

    # convert to tidy DataFrame then pivot so index = time and columns = class_ 1..class_26
    df_tidy = inflow_material.to_dataframe(name="value").reset_index()
    df_pivot = df_tidy.pivot(index=time_dim, columns=region_dim, values="value")

    # ensure exact column order and fill missing classes with 0
    desired = [f"class_ {i}" for i in range(1, 27)]
    df_pivot = df_pivot.reindex(columns=desired, fill_value=0)

    # name the index column "time" in the CSV header
    df_pivot.index.name = "time"
    
    assert int(df_pivot.sum().sum()) == int(inflow_compare.sum().sum())

    # save with using the save path and Pathlib
    save_path = Path(save_path)
    
    if material_name in ("cement", "sand_gravel_crushed_rock", "concrete"):
        df_pivot.to_csv(save_path / f"raw/rest-of/nmm/image_materials_{material_name}.csv")
    elif material_name in ("steel", "aluminium", "copper"):
        df_pivot.to_csv(save_path / f"raw/rest-of/metals/image_materials_{material_name}.csv")