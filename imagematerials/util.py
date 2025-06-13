from pathlib import Path
import sys
from typing import Optional, Union

import netCDF4
import numpy as np
import pandas as pd
import xarray as xr
from collections import defaultdict

if sys.version_info < (3, 11):
    import tomli as tomllib
else:
    import tomllib

from imagematerials.concepts import KnowledgeGraph
from imagematerials.constants import SUBTYPE_SEPARATOR
from imagematerials.distribution import ALL_DISTRIBUTIONS, NAME_TO_DIST


NONE_SENTINEL = "__NETCDF_NONE_SENTINEL__"

def pandas_to_xarray(df, unit_mapping):
    ds = df.to_xarray()
    # Apply units to each dimension
    for dim in ds.dims:
        if dim in unit_mapping:
            ds[dim].attrs['units'] = unit_mapping[dim]
    return ds.pint.quantify()


def dataset_to_array(xar_dataset: xr.Dataset, main_coor: list[str], extra_dims: list[str],
                     merge: Optional[dict[str, list[str]]]=None) -> xr.DataArray:
    """Convert an xarray dataset to an xarray dataarray.

    Parameters
    ----------
    xar_dataset
        Input xArray dataset.
    main_coor:
        Names of the main coordinate(s) of the dataset, usually time or cohort.
    extra_dims
        Coordinates for the data variables for the input dataset in the correct order.
    merge, optional
        Merge columns in the dataframe for sub modes mainly, by default None, in which case
        nothing is merged. The key should be the name of the resulting coordinate,
        while the value is a list of columns that are the source of this resulting coordinate.
        The values in this list should be also present in the extra_dims argument.
        Merged columns will have their dimensions concatenated with a hyphen
        in between.

    Returns
    -------
        A xArray DataArray with the correct coordinates and dimensions.

    Examples
    --------
    >>> dataset_to_array(xr_data, ["mode", "type", "battery"], merge={"merged_mode": ["mode", "type"]})

    """
    coords = {}
    rename_coords = {}
    for i_coor, coor_name in enumerate(xar_dataset.coords.keys()):
        coords[coor_name] = xar_dataset.coords[coor_name]
        rename_coords[coor_name] = main_coor[i_coor]

    extra_dims = [("mode" if x is None else x) for x in extra_dims]

    # Reverse lookup for the merge dictionary.
    merge_dims = {}
    if merge is not None:
        for key, vals in merge.items():
            merge_dims.update({val: key for val in vals})

    # Compute all the dimensions for the coordinates.
    for i_dim, dim_name in enumerate(extra_dims):
        new_dim_name = dim_name
        if len(extra_dims) == 1 and merge is None:
            dim_values = np.unique([str(x) for x in xar_dataset.data_vars])
        elif dim_name in merge_dims:
            new_dim_name = merge_dims[dim_name]
            cur_merge_dims = merge[new_dim_name]
            if new_dim_name in coords:
                continue
            dim_values = []
            dim_idx = [extra_dims.index(name) for name in cur_merge_dims]
            for cur_data_var in xar_dataset.data_vars:
                dim_values.append(SUBTYPE_SEPARATOR.join(str(cur_data_var[i_dim])
                                                         for i_dim in dim_idx))
            dim_values = np.unique(dim_values)
        else:
            dim_values = np.unique([str(x[i_dim]) for x in xar_dataset.data_vars])

        coords[new_dim_name] = dim_values

    # Create the data array.
    result_array = xr.DataArray(0.0, dims = tuple(coords),
                                coords=coords)
    for dv in xar_dataset.data_vars:
        if len(extra_dims) == 1 and merge is None:
            loc = {extra_dims[0]: dv}
        elif merge is not None:
            loc = {key: str(value) for key, value in zip(extra_dims, dv) if key not in merge_dims}
            for dst, sources in merge.items():
                dst_idx = [extra_dims.index(name) for name in sources]
                dst_key = SUBTYPE_SEPARATOR.join(str(dv[idx]) for idx in dst_idx)
                loc[dst] = dst_key
        else:
            loc = {key: str(value) for key, value in zip(extra_dims, dv)}
        result_array.loc[loc] = xar_dataset[dv].to_numpy()
    return result_array.rename(rename_coords)


def merge_dims(xr_array, dim_one, dim_two):
    if dim_one not in xr_array.coords:
        raise ValueError(f"Dimension {dim_one} not found in data array.")
    if dim_two not in xr_array.coords:
        raise ValueError(f"Dimension {dim_two} not found in data array.")

    new_types = []
    for cur_coor_one in xr_array.coords[dim_one].values:
        for cur_coor_two in xr_array.coords[dim_two].values:
            new_types.append(SUBTYPE_SEPARATOR.join((cur_coor_one, cur_coor_two)))

    new_dims = [dim for dim in xr_array.coords.keys() if dim != dim_two]
    new_coords = {dim: coord.values for dim, coord in xr_array.coords.items() if dim not in [dim_one, dim_two]}
    new_coords.update({dim_one: new_types})
    new_array = xr.DataArray(
        0.0, dims=new_dims,
        coords = new_coords
    )
    for cur_coor_one in xr_array.coords[dim_one].values:
        for cur_coor_two in xr_array.coords[dim_two].values:
            new_coor_one = SUBTYPE_SEPARATOR.join((cur_coor_one, cur_coor_two))
            new_array.loc[{dim_one: new_coor_one}] = xr_array.loc[{dim_one: cur_coor_one, dim_two: cur_coor_two}]
    return new_array


def export_to_netcdf(prep_data: dict, out_fp):
    """Export the xarray data to a netcdf4 file.

    Parameters
    ----------
    prep_data
        xArray data from the preprocessing steps.
    out_fp
        Netcdf4 file to write to, recommended extension is .nc.

    """
    new_prep_data = {key: val for key, val in prep_data.items()}
    lifetimes = new_prep_data.pop("lifetimes")
    # xr.Dataset(new_prep_data).to_netcdf(out_fp, group="main", engine="netcdf4")
    xr.Dataset(lifetimes).to_netcdf(out_fp, group="lifetimes", mode="a", engine="netcdf4")
    for key, data in new_prep_data.items():
        try:
            data.to_netcdf(out_fp, group=key, mode="a", engine="netcdf4")
        except AttributeError:
            with netCDF4.Dataset(out_fp, "a") as rootgrp:
                if data is None:
                    setattr(rootgrp, key, NONE_SENTINEL)
                else:
                    setattr(rootgrp, key, data)

def import_from_netcdf(in_fp) -> dict:
    """Import the xarray data from a netcdf4 file.

    Parameters
    ----------
    in_fp
        File to read the xarray data file from (usualy with *.nc).

    Returns
    -------
        Dictionary containing the data arrays and datasets.

    """
    prep_data_dict = {}
    lt = xr.open_dataset(in_fp, group="lifetimes", engine="netcdf4").load()
    prep_data_dict["lifetimes"] = {dist_name: arr.dropna("Type")
                                            for dist_name, arr in lt.items()}
    # prep_data = xr.open_dataset(in_fp, group="main", engine="netcdf4").load()
    # prep_data_dict = {key: value for key, value in prep_data.items()}
    with netCDF4.Dataset(in_fp, "r") as data:
        all_groups = list(data.groups.keys())
        # Get attributes
        for key, val in data.__dict__.items():
            if val == NONE_SENTINEL:
                prep_data_dict[key] = None
            else:
                prep_data_dict[key] = val
    all_groups.remove("lifetimes")
    for key in all_groups:
        prep_data_dict[key] = xr.open_dataarray(in_fp, group=key, engine="netcdf4").load()
        if key == "knowledge_graph":
            prep_data_dict[key] = KnowledgeGraph.from_dataarray(prep_data_dict[key])
    return prep_data_dict

def summarize_prep_data(data):
    all_summary = {}
    for data_name, array in data.items():
        if isinstance(array, (dict, xr.Dataset)):
            all_summary[data_name] = summarize_prep_data(array)
        elif isinstance(array, xr.DataArray):
            all_summary[data_name] = _summarize_array(array)
        elif isinstance(array, KnowledgeGraph):
            continue
        elif array is None:
            all_summary[data_name] = array
        else:
            raise ValueError(f"Cannot compare data with name '{data_name}' with type {type(array)}")
    return all_summary

def _summarize_array(array):
    all_summary = {}
    for drop_coor in array.coords.keys():
        sum_over = set(x for x in list(array.coords.keys())) - set({drop_coor})
        summary = array.sum(sum_over)
        all_summary[drop_coor] = _listify(summary.to_dict())
    return all_summary

def _listify(data):
    if isinstance(data, dict):
        return {key: _listify(value) for key, value in data.items()}
    elif isinstance(data, tuple):
        return list(data)
    return data

def rebroadcast_prep_data(prep_data, knowledge_graph, dim, output_coords):
    new_prep_data = {}
    for data_name, data in prep_data.items():
        if not isinstance(data, xr.DataArray) or dim not in data.coords:
            new_prep_data[data_name] = data
        else:
            new_prep_data[data_name] = knowledge_graph.rebroadcast_xarray(data, output_coords, dim=dim)
    return new_prep_data

  
def read_climate_policy_config(scenario_folder) -> dict:
    """
    Extracts data from a .toml-file.

    Parameters
    ----------
    scenario_folder
        Path to file that must be read

    Returns
    -------
        Dictionary containing the contents of the toml-file
    """
    return _read_config(scenario_folder)

def read_circular_economy_config(scenario_folders: dict) -> dict:
    """
    Extracts data from multiple .toml-files and joins it together.

    Parameters
    ----------
    scenario_folders
        Dictionary with labelled paths to the files that must be read.

    Returns
    -------
        Dictionary containing the contents of all toml-file, accessible
        under the specified labels.
    """
    config_dict = {}
    for key, scenario_folder in scenario_folders.items():
        config_dict[key] = _read_config(scenario_folder)
    return config_dict

def _read_config(scenario_folder) -> dict:
    """
    Extracts data from a .toml-file.

    Parameters
    ----------
    scenario_folder
        Path to file that must be read

    Returns
    -------
        Dictionary containing the contents of the toml-file
    """
    # Turn the path into a Path object, if it wasn't already
    scenario_folder = Path(scenario_folder)
    with open(scenario_folder / "config.toml", "rb") as f:
        config_dict = tomllib.load(f)
    
    config_dict['config_file_path'] = scenario_folder.resolve()

    return config_dict


def convert_life_time_vehicles(life_time_vehicles: xr.Dataset) -> dict[str, xr.DataArray]:
    """Convert lifetime vehicles dataset to a more appropriate data format.

    This conversion should probably move to the preprocessing stage after we figure out
    the exact details of what the output should look like.

    Parameters
    ----------
    life_time_vehicles
        The input life_time_vehicles xarray dataset. It is supposed to be in a very particular format:
        it contains the parameters for each of the modes. However, the distribution types are not
        the same for each of the modes. Thus, the distribution types need to be inferred from the
        names of the parameters that are given. If multiple parameter sets for multiple distributions
        are given, the Weibull distribution is given preference over the FoldedNormal distribution.

    Returns
    -------
        A dictionary that contains a data array for each of the distributions. Given the setup there is
        an implicit assumption that only one distribution is used for each of the modes. If distribution
        types change over time, this data structure needs to be adjusted.

    """
    # Create a dictionary to find which parameters are available for which mode.
    mode_param = defaultdict(list)
    for mode, par in life_time_vehicles.data_vars:
        mode_param[mode].append(par)

    # Create a dictionary that says which modes are tied to which distribution.
    dist_mode = defaultdict(list)
    modes_done = set()  # temporary
    for dist in ALL_DISTRIBUTIONS:
        for mode, param in mode_param.items():
            if mode not in modes_done and dist.has_param(param):
                dist_mode[dist.name].append(mode)
                modes_done.add(mode)

    # Iterate over all distributions to create a data array for each of them.
    ret_scipy_params = {}
    for dist_name, mode_list in dist_mode.items():
        if len(mode_list) == 0:
            continue
        dist = NAME_TO_DIST[dist_name]

        # param_arrays = {}
        array = xr.DataArray(
            0.0, dims=("Time", "Type", "ScipyParam"),
            coords={
                "Time": life_time_vehicles.coords["Year"].to_numpy(),
                "Type": mode_list,
                "ScipyParam": dist.variable_scipy_param})
        for mode in mode_list:
            orig_param_dict = {}
            for param in dist.params:
                orig_param_dict[param] = life_time_vehicles.data_vars[str(mode), param].to_numpy()
            scipy_params = dist.get_param(orig_param_dict)
            for cur_scipy_key, cur_scipy_par in scipy_params.items():
                if cur_scipy_key in dist.variable_scipy_param:
                    array.loc[:, str(mode), cur_scipy_key] = cur_scipy_par
                else:
                    array.attrs[cur_scipy_key] = cur_scipy_par
        ret_scipy_params[dist_name] = array
    return ret_scipy_params


def scenario_change(arr: xr.DataArray, base_year: int, target_year: int, change: dict,
                    implementation_rate: str, data_type: Optional[str]=None, steepness: float=0.5) -> xr.DataArray:
    """
    Applies a time-based change to values in a DataFrame between a base and target year using a specified implementation method.

    Parameters
    ----------
    arr
        A time-indexed DataFrame containing mode-specific values, such as lifetime or mileage.
    base_year
        The starting year for the change.
    target_year
        The year by which the full change should be achieved.
    change
        A dictionary mapping modes to percentage increases (e.g., {'Cars': 20} for +20%).
    implementation_rate
        The implementation method; one of 'linear', 'immediate', or 's-curve'.
    data_type
        Indicates what kind of data is being modified; one of 'lifetime' or 'mileages'.
    steepness
        Steepness parameter for the 's-curve' implementation; default is 0.5.

    Returns
    -------
        A new DataFrame with updated values for each year between base_year and target_year, and interpolated values where necessary.

    Raises
    ------
    ValueError
        If the implementation method is unsupported or if the specified column is not found in the DataFrame.

    Notes
    -----
    For verhicles, this function has an implementation that works on Pandas dataframes.
    """
    result = arr.where(arr.Time <= base_year, np.nan)
    result.loc[{"Time": target_year}] = result.loc[{"Time": base_year}]

    for region, increase in change.items():
        base_val = result.loc[{"Time": target_year, "Region": region}]
        if region in result.Region:
            if implementation_rate =='linear':
                result.loc[{"Time": target_year, "Region": region}] *= (1 + increase / 100)
            elif implementation_rate =='immediate':
                result.loc[{"Time": base_year + 1, "Region": region}] = \
                    base_val * (1 + increase / 100)
                result.loc[{"Time": target_year, "Region": region}] *= (1 + increase / 100)
            elif implementation_rate =='s-curve':
                years = list(range(base_year, target_year + 1))
                mid_year = (base_year + target_year) / 2
                target_val = base_val * (1 + increase / 100)
                for year in years:
                    progress = 1 / (1 + np.exp(-steepness * (year - mid_year)))
                    result.loc[{"Time": year, "Region": region}] = \
                        base_val + (target_val - base_val) * progress
            else: 
                raise ValueError(f"Unknown implementation method: '{implementation_rate}'. "
                                  "Supported methods are 'immediate', 'linear', and 's-curve'.")
        else:
            raise ValueError(f"Region {region} not found in DataArray.")
    return result.interpolate_na("Time", method="linear")


def apply_change_per_region(arr: xr.DataArray, base_year: int, target_year: int, increase: float,
                            implementation_rate: str, data_type: Optional[str]=None,
                            steepness: float=0.5) -> xr.DataArray:
    """
    Applies a uniform percentage increase across all regions (columns) in the DataFrame using a specified implementation method.

    Parameters:
        df (pd.DataFrame): A time-indexed DataFrame with regions as columns and a common structure across all regions.
        base_year (int): The starting year for the change.
        target_year (int): The year by which the full change should be achieved.
    increase
        The percentage increase to apply to all regions (e.g., 10 for +10%).
        implementation_rate (str): The implementation method; one of 'linear', 'immediate', or 's-curve'.
        data_type (str): Indicates what kind of data is being modified; one of 'lifetime' or 'mileages'.
        steepness (float, optional): Steepness parameter for the 's-curve' implementation; default is 0.5.

    Returns:
        pd.DataFrame: A DataFrame with updated values for each region, aligned by year (index).
    """
    results = []
    for region, subarr in arr.groupby('Region'):
        regional_subarr = subarr.copy()  # Keep as DataFrame for compatibility
        result = scenario_change(
            regional_subarr, 
            base_year=base_year, 
            target_year=target_year, 
            change={region: increase.loc[{"Region": region}]}, 
            implementation_rate=implementation_rate, 
            data_type=data_type, 
            steepness=steepness
        )
        results.append(result)
    # Concatenate results along columns (axis=1), aligning on index
    return xr.concat(results, 'Region')
