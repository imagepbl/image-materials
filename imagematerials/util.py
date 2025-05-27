from pathlib import Path
import sys
from typing import Optional, Union

import netCDF4
import numpy as np
import xarray as xr

if sys.version_info < (3, 11):
    import tomli as tomllib
else:
    import tomllib

from imagematerials.concepts import KnowledgeGraph
from imagematerials.constants import SUBTYPE_SEPARATOR


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
        data.to_netcdf(out_fp, group=key, mode="a", engine="netcdf4")

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
