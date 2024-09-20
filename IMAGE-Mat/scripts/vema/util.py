import numpy as np
import xarray as xr

from typing import Optional


def pandas_to_xarray(df, unit_mapping):
    ds = df.to_xarray()
    # Apply units to each dimension
    for dim in ds.dims:
        if dim in unit_mapping:
            ds[dim].attrs['units'] = unit_mapping[dim]
    return ds.pint.quantify()


def dataset_to_array(xar_dataset: xr.Dataset, extra_dims: list[str],
                     merge: Optional[dict[str, list[str]]]=None) -> xr.DataArray:
    """Convert an xarray dataset to an xarray dataarray.

    Parameters
    ----------
    xar_dataset
        Input xArray dataset.
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
    for coor_name in xar_dataset.coords.keys():
        coords[coor_name] = xar_dataset.coords[coor_name]

    extra_dims = [("mode" if x is None else x) for x in extra_dims]
    merge_dims = {}
    if merge is not None:
        for key, vals in merge.items():
            merge_dims.update({val: key for val in vals})

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
                dim_values.append(" - ".join(str(cur_data_var[i_dim]) for i_dim in dim_idx))
            dim_values = np.unique(dim_values)
        else:
            dim_values = np.unique([str(x[i_dim]) for x in xar_dataset.data_vars])

        coords[new_dim_name] = dim_values

    result_array = xr.DataArray(0.0, dims = tuple(coords),
                                coords=coords)
    for dv in xar_dataset.data_vars:
        if len(extra_dims) == 1 and merge is None:
            loc = {extra_dims[0]: dv}
        elif merge is not None:
            loc = {key: str(value) for key, value in zip(extra_dims, dv) if key not in merge_dims}
            for dst, sources in merge.items():
                dst_idx = [extra_dims.index(name) for name in sources]
                dst_key = " - ".join(str(dv[idx]) for idx in dst_idx)
                loc[dst] = dst_key
        else:
            loc = {key: str(value) for key, value in zip(extra_dims, dv)}
        result_array.loc[loc] = xar_dataset[dv].to_numpy()
    return result_array
