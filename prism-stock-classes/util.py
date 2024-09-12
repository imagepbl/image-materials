import numpy as np
import xarray as xr


# def convert_vehicles(vehicle_nr):
#         xr_vehicles_orig = vehicle_nr.to_xarray()
#         time_series = xr_vehicles_orig.coords["time"]
#         modes = np.unique([x[0] for x in xr_vehicles_orig.data_vars])
#         region_int = np.sort(np.unique([x[1] for x in xr_vehicles_orig.data_vars]))
#         region_dim = np.array([str(x) for x in region_int])


#         xr_vehicles = xr.DataArray(0.0, dims=("time", "mode", "region"),
#                 coords={"time": time_series,
#                         "mode": modes,
#                         "region": region_dim})
#         for dv in xr_vehicles_orig.data_vars:
#                 xr_vehicles.loc[:, dv[0], str(dv[1])] = xr_vehicles_orig.data_vars[dv]
#         return xr_vehicles


def dataset_to_array(xar_dataset, extra_dims):
    coords = {}
    for coor_name in xar_dataset.coords.keys():
        coords[coor_name] = xar_dataset.coords[coor_name]

    for i_dim, dim_name in enumerate(extra_dims):
        dim_values = np.unique([str(x[i_dim]) for x in xar_dataset.data_vars])
        coords[dim_name] = dim_values

    result_array = xr.DataArray(0.0, dims = tuple(coords),
                                coords=coords)
    for dv in xar_dataset.data_vars:
        result_array.loc[{key: str(value) for key, value in zip(extra_dims, dv)}] = xar_dataset[dv].to_numpy()
    return result_array

