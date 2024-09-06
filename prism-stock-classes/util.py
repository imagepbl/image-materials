import numpy as np
import xarray as xr


def convert_vehicles(vehicle_nr):
        xr_vehicles_orig = vehicle_nr.to_xarray()
        time_series = xr_vehicles_orig.coords["time"]
        modes = np.unique([x[0] for x in xr_vehicles_orig.data_vars])
        region_dim = np.unique([x[1] for x in xr_vehicles_orig.data_vars])


        xr_vehicles = xr.DataArray(0.0, dims=("time", "mode", "region"),
                coords={"time": time_series,
                        "mode": modes,
                        "region": region_dim})
        for dv in xr_vehicles_orig.data_vars:
                xr_vehicles.loc[:, dv[0], dv[1]] = xr_vehicles_orig.data_vars[dv]
        return xr_vehicles

