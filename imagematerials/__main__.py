import cProfile
from pathlib import Path
from typing import Callable

import xarray as xr

import prism
from imagematerials.stock import compute_dynamic_stock_driven, compute_historic, material_computation, battery_computation
from imagematerials.survival import ScipySurvival, SurvivalMatrix
from imagematerials.util import (
    import_from_netcdf,
)


def export_model_netcdf(model, output_fp):
    time_coor = model.time_coor
    data_dict = {
        "inflow": _convert_timevar(model.inflow, time_coor),
        "outflow_by_cohort": _convert_timevar(model.outflow_by_cohort, time_coor),
        "stock_by_cohort": model.stock_by_cohort,
    }
    for array in data_dict.values():
        for coord in array.coords.values():
            coord.attrs.pop("unit", None)
    xr.Dataset(data_dict).drop_encoding().to_netcdf(output_fp, engine="netcdf4")

def export_summary_netcdf(model, output_fp):
    time_coor = model.time_coor
    data_dict = {
        "inflow": _convert_timevar(model.inflow, time_coor),
        "outflow_by_cohort": _convert_timevar(model.outflow_by_cohort, time_coor),
        "stock_by_cohort": model.stock_by_cohort,
    }
    if Path(output_fp).is_file():
        Path(output_fp).unlink()
    all_keys = []
    for data_name, array in data_dict.items():
        for drop_coor in array.coords.keys():
            sum_over = set(x for x in list(array.coords.keys())) - set({drop_coor})
            summary = array.sum(sum_over)
            summary.attrs.pop("units", None)
            summary.coords[drop_coor].attrs.pop("units", None)
            key = f"{data_name}-{drop_coor}"
            summary.drop_encoding().to_netcdf(output_fp, group=key, mode="a", engine="netcdf4")
            all_keys.append(key)
    empty = xr.DataArray(0.0)
    empty.attrs["summary_names"] = all_keys
    empty.to_netcdf(output_fp, group="summary", mode="a", engine="netcdf4")

def _convert_timevar(time_var, time_coor):
    random_t = time_coor.coords["Time"].values[0]
    coords = dict(time_var[random_t].coords.items())
    coords["Time"] = time_coor
    array = xr.DataArray(0.0, dims=list(coords), coords=coords)
    for t in time_coor.values:
        array.loc[{"Time": t}] = time_var[t]
    return array
