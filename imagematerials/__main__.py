import cProfile
from pathlib import Path
from typing import Callable

import xarray as xr

import prism
from imagematerials.stock import compute_dynamic_stock_driven, compute_historic
from imagematerials.survival import ScipySurvival, SurvivalMatrix
from imagematerials.util import (
    import_from_netcdf,
)


def simulate_stocks(prep_data):
    total_stocks = prep_data['stocks']
    lifetimes = prep_data["lifetimes"]
    shares = prep_data.get("shares", None)  # If the shares are not present they will be set to None
    # Type means: E.g. type of building (Appartment)/ vehicles / etc.
    # Currently only one way of computing the survival matrix exists: ScipySurvival
    # This computes the survival matrix on the fly, values are only computed when accessed.
    # Other ways of computing the survival matrix could be implemented.
    survival_matrix = SurvivalMatrix(ScipySurvival(lifetimes, total_stocks.coords["Type"]))

    start_simulation = 1970
    end_simulation = total_stocks.coords["Time"].max()
    region = prism.Dimension("Region", coords=[str(x) for x in total_stocks.coords["Region"].values])
    stock_type = prism.Dimension("Type", coords=[str(x) for x in total_stocks["Type"].to_numpy()])
    cohort = prism.Dimension("Cohort", coords=[x for x in total_stocks["Time"].to_numpy()])

    # Create stock_by_cohort as a data array instead of a prism array, because this is much faster.
    stock_by_cohort = xr.DataArray(0.0, dims=("Time", "Cohort", "Region", "Type"),
                                   coords={"Time": total_stocks.coords["Time"],
                                           "Cohort": total_stocks.coords["Time"].to_numpy(),
                                           "Region": total_stocks.coords["Region"],
                                           "Type": total_stocks.coords["Type"]})

    @prism.interface
    class Stocks(prism.Model):
        start_simulation: int
        survival_matrix: SurvivalMatrix
        stock: prism.TimeVariable[region, stock_type] #TODO check how to have property that can be both input and output within prism
        stock_function: Callable    # defines the stock function to use e.g. stock or inflow driven
        stock_by_cohort: xr.DataArray
        shares: prism.TimeVariable[region, stock_type]

        # stock_by_cohort: prism.TimeVariable[Region, Mode, Cohort, "count"] = prism.export(initial_value = prism.Array[Region, Mode, Cohort, 'count'](0.0)) 
        inflow: prism.TimeVariable[region, stock_type, "count"] = prism.export(initial_value = prism.Array[region, stock_type, 'count'](0.0))
        outflow_by_cohort: prism.TimeVariable[region, stock_type, cohort, "count"] = prism.export(initial_value = prism.Array[region, stock_type, cohort, 'count'](0.0))

        def compute_initial_values(self, time: prism.Timeline):
            compute_historic(self.stock, self.survival_matrix, self.start_simulation,
                            self.stock_by_cohort, self.inflow, self.outflow_by_cohort,
                            self.shares,
                            self.stock_function)

        def compute_values(self, time: prism.Time):
            t, dt = time.t, time.dt
            self.stock_function(self.stock, self.stock_by_cohort,  self.inflow, self.outflow_by_cohort,
                                self.survival_matrix, t, self.shares)

    timeline = prism.Timeline(start=total_stocks.coords["Time"][0],
                              end=end_simulation, stepsize=1)
    timeline_simulate = prism.Timeline(start=start_simulation,
                                       end=end_simulation, stepsize=1)
    model = Stocks(timeline, start_simulation=start_simulation, survival_matrix=survival_matrix,
                   stock=total_stocks, stock_function = compute_dynamic_stock_driven,
                   stock_by_cohort=stock_by_cohort, shares=shares)
    model.simulate(timeline_simulate)
    model.time_coor = total_stocks.coords["Time"]
    return model

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

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    simulate_stocks(import_from_netcdf("prep_vema.nc"))
    profiler.disable()
    profiler.dump_stats("all.prof")
