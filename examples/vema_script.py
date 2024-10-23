import cProfile
from pathlib import Path
from typing import Callable

import prism
import xarray as xr

from imagematerials.stock import compute_dynamic_stock_driven, compute_historic
from imagematerials.survival import ScipySurvival, SurvivalMatrix
from imagematerials.vehicles.preprocessing import (
    export_to_netcdf,
    import_from_netcdf,
    preprocessing,
)


def simulate_stocks(prep_data):
    # prep_data = import_from_netcdf(prep_fp)

    vehicle_nr = prep_data['total_nr_vehicles']
    life_time_vehicles = prep_data["lifetimes_vehicles"]
    vehicle_shares = prep_data["vehicle_shares"]
    survival_matrix = SurvivalMatrix(ScipySurvival(life_time_vehicles, vehicle_nr.coords["mode"]))

    start_simulation = 1970
    end_simulation = vehicle_nr.coords["time"].max()
    Region = prism.Dimension("region", coords=[str(x) for x in vehicle_nr.coords["region"].values])
    Mode = prism.Dimension("mode", coords=[str(x) for x in vehicle_nr["mode"].to_numpy()])
    Cohort = prism.Dimension("cohort", coords=[x for x in vehicle_nr["time"].to_numpy()])
    stock_by_cohort = xr.DataArray(0.0, dims=("time", "cohort", "region", "mode"),
                                   coords={"time": vehicle_nr.coords["time"],
                                           "cohort": vehicle_nr.coords["time"].to_numpy(),
                                           "region": vehicle_nr.coords["region"],
                                           "mode": vehicle_nr.coords["mode"]})

    @prism.interface
    class Stocks(prism.Model):
        start_simulation: int
        survival_matrix: SurvivalMatrix
        stock: prism.TimeVariable[Region, Mode] #TODO check how to have property that can be both input and output within prism
        stock_function: Callable    # defines the stock function to use e.g. stock or inflow driven
        stock_by_cohort: xr.DataArray
        shares: prism.TimeVariable[Region, Mode]

        # stock_by_cohort: prism.TimeVariable[Region, Mode, Cohort, "count"] = prism.export(initial_value = prism.Array[Region, Mode, Cohort, 'count'](0.0)) 
        inflow: prism.TimeVariable[Region, Mode, "count"]         = prism.export(initial_value = prism.Array[Region, Mode, 'count'](0.0))   
        outflow_by_cohort: prism.TimeVariable[Region, Mode, Cohort, "count"] = prism.export(initial_value = prism.Array[Region, Mode, Cohort, 'count'](0.0))

        def compute_initial_values(self, time: prism.Timeline):
            compute_historic(self.stock, self.survival_matrix, self.start_simulation,
                            self.stock_by_cohort, self.inflow, self.outflow_by_cohort,
                            self.shares,
                            self.stock_function)

        def compute_values(self, time: prism.Time):
            t, dt = time.t, time.dt
            self.stock_function(self.stock, self.stock_by_cohort,  self.inflow, self.outflow_by_cohort,
                                self.survival_matrix, t, self.shares)

    timeline = prism.Timeline(start=vehicle_nr.coords["time"][0],
                            end=end_simulation, stepsize=1)
    timeline_simulate = prism.Timeline(start=start_simulation,
                            end=end_simulation, stepsize=1)
    model = Stocks(timeline, start_simulation=start_simulation, survival_matrix=survival_matrix,
                   stock=vehicle_nr, stock_function = compute_dynamic_stock_driven,
                   stock_by_cohort=stock_by_cohort, shares=vehicle_shares)
    model.simulate(timeline_simulate)
    model.time_coor = vehicle_nr.coords["time"]
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
    for data_name, array in data_dict.items():
        for drop_coor in array.coords.values():
            sum_over = set(array.coords.values().values) - set(drop_coor)
            summary = array.sum(sum_over)
            summary.to_netcdf(output_fp, group=f"{data_name}-{drop_coor}", mode="a",
                              engine="netcdf4")

def _convert_timevar(time_var, time_coor):
    random_t = time_coor.coords["time"].values[0]
    coords = dict(time_var[random_t].coords.items())
    coords["time"] = time_coor
    array = xr.DataArray(0.0, dims=list(coords), coords=coords)
    for t in time_coor.values:
        array.loc[{"time": t}] = time_var[t]
    return array

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    simulate_stocks(import_from_netcdf("prep_vema.nc"))
    profiler.disable()
    profiler.dump_stats("all.prof")
