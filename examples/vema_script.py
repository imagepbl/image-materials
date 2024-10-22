from pathlib import Path
from typing import Callable

import prism
import cProfile

from imagematerials.stock import compute_dynamic_stock_driven, compute_historic
from imagematerials.survival import ScipySurvival, SurvivalMatrix
from imagematerials.vehicles.preprocessing import (
    export_to_netcdf,
    import_from_netcdf,
    preprocessing,
)


def main():
    base_dir = "../data/raw"
    prep_fp = Path("prep_vema.nc")

    if not prep_fp.is_file():
        _, orig_prep_data = preprocessing(base_dir)
        export_to_netcdf(orig_prep_data, prep_fp)
    prep_data = import_from_netcdf(prep_fp)

    vehicle_nr = prep_data['total_nr_vehicles']
    life_time_vehicles = prep_data["lifetimes_vehicles"]

    survival_matrix = SurvivalMatrix(ScipySurvival(life_time_vehicles, vehicle_nr.coords["mode"]))

    start_simulation = 1970
    end_simulation = vehicle_nr.coords["time"].max()
    Region = prism.Dimension("region", coords=[str(x) for x in vehicle_nr.coords["region"].values])
    Mode = prism.Dimension("mode", coords=[str(x) for x in vehicle_nr["mode"].to_numpy()])
    Cohort = prism.Dimension("cohort", coords=[str(x) for x in vehicle_nr["time"].to_numpy()])


    @prism.interface
    class Stocks(prism.Model):
        start_simulation: int
        survival_matrix: SurvivalMatrix
        stock: prism.TimeVariable[Region, Mode] #TODO check how to have property that can be both input and output within prism
        stock_function: Callable    # defines the stock function to use e.g. stock or inflow driven

        stock_by_cohort: prism.TimeVariable[Region, Mode, Cohort, "count"] = prism.export(initial_value = prism.Array[Region, Mode, Cohort, 'count'](0.0)) 
        inflow: prism.TimeVariable[Region, Mode, "count"]         = prism.export(initial_value = prism.Array[Region, Mode, 'count'](0.0))   
        outflow_by_cohort: prism.TimeVariable[Region, Mode, Cohort, "count"] = prism.export(initial_value = prism.Array[Region, Mode, Cohort, 'count'](0.0))

        def compute_initial_values(self, time: prism.Timeline):
            compute_historic(self.stock, self.survival_matrix, self.start_simulation,
                            self.stock_by_cohort, self.inflow, self.outflow_by_cohort,
                            self.stock_function)

        def compute_values(self, time: prism.Time):
            t, dt = time.t, time.dt
            self.stock_function(self.stock, self.stock_by_cohort,  self.inflow, self.outflow_by_cohort, self.survival_matrix, t)

    timeline = prism.Timeline(start=vehicle_nr.coords["time"][0],
                            end=end_simulation, stepsize=1)
    timeline_simulate = prism.Timeline(start=start_simulation,
                            end=end_simulation, stepsize=1)
    model = Stocks(timeline, start_simulation=start_simulation, survival_matrix=survival_matrix, stock=vehicle_nr, stock_function = compute_dynamic_stock_driven)
    model.simulate(timeline_simulate)

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    profiler.dump_stats("all.prof")
