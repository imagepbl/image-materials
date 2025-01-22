from typing import Callable, ClassVar, Optional

import xarray as xr

import prism
from imagematerials.stock import (
    compute_dynamic_stock_driven,
    compute_historic,
)
from imagematerials.survival import ScipySurvival, SurvivalMatrix


class BaseModule():
    name = "base"
    input_data = []
    optional_input_data = []
    output_data = []

    def __init__(self, prep_data):
        for data_name in self.input_data:
            try:
                setattr(self, data_name, prep_data[data_name])
            except KeyError:
                raise KeyError(f"Cannot find {data_name} in preprocessing data.")
        for data_name in self.optional_input_data:
            setattr(self, data_name, prep_data.get(data_name))


class StockModule(BaseModule):
    name = "stock"
    input_data = ["stocks", "lifetimes"]
    optional_input_data = ["shares"]
    output_data = ["outflow_by_cohort", "inflow", "stock_by_cohort"]

    def create_model(self, coordinates):
        survival_matrix = SurvivalMatrix(ScipySurvival(self.lifetimes, self.stocks.coords["Type"]))
        start_simulation = 1970
        end_simulation = coordinates["Time"].max()
        region = prism.Dimension("Region", coords=list(coordinates["Region"].values))
        stock_type = prism.Dimension("Type", coords=list(coordinates["Type"].values))
        cohort = prism.Dimension("Cohort", coords=list(coordinates["Time"].values))
        stock_by_cohort = xr.DataArray(0.0, dims=("Time", "Cohort", "Region", "Type"),
                                    coords={"Time": coordinates["Time"],
                                            "Cohort": coordinates["Time"].values,
                                            "Region": coordinates["Region"],
                                            "Type": coordinates["Type"]})

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

        timeline = prism.Timeline(start=self.stocks.coords["Time"][0],
                                  end=end_simulation, stepsize=1)

        return Stocks(timeline, start_simulation=start_simulation, survival_matrix=survival_matrix,
                   stock=self.stocks, stock_function = compute_dynamic_stock_driven,
                   stock_by_cohort=stock_by_cohort,
                   shares=self.shares)

class MaterialModule(BaseModule):
    name = "material"
    input_data = ["inflow", "material_fractions", "outflow_by_cohort", "weights", "stock_by_cohort"]
    optional_input_data = ["shares"]
    output_data = ["outflow_by_cohort_materials", "inflow_materials", "stock_by_cohort_materials"]

    def create_model(self, coordinates):
        external = self
        material_type = prism.Dimension("material", coords=list(coordinates["material"].values))
        region = prism.Dimension("Region", coords=list(coordinates["Region"].values))
        stock_type = prism.Dimension("Type", coords=list(coordinates["Type"].values))
        cohort = prism.Dimension("Cohort", coords=list(coordinates["Time"].values))
        end_simulation = coordinates["Time"].max()
        start_simulation = coordinates["Time"].min()

        stock_by_cohort_materials = xr.DataArray(0.0, dims=("Time", "Region", "Type", "material"),
                                    coords={"Time": coordinates["Time"],
                                            # "Cohort": coordinates["Time"].values,
                                            "Region": coordinates["Region"],
                                            "Type": coordinates["Type"],
                                            "material": coordinates["material"]})

        @prism.interface
        class Materials(prism.Model):
            weights: xr.DataArray
            material_fractions: xr.DataArray
            stock_by_cohort_materials: xr.DataArray

            inflow_materials: prism.TimeVariable[region, stock_type, material_type, "count"] = prism.export(initial_value = prism.Array[region, stock_type, material_type, 'count'](0.0))
            outflow_by_cohort_materials: prism.TimeVariable[region, stock_type, material_type, "count"] = prism.export(initial_value = prism.Array[region, stock_type, material_type, cohort, 'count'](0.0))

            def compute_values(self, time: prism.Time):
                t, dt = time.t, time.dt
                self.inflow_materials[t] = external.inflow[t]*self.material_fractions.sel(Cohort=t).drop_vars("Cohort")*self.weights.sel(Cohort=t).drop_vars("Cohort")
                self.outflow_by_cohort_materials[t] = (external.outflow_by_cohort[t]*self.material_fractions*self.weights).sum("Cohort")
                self.stock_by_cohort_materials.loc[t] = (external.stock_by_cohort.loc[t]*self.material_fractions*self.weights).sum("Cohort")

        timeline = prism.Timeline(start=start_simulation,
                                  end=end_simulation, stepsize=1)
        mat_model = Materials(timeline, weights=self.weights, material_fractions=self.material_fractions,
                             stock_by_cohort_materials=stock_by_cohort_materials)
        return mat_model




class ModelFactory():
    all_modules = [StockModule, MaterialModule]

    def __init__(self, prep_data):
        self.prep_data = prep_data
        self.models = []
        self.coordinates = self.create_coordinates()
        self.all_data = {key: value for key, value in prep_data.items()}

    def create_coordinates(self):
        coordinates = {}
        for data_name, data_obj in self.prep_data.items():
            if not isinstance(data_obj, xr.DataArray):
                continue
            for coor in data_obj.coords.values():
                if coor.name not in coordinates:
                    coordinates[coor.name] = coor
                elif not all(coor == coordinates[coor.name]):
                    raise ValueError(f"Mismatched dimensions in input data for {data_name} for coordinates {coor.name}")
        return coordinates

    def add(self, module_name: str):
        module_class = None
        for module in self.all_modules:
            if module_name == module.name:
                module_class = module
        if module_class is None:
            raise ValueError(f"Cannot find module with name '{module_name}'.")
        new_sub_model = module_class(self.all_data).create_model(self.coordinates)
        self.models.append(new_sub_model)

        for output_name in module_class.output_data:
            self.all_data[output_name] = getattr(new_sub_model, output_name)

        return self

    def get_model(self, module_name):
        module_class = None
        for module in self.all_modules:
            if module_name == module.name:
                module_class = module
        if module_class is None:
            raise KeyError(f"Cannot find module with name '{module_name}', available: {self.all_modules}.")
        return module_class


    def run(self):
        factory = self
        @prism.interface
        class MainModule(prism.Model):
            def init_submodels(self, timeline: prism.Timeline):
                self.submodels = factory.models
            def compute_initial_values(self, time: prism.Time):
                for model in self.submodels:
                    try:
                        model.compute_initial_values(time)
                    except AttributeError:
                        pass
            def compute_values(self, time: prism.Time):
                for model in self.submodels:
                    model.compute_values(time)
        main_model = MainModule(self.timeline)
        main_model.simulate(self.timeline_simulate)
        main_model.time_coor = self.time_coordinates
        for data_name, data in self.all_data.items():
            setattr(main_model, data_name, data)
        return main_model

    @property
    def time_coordinates(self):
        for data_obj in self.prep_data.values():
            if isinstance(data_obj, xr.DataArray) and "Time" in data_obj.coords:
                return data_obj.coords["Time"].values
        raise ValueError("Cannot find any time coordinates in the data.")

    @property
    def historic_start(self):
        return self.time_coordinates.min()

    @property
    def simulation_start(self):
        return 1970

    @property
    def simulation_end(self):
        return self.time_coordinates.max()

    @property
    def timeline(self):
        return prism.Timeline(start=self.historic_start,
                              end=self.simulation_end, stepsize=1)

    @property
    def timeline_simulate(self):
        return prism.Timeline(start=self.simulation_start, end=self.simulation_end,
                              stepsize=1)
