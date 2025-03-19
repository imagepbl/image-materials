from typing import Callable, ClassVar, Optional

import xarray as xr

import prism
from imagematerials.stock import (
    compute_dynamic_stock_driven,
)
from imagematerials.survival import ScipySurvival, SurvivalMatrix

REGION = prism.Dimension("Region")
STOCK_TYPE = prism.Dimension("Type")
COHORT = prism.Dimension("Cohort")
TIME = prism.Dimension("Time")
MATERIAL_TYPE = prism.Dimension("material")


@prism.interface
class GenericStocks(prism.Model):
    """Stock class that can be used for different products.
    A model class for managing stocks and their inflows and outflows over time, 
    including the computation of initial and dynamic stock values based on input data.
    Attributes
    ----------
    Region : prism.Coords[REGION]
        Defines the regions for the stock.
    Type : prism.Coords[STOCK_TYPE]
        Defines the stock types (e.g., vehicles, buildings).
    Cohort : prism.Coords[COHORT]
        Defines the cohorts (e.g., different age groups of stock).
    Time : prism.Coords[TIME]
        Defines the time steps for the stock simulation.
    lifetimes : xr.DataArray
        Expected lifetimes for each stock type.
    stocks : xr.DataArray
        Initial stock values.
    shares : Optional[xr.DataArray]
        Optional share data for the stock subtypes.
    input_data : tuple of str
        Tuple of input data variable names.
    output_data : tuple of str
        Tuple of output data variable names.
    """
     
    # Dimensions
    Region: prism.Coords[REGION]
    Type: prism.Coords[STOCK_TYPE]
    Cohort: prism.Coords[COHORT]
    Time: prism.Coords[TIME]

    # Inputs
    lifetimes: xr.DataArray
    stocks: xr.DataArray #TODO check how to have property that can be both input and output within prism
    # stock_function: Callable    # defines the stock function to use e.g. stock or inflow driven
    shares: Optional[xr.DataArray]

    # For module dependency, ignored by prism
    input_data: tuple[str] = ("stocks", "lifetimes")
    optional_input_data: tuple[str] = ("shares",)
    output_data: tuple[str] = ("outflow_by_cohort", "inflow", "stock_by_cohort")

    # stock_by_cohort: prism.TimeVariable[Region, Mode, Cohort, "count"] = prism.export(initial_value = prism.Array[Region, Mode, Cohort, 'count'](0.0))
    inflow: prism.TimeVariable[REGION, STOCK_TYPE, "count"] = prism.export()
    outflow_by_cohort: prism.TimeVariable[REGION, STOCK_TYPE, COHORT, "count"] = prism.export()

    def compute_initial_values(self, time: prism.Timeline):
        """Compute the initial values for stocks and the survival matrix.
        
        Parameters
        ----------
        time : prism.Timeline
            The simulation timeline.
        """

        self.survival_matrix = SurvivalMatrix(ScipySurvival(self.lifetimes, self.stocks.coords["Type"]))
        self.stock_by_cohort = xr.DataArray(
            0.0,
            dims=("Time", "Cohort", "Region", "Type"),
            coords={"Time": self.Time,
                    "Cohort": self.Cohort,
                    "Region": self.Region,
                    "Type": self.Type})

    def compute_values(self, time: prism.Time):
        """
        Computes the stock values at each time step, including inflow and outflow by cohort.
        
        Parameters
        ----------
        time : prism.Time
            The current simulation time step.
        """
         
        t, dt = time.t, time.dt
        self.inflow[t].loc[:] = 0.0
        self.outflow_by_cohort[t].loc[:] = 0.0

        compute_dynamic_stock_driven(
            self.stocks, self.stock_by_cohort,  self.inflow, self.outflow_by_cohort,
            self.survival_matrix, t, self.shares)

@prism.interface
class GenericMaterials(prism.Model):
    """
    A model class for managing materials used in stock cohorts, including 
    inflows and outflows of materials and the computation of stock-by-cohort 
    material use over time.
    Attributes
    ----------
    weights : xr.DataArray
        Weight data for the respective product.
    material_fractions : xr.DataArray
        Material composition of the stock, varying by cohort.
    Region : prism.Coords[REGION]
        The region where material use is calculated.
    Type : prism.Coords[STOCK_TYPE]
        The type of stock (e.g., Cars - ICE).
    Cohort : prism.Coords[COHORT]
        Cohort groups within the stock (e.g., yearly age groups).
    material : prism.Coords[MATERIAL_TYPE]
        The material types used in the model.
    input_data : tuple of str
        Tuple of input data variable names.
    output_data : tuple of str
        Tuple of output data variable names.
    """

    # Input data
    weights: xr.DataArray
    material_fractions: xr.DataArray

    # Dimensions
    Region: prism.Coords[REGION]
    Type: prism.Coords[STOCK_TYPE]
    Cohort: prism.Coords[COHORT]
    Time: prism.Coords[TIME]
    material: prism.Coords[MATERIAL_TYPE]

    # Data dependencies
    input_data: tuple[str] = ("weights", "material_fractions", "inflow",
                              "stock_by_cohort", "outflow_by_cohort")
    output_data: tuple[str] = ("stock_by_cohort_materials", "inflow_materials",
                               "outflow_by_cohort_materials")

    # Output data
    inflow_materials: prism.TimeVariable[REGION, STOCK_TYPE, MATERIAL_TYPE, "count"] = prism.export()
    outflow_by_cohort_materials: prism.TimeVariable[REGION, STOCK_TYPE, MATERIAL_TYPE, "count"] = prism.export()

    def compute_initial_values(self, time: prism.Timeline):
        """
        Computes the initial values for materials used in each stock cohort.
        
        Parameters
        ----------
        time : prism.Timeline
            The simulation timeline.
        """
                
        self.stock_by_cohort_materials = xr.DataArray(
            0.0, dims=("Time", "Region", "Type", "material"),
            coords={"Time": self.Time,
                    # "Cohort": coordinates["Time"].values,
                    "Region": self.Region,
                    "Type": self.Type,
                    "material": self.material})

    def compute_values(self, time: prism.Time, inflow, stock_by_cohort, outflow_by_cohort):
        """
        Computes the material inflows, outflows, and stock usage by cohort 
        at each time step.
        
        Parameters
        ----------
        time : prism.Time
            The current simulation time step.
        inflow : xr.DataArray
            Inflow data for the stocks.
        stock_by_cohort : xr.DataArray
            Stock-by-cohort data.
        outflow_by_cohort : xr.DataArray
            Outflow data by cohort.
        """
         
        t, dt = time.t, time.dt
        self.inflow_materials[t] = inflow[t]*self.material_fractions.sel(Cohort=t).drop_vars("Cohort")*self.weights.sel(Cohort=t).drop_vars("Cohort")
        self.outflow_by_cohort_materials[t] = (outflow_by_cohort[t]*self.material_fractions*self.weights).sum("Cohort")
        self.stock_by_cohort_materials.loc[t] = (stock_by_cohort.loc[t]*self.material_fractions*self.weights).sum("Cohort")

@prism.interface
class MaterialIntensities(prism.Model):
    # Input data
    material_intensities: xr.DataArray

    # Dimensions
    Region: prism.Coords[REGION]
    Type: prism.Coords[STOCK_TYPE]
    Cohort: prism.Coords[COHORT]
    Time: prism.Coords[TIME]
    material: prism.Coords[MATERIAL_TYPE]

    # Data dependencies
    input_data: tuple[str] = ("material_intensities", "inflow",
                              "stock_by_cohort", "outflow_by_cohort")
    output_data: tuple[str] = ("stock_by_cohort_materials", "inflow_materials",
                               "outflow_by_cohort_materials")

    # Output data
    inflow_materials: prism.TimeVariable[REGION, STOCK_TYPE, MATERIAL_TYPE, "count"] = prism.export()
    outflow_by_cohort_materials: prism.TimeVariable[REGION, STOCK_TYPE, MATERIAL_TYPE, "count"] = prism.export()

    def compute_initial_values(self, time: prism.Timeline):
        self.stock_by_cohort_materials = xr.DataArray(
            0.0, dims=("Time", "Region", "Type", "material"),
            coords={"Time": self.Time,
                    # "Cohort": coordinates["Time"].values,
                    "Region": self.Region,
                    "Type": self.Type,
                    "material": self.material})

    def compute_values(self, time: prism.Time, inflow, stock_by_cohort, outflow_by_cohort):
        t, dt = time.t, time.dt
        self.inflow_materials[t] = inflow[t]*self.material_intensities.sel(Cohort=t).drop_vars("Cohort")
        self.outflow_by_cohort_materials[t] = (outflow_by_cohort[t]*self.material_intensities).sum("Cohort")
        self.stock_by_cohort_materials.loc[t] = (stock_by_cohort.loc[t]*self.material_intensities).sum("Cohort")


@prism.interface
class Maintenance(prism.Model):
    """
    A model class for managing maintenance-related materials used over time, 
    including stock-by-cohort maintenance material use and computation of values.
    Attributes
    ----------
    weights : xr.DataArray
        Weight data for materials used in the product for maintaining it.
    maintenance_material_fractions : xr.DataArray
        Fractions of materials used for maintenance.
    Region : prism.Coords
        The region for the stock.
    Type : prism.Coords
        The type of stock (e.g., vehicles).
    Cohort : prism.Coords
        Cohort groups within the stock (e.g., age groups).
    Time : prism.Coords
        Time steps in the simulation.
    material : prism.Coords
        The material type used in the model.
    input_data : tuple of str
        Tuple of input data variables.
    output_data : tuple of str
        Tuple of output data variables.
    """

    # Input data
    weights: xr.DataArray
    maintenance_material_fractions: xr.DataArray

    # Dimensions
    Region: prism.Coords[REGION]
    Type: prism.Coords[STOCK_TYPE]
    Cohort: prism.Coords[COHORT]
    Time: prism.Coords[TIME]
    material: prism.Coords[MATERIAL_TYPE]

    # Data dependencies
    input_data: tuple[str] = ("weights", "maintenance_material_fractions",
                              "stock_by_cohort")
    output_data: tuple[str] = ("stock_by_cohort_maintenance_materials", )
    #, "inflow_maintenance_materials",
    #                           "outflow_by_cohort_maintenance_materials")

    def compute_initial_values(self, time: prism.Timeline):
        """
        Computes the initial values for maintenance materials used by stock cohorts.
        
        Parameters
        ----------
        time : prism.Timeline
            The simulation timeline.
        """
         
        self.stock_by_cohort_maintenance_materials = xr.DataArray(
            0.0, dims=("Time", "Region", "Type", "material"),
            coords={"Time": self.Time,
                    # "Cohort": coordinates["Time"].values,
                    "Region": self.Region,
                    "Type": self.Type,
                    "material": self.material})

    def compute_values(self, time: prism.Time, stock_by_cohort):
        """
        Computes the maintenance material usage by stock cohort at each time step.
        
        Parameters
        ----------
        time : prism.Time
            The current simulation time step.
        stock_by_cohort : xr.DataArray
            The stock-by-cohort data.
        """
         
        t, dt = time.t, time.dt
        self.stock_by_cohort_maintenance_materials.loc[t] = (stock_by_cohort.loc[t]*self.maintenance_material_fractions*self.weights).sum("Cohort")


@prism.interface
class GenericMainModel(prism.Model):
    """
    The main model class that integrates multiple submodels (stocks, materials, maintenance).
    It initializes submodels based on the configuration and computes values over time.
    
    Attributes
    ----------
    prep_data : dict
        The preparation data for the model.
    compute_materials : bool
        Flag to compute materials data.
    compute_battery_materials : bool
        Flag to compute battery materials data.
    compute_maintenance_materials : bool
        Flag to compute maintenance materials data.
    Region : prism.Coords
        The region for the stock.
    Type : prism.Coords
        The type of stock (e.g., vehicles).
    Cohort : prism.Coords
        Cohort groups within the stock.
    Time : prism.Coords
        Time steps in the simulation.
    material : prism.Coords
        Material type used in the model.
    """

    prep_data: dict
    compute_materials: bool
    compute_battery_materials: bool
    compute_maintenance_materials: bool

    Region: prism.Coords[REGION]
    Type: prism.Coords[STOCK_TYPE]
    Cohort: prism.Coords[COHORT]
    Time: prism.Coords[TIME]
    material: prism.Coords[MATERIAL_TYPE]

    def compute_initial_values(self, timeline: prism.Timeline):
        """
        Initializes the simulation by setting the historic tail as not computed.
        
        Parameters
        ----------
        timeline : prism.Timeline
            The simulation timeline.
        """
         
        self.historic_tail_computed = False

    def init_submodels(self, timeline: prism.Timeline):
        """
        Initializes all necessary submodels based on flags.
        
        Parameters
        ----------
        timeline : prism.Timeline
            The simulation timeline.
        """

        self.complete_timeline = timeline
        self.stock_model = GenericStocks(
            self.complete_timeline, Region=self.Region, Type=self.Type, Cohort=self.Cohort, Time=self.Time,
            stocks=self.prep_data["stocks"],
            lifetimes=self.prep_data["lifetimes"], shares=self.prep_data.get("shares")
        )
        self.stock_model.compute_initial_values(timeline)

        # Material module
        if self.compute_materials:
            self.material_model = GenericMaterials(
                self.complete_timeline, Region=self.Region, Type=self.Type, Cohort=self.Cohort, Time=self.Time,
                material=self.material, weights=self.prep_data["weights"],
                material_fractions=self.prep_data["material_fractions"]
            )
            self.material_model.compute_initial_values(timeline)

        # Battery materials
        if self.compute_battery_materials:
            self.material_model = GenericMaterials(
                self.complete_timeline, Region=self.Region, Type=self.Type, Cohort=self.Cohort, Time=self.Time,
                material=self.material, weights=self.prep_data["weights"],
                material_fractions=self.prep_data["battery_material_fractions"]
            )
            self.material_model.compute_initial_values(timeline)

        # Maintenance materials
        if self.compute_maintenance_materials:
            self.maintenance_model = Maintenance(
                self.complete_timeline, Region=self.Region, Type=self.Type, Cohort=self.Cohort, Time=self.Time,
                material=self.material, weights=self.prep_data["weights"],
                maintenance_material_fractions=self.prep_data["maintenance_material_fractions"]
            )
            self.maintenance_model.compute_initial_values(timeline)


    def compute_values(self, time: prism.Time):
        """
        Computes values for the main model based on submodels and configurations.
        
        Parameters
        ----------
        time : prism.Time
            The current simulation time step.
        """
         
        t, dt = time.t, time.dt
        if not self.historic_tail_computed:
            for historic_time in self.complete_timeline:
                prism_time = prism.Time(self.complete_timeline.start, self.complete_timeline.end,
                                        dt, historic_time)
                if historic_time == self.complete_timeline.start:
                    continue
                if historic_time >= t:
                    break
                self._compute_one_timestep(prism_time)
            self.historic_tail_computed = True
        self._compute_one_timestep(time)

    def _compute_one_timestep(self, time: prism.Time):
        """
        Computes one timestep of values for all submodels.
        #TODO explain why this was created
        
        Parameters
        ----------
        time : prism.Time
            The current simulation time step.
        """
         
        self.stock_model.compute_values(time)
        if self.compute_materials:
            self.material_model.compute_values(time, inflow=self.stock_model.inflow,
                                               stock_by_cohort=self.stock_model.stock_by_cohort,
                                               outflow_by_cohort=self.stock_model.outflow_by_cohort)
        if self.compute_maintenance_materials:
            self.maintenance_model.compute_values(time, stock_by_cohort=self.stock_model.stock_by_cohort)

