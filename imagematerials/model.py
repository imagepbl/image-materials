
from typing import Callable, ClassVar, Optional
import pint_xarray
from pint import UnitRegistry
from pathlib import Path
from importlib.resources import files

import prism
import xarray as xr
import numpy as np

from imagematerials.concepts import KnowledgeGraph
from imagematerials.maintenance import Maintenance
from imagematerials.survival import ScipySurvival, SurvivalMatrix
from imagematerials.vehicles.battery import ElectricVehicleBatteries, BatteryMaterials

REGION = prism.Dimension("Region")
STOCK_TYPE = prism.Dimension("Type")
STOCK_SUPERTYPE = prism.Dimension("SuperType")
COHORT = prism.Dimension("Cohort")
TIME = prism.Dimension("Time")
MATERIAL_TYPE = prism.Dimension("material")
BATTERY_TYPE = prism.Dimension("battery")
EOL_TYPE = prism.Dimension("eoltype")
UnitFlexibleStock = prism.DynamicUnit("my_unit_stock")

prism.unit_registry.load_definitions(files(__package__) / "units.txt")

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
        Defines the stock types (e.g., vehicle types, buildings).
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
    knowledge_graph: KnowledgeGraph
    # set a flexible unit that can be changed depending on type of stock - is passed by preprocessing
    set_unit_flexible: prism.VarUnit[UnitFlexibleStock]

    # For module dependency, ignored by prism
    input_data: tuple[str] = ("stocks", "lifetimes", "knowledge_graph", "set_unit_flexible")
    output_data: tuple[str] = ("outflow_by_cohort", "inflow", "stock_by_cohort")

    # stock_by_cohort: prism.TimeVariable[Region, Mode, Cohort, "count"] = prism.export(initial_value = prism.Array[Region, Mode, Cohort, 'count'](0.0))
    inflow: prism.TimeVariable[REGION, STOCK_TYPE, UnitFlexibleStock] = prism.export()
    outflow_by_cohort: prism.TimeVariable[REGION, STOCK_TYPE, COHORT, UnitFlexibleStock] = prism.export()

    def compute_initial_values(self, time: prism.Timeline):
        """Compute the initial values for stocks and the survival matrix.
        Note:
        stock_by_cohort is defined here instead of together with the other output data (inflow, outflow_by_cohort) to increase 
        computational efficiency, might need to change when coupling to TIMER (should be prism.TimeVariable as well then)
       
        Parameters
        ----------
        time : prism.Timeline
            The simulation timeline.
        """
  
        survival = ScipySurvival(self.lifetimes, self.stocks.coords["Type"],
                                 knowledge_graph=self.knowledge_graph)
        self.survival_matrix = SurvivalMatrix(survival)
        self.stock_by_cohort = xr.DataArray(
            0.0,
            dims=("Time", "Cohort", "Region", "Type"),
            coords={"Time": self.Time,
                    "Cohort": self.Cohort,
                    "Region": self.Region,
                    "Type": self.Type})
        self.stock_by_cohort = prism.Q_(self.stock_by_cohort, self.set_unit_flexible) # pass unit from stocks: set_unit_flexible should contain stocks unit

    def compute_values(self, time: prism.Time):
        """
        Computes the stock values at each time step, including inflow and outflow by cohort.
        
        Parameters
        ----------
        time : prism.Time
            The current simulation time step.
        """
        
        t, dt = time.t, time.dt
        self.inflow[t].loc[:] = prism.Q_(0.0, self.set_unit_flexible)
        self.outflow_by_cohort[t].loc[:] = prism.Q_(0.0, self.set_unit_flexible)

        # copy only for readability
        stock_demand = self.stocks
        # calculate missing stock to fulfill demand (input stock)
        stock_diff = stock_demand.loc[t] - self.stock_by_cohort.loc[t].sum("Cohort")
        # stock_diff cannot be negative (no negative inflow); when positive, divide by survival matrix in case there is a loss in the first year (inflow needs to be larger than input stock)
        stock_diff = xr.where(stock_diff>0, stock_diff/self.survival_matrix[t, t].drop("Cohort"), 0)

        self.inflow[t] = stock_diff
        # calculate future development of the current cohort (inflow at time t; t: = time from current time onwards, t = cohort of time t)
        self.stock_by_cohort.loc[t:, t] = self.inflow[t] * self.survival_matrix[t:, t]
        # for t_future in stock_by_cohort[t].coords["Cohort"].loc[t_str:]:
            # t_future = int(t_future)
            # stock_by_cohort[t_future].loc[{"Cohort": t_str}] = inflow[t]*survival[t_future, t]

        # Prevent out of bounds error, assume first outflow to be 0.
        if t-1 < time.start:
            self.outflow_by_cohort[t] = prism.Q_(0.0, self.set_unit_flexible)
        else:
            # for previous cohorts: calculate outflow by subtracting stocks of previous - current year
            self.outflow_by_cohort[t].loc[:, :, :t-1] = self.stock_by_cohort.loc[t-1, :t-1] - self.stock_by_cohort.loc[t, :t-1]
            # for current cohort: calculate outflow by inflow * (1-survival matrix) as stock at t-1 is not existent
            self.outflow_by_cohort[t].loc[:, :, t] = self.inflow[t] * (1-self.survival_matrix[t, t])


@prism.interface
class SharesInflowStocks(prism.Model):
    """Stock class in case that different sub-technologies with different lifetimes are 
    present for which the market shares are known for the INFLOW.
    A model class for managing stocks and their inflows and outflows over time, 
    including the computation of initial and dynamic stock values based on input data.
    
    Attributes
    ----------
    Region : prism.Coords[REGION]
        Defines the regions for the stock.
    SuperType : prism.Coords[STOCK_SUPERTYPE]
        Defines the stock super types (e.g., electricity storage type ("Other storage")).
    Type : prism.Coords[STOCK_TYPE]
        Defines the stock sub types (e.g., different storage sub types (NIMH, Hydrogen, Lithium Sulfur etc.)).
    Cohort : prism.Coords[COHORT]
        Defines the cohorts (e.g., different age groups of stock).
    Time : prism.Coords[TIME]
        Defines the time steps for the stock simulation.
    lifetimes : xr.DataArray
        Expected lifetimes for each stock type.
    stocks : xr.DataArray
        Prescribed stock values of supertype (e.g. how many MWh "Other storage" capacity per year is needed).
    shares : xr.DataArray
        Percentage shares of the INFLOW subtypes (e.g. shares of NIMH, Hydrogen, Lithium Sulfur etc. of "Other storage" technologies).
    input_data : tuple of str
        Tuple of input data variable names.
    output_data : tuple of str
        Tuple of output data variable names.
    """
     
    # Dimensions
    Region:     prism.Coords[REGION]
    SuperType:  prism.Coords[STOCK_SUPERTYPE]
    Type:       prism.Coords[STOCK_TYPE]
    Cohort:     prism.Coords[COHORT]
    Time:       prism.Coords[TIME]
    # generation: Super-Type: wind, solar PV, ... | (Sub-)Type: wind-direct drive, wind-geared, solar PV-crystalline, solar PV-thin film, ...
    # storage:    Super-Type: Other storage       | (Sub-)Type: NiMH, Lithium-Sulfur, hydrogen, ...

    # Inputs
    lifetimes:          xr.DataArray
    stocks:             xr.DataArray
    shares:             xr.DataArray
    knowledge_graph:    KnowledgeGraph
    set_unit_flexible:  prism.VarUnit[UnitFlexibleStock] # set a flexible unit that can be changed depending on type of stock - is passed by preprocessing

    # For module dependency, ignored by prism
    input_data:  tuple[str] = ("stocks", "lifetimes", "knowledge_graph","shares", "set_unit_flexible")
    # "stocks" is by supertype, "lifetimes" & "shares" by subtype
    output_data: tuple[str] = ("outflow_by_cohort", "inflow", "stock_by_cohort") 
    # stock_by_cohort, inflow & outflow_by_cohort by subtype

    # stock_by_cohort:    prism.TimeVariable[Region, Mode, Cohort, "count"] = prism.export(initial_value = prism.Array[Region, Mode, Cohort, 'count'](0.0))
    inflow:             prism.TimeVariable[REGION, STOCK_TYPE, UnitFlexibleStock] = prism.export()
    outflow_by_cohort:  prism.TimeVariable[REGION, STOCK_TYPE, COHORT, UnitFlexibleStock] = prism.export()

    def compute_initial_values(self, time: prism.Timeline):
        """
        Initializes stocks and the survival matrix.

        Creates the survival matrix from lifetime distributions and type shares,
        and initializes the stock-by-cohort array with zeros (carrying the units
        from `self.stocks`).
        
        Parameters
        ----------
        time : prism.Timeline
            The simulation timeline.
        """
        
        survival = ScipySurvival(self.lifetimes, self.shares.coords["Type"],
                                 knowledge_graph=self.knowledge_graph)
        self.survival_matrix = SurvivalMatrix(survival)
        self.stock_by_cohort = xr.DataArray(
            0.0,
            dims=("Time", "Cohort", "Region", "Type"),
            coords={"Time":     self.Time,
                    "Cohort":   self.Cohort,
                    "Region":   self.Region,
                    "Type":     self.shares.coords["Type"]})
        self.stock_by_cohort = prism.Q_(self.stock_by_cohort, self.set_unit_flexible) # pass unit from stocks


    def compute_values(self, time: prism.Time):
        """
        Computes stocks, inflows by cohort, and outflows by cohort for the current time step.

        At time `t`, this method:
        - Computes the inflow required to meet the target stock, distributed across sub-technologies 
        according to shares in the inflow.
        - Calculates future stock development of the current cohort `t` (= inflow in year t) based on 
        the survival matrix.
        - Computes outflow by cohort in year t
        
        Parameters
        ----------
        time : prism.Time
            The current simulation time step.
        """
        # pass unit from stocks
        t, dt = time.t, time.dt
        self.inflow[t].loc[:] = prism.Q_(0.0, self.set_unit_flexible)
        self.outflow_by_cohort[t].loc[:] = prism.Q_(0.0, self.set_unit_flexible)

        stock_demand = self.stocks # copy only for readability
        # calculate missing stock to fulfill demand (input stock) -> for this aggregate over sub-technologies in stock_by_cohort to compare to stock_demand which is by super-type
        stock_diff = stock_demand.loc[t] - self.knowledge_graph.aggregate_sum(self.stock_by_cohort.loc[t].sum("Cohort"), self.stocks.coords["SuperType"].values, dim="Type").rename({"Type": "SuperType"}) # rename needed for the difference calculation - cannot subtract over different dims
        # calculate the inflow by sub-technology by rebroadcasting the stock_diff (by super-type) to sub-technologies according to their shares in the inflow
        inflow_tech = self.knowledge_graph.rebroadcast_xarray(stock_diff, self.stock_by_cohort.coords["Type"].values, dim="SuperType", shares=self.shares.sel(Cohort=t), dim_shares="Type").rename({"SuperType": "Type"})
        # stock_diff cannot be negative (no negative inflow); when positive, divide by survival matrix in case there is a loss in the first year (inflow needs to be larger than input stock)
        inflow_tech = xr.where(inflow_tech>0, inflow_tech/self.survival_matrix[t, t].drop("Cohort"), 0)
        self.inflow[t] = inflow_tech

        # calculate future development of the current cohort (inflow at time t; t: = time from current time onwards, t = cohort of time t)
        self.stock_by_cohort.loc[t:, t] = self.inflow[t] * self.survival_matrix[t:, t]

        # Prevent out of bounds error, assume first outflow to be 0.
        if t-1 < time.start:
            self.outflow_by_cohort[t] = prism.Q_(0.0, self.set_unit_flexible)
        else:
            # for previous cohorts: calculate outflow by subtracting stocks of previous - current year
            self.outflow_by_cohort[t].loc[:, :, :t-1] = self.stock_by_cohort.loc[t-1, :t-1] - self.stock_by_cohort.loc[t, :t-1]
            # for current cohort: calculate outflow by inflow * (1-survival matrix) as stock at t-1 is not existent
            self.outflow_by_cohort[t].loc[:, :, t] = self.inflow[t] * (1-self.survival_matrix[t, t])




@prism.interface
class GenericMaterials(prism.Model):
    """
    A model class for managing materials used in stock cohorts, including 
    inflows and outflows of materials. This version uses stock weights and material 
    fractions to calculate material use.

    Attributes
    ----------
    weights : xr.DataArray
        Weight data for the respective product/stock.
    material_fractions : xr.DataArray
        Material composition of the stock (weight %), varying by cohort.
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
    # stock_by_cohort_materials & outflow_by_cohort_materials is NOT by cohort as it currently requires too much memory

    # Output data
    inflow_materials: prism.TimeVariable[REGION, STOCK_TYPE, MATERIAL_TYPE, "kg"] = prism.export()
    outflow_by_cohort_materials: prism.TimeVariable[REGION, STOCK_TYPE, MATERIAL_TYPE, "kg"] = prism.export()

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
        self.stock_by_cohort_materials = prism.Q_(self.stock_by_cohort_materials, "kg")

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

        self.material_fractions = prism.Q_(self.material_fractions, "1")
        self.weights = prism.Q_(self.weights, "kg/count")
       
        t, dt = time.t, time.dt
        self.inflow_materials[t] = inflow[t]*self.material_fractions.sel(Cohort=t).drop_vars("Cohort")*self.weights.sel(Cohort=t).drop_vars("Cohort")
        self.outflow_by_cohort_materials[t] = (outflow_by_cohort[t]*self.material_fractions*self.weights).sum("Cohort")
        self.stock_by_cohort_materials.loc[t] = (stock_by_cohort.loc[t]*self.material_fractions*self.weights).sum("Cohort")



@prism.interface
class MaterialIntensities(prism.Model):
    """
    A model class for managing materials used in stock cohorts, including 
    inflows and outflows of materials. This version uses material intensities
    (material per unit of stock) instead of weights and material fractions.

    Attributes
    ----------
    material_intensities : xr.DataArray
        Material intenisities applied to the inflow (kg steel/unit of stock), varying by cohort.
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
    # stock_by_cohort_materials & outflow_by_cohort_materials is NOT by cohort as it currently requires too much memory

    # Output data
    inflow_materials: prism.TimeVariable[REGION, STOCK_TYPE, MATERIAL_TYPE, "kg"] = prism.export()
    outflow_by_cohort_materials: prism.TimeVariable[REGION, STOCK_TYPE, MATERIAL_TYPE, "kg"] = prism.export()

    def compute_initial_values(self, time: prism.Timeline):
        """
        Create and initialize the `stock_by_cohort_materials` data array.

        The array is constructed with dimensions (Time, Region, Type, material),
        filled with zeros, and assigned units of kilograms.
        
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
        self.stock_by_cohort_materials = prism.Q_(self.stock_by_cohort_materials, "kg") 

    def compute_values(self, time: prism.Time, inflow, stock_by_cohort, outflow_by_cohort):
        """
        Computes the material inflows, outflows, and stock usage by cohort for the current time step.
        
        At time `t`, computes:
        - material inflows from stock `inflow` and cohort-specific material intensities,
        - material outflows by cohort aggregated across cohorts,
        - material stocks by cohort aggregated across cohorts.
        
        Parameters
        ----------
        time : prism.Time
            The current simulation time step.
        inflow : xr.DataArray
            Inflow data for the stocks.
        stock_by_cohort : xr.DataArray
            Stock-by-cohort data.
        outflow_by_cohort : xr.DataArray
            Outflow data of the stock given by cohort.
        """

        t, dt = time.t, time.dt
        self.inflow_materials[t] = inflow[t]*self.material_intensities.sel(Cohort=t).drop_vars("Cohort")
        self.outflow_by_cohort_materials[t] = (outflow_by_cohort[t]*self.material_intensities).sum("Cohort")
        self.stock_by_cohort_materials.loc[t] = (stock_by_cohort.loc[t]*self.material_intensities).sum("Cohort")


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
    battery: prism.Coords[BATTERY_TYPE]

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

        # EV Battery materials
        if self.compute_battery_materials:
            self.battery_model = Battery(
                self.complete_timeline, Region=self.Region, Type=self.Type, Cohort=self.Cohort, Time=self.Time, 
                material=self.material, battery  = self.battery, battery_weights=self.prep_data["battery_weights"],
                battery_material_fractions=self.prep_data["battery_materials"], battery_shares = self.prep_data["battery_shares"]
            )
            self.battery_model.compute_initial_values(timeline)

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
            self.maintenance_model.compute_values(time, 
                                                  stock_by_cohort=self.stock_model.stock_by_cohort,)


@prism.interface
class EndOfLife(prism.Model):

    # Input data
    #collection: xr.DataArray
    #reuse: xr.DataArray
    #recycling: xr.DataArray

    # Dimensions
    Region: prism.Coords[REGION]
    Type: prism.Coords[STOCK_TYPE]
    Time: prism.Coords[TIME]
    material: prism.Coords[MATERIAL_TYPE]

    # Data dependencies
    input_data: tuple[str] = ("collection", "reuse", "recycling", "inflow_materials", "outflow_by_cohort_materials") # outflow_by_cohort_materials currently summed over cohorts
    output_data: tuple[str] = ("sum_outflow","sum_inflow", "collected_materials","reusable_materials", "recyclable_materials","losses_materials", "virgin_materials")

    # Output data
    sum_outflow: prism.TimeVariable[REGION, STOCK_TYPE, MATERIAL_TYPE, "kg"] = prism.export()
    sum_inflow: prism.TimeVariable[REGION, STOCK_TYPE, MATERIAL_TYPE, "kg"] = prism.export()
    collected_materials: prism.TimeVariable[REGION, STOCK_TYPE, MATERIAL_TYPE, "kg"] = prism.export()
    reusable_materials: prism.TimeVariable[REGION, STOCK_TYPE, MATERIAL_TYPE, "kg"] = prism.export()
    remaining_materials: prism.TimeVariable[REGION, STOCK_TYPE, MATERIAL_TYPE, "kg"] = prism.export()
    recyclable_materials: prism.TimeVariable[REGION, STOCK_TYPE, MATERIAL_TYPE, "kg"] = prism.export()
    losses_materials: prism.TimeVariable[REGION, STOCK_TYPE, MATERIAL_TYPE, "kg"] = prism.export()
    virgin_materials: prism.TimeVariable[REGION, STOCK_TYPE, MATERIAL_TYPE, "kg"] = prism.export()

    def compute_initial_values(self, timeline: prism.Timeline):
        pass

    def compute_values(self, time: prism.Time, inflow_materials, outflow_by_cohort_materials, collection, reuse, recycling):
        """
        Computes reused material within sector and recyclable material from each sector/type, given collection, 
        reuse and recycling rates at each time step 

        Parameters
        ----------
        time : prism.Time
            The current simulation time step.
        inflow_materials : xr.DataArray
            Inflow materials data. 
        outflow_by_cohort_materials : xr.DataArray
            Outflow data after lifetime.
        collection : xr.DataArray
            Collection rate data by material and type. 
        reuse: xr.DataArray
            Reuse rate data by material and type.
        recycling: xr.DataArray
            Recycling rate data by material and type.
        """
        t, dt = time.t, time.dt
        type_dict = {
            'passenger': ['Bikes', 
                          'Cars - BEV', 'Cars - FCV', 'Cars - HEV','Cars - ICE', 'Cars - PHEV', 'Cars - Trolley', 
                          'Light Commercial Vehicles - BEV','Light Commercial Vehicles - FCV','Light Commercial Vehicles - HEV','Light Commercial Vehicles - ICE','Light Commercial Vehicles - PHEV','Light Commercial Vehicles - Trolley',
                          'Regular Buses - PHEV', 'Regular Buses - Trolley','Regular Buses - BEV', 'Regular Buses - FCV','Regular Buses - HEV', 'Regular Buses - ICE',
                          'Midi Buses - BEV','Midi Buses - FCV', 'Midi Buses - HEV', 'Midi Buses - ICE','Midi Buses - PHEV', 'Midi Buses - Trolley',                        
                          'Trains','High Speed Trains',
                          'Passenger Planes',       
        ],
            'freight': ['Freight Planes',
                        'Medium Freight Trucks - BEV', 'Medium Freight Trucks - FCV','Medium Freight Trucks - HEV', 'Medium Freight Trucks - ICE','Medium Freight Trucks - PHEV', 'Medium Freight Trucks - Trolley',
                        'Freight Trains',
                        'Small Ships','Inland Ships','Medium Ships','Large Ships', 'Very Large Ships',
                        'Heavy Freight Trucks - BEV', 'Heavy Freight Trucks - FCV','Heavy Freight Trucks - HEV', 'Heavy Freight Trucks - ICE','Heavy Freight Trucks - PHEV', 'Heavy Freight Trucks - Trolley'
        ],
            'urban': ["Appartment - Urban","Detached - Urban","High-rise - Urban", "Semi-detached - Urban",    
        ],

            'rural': ["Appartment - Rural","Detached - Rural","High-rise - Rural", "Semi-detached - Rural",
        ],
            
            'commercial': ["Office","Retail+","Hotels+","Govt+"
        ],
            'generation':[ 'SPV', 'SPVR', 'CSP', 'WON', 'WOFF',
            'WAVE', 'HYD', 'OREN', 'GEO', 'H2P', 'NUC', 'FREE12', 'ClST', 'OlST',
            'NGOT', 'BioST', 'IGCC', 'OlCC', 'NGCC', 'BioCC', 'ClCS', 'OlCS',
            'NGCS', 'BioCS', 'ClCHP', 'OlCHP', 'NGCHP', 'BioCHP', 'ClCHPCS',
            'OlCHPCS', 'NGCHPCS', 'BioCHPCS', 'GeoCHP', 'H2CHP'
            ],

            'grid':['HV - Lines - Overhead','HV - Lines - Underground','HV - Substations', 'HV - Transformers',
                    'LV - Lines - Overhead','LV - Lines - Underground','LV - Substations', 'LV - Transformers', 
                    'MV - Lines - Overhead', 'MV - Lines - Underground','MV - Substations', 'MV - Transformers' 
            ],

            'storage': ["PHS", 'Compressed Air', 'Deep-cycle Lead-Acid', 'Flywheel', 'Hydrogen FC', 'LFP', 'LMO', 'LTO', 'Lithium Ceramic', 'Lithium Sulfur', 'Lithium-air',
                        'NCA', 'NMC', 'NiMH', 'Sodium-Sulfur', 'Vanadium Redox', 'ZEBRA', 'Zinc-Bromide']
        }

        self.sum_outflow[t] = prism.Q_(0.0, 'kg')
        for outflow in outflow_by_cohort_materials:
            outflow_t = outflow[t] 

            # sum over subtypes to supertypes
            for supertype, subtypes in type_dict.items():
                available = set(outflow_t.coords["Type"].values)                        # available subtypes in outflow 
                present = [subtype for subtype in subtypes if subtype in available]     # present subtypes in outflow (defined above)
                if not present:                                                         # if a subtype of this supertype is not present in outflow, continue
                        continue
                
                # calculate sum outflow by supertype 
                sum_outflow = outflow_t.sel(Type=present).sum("Type")

                # --- reindex to target materials ---
                target_mats = self.sum_outflow.to_array().coords["material"]  #  defining the ordering of material coords
                sum_outflow = sum_outflow.reindex(material=target_mats, fill_value=0)
                # --- harmonize coordinates ---
                coords = {"Type": supertype, "material": target_mats, "Region": sum_outflow.coords["Region"]}    
                input_coords = {"Time":t, "Type":supertype} 
                
                # store sum outflow
                self.sum_outflow[t].loc[coords] = sum_outflow

                # calculate collected, reusable, recyclable materials and losses
                collected_materials = collection.loc[input_coords]*sum_outflow
                reusable_materials = collected_materials*reuse.loc[input_coords]
                remaining_materials = collected_materials-reusable_materials                        # non-reused but collected waste
                recyclable_materials = remaining_materials*recycling.loc[input_coords]
                losses_materials = (sum_outflow-collected_materials                                 # non-collected waste
                                    +remaining_materials-recyclable_materials)                      # non-reused/recycled but collected waste
                
                # --- reindex to target materials ---
                collected_materials = collected_materials.reindex(material=target_mats, fill_value=0)
                reusable_materials = reusable_materials.reindex(material=target_mats, fill_value=0)
                remaining_materials = remaining_materials.reindex(material=target_mats, fill_value=0)
                recyclable_materials = recyclable_materials.reindex(material=target_mats, fill_value=0)
                losses_materials = losses_materials.reindex(material=target_mats, fill_value=0)

                # store results
                self.collected_materials[t].loc[coords] = collected_materials
                self.reusable_materials[t].loc[coords] = reusable_materials
                self.remaining_materials[t].loc[coords] = remaining_materials
                self.recyclable_materials[t].loc[coords] = recyclable_materials
                self.losses_materials[t].loc[coords] = losses_materials
        
        # calculate virgin materials needed (based on recyclables and reusables, fix this in hybrid version)
        for inflow in inflow_materials:
            inflow_t = inflow[t]
            for supertype, subtypes in type_dict.items():                   # sum over subtypes to supertypes
                available = set(map(str, inflow_t.coords["Type"].values))   # available subtypes in inflow
                present = [st for st in subtypes if st in available]        # present subtypes in inflow (defined above)

                if not present:                                             # if a subtype of this supertype is not present in inflow, continue
                    continue

                sum_inflow = inflow_t.sel(Type=present).sum("Type")

                # --- harmonize coordinates ---
                sum_inflow = sum_inflow.reindex_like(collection, fill_value=0)
            
                coords = {"Type": supertype, "material": sum_inflow.coords["material"], "Region": sum_inflow.coords["Region"]}
                input_coords = {"Time":t, "Type":supertype}

                self.sum_inflow[t].loc[coords] = sum_inflow

                virgin_materials = xr.where(                                    # quick fix... TODO: redefine this when trade is implemented 
                    sum_inflow - reusable_materials - recyclable_materials < 0,
                    0,
                    sum_inflow - reusable_materials - recyclable_materials
                )
                virgin_materials = virgin_materials.reindex_like(self.virgin_materials[t], fill_value=0)
                self.virgin_materials[t].loc[coords] = virgin_materials

@prism.interface
class RestOf(prism.Model):

    # Dimensions
    Region: prism.Coords[REGION]
    Time: prism.Coords[TIME]
    material: prism.Coords[MATERIAL_TYPE]

    # Data dependencies
    input_data: tuple[str] = ("gompertz_coefs", "gdp_per_capita", "population", 
                              "historic_diff_consumption_mean", "historic_diff_consumption_total")
    output_data: tuple[str] = ("inflow_materials_rest",)

    # Output data inflow_materials_rest
    # inflow_materials_rest: prism.TimeVariable[REGION, MATERIAL_TYPE] = prism.export()

    def compute_initial_values(self, time: prism.Timeline):
        self.inflow_materials_rest = xr.DataArray(
            0.0, dims=("Time", "Region", "material"),
            coords={
                "Time": self.Time,
                "Region": self.Region,  
                "material": self.material
            }
        )
        self.inflow_materials_rest = prism.Q_(self.inflow_materials_rest, "t")
        
    def compute_values(self, time: prism.Time, gompertz_coefs, gdp_per_capita, population, 
                       historic_diff_consumption_mean, historic_diff_consumption_total):
        t, dt = time.t, time.dt

        if t > 1970:
            # Select coefficients for all regions/materials
            a = gompertz_coefs.sel(coef='a', Time = t)
            b = gompertz_coefs.sel(coef='b', Time = t)
            c = gompertz_coefs.sel(coef='c', Time = t)
            self.inflow_per_capita_rest = (a * np.exp(-b * np.exp(-c * gdp_per_capita.loc[t])))
            self.inflow_per_capita_rest = prism.Q_(self.inflow_per_capita_rest, "t/person")
            self.inflow_materials_rest.loc[t] = self.inflow_per_capita_rest * population.loc[t]
            
            # Create a mask of where values are nan
            mask = np.isnan(self.inflow_materials_rest.loc[t])
           
            # Use the mask to fill nans with historic_diff_consumption
            # Align historic_diff_consumption to the same dims/order as inflow_materials_rest

            self.inflow_materials_rest.loc[t] = xr.where(
                mask,
                historic_diff_consumption_mean.transpose(*self.inflow_materials_rest.loc[t].dims),
                self.inflow_materials_rest.loc[t]
            )
            
            # check if real historic data is available (not nan)
            real_historic_data_mask = ~np.isnan(historic_diff_consumption_total.sel(Time=t))
            self.inflow_materials_rest.loc[t] = xr.where(
                real_historic_data_mask,
                historic_diff_consumption_total.sel(Time=t),
                self.inflow_materials_rest.loc[t]
            )

        else:
            pass # No inflow before 1970
        