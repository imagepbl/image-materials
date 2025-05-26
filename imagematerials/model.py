from typing import Callable, ClassVar, Optional

import prism
import xarray as xr

from imagematerials.concepts import KnowledgeGraph
from imagematerials.maintenance import Maintenance
from imagematerials.survival import ScipySurvival, SurvivalMatrix
from imagematerials.vehicles.battery import Battery

REGION = prism.Dimension("Region")
STOCK_TYPE = prism.Dimension("Type")
COHORT = prism.Dimension("Cohort")
TIME = prism.Dimension("Time")
MATERIAL_TYPE = prism.Dimension("material")
<<<<<<< HEAD
BATTERY_TYPE = prism.Dimension("battery")

=======
EOL_TYPE = prism.Dimension("eol")
>>>>>>> 820f8ea (add structure for EndOfLife class)

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
    knowledge_graph: KnowledgeGraph

    # For module dependency, ignored by prism
    input_data: tuple[str] = ("stocks", "lifetimes", "knowledge_graph")
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

        input_stock = self.stocks
        stock_diff = input_stock.loc[t] - self.stock_by_cohort.loc[t].sum("Cohort")
        # Drop dimension cohort
        stock_diff = xr.where(stock_diff>0, stock_diff/self.survival_matrix[t, t].drop("Cohort"), 0)
        self.inflow[t] = stock_diff
        self.stock_by_cohort.loc[t:, t] = self.inflow[t] * self.survival_matrix[t:, t]
        # for t_future in stock_by_cohort[t].coords["Cohort"].loc[t_str:]:
            # t_future = int(t_future)
            # stock_by_cohort[t_future].loc[{"Cohort": t_str}] = inflow[t]*survival[t_future, t]

        # Prevent out of bounds error, assume first outflow to be 0.
        if t-1 < time.start:
            self.outflow_by_cohort[t] = 0.0
        else:
            self.outflow_by_cohort[t] = self.stock_by_cohort.loc[t-1] - self.stock_by_cohort.loc[t]


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
class RestModel(prism.Model):
    # Input data
    gdp_per_capita: xr.DataArray  # Will be a prism time variable probably
    population: xr.DataArray

    # Output data
    inflow_materials_rest: prism.TimeVariable[REGION, STOCK_TYPE, MATERIAL_TYPE, "count"] = prism.export()

    input_data: tuple[str] = ("total_inflow_materials_class", "gdp_per_capita", "population")
    output_data: tuple[str] = ("inflow_materials_rest")

    def compute_values(self, time: prism.Time, total_inflow_materials_class):
        t = time.t
        self.total_inflow_materials_rest[t] = self.total_inflow_materials_class.predict(self.gdp_per_capita)*self.population


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

        # Battery materials
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
class GenericEndOfLife(prism.Model):

    # Input data
    collection: xr.DataArray
    reuse: xr.DataArray
    recycling: xr.DataArray

    # Dimensions
    Region: prism.Coords[REGION]
    Type: prism.Coords[STOCK_TYPE]
    Time: prism.Coords[TIME]
    material: prism.Coords[MATERIAL_TYPE]
    eol: prism.Coords [EOL_TYPE]

    # Data dependencies
    input_data: tuple[str] = ("collection", "reuse", "recycling",
                              "outflow_by_cohort_materials")        # I think that's the variable needed as input from all sectors, but I'm not sure. I'm assuming this is the material outflows from each sector and type, per region and material   
   
    output_data: tuple[str] = ("end_of_life_materials")

    # Output data
    end_of_life_materials: prism.TimeVariable[REGION, STOCK_TYPE, MATERIAL_TYPE, EOL_TYPE, "count"] = prism.export()

    def compute_initial_values(self, time: prism.Timeline):
        self.end_of_life_materials = xr.DataArray(
            0.0, dims=("Time", "Region", "Type", "material","eol"),
            coords={"Time": self.Time,
                    "Region": self.Region,
                    "Type": self.Type,
                    "material": self.material,
                    "eol": self.eol
                    }
                    )

    def compute_values(self, time: prism.Time, outflow_by_cohort_materials, collection, reuse, recycling): 
        """
        Computes reused material within sector and recyclable material from each sector/type, given collection, 
        reuse and recycling rates at each time step 

        Parameters
        ----------
        time : prism.Time
            The current simulation time step.
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
        self.collected[t] = outflow_by_cohort_materials[t] * self.collection        
        #self.losses[t] = outflow_by_cohort_materials[t] - self.collected[t]     # non-collected waste
        #self.reusable [t] = self.collected[t] * self.reuse 
        #self.remaining [t] = self.collected[t] - self.reusable [t]              # non-reused but collected waste
        #self.recyclable [t] = self.remaining[t] * self.recycling 
        #self.losses [t] += (self.remaining[t] -  self.recyclable [t])           # non-reused/recycled but collected waste

