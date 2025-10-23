import prism
import xarray as xr

REGION = prism.Dimension("Region")
STOCK_TYPE = prism.Dimension("Type")
COHORT = prism.Dimension("Cohort")
TIME = prism.Dimension("Time")
MATERIAL_TYPE = prism.Dimension("material")
BATTERY_TYPE = prism.Dimension("battery")

@prism.interface
class ElectricVehicleBatteries(prism.Model):
    """ Module to calculate material use for vehicle batteries.

    Calculates inflow, stock, and outflow of batteries in vehicles based on the battery weights, 
    inflow shares, and materials.

    Assumptions:
    - Battery lifetime is equal to vehicle lifetime (no explicit battery stock calculation)
    - Battery type shares do not vary with vehicle types (all vehicle types have same batteries - to be adjusted later?)

    Attributes
    ----------
    battery_weights : xr.DataArray
        Weights of batteries in kg (?). Defined per cohort, vehicle type.
    battery_shares : xr.DataArray
        Shares of battery types in the inflow, defined per cohort, battery type.
    battery_materials : xr.DataArray
        Materials used in batteries, defined per cohort, battery type, material.
    Region : prism.Coords
        #The region for the stock.
    Type : prism.Coords
        Vehicles types.
    Cohort : prism.Coords
        Cohort groups within the stock (e.g., age groups).
    Time : prism.Coords
        Time steps in the simulation.
    material : prism.Coords
        The material type used in the model.
    battery : prism.Coords
        Battery types.
    input_data : tuple of str
        Tuple of input data variables.
    output_data : tuple of str
        Tuple of output data variables.
    """

    # Input data
    battery_weights: xr.DataArray
    battery_shares: xr.DataArray
    battery_materials: xr.DataArray

    # Dimensions
    Region: prism.Coords[REGION]
    Type: prism.Coords[STOCK_TYPE]
    Cohort: prism.Coords[COHORT]
    Time: prism.Coords[TIME]
    material: prism.Coords[MATERIAL_TYPE]
    battery: prism.Coords[BATTERY_TYPE] 

    # Data dependencies
    input_data: tuple[str] = ("battery_weights", "battery_materials", "battery_shares",
                              "stock_by_cohort", "inflow", "outflow_by_cohort")
    output_data: tuple[str] = ("inflow_battery","stock_battery","outflow_battery")

    
    def compute_initial_values(self, time: prism.Timeline):
        """
        Computes the initial values for battery materials (initilizes the DataArrays)
        
        Parameters
        ----------
        time : prism.Timeline
            The simulation timeline.
        """
        self.inflow_battery = xr.DataArray(
            0.0,
            dims=("Time", "Region", "Type","material","battery"),
            coords={"Time": self.Time,
                    "Region": self.Region,
                    "Type": self.Type,
                    "material": self.material})
        
        self.stock_battery = xr.DataArray(
            0.0,
            dims=("Time", "Region", "Type","material","battery"),
            coords={"Time": self.Time,
                    "Region": self.Region,
                    "Type": self.Type,
                    "material": self.material})
        
        self.outflow_battery = xr.DataArray(
            0.0,
            dims=("Time", "Region", "Type","material","battery"),
            coords={"Time": self.Time,
                    "Region": self.Region,
                    "Type": self.Type,
                    "material": self.material})
        

    def compute_values(self, time: prism.Time, inflow, stock_by_cohort, outflow_by_cohort):
        """
        Computes the battery materials at each time step for each cohort.
        
        Parameters
        ----------
        time : prism.Time
            The current simulation time step.
        inflow : TimeVariable
            Inflow data for vehicles, defined per region, cohort and vehicle type.
        stock_by_cohort : xr.DataArray
            The stock-by-cohort data for vehicles per region, cohort and vehicle type.
        outflow_by_cohort : TimeVariable
            Outflow data for vehicles, defined per region, cohort and vehicle type.

        Returns
        -------
        inflow_battery : xr.DataArray
            Inflow of battery materials at the current time step, defined per region, vehicle type, material.
        stock_battery : xr.DataArray
            Stock of battery materials at the current time step, defined per region, vehicle type, material.
        outflow_battery : xr.DataArray
            Outflow of battery materials at the current time step, defined per region, vehicle type, material.
        """
         
        t, dt = time.t, time.dt

        # battery_per_vehicles is defined for cohort, battery type, vehicle type, and material
        battery_per_vehicle = self.battery_weights * self.battery_shares * self.battery_materials
        # drop battery dimension to reduce memory usage
        self.inflow_battery.loc[t] = (inflow[t] * battery_per_vehicle.sel(Cohort = t)).sum("battery")
        # assumption: battery lifetime = vehicle lifetime (no explicit battery stock calculation)
        # cohort dimension calculation done internally by xarray (in that way battery shares of stock are used which are different from inflow shares)
        self.stock_battery.loc[t] = (stock_by_cohort.loc[t]*battery_per_vehicle).sum(["Cohort","battery"])
        self.outflow_battery.loc[t] = (outflow_by_cohort[t]*battery_per_vehicle).sum(["Cohort","battery"])


@prism.interface
class Battery(prism.Model):
    """ Module to calculate material use for vehicle batteries.

    Calculates inflow, stock, and outflow of batteries in vehicles based on the battery weights, 
    inflow shares, and materials.

    Assumptions:
    - Battery lifetime is equal to vehicle lifetime (no explicit battery stock calculation)
    - Battery type shares do not vary with vehicle types (all vehicle types have same batteries - to be adjusted later?)

    Attributes
    ----------
    battery_weights : xr.DataArray
        Weights of batteries in kg (?). Defined per cohort, vehicle type.
    battery_shares : xr.DataArray
        Shares of battery types in the inflow, defined per cohort, battery type.
    battery_materials : xr.DataArray
        Materials used in batteries, defined per cohort, battery type, material.
    Region : prism.Coords
        #The region for the stock.
    Type : prism.Coords
        Vehicles types.
    Cohort : prism.Coords
        Cohort groups within the stock (e.g., age groups).
    Time : prism.Coords
        Time steps in the simulation.
    material : prism.Coords
        The material type used in the model.
    battery : prism.Coords
        Battery types.
    input_data : tuple of str
        Tuple of input data variables.
    output_data : tuple of str
        Tuple of output data variables.
    """

    # Input data
    battery_weights: xr.DataArray
    battery_shares: xr.DataArray
    battery_materials: xr.DataArray

    # Dimensions
    Region: prism.Coords[REGION]
    Type: prism.Coords[STOCK_TYPE]
    Cohort: prism.Coords[COHORT]
    Time: prism.Coords[TIME]
    material: prism.Coords[MATERIAL_TYPE]
    battery: prism.Coords[BATTERY_TYPE] 

    # Data dependencies
    input_data: tuple[str] = ("battery_weights", "battery_materials", "battery_shares",
                              "stock_by_cohort", "inflow", "outflow_by_cohort")
    output_data: tuple[str] = ("inflow_battery","stock_battery","outflow_battery")

    
    def compute_initial_values(self, time: prism.Timeline):
        """
        Computes the initial values for battery materials (initilizes the DataArrays)
        
        Parameters
        ----------
        time : prism.Timeline
            The simulation timeline.
        """
        self.inflow_battery = xr.DataArray(
            0.0,
            dims=("Time", "Region", "Type","material","battery"),
            coords={"Time": self.Time,
                    "Region": self.Region,
                    "Type": self.Type,
                    "material": self.material})
        
        self.stock_battery = xr.DataArray(
            0.0,
            dims=("Time", "Region", "Type","material","battery"),
            coords={"Time": self.Time,
                    "Region": self.Region,
                    "Type": self.Type,
                    "material": self.material})
        
        self.outflow_battery = xr.DataArray(
            0.0,
            dims=("Time", "Region", "Type","material","battery"),
            coords={"Time": self.Time,
                    "Region": self.Region,
                    "Type": self.Type,
                    "material": self.material})
        

    def compute_values(self, time: prism.Time, inflow, stock_by_cohort, outflow_by_cohort):
        """
        Computes the battery materials at each time step for each cohort.
        
        Parameters
        ----------
        time : prism.Time
            The current simulation time step.
        inflow : TimeVariable
            Inflow data for vehicles, defined per region, cohort and vehicle type.
        stock_by_cohort : xr.DataArray
            The stock-by-cohort data for vehicles per region, cohort and vehicle type.
        outflow_by_cohort : TimeVariable
            Outflow data for vehicles, defined per region, cohort and vehicle type.

        Returns
        -------
        inflow_battery : xr.DataArray
            Inflow of battery materials at the current time step, defined per region, vehicle type, material.
        stock_battery : xr.DataArray
            Stock of battery materials at the current time step, defined per region, vehicle type, material.
        outflow_battery : xr.DataArray
            Outflow of battery materials at the current time step, defined per region, vehicle type, material.
        """
         
        t, dt = time.t, time.dt

        # battery_per_vehicles is defined for cohort, battery type, vehicle type, and material
        battery_per_vehicle = self.battery_weights * self.battery_shares * self.battery_materials
        # drop battery dimension to reduce memory usage
        self.inflow_battery.loc[t] = (inflow[t] * battery_per_vehicle.sel(Cohort = t)).sum("battery")
        # assumption: battery lifetime = vehicle lifetime (no explicit battery stock calculation)
        # cohort dimension calculation done internally by xarray (in that way battery shares of stock are used which are different from inflow shares)
        self.stock_battery.loc[t] = (stock_by_cohort.loc[t]*battery_per_vehicle).sum(["Cohort","battery"])
        self.outflow_battery.loc[t] = (outflow_by_cohort[t]*battery_per_vehicle).sum(["Cohort","battery"])
                                  
