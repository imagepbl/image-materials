import prism
import xarray as xr
import numpy as np

from imagematerials.concepts import KnowledgeGraph

REGION = prism.Dimension("Region")
STOCK_TYPE = prism.Dimension("Type")
STOCK_SUPERTYPE = prism.Dimension("SuperType")
COHORT = prism.Dimension("Cohort")
TIME = prism.Dimension("Time")
MATERIAL_TYPE = prism.Dimension("material")
BATTERY_TYPE = prism.Dimension("BatteryType")



@prism.interface
class ElectricVehicleBatteries(prism.Model):
    """ Calculates # batteries, energy capacity and materials per battery type for inflow, stock, outflow
    in electric vehicles.


    Notes
    -----
        Assumption: battery lifetime = vehicle lifetime (no explicit battery stock calculation)
    """

    # Input data
    weights:             xr.DataArray
    shares:              xr.DataArray
    material_fractions:  xr.DataArray
    energy_density:      xr.DataArray
    vhc_fraction_v2g:    xr.DataArray
    capacity_fraction_v2g: xr.DataArray
    knowledge_graph_vhc: KnowledgeGraph
    

    # Dimensions
    Type:         prism.Coords[STOCK_TYPE]
    BatteryType:  prism.Coords[BATTERY_TYPE]
    Region:       prism.Coords[REGION]
    Cohort:       prism.Coords[COHORT]
    material:     prism.Coords[MATERIAL_TYPE]
    Time:         prism.Coords[TIME]

    # Data dependencies
    input_data: tuple[str] = ("shares", "weights", "material_fractions", "energy_density", "knowledge_graph_vhc", "vhc_fraction_v2g", "capacity_fraction_v2g", # input from battery preprocessing
                              "stock_by_cohort", "inflow", "outflow_by_cohort") # input from vehicle stock module
    output_data: tuple[str] = ("stock_battery_kWh_v2g", #"outflow_battery_kWh_v2g",#"inflow_battery_kWh_v2g",
                               "inflow_battery_kWh","stock_battery_kWh","outflow_battery_kWh",
                               "inflow_battery_materials","stock_battery_materials","outflow_battery_materials")

    # Output
    inflow_battery_kWh:        prism.TimeVariable[BATTERY_TYPE, STOCK_TYPE, REGION, "kWh"] = prism.export()
    stock_battery_kWh:         prism.TimeVariable[BATTERY_TYPE, STOCK_TYPE, REGION, COHORT, "kWh"] = prism.export()
    outflow_battery_kWh:       prism.TimeVariable[BATTERY_TYPE, STOCK_TYPE, REGION, COHORT, "kWh"] = prism.export()
    # inflow_battery_kWh_v2g:    prism.TimeVariable[BATTERY_TYPE, STOCK_TYPE, REGION, "kWh"] = prism.export()
    # stock_battery_kWh_v2g:     prism.TimeVariable[BATTERY_TYPE, STOCK_TYPE, REGION, COHORT, "kWh"] = prism.export()
    # outflow_battery_kWh_v2g:   prism.TimeVariable[BATTERY_TYPE, STOCK_TYPE, REGION, COHORT, "kWh"] = prism.export()
    inflow_battery_materials:  prism.TimeVariable[BATTERY_TYPE, STOCK_TYPE, REGION, MATERIAL_TYPE, "kg"] = prism.export()
    stock_battery_materials:   prism.TimeVariable[BATTERY_TYPE, STOCK_TYPE, REGION, MATERIAL_TYPE, "kg"] = prism.export() # COHORT,
    outflow_battery_materials: prism.TimeVariable[BATTERY_TYPE, STOCK_TYPE, REGION, MATERIAL_TYPE, "kg"] = prism.export() # COHORT,


    def compute_initial_values(self, time: prism.Timeline):
        """
        """
        # vhc_fraction_v2g only contains Types that are V2G capable (e.g. "Cars - BEV")
        self._types_v2g = self.vhc_fraction_v2g.Type
        self.ROAD_VEHICLE_TYPES = ['Cars - BEV', 'Cars - FCV', 'Cars - HEV', 'Cars - ICE', 
       'Cars - PHEV', 'Cars - Trolley', 'Heavy Freight Trucks - BEV',
       'Heavy Freight Trucks - FCV', 'Heavy Freight Trucks - HEV',
       'Heavy Freight Trucks - ICE', 'Heavy Freight Trucks - PHEV',
       'Heavy Freight Trucks - Trolley', 'Light Commercial Vehicles - BEV',
       'Light Commercial Vehicles - FCV', 'Light Commercial Vehicles - HEV',
       'Light Commercial Vehicles - ICE', 'Light Commercial Vehicles - PHEV',
       'Light Commercial Vehicles - Trolley', 'Medium Freight Trucks - BEV',
       'Medium Freight Trucks - FCV', 'Medium Freight Trucks - HEV',
       'Medium Freight Trucks - ICE', 'Medium Freight Trucks - PHEV',
       'Medium Freight Trucks - Trolley', 'Midi Buses - BEV',
       'Midi Buses - FCV', 'Midi Buses - HEV', 'Midi Buses - ICE',
       'Midi Buses - PHEV', 'Midi Buses - Trolley', 'Regular Buses - BEV',
       'Regular Buses - FCV', 'Regular Buses - HEV', 'Regular Buses - ICE',
       'Regular Buses - PHEV', 'Regular Buses - Trolley']

        self.stock_battery_kWh_v2g = xr.DataArray(
            0.0,
            dims=("Time", "Type", "BatteryType", "Region"), # 
            coords={"Time":         self.Time,
                    "BatteryType":  self.BatteryType,
                    "Type":         self._types_v2g,
                    "Region":       self.Region})
        self.stock_battery_kWh_v2g = prism.Q_(self.stock_battery_kWh_v2g, "kWh")

        
    def compute_values(self, time: prism.Time, inflow, stock_by_cohort, outflow_by_cohort):
        """
        
        """
         
        t, dt = time.t, time.dt

        # select road vehicles only, as only these have batteries
        inflow_t  = inflow[t].sel(Type=self.ROAD_VEHICLE_TYPES)
        stock_t   = stock_by_cohort.loc[t].sel(Type=self.ROAD_VEHICLE_TYPES)
        outflow_t = outflow_by_cohort[t].sel(Type=self.ROAD_VEHICLE_TYPES)

        # 1. Calculate battery mass inflow, stock, outflow (kg)
        inflow_battery_kg  = inflow_t * self.shares.sel(Cohort = t) * self.weights.sel(Cohort = t)
        stock_battery_kg   = (stock_t * self.shares * self.weights)
        outflow_battery_kg = (outflow_t * self.shares * self.weights)    

        # 2. Calculate battery materials (copper, ..) inflow, stock, outflow (kg)
        self.inflow_battery_materials[t]  = (inflow_battery_kg * self.material_fractions.sel(Cohort = t))
        self.stock_battery_materials[t]   = (stock_battery_kg * self.material_fractions).sum(["Cohort"])
        self.outflow_battery_materials[t] = (outflow_battery_kg * self.material_fractions).sum(["Cohort"])

        # 3. Calculate battery energy capacity inflow, stock, outflow (kWh)
        self.inflow_battery_kWh[t]  = inflow_battery_kg / self.energy_density.sel(Cohort = t)
        self.stock_battery_kWh[t]   = stock_battery_kg / self.energy_density
        self.outflow_battery_kWh[t] = outflow_battery_kg / self.energy_density

        # 4. Calculate V2G capable battery energy capacity stock (kWh) (to calculate "Other storage" only the V2G battery stock is needed)
        stock_v2g = self.stock_battery_kWh[t].sel(Type=self._types_v2g) * self.vhc_fraction_v2g
        self.stock_battery_kWh_v2g.loc[dict(Time=t, Type=self._types_v2g)]  = (stock_v2g.sel(Time=t).drop_vars("Time") * self.capacity_fraction_v2g).sum(["Cohort"])
        


@prism.interface
class ElectricVehicleBatteries_TV(prism.Model):
    """ Calculates # batteries, energy capacity and materials per battery type for inflow, stock, outflow
    in electric vehicles.


    Notes
    -----
        Assumption: battery lifetime = vehicle lifetime (no explicit battery stock calculation)
    """

    # Input data
    weights:             xr.DataArray
    shares:              xr.DataArray
    material_fractions:  xr.DataArray
    energy_density:      xr.DataArray
    knowledge_graph_vhc: KnowledgeGraph
    # inflow:              prism.TimeVariable
    # stock_by_cohort:     xr.DataArray   
    # outflow_by_cohort:   prism.TimeVariable
    

    # Dimensions
    Type:         prism.Coords[STOCK_TYPE]
    BatteryType:  prism.Coords[BATTERY_TYPE]
    Region:       prism.Coords[REGION]
    Cohort:       prism.Coords[COHORT]
    material:     prism.Coords[MATERIAL_TYPE]
    Time:         prism.Coords[TIME]

    # Data dependencies
    input_data: tuple[str] = ("shares", "weights", "material_fractions", "energy_density", "knowledge_graph_vhc",
                              "stock_by_cohort", "inflow", "outflow_by_cohort") # from vehicle stock module
    output_data: tuple[str] = ("inflow_battery","stock_battery","outflow_battery",
                               "inflow_battery_kWh","stock_battery_kWh","outflow_battery_kWh",
                               "inflow_battery_materials","stock_battery_materials","outflow_battery_materials")

    # Output
    inflow_battery:            prism.TimeVariable[BATTERY_TYPE, STOCK_TYPE, REGION, "count"] = prism.export()
    stock_battery:             prism.TimeVariable[BATTERY_TYPE, STOCK_TYPE, REGION, COHORT, "count"] = prism.export()
    outflow_battery:           prism.TimeVariable[BATTERY_TYPE, STOCK_TYPE, REGION, COHORT, "count"] = prism.export()
    inflow_battery_kWh:        prism.TimeVariable[BATTERY_TYPE, STOCK_TYPE, REGION, "kWh"] = prism.export()
    stock_battery_kWh:         prism.TimeVariable[BATTERY_TYPE, STOCK_TYPE, REGION, COHORT, "kWh"] = prism.export()
    outflow_battery_kWh:       prism.TimeVariable[BATTERY_TYPE, STOCK_TYPE, REGION, COHORT, "kWh"] = prism.export()
    inflow_battery_materials:  prism.TimeVariable[BATTERY_TYPE, STOCK_TYPE, REGION, MATERIAL_TYPE, "kg"] = prism.export()
    stock_battery_materials:   prism.TimeVariable[BATTERY_TYPE, STOCK_TYPE, REGION, MATERIAL_TYPE, COHORT, "kg"] = prism.export()
    outflow_battery_materials: prism.TimeVariable[BATTERY_TYPE, STOCK_TYPE, REGION, MATERIAL_TYPE, COHORT, "kg"] = prism.export()


    def compute_initial_values(self, time: prism.Timeline):
        """
        """

        pass

        

    def compute_values(self, time: prism.Time, inflow, stock_by_cohort, outflow_by_cohort):
        """
        
        """
         
        t, dt = time.t, time.dt

        # drop non-electric vehicles, otherwise not compatible with battery data
        inflow = inflow.to_array()
        inflow = inflow.drop_sel(Type=["Bikes", "Freight Planes", "Passenger Planes","Small Ships", "Medium Ships", 
                                                 "Large Ships", "Very Large Ships", "Inland Ships", "Freight Trains", "Trains", "High Speed Trains"])
        outflow_by_cohort = outflow_by_cohort.to_array()
        outflow_by_cohort = outflow_by_cohort.drop_sel(Type=["Bikes", "Freight Planes", "Passenger Planes","Small Ships", "Medium Ships", 
                                                 "Large Ships", "Very Large Ships", "Inland Ships", "Freight Trains", "Trains", "High Speed Trains"])
        stock_by_cohort = stock_by_cohort.drop_sel(Type=["Bikes", "Freight Planes", "Passenger Planes","Small Ships", "Medium Ships", 
                                                 "Large Ships", "Very Large Ships", "Inland Ships", "Freight Trains", "Trains", "High Speed Trains"])


        # 1. Calculate batteries inflow, stock, outflow ("count")
        self.inflow_battery[t]  = inflow.loc[t] * self.shares.sel(Cohort = t).drop_vars("Cohort")
        self.stock_battery[t]   = (stock_by_cohort.loc[t] * self.shares) #.sum(["Cohort","battery"])
        self.outflow_battery[t] = (outflow_by_cohort.loc[t] * self.shares)

        # 2. Intermediate variable: battery mass inflow, stock, outflow (kg)
        inflow_battery_kg   = self.inflow_battery[t]  * self.weights.sel(Cohort = t).drop_vars("Cohort")
        stock_battery_kg    = self.stock_battery[t]  * self.weights
        outflow_battery_kg  = self.outflow_battery[t]  * self.weights

        # 3. Calculate battery materials (copper, ..) inflow, stock, outflow (kg)
        self.inflow_battery_materials[t]  = (inflow_battery_kg * self.material_fractions.sel(Cohort = t).drop_vars("Cohort"))
        self.stock_battery_materials[t]   = (stock_battery_kg * self.material_fractions)
        self.outflow_battery_materials[t] = (outflow_battery_kg * self.material_fractions)

        # 4. Calculate battery energy capacity inflow, stock, outflow (kWh)
        self.inflow_battery_kWh[t]  = inflow_battery_kg / self.energy_density.sel(Cohort = t).drop_vars("Cohort")
        self.stock_battery_kWh[t]   = stock_battery_kg / self.energy_density
        self.outflow_battery_kWh[t] = outflow_battery_kg / self.energy_density





@prism.interface
class BatteryMaterials(prism.Model):
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
    output_data: tuple[str] = ("inflow_battery_materials","stock_battery_materials","outflow_battery_materials")

    
    def compute_initial_values(self, time: prism.Timeline):
        """
        Computes the initial values for battery materials (initilizes the DataArrays)
        
        Parameters
        ----------
        time : prism.Timeline
            The simulation timeline.
        """
        self.inflow_battery_materials = xr.DataArray(
            0.0,
            dims=("Time", "Region", "Type","material","battery"),
            coords={"Time": self.Time,
                    "Region": self.Region,
                    "Type": self.Type,
                    "material": self.material})
        
        self.stock_battery_materials = xr.DataArray(
            0.0,
            dims=("Time", "Region", "Type","material","battery"),
            coords={"Time": self.Time,
                    "Region": self.Region,
                    "Type": self.Type,
                    "material": self.material})
        
        self.outflow_battery_materials = xr.DataArray(
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
        inflow_battery_materials : xr.DataArray
            Inflow of battery materials at the current time step, defined per region, vehicle type, material.
        stock_battery_materials : xr.DataArray
            Stock of battery materials at the current time step, defined per region, vehicle type, material.
        outflow_battery_materials : xr.DataArray
            Outflow of battery materials at the current time step, defined per region, vehicle type, material.
        """
         
        t, dt = time.t, time.dt

        # battery_per_vehicles is defined for cohort, battery type, vehicle type, and material
        battery_materials_per_vehicle = self.battery_weights * self.battery_shares * self.battery_materials
        # drop battery dimension to reduce memory usage
        self.inflow_battery_materials.loc[t] = (inflow[t] * battery_materials_per_vehicle.sel(Cohort = t)).sum("battery")
        # assumption: battery lifetime = vehicle lifetime (no explicit battery stock calculation)
        # cohort dimension calculation done internally by xarray (in that way battery shares of stock are used which are different from inflow shares)
        self.stock_battery_materials.loc[t] = (stock_by_cohort.loc[t]*battery_materials_per_vehicle).sum(["Cohort","battery"])
        self.outflow_battery_materials.loc[t] = (outflow_by_cohort[t]*battery_materials_per_vehicle).sum(["Cohort","battery"])
                                  
