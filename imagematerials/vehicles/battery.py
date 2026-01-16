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
    knowledge_graph_vhc: KnowledgeGraph
    # inflow:              prism.TimeVariable
    # stock_by_cohort:     xr.DataArray   
    # outflow_by_cohort:   prism.TimeVariable
    

    # Dimensions
    Type:         prism.Coords[STOCK_TYPE]
    BatteryType:  prism.Coords[BATTERY_TYPE]
    Region:       prism.Coords[REGION]
    Cohort:       prism.Coords[COHORT]
    # Time:         prism.Coords[TIME]
    material:     prism.Coords[MATERIAL_TYPE]

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
    # inflow_battery_kg:      prism.TimeVariable[BATTERY_TYPE, STOCK_TYPE, REGION, MATERIAL_TYPE, "kg"] = prism.export()
    # stock_battery_kg:       prism.TimeVariable[BATTERY_TYPE, STOCK_TYPE, REGION, MATERIAL_TYPE, COHORT, "kg"] = prism.export()
    # outflow_battery_kg:     prism.TimeVariable[BATTERY_TYPE, STOCK_TYPE, REGION, MATERIAL_TYPE, COHORT, "kg"] = prism.export()

    def compute_initial_values(self, time: prism.Timeline):
        """
        """
        inflow = inflow.to_array()
        inflow = inflow.drop_sel(Type=["Bikes", "Freight Planes", "Passenger Planes","Small Ships", "Medium Ships", 
                                                 "Large Ships", "Very Large Ships", "Inland Ships", "Freight Trains", "Trains", "High Speed Trains"])
        outflow_by_cohort = outflow_by_cohort.to_array()
        outflow_by_cohort = outflow_by_cohort.drop_sel(Type=["Bikes", "Freight Planes", "Passenger Planes","Small Ships", "Medium Ships", 
                                                 "Large Ships", "Very Large Ships", "Inland Ships", "Freight Trains", "Trains", "High Speed Trains"])
        stock_by_cohort = stock_by_cohort.drop_sel(Type=["Bikes", "Freight Planes", "Passenger Planes","Small Ships", "Medium Ships", 
                                                 "Large Ships", "Very Large Ships", "Inland Ships", "Freight Trains", "Trains", "High Speed Trains"])
        # self.inflow = self.knowledge_graph_vhc.rebroadcast_xarray(self.inflow, output_coords = self.weights.coords["Type"].values, dim="Type")
        # print(self.inflow.coords["Type"].values)

        pass

        

    def compute_values(self, time: prism.Time, inflow, stock_by_cohort, outflow_by_cohort):
        """
        
        """
         
        t, dt = time.t, time.dt

        print("inflow is zero:")
        print(bool(np.isclose(self.inflow.pint.magnitude, 0).all()))

        # 1. Calculate batteries inflow, stock, outflow ("count")
        self.inflow_battery[t]  = inflow.loc[t] * self.shares.sel(Cohort = t).drop_vars("Cohort")#.sum("Type")
        self.stock_battery[t]   = (stock_by_cohort.loc[t] * self.shares)#.sum("Type") #.sum(["Cohort","battery"])
        self.outflow_battery[t] = (outflow_by_cohort.loc[t] * self.shares)#.sum("Type") #.sum(["Cohort","battery"])

        print("inflow_battery is zero:")
        print(bool(np.isclose(prism.M_(self.inflow_battery), 0).all()))

        # 2. Intermediate variable: battery mass inflow, stock, outflow (kg)
        inflow_battery_kg   = self.inflow_battery[t]  * self.weights.sel(Cohort = t).drop_vars("Cohort")
        stock_battery_kg    = self.stock_battery[t]  * self.weights
        outflow_battery_kg  = self.outflow_battery[t]  * self.weights

        print("inflow_battery_kg is zero:")
        print(bool(np.isclose(prism.M_(inflow_battery_kg), 0).all()))

        # 3. Calculate battery materials (copper, ..) inflow, stock, outflow (kg)
        self.inflow_battery_materials[t]  = (inflow_battery_kg * self.material_fractions.sel(Cohort = t).drop_vars("Cohort"))
        self.stock_battery_materials[t]   = (stock_battery_kg * self.material_fractions)
        self.outflow_battery_materials[t] = (outflow_battery_kg * self.material_fractions)

        print("inflow_battery_materials is zero:")
        print(bool(np.isclose(prism.M_(self.inflow_battery_materials), 0).all()))

        # 4. Calculate battery energy capacity inflow, stock, outflow (kWh)
        self.inflow_battery_kWh[t]  = inflow_battery_kg / self.energy_density.sel(Cohort = t).drop_vars("Cohort")
        self.stock_battery_kWh[t]   = stock_battery_kg / self.energy_density
        self.outflow_battery_kWh[t] = outflow_battery_kg / self.energy_density

        print("inflow_battery_kWh is zero:")
        print(bool(np.isclose(prism.M_(self.inflow_battery_kWh), 0).all()))
        
        # cohort dimension calculation done internally by xarray (in that way battery shares of stock are used which are different from inflow shares)
        




@prism.interface
class ElectricVehicleBatteries_weight(prism.Model):
    """ Calculates # batteries per battery type and their weight
    """

    # Input data
    battery_weights: xr.DataArray
    battery_shares: xr.DataArray

    # Dimensions
    Region: prism.Coords[REGION]
    Type: prism.Coords[STOCK_TYPE]
    battery: prism.Coords[BATTERY_TYPE] 
    Cohort: prism.Coords[COHORT]
    Time: prism.Coords[TIME]

    # Data dependencies
    input_data: tuple[str] = ("battery_shares", "battery_weights",
                              "stock_by_cohort", "inflow", "outflow_by_cohort") #
    output_data: tuple[str] = ("inflow_battery_kg","stock_battery_kg","outflow_battery_kg")

    
    def compute_initial_values(self, time: prism.Timeline):
        """
        """
        self.inflow_battery_kg = xr.DataArray(
            0.0,
            dims=("Time", "Cohort", "Region", "battery", "Type"),
            coords={"Time":      self.Time,
                    "Cohort":    self.Cohort,
                    "Region":    self.Region,
                    "battery":   self.battery,
                    "Type":      self.Type})
        self.inflow_battery_kg = prism.Q_(self.inflow_battery_kg, "kg")
        
        self.stock_battery_kg = xr.DataArray(
            0.0,
            dims=("Time", "Cohort", "Region", "battery", "Type"),
            coords={"Time":      self.Time,
                    "Cohort":    self.Cohort,
                    "Region":    self.Region,
                    "battery":   self.battery,
                    "Type":      self.Type})
        self.stock_battery_kg = prism.Q_(self.stock_battery_kg, "kg")
        
        self.outflow_battery_kg = xr.DataArray(
            0.0,
            dims=("Time", "Cohort", "Region", "battery", "Type"),
            coords={"Time":      self.Time,
                    "Cohort":    self.Cohort,
                    "Region":    self.Region,
                    "battery":   self.battery,
                    "Type":      self.Type})
        self.outflow_battery_kg = prism.Q_(self.outflow_battery_kg, "kg")
        

    def compute_values(self, time: prism.Time, inflow, stock_by_cohort, outflow_by_cohort):
        """
        
        """
         
        t, dt = time.t, time.dt


        # # battery_per_vehicles is defined for cohort, battery type, vehicle type, and material
        # battery_per_vehicle = self.battery_weights * self.battery_shares * self.battery_materials
        # drop battery dimension to reduce memory usage
        self.inflow_battery_kg.loc[t] = inflow[t] * self.battery_shares.sel(Cohort = t) * self.battery_weights.sel(Cohort = t)#.sum("Type")
        # assumption: battery lifetime = vehicle lifetime (no explicit battery stock calculation)
        # cohort dimension calculation done internally by xarray (in that way battery shares of stock are used which are different from inflow shares)
        self.stock_battery_kg.loc[t] = (stock_by_cohort.loc[t] * self.battery_shares * self.battery_weights)#.sum("Type") #.sum(["Cohort","battery"])
        self.outflow_battery_kg.loc[t] = (outflow_by_cohort[t] * self.battery_shares * self.battery_weights)#.sum("Type") #.sum(["Cohort","battery"])



@prism.interface
class ElectricVehicleBatteries_number(prism.Model):
    """ Calculates # batteries per battery type
    """

    # Input data
    # battery_weights: xr.DataArray
    battery_shares: xr.DataArray

    # Dimensions
    Region: prism.Coords[REGION]
    Type: prism.Coords[STOCK_TYPE]
    battery: prism.Coords[BATTERY_TYPE] 
    Cohort: prism.Coords[COHORT]
    Time: prism.Coords[TIME]

    # Data dependencies
    input_data: tuple[str] = ("battery_shares",
                              "stock_by_cohort", "inflow", "outflow_by_cohort") #"battery_weights"
    output_data: tuple[str] = ("inflow_battery","stock_battery","outflow_battery")

    
    def compute_initial_values(self, time: prism.Timeline):
        """
        """

        # vehicles should be SupeType relative to the batteries that are Type
        # inflow = inflow.rename({"Type": "SuperType"})
        # stock_by_cohort = stock_by_cohort.rename({"Type": "SuperType"})
        # outflow_by_cohort = outflow_by_cohort.rename({"Type": "SuperType"})

        self.inflow_battery = xr.DataArray(
            0.0,
            dims=("Time", "Cohort", "Region", "battery", "Type"),
            coords={"Time":      self.Time,
                    "Cohort":    self.Cohort,
                    "Region":    self.Region,
                    "battery":   self.battery,
                    "Type":      self.Type})
        
        self.stock_battery = xr.DataArray(
            0.0,
            dims=("Time", "Cohort", "Region", "battery", "Type"),
            coords={"Time":      self.Time,
                    "Cohort":    self.Cohort,
                    "Region":    self.Region,
                    "battery":   self.battery,
                    "Type":      self.Type})
        
        self.outflow_battery = xr.DataArray(
            0.0,
            dims=("Time", "Cohort", "Region", "battery", "Type"),
            coords={"Time":      self.Time,
                    "Cohort":    self.Cohort,
                    "Region":    self.Region,
                    "battery":   self.battery,
                    "Type":      self.Type})
        

    def compute_values(self, time: prism.Time, inflow, stock_by_cohort, outflow_by_cohort):
        """
        
        """
         
        t, dt = time.t, time.dt


        # # battery_per_vehicles is defined for cohort, battery type, vehicle type, and material
        # battery_per_vehicle = self.battery_weights * self.battery_shares * self.battery_materials
        # drop battery dimension to reduce memory usage
        self.inflow_battery.loc[t] = (inflow[t] * self.battery_shares.sel(Cohort = t)) #.sum("Type")
        # assumption: battery lifetime = vehicle lifetime (no explicit battery stock calculation)
        # cohort dimension calculation done internally by xarray (in that way battery shares of stock are used which are different from inflow shares)
        self.stock_battery.loc[t] = (stock_by_cohort.loc[t]*self.battery_shares) #.sum(["Cohort","battery"])
        self.outflow_battery.loc[t] = (outflow_by_cohort[t]*self.battery_shares) #.sum(["Cohort","battery"])


# @prism.interface
# class ElectricVehicleBatteries(prism.Model):
#     """ Module to calculate material use for vehicle batteries.

#     Calculates inflow, stock, and outflow of batteries in vehicles based on the battery weights, 
#     inflow shares, and materials.

#     Assumptions:
#     - Battery lifetime is equal to vehicle lifetime (no explicit battery stock calculation)
#     - Battery type shares do not vary with vehicle types (all vehicle types have same batteries - to be adjusted later?)

#     Attributes
#     ----------
#     battery_weights : xr.DataArray
#         Weights of batteries in kg (?). Defined per cohort, vehicle type.
#     battery_shares : xr.DataArray
#         Shares of battery types in the inflow, defined per cohort, battery type.
#     Region : prism.Coords
#         #The region for the stock.
#     Type : prism.Coords
#         Vehicles types.
#     Cohort : prism.Coords
#         Cohort groups within the stock (e.g., age groups).
#     Time : prism.Coords
#         Time steps in the simulation.
#     material : prism.Coords
#         The material type used in the model.
#     battery : prism.Coords
#         Battery types.
#     input_data : tuple of str
#         Tuple of input data variables.
#     output_data : tuple of str
#         Tuple of output data variables.
#     """

#     # Input data
#     # battery_weights: xr.DataArray
#     battery_shares: xr.DataArray

#     # Dimensions
#     Region: prism.Coords[REGION]
#     SuperType:  prism.Coords[STOCK_SUPERTYPE]
#     Type: prism.Coords[STOCK_TYPE]
#     Cohort: prism.Coords[COHORT]
#     Time: prism.Coords[TIME]

#     # Data dependencies
#     input_data: tuple[str] = ("battery_shares",
#                               "stock_by_cohort", "inflow", "outflow_by_cohort") #"battery_weights"
#     output_data: tuple[str] = ("inflow_battery","stock_battery","outflow_battery")

    
#     def compute_initial_values(self, time: prism.Timeline, inflow, stock_by_cohort, outflow_by_cohort):
#         """
#         Computes the initial values for battery materials (initilizes the DataArrays)
        
#         Parameters
#         ----------
#         time : prism.Timeline
#             The simulation timeline.
#         """

#         # vehicles should be SupeType relative to the batteries that are Type
#         inflow = inflow.rename({"Type": "SuperType"})
#         stock_by_cohort = stock_by_cohort.rename({"Type": "SuperType"})
#         outflow_by_cohort = outflow_by_cohort.rename({"Type": "SuperType"})

#         self.inflow_battery = xr.DataArray(
#             0.0,
#             dims=("Time", "Cohort", "Region", "SuperType", "Type"),
#             coords={"Time":      self.Time,
#                     "Cohort":    self.Cohort,
#                     "Region":    self.Region,
#                     "SuperType": self.SuperType,
#                     "Type":      self.Type})
        
#         self.stock_battery = xr.DataArray(
#             0.0,
#             dims=("Time", "Cohort", "Region", "SuperType", "Type"),
#             coords={"Time":      self.Time,
#                     "Cohort":    self.Cohort,
#                     "Region":    self.Region,
#                     "SuperType": self.SuperType,
#                     "Type":      self.Type})
        
#         self.outflow_battery = xr.DataArray(
#             0.0,
#             dims=("Time", "Cohort", "Region", "SuperType", "Type"),
#             coords={"Time":      self.Time,
#                     "Cohort":    self.Cohort,
#                     "Region":    self.Region,
#                     "SuperType": self.SuperType,
#                     "Type":      self.Type})
        

#     def compute_values(self, time: prism.Time, inflow, stock_by_cohort, outflow_by_cohort):
#         """
#         Computes the battery materials at each time step for each cohort.
        
#         Parameters
#         ----------
#         time : prism.Time
#             The current simulation time step.
#         inflow : TimeVariable
#             Inflow data for vehicles, defined per region, cohort and vehicle type.
#         stock_by_cohort : xr.DataArray
#             The stock-by-cohort data for vehicles per region, cohort and vehicle type.
#         outflow_by_cohort : TimeVariable
#             Outflow data for vehicles, defined per region, cohort and vehicle type.

#         Returns
#         -------
#         inflow_battery : xr.DataArray
#             Inflow of battery materials at the current time step, defined per region, vehicle type, material.
#         stock_battery : xr.DataArray
#             Stock of battery materials at the current time step, defined per region, vehicle type, material.
#         outflow_battery : xr.DataArray
#             Outflow of battery materials at the current time step, defined per region, vehicle type, material.
#         """
         
#         t, dt = time.t, time.dt


#         # # battery_per_vehicles is defined for cohort, battery type, vehicle type, and material
#         # battery_per_vehicle = self.battery_weights * self.battery_shares * self.battery_materials
#         # drop battery dimension to reduce memory usage
#         self.inflow_battery.loc[t] = (inflow[t] * self.battery_shares.sel(Cohort = t)) #.sum("Type")
#         # assumption: battery lifetime = vehicle lifetime (no explicit battery stock calculation)
#         # cohort dimension calculation done internally by xarray (in that way battery shares of stock are used which are different from inflow shares)
#         self.stock_battery.loc[t] = (stock_by_cohort.loc[t]*self.battery_shares) #.sum(["Cohort","battery"])
#         self.outflow_battery.loc[t] = (outflow_by_cohort[t]*self.battery_shares) #.sum(["Cohort","battery"])




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
                                  
