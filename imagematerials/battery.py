import prism
import xarray as xr

REGION = prism.Dimension("Region")
STOCK_TYPE = prism.Dimension("Type")
COHORT = prism.Dimension("Cohort")
TIME = prism.Dimension("Time")
MATERIAL_TYPE = prism.Dimension("material")
BATTERY_TYPE = prism.Dimension("battery")

@prism.interface
class Battery(prism.Model):
    """ Module to calculate material use for product stock maintenance

    A model class for managing maintenance-related materials used over time, 
    including stock-by-cohort maintenance material use and computation of values.
    Attributes
    ----------
    weights : xr.DataArray
        Weight data for materials used in the product for maintaining it.
    maintenance_material_fractions : xr.DataArray
        Fractions of materials used for maintenance.
    Region : prism.Coords
        #The region for the stock.
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
    battery_weights: xr.DataArray
    battery_shares: xr.DataArray
    battery_material_fractions: xr.DataArray

    # Dimensions
    Region: prism.Coords[REGION]
    Type: prism.Coords[STOCK_TYPE]
    Cohort: prism.Coords[COHORT]
    Time: prism.Coords[TIME]
    material: prism.Coords[MATERIAL_TYPE]
    battery: prism.Coords[BATTERY_TYPE] 

    # Data dependencies
    input_data: tuple[str] = ("battery_weights", "battery_material_fractions",
                              "stock_by_cohort", "inflow")
    output_data: tuple[str] = ("inflow_battery",
                               "stock_battery")

    # Output data
    inflow_battery: prism.TimeVariable[REGION, BATTERY_TYPE, MATERIAL_TYPE, "count"] = prism.export()
    stock_battery: prism.TimeVariable[REGION, BATTERY_TYPE, MATERIAL_TYPE, "count"] = prism.export()

    def compute_initial_values(self, time: prism.Timeline):
        """
        Computes the initial values for maintenance materials used by stock cohorts.
        
        Parameters
        ----------
        time : prism.Timeline
            The simulation timeline.
        """
        

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

        self.inflow_battery[t] = (stock_by_cohort.loc[t]*battery_weights[t]).sum("Type")
                                  
        #                          self.maintenance_material_fractions*
        #                              self.weights.sel(Cohort=t).drop_vars("Cohort")).sum("Cohort")
        #self.stock_battery[t] = 