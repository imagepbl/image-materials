import prism
import xarray as xr

REGION = prism.Dimension("Region")
STOCK_TYPE = prism.Dimension("Type")
COHORT = prism.Dimension("Cohort")
TIME = prism.Dimension("Time")
MATERIAL_TYPE = prism.Dimension("material")

@prism.interface
class Maintenance(prism.Model):
    """Module to calculate material use for product stock maintenance.

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
    output_data: tuple[str] = ("inflow_maintenance",
                               "outflow_maintenance")

    # Output data
    inflow_maintenance: prism.TimeVariable[REGION, STOCK_TYPE, MATERIAL_TYPE, "count"] = prism.export()
    outflow_maintenance: prism.TimeVariable[REGION, STOCK_TYPE, MATERIAL_TYPE, "count"] = prism.export()

    def compute_initial_values(self, time: prism.Timeline):
        """Compute the initial values for maintenance materials used by stock cohorts.

        Parameters
        ----------
        time : prism.Timeline
            The simulation timeline.

        """

    def compute_values(self, time: prism.Time, stock_by_cohort):
        """Compute the maintenance material usage by stock cohort at each time step.

        Parameters
        ----------
        time : prism.Time
            The current simulation time step.
        stock_by_cohort : xr.DataArray
            The stock-by-cohort data.

        """
        t, dt = time.t, time.dt

        self.inflow_maintenance[t] = (stock_by_cohort.loc[t]*self.maintenance_material_fractions*
                                      self.weights.sel(Cohort=t).drop_vars("Cohort")).sum("Cohort")
        self.outflow_maintenance[t] = self.inflow_maintenance[t]