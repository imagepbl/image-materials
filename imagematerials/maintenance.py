import prism
import xarray as xr

from imagematerials.constants import modes
from imagematerials.vehicles.constants import maintenance_lifetime_per_mode
from imagematerials.concepts import create_vehicle_graph


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

@prism.interface
class MaintenanceLinear(prism.Model):
    """
    Enhanced Maintenance model that includes linear maintenance factors.
    Maintenance increases linearly over vehicle lifetime while preserving total maintenance.
    """
    
    # Input data (matching original + vehicle_lifetimes)
    weights: xr.DataArray
    maintenance_material_fractions: xr.DataArray
    #maintenance_lifetime_per_mode: xr.DataArray 
    
    # Dimensions (matching original)
    Region: prism.Coords[REGION]
    Type: prism.Coords[STOCK_TYPE]
    Cohort: prism.Coords[COHORT]
    Time: prism.Coords[TIME]
    material: prism.Coords[MATERIAL_TYPE]
    
    # Data dependencies (matching original + vehicle_lifetimes)
    input_data: tuple[str] = ("weights", "maintenance_material_fractions", "stock_by_cohort")#, "maintenance_lifetime_per_mode")
    output_data: tuple[str] = ("inflow_maintenance", "outflow_maintenance")
    
    # Output data (matching original)
    inflow_maintenance: prism.TimeVariable[REGION, STOCK_TYPE, MATERIAL_TYPE, "count"] = prism.export()
    outflow_maintenance: prism.TimeVariable[REGION, STOCK_TYPE, MATERIAL_TYPE, "count"] = prism.export()
    
    # Configuration parameters for age-dependent maintenance
    end_multiplier: float = 10  
    
    def compute_initial_values(self, time: prism.Timeline):
        """Compute the initial values for maintenance materials used by stock cohorts.

        Parameters
        ----------
        time : prism.Timeline
            The simulation timeline.
        """

        maintenance_lifetime_per_mode_ = maintenance_lifetime_per_mode.copy()
        maintenance_lifetime_per_mode_['Vehicles'] = 0
        modes_lifetime = list(maintenance_lifetime_per_mode_.keys())
        expected_lifetimes = xr.DataArray(
                data=[maintenance_lifetime_per_mode_[mode] for mode in modes_lifetime],
                dims=["Type"],
                coords={"Type": modes_lifetime},
                name="vehicle_lifetime"
            )

        self.vehicle_lifetime_maintenance = create_vehicle_graph().rebroadcast_xarray(expected_lifetimes, output_coords=modes, dim="Type")

        
        # Pre-identify which vehicle types have maintenance 
        self.maintenance_types = set(self.maintenance_material_fractions.coords["Type"].values)
        
        print(f"Initialized MaintenanceWithAge with {len(self.maintenance_types)} maintenance types")
    
    def compute_initial_values(self, time: prism.Timeline):
        """Setup vehicle lifetimes and maintenance types."""
        # Create vehicle lifetimes
        maintenance_lifetime_per_mode_ = maintenance_lifetime_per_mode.copy()
        maintenance_lifetime_per_mode_['Vehicles'] = 0
        modes_lifetime = list(maintenance_lifetime_per_mode_.keys())
        
        expected_lifetimes = xr.DataArray(
            data=[maintenance_lifetime_per_mode_[mode] for mode in modes_lifetime],
            dims=["Type"],
            coords={"Type": modes_lifetime},
            name="vehicle_lifetime"
        )
        
        self.vehicle_lifetime_maintenance = create_vehicle_graph().rebroadcast_xarray(
            expected_lifetimes, output_coords=modes, dim="Type"
        )
        
        # Cache maintenance types for performance
        self.maintenance_types = set(self.maintenance_material_fractions.coords["Type"].values)
        
        print(f"Initialized MaintenanceLinear with {len(self.maintenance_types)} maintenance types")
    
    def compute_values(self, time: prism.Time, stock_by_cohort):
        """Compute maintenance with age-dependent linear factors (vectorized)."""
        t, dt = time.t, time.dt
        current_stock = stock_by_cohort.loc[t]
        
        # Calculate vehicle ages
        #cohort_years = current_stock.coords["Cohort"].values.astype(int)
        #ages = xr.DataArray(
        #    t - cohort_years, 
        #    dims=["Cohort"], 
        #    coords={"Cohort": current_stock.coords["Cohort"]}
        #)

        ages = t-current_stock.coords["Cohort"]
        
        # Get vehicle lifetimes
        lifetimes = self.vehicle_lifetime_maintenance.sel(Type=current_stock.coords["Type"])
        
        # Calculate linear age factors (preserves total maintenance over lifetime)
        start_factor = 2.0 / (self.end_multiplier + 1)

        ages_bc, lifetimes_bc = xr.broadcast(ages, lifetimes)
        slope = xr.where(lifetimes_bc > 0, ((start_factor * self.end_multiplier) - start_factor) / lifetimes_bc, 0.0)
        age_factors = start_factor + slope * ages_bc

        has_maintenance = xr.DataArray(
            [str(vtype) in self.maintenance_types for vtype in current_stock.coords["Type"].values],
            dims=["Type"], 
            coords={"Type": current_stock.coords["Type"]}
        )

        age_factors = xr.where(has_maintenance, age_factors, 0)
        
        # Calculate maintenance with age factors and sum over cohorts
        maintenance = (
            current_stock * 
            self.maintenance_material_fractions * 
            self.weights.sel(Cohort=current_stock.coords["Cohort"]) *
            age_factors
        ).sum("Cohort")
        
        self.inflow_maintenance[t] = maintenance
        self.outflow_maintenance[t] = maintenance

@prism.interface
class MaintenanceLinearAgeBins(prism.Model):
    """
    Enhanced Maintenance model that includes linear maintenance factors.
    Maintenance increases linearly over vehicle lifetime while preserving total maintenance.
    """
    
    # Input data (matching original + vehicle_lifetimes)
    weights: xr.DataArray
    maintenance_material_fractions: xr.DataArray
    #maintenance_lifetime_per_mode: xr.DataArray 
    
    # Dimensions (matching original)
    Region: prism.Coords[REGION]
    Type: prism.Coords[STOCK_TYPE]
    Cohort: prism.Coords[COHORT]
    Time: prism.Coords[TIME]
    material: prism.Coords[MATERIAL_TYPE]
    
    # Data dependencies (matching original + vehicle_lifetimes)
    input_data: tuple[str] = ("weights", "maintenance_material_fractions", "stock_by_cohort")#, "maintenance_lifetime_per_mode")
    output_data: tuple[str] = ("inflow_maintenance", "outflow_maintenance")
    
    # Output data (matching original)
    inflow_maintenance: prism.TimeVariable[REGION, STOCK_TYPE, MATERIAL_TYPE, "count"] = prism.export()
    outflow_maintenance: prism.TimeVariable[REGION, STOCK_TYPE, MATERIAL_TYPE, "count"] = prism.export()
    
    # Configuration parameters for age-dependent maintenance
    end_multiplier: float = 10  
    
    def compute_initial_values(self, time: prism.Timeline):
        """Setup vehicle lifetimes and maintenance types."""
        # Create vehicle lifetimes
        maintenance_lifetime_per_mode_ = maintenance_lifetime_per_mode.copy()
        maintenance_lifetime_per_mode_['Vehicles'] = 0
        modes_lifetime = list(maintenance_lifetime_per_mode_.keys())
        
        expected_lifetimes = xr.DataArray(
            data=[maintenance_lifetime_per_mode_[mode] for mode in modes_lifetime],
            dims=["Type"],
            coords={"Type": modes_lifetime},
            name="vehicle_lifetime"
        )
        
        self.vehicle_lifetime_maintenance = create_vehicle_graph().rebroadcast_xarray(
            expected_lifetimes, output_coords=modes, dim="Type"
        )
        
        # Cache maintenance types for performance
        self.maintenance_types = set(self.maintenance_material_fractions.coords["Type"].values)
        
        self.age_distribution = None
        self.age_categories = None
        
                
        # Define age categories
        age_bins = [0, 5, 10, 15, 20, 25, 30, 40, 100]  # Age groups
        age_labels = ['0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-40', '40+']
        
        # Get coordinates from the model
        time_coords = time.coords
        type_coords = self.maintenance_material_fractions.coords["Type"]
        region_coords = self.maintenance_material_fractions.coords["Region"]
        
        # Create age distribution array
        self.age_distribution = prism.Variable(
            dims=["time", "Region", "Type", "age_category"],
            coords={
                "time": time_coords,
                "Region": region_coords, 
                "Type": type_coords,
                "age_category": age_labels
            },
            name="age_distribution"
        )
        
        # Store age bins for categorization
        self.age_bins = age_bins
        self.age_labels = age_labels

        print(f"Initialized MaintenanceLinear with {len(self.maintenance_types)} maintenance types")
           
    def compute_values(self, time: prism.Time, stock_by_cohort):
        t, dt = time.t, time.dt
        current_stock = stock_by_cohort.loc[t]
        
        # Calculate ages
        ages = t - current_stock.coords["Cohort"]
        
        # Get vehicle lifetimes
        lifetimes = self.vehicle_lifetime_maintenance.sel(Type=current_stock.coords["Type"])
        
        # Calculate maintenance (existing code)
        start_factor = 2.0 / (self.end_multiplier + 1)
        ages_bc, lifetimes_bc = xr.broadcast(ages, lifetimes)
        slope = xr.where(lifetimes_bc > 0, ((start_factor * self.end_multiplier) - start_factor) / lifetimes_bc, 0.0)
        age_factors = start_factor + slope * ages_bc
        
        # Cap age factors
        max_age_factor = start_factor * self.end_multiplier
        age_factors = xr.where(ages_bc <= lifetimes_bc, age_factors, max_age_factor)
        
        # Apply maintenance mask
        has_maintenance = xr.DataArray(
            [str(vtype) in self.maintenance_types for vtype in current_stock.coords["Type"].values],
            dims=["Type"], 
            coords={"Type": current_stock.coords["Type"]}
        )
        age_factors = xr.where(has_maintenance, age_factors, 0)
        
        # Calculate maintenance (existing code)
        maintenance = (
            current_stock * 
            self.maintenance_material_fractions * 
            self.weights.sel(Cohort=current_stock.coords["Cohort"]) *
            age_factors
        ).sum("Cohort")
        
        self.inflow_maintenance[t] = maintenance
        

        """Calculate age distribution by categorizing vehicles into age groups."""
        import pandas as pd
        
        # Convert to pandas for easier manipulation
        stock_df = stock.to_pandas()
        ages_df = ages.to_pandas()
        lifetimes_df = lifetimes.to_pandas()
        
        # Initialize age distribution for this time step
        age_dist = {}
        
        for region in stock.coords["Region"].values:
            age_dist[region] = {}
            
            for vtype in stock.coords["Type"].values:
                age_dist[region][vtype] = {}
                
                # Get stock and ages for this region/type
                if hasattr(stock_df.index, 'levels'):  # MultiIndex
                    stock_slice = stock_df.loc[(region, vtype), :]
                    ages_slice = ages_df.loc[(region, vtype), :]
                else:
                    stock_slice = stock_df.loc[region, vtype, :]
                    ages_slice = ages_df.loc[region, vtype, :]
                
                lifetime = lifetimes_df.loc[vtype] if vtype in lifetimes_df.index else 0
                
                # Categorize by age groups
                for i, (age_label, age_min, age_max) in enumerate(zip(self.age_labels, self.age_bins[:-1], self.age_bins[1:])):
                    if age_max == 100:  # Last category is open-ended
                        mask = ages_slice >= age_min
                    else:
                        mask = (ages_slice >= age_min) & (ages_slice < age_max)
                    
                    total_stock = stock_slice[mask].sum() if mask.any() else 0
                    age_dist[region][vtype][age_label] = total_stock
                
                # Add special categories
                if lifetime > 0:
                    # Vehicles beyond lifetime
                    beyond_lifetime_mask = ages_slice > lifetime
                    beyond_lifetime_stock = stock_slice[beyond_lifetime_mask].sum() if beyond_lifetime_mask.any() else 0
                    age_dist[region][vtype]['beyond_lifetime'] = beyond_lifetime_stock
                    
                    # Vehicles near end of life (last 20% of lifetime)
                    near_eol_mask = ages_slice > (lifetime * 0.8)
                    near_eol_stock = stock_slice[near_eol_mask].sum() if near_eol_mask.any() else 0
                    age_dist[region][vtype]['near_end_of_life'] = near_eol_stock
        
        # Store in the xarray structure
        for age_label in self.age_labels:
            age_data = []
            for region in stock.coords["Region"].values:
                region_data = []
                for vtype in stock.coords["Type"].values:
                    region_data.append(age_dist[region][vtype].get(age_label, 0))
                age_data.append(region_data)
            
            self.age_distribution[t].loc[dict(age_category=age_label)] = age_data