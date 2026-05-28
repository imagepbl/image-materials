"""Vehicle stocks prism model for yearly stock calculations.

This module provides a prism.Model that calculates vehicle stocks from yearly
passenger and tonne kilometer demands, using the shares and conversion factors
determined during preprocessing.
"""

import logging
import xarray as xr
import numpy as np
import pandas as pd
import prism
from pathlib import Path

from imagematerials.concepts import KnowledgeGraph, create_region_graph
from imagematerials.constants import IMAGE_REGIONS, base_directory
from imagematerials.preprocessing import get_preprocessing_data
from imagematerials.util import read_climate_policy_config
from imagematerials.vehicles.constants import (
    END_YEAR,
    LIGHT_COMMERCIAL_VEHICLE_SHARE,
    MEGA_TO_TERA,
    PKMS_TO_VKMS,
    REGIONS,
    START_YEAR,
    all_modes
)
#from imagematerials.vehicles.preprocessing.main_prism import vehicles_preprocessing_integration
from imagematerials.vehicles.shares_prism import get_vehicle_shares_prism
from imagematerials.vehicles.preprocessing.util import (
    xarray_conversion
)

REGION = prism.Dimension("Region")
STOCK_TYPE = prism.Dimension("Type")
TIME = prism.Dimension("Time")
UnitFlexibleStock = prism.DynamicUnit("my_unit_stock")


@prism.interface
class VehicleStocks(prism.Model):
    """Vehicle stocks prism model for yearly calculations.
    
    Calculates vehicle stocks from yearly passenger and tonne kilometer demands,
    applying vehicle technology shares and conversion factors to determine the
    number of vehicles needed in each year and region.
    
    Attributes
    ----------
    Region : list
        Geographic regions (derived from preprocessing data)
    Type : list
        Vehicle types (including subtypes with fuel/technology, derived from preprocessing data)
    Time : list
        Time steps for simulation (derived from preprocessing data)
    passengerkms : prism.TimeVariable[Region, Type]
        Yearly passenger kilometers demanded
    tonekms : prism.TimeVariable[Region, Type]
        Yearly tonne kilometers demanded
    conversion_factor_tkms : xr.DataArray
        Conversion factor from tkms to vehicle numbers
    first_year_vehicle : pd.DataFrame
        First year of operation for each vehicle type
    market_share : xr.DataArray
        Market share of vehicle types (passenger vs freight)
    vehicle_shares : prism.TimeVariable[Type, SubType, Region]
        Technology and fuel type shares for vehicles over time
    knowledge_graph : KnowledgeGraph
        Vehicle knowledge graph for rebroadcasting
    set_unit_flexible : str
        Unit designation for stock (typically "count")
    
    Outputs
    -------
    stocks : prism.TimeVariable[Region, Type]
        Number of vehicles by type and region for each year
    """
    
    # Dimensions - derived dynamically from preprocessing data in compute_initial_values
    # Region: prism.Coords[REGION]
    # Type: prism.Coords[STOCK_TYPE]
    # Time: prism.Coords[TIME]
    
    # Inputs - Time-varying
    #passengerkms: prism.TimeVariable
    #tonekms: prism.TimeVariable
    #vehicle_shares: prism.TimeVariable
    
    # Output - dimensions are set dynamically in compute_initial_values
    stocks: prism.TimeVariable = prism.export()
    
    input_data: tuple[str] = (
        "passengerkms", "tonekms", "vehicle_shares", "conversion_factor_tkms", 
        "first_year_vehicle", "market_share", "knowledge_graph", "set_unit_flexible"
    )
    output_data: tuple[str] = ("stocks",)

    climate_policy_scenario_dir: Path | None = None

    @staticmethod
    def _normalize_year_key(year_value):
        year_key = getattr(year_value, "m", getattr(year_value, "magnitude", year_value))
        if isinstance(year_key, np.ndarray) and year_key.size == 1:
            year_key = year_key.item()
        if isinstance(year_key, (float, np.floating)) and year_key.is_integer():
            year_key = int(year_key)
        if isinstance(year_key, (np.integer, int)):
            return int(year_key)
        return year_key

    def compute_initial_values(self, 
                               time: prism.Timeline, 
                               climate_policy_scenario_dir: Path | None = None):
        """Initialize the stocks TimeVariable with zeros.
        
        Parameters
        ----------
        time : prism.Timeline
            The simulation timeline
        """
        if climate_policy_scenario_dir is None:
            climate_policy_scenario_dir = self.climate_policy_scenario_dir

        # Load preprocessing data first
        # TODO: make this flexible with scenarios etc set in settings/constants file
        vehicle_preprocessing = get_preprocessing_data(
            "vehicles", 
            base_directory,
            climate_policy_scenario_dir=climate_policy_scenario_dir,
            circular_economy_scenario_dirs=None,
            integration_preprocessing = True
        )
        
        # TODO: fix dimensions here such that they get imported from the data
        # Extract dimensions from preprocessing data if not already set
        if not hasattr(self, 'Region') or self.Region is None:
            self.Region = list(vehicle_preprocessing.coordinates["Region"])
        if not hasattr(self, 'Type') or self.Type is None:
            self.Type = list(vehicle_preprocessing.coordinates["Type"])
        if not hasattr(self, 'Time') or self.Time is None:
            self.Time = list(vehicle_preprocessing.coordinates["Time"])
        
        # Extract unit if not set
        if not hasattr(self, 'set_unit_flexible') or self.set_unit_flexible is None:
            self.set_unit_flexible = vehicle_preprocessing.prep_data.get("set_unit_flexible", "count")
        
        unit = str(self.set_unit_flexible)
        self.stocks = xr.DataArray(
            0.0,
            dims=("Time", "Region", "Type"),
            coords={
                "Time": self.Time,
                "Region": self.Region,
                "Type": self.Type
            }
        )
        self.stocks = prism.Q_(self.stocks, unit)

        # Note: add flag to set settings 
        vehicle_data = vehicle_preprocessing.all_data
        self.knowledge_graph = vehicle_data["knowledge_graph"]
        self.lifetimes = vehicle_data["lifetimes"]
        self.maintenance_material_fractions = vehicle_data["maintenance_material_fractions"]
        self.material_fractions = vehicle_data["material_fractions"]
        self.conversion_factor_tkms = vehicle_data["conversion_factor_tkms"]
        self.first_year_vehicle = vehicle_data["first_year_vehicle"]
        self.market_share = vehicle_data["market_share"]
        self.weights = vehicle_data["weights"]

        # Vehicle shares may be absent in integration preprocessing; compute them on demand.
        if "vehicle_shares" in vehicle_data:
            self.vehicle_shares = vehicle_data["vehicle_shares"]
        elif "shares" in vehicle_data:
            self.vehicle_shares = vehicle_data["shares"]
        else:
            climate_policy_config = read_climate_policy_config(climate_policy_scenario_dir)
            self.vehicle_shares = get_vehicle_shares_prism(
                climate_policy_scenario_dir,
                climate_policy_config,
                self.knowledge_graph,
            )

        # dimension check?
        # compute historic tail
    
    def compute_values(self, time: prism.Time):
        """Calculate vehicle stocks for the current timestep.
        
        Computes the number of vehicles needed to satisfy the yearly passenger
        and tonne kilometer demands, using conversion factors and vehicle shares.
        
        Parameters
        ----------
        time : prism.Time
            Current simulation time step
        """
        t = time.t
        year_key = self._normalize_year_key(t)
        unit = str(self.set_unit_flexible)

        # Get the km data for this year and keep xarray coordinates intact.
        pkms_year = self.passenger_kms[t]
        if hasattr(pkms_year, "m"):
            pkms_year = pkms_year.m
        elif hasattr(pkms_year, "magnitude"):
            pkms_year = pkms_year.magnitude

        tkm_source = getattr(self, "tone_kms", None)
        if tkm_source is None:
            tkm_source = getattr(self, "tonkms", None)
        if tkm_source is None:
            raise AttributeError("VehicleStocks requires `tone_kms` or `tonkms` input")

        tkms_year = tkm_source[t]
        if hasattr(tkms_year, "m"):
            tkms_year = tkms_year.m
        elif hasattr(tkms_year, "magnitude"):
            tkms_year = tkms_year.magnitude
        
        # Calculate total vehicles for simple types and aggregated freight
        total_vehicles = self._calculate_vehicle_stocks(pkms_year, tkms_year, year_key)
        
        # Align coordinates to the stocks variable to avoid coordinate-order conflicts.
        total_vehicles = total_vehicles.reindex(
            Type=self.stocks.coords["Type"].values,
            Region=self.stocks.coords["Region"].values,
            fill_value=0.0,
        )

        # Store result using label-based Time indexing (calendar year, not positional index).
        self.stocks.loc[dict(Time=year_key)] = prism.Q_(total_vehicles, unit)
    
    def _calculate_vehicle_stocks(self, pkms_year, tkms_year, year):
        """Calculate vehicle stocks from yearly km demands.
        
        Computes the number of vehicles needed for each type and region based on
        yearly passenger and tonne kilometer demands.
        
        Parameters
        ----------
        pkms_year : xr.DataArray
            Passenger kilometers for the year (dimensions: Region, Type)
        tkms_year : xr.DataArray
            Tonne kilometers for the year (dimensions: Region, Type)
        year : int
            Current year
            
        Returns
        -------
        xr.DataArray
            Number of vehicles by type and region (dimensions: Type, Region)
        """
        # Calculate total vehicles by type
        total_vehicles = self._calculate_total_vehicles_by_type(pkms_year, tkms_year, year)
        
        # Apply vehicle shares to expand to subtypes (fuel/technology variants)
        stocks_with_subtypes = self._apply_vehicle_shares(total_vehicles, year)
        
        return stocks_with_subtypes
    
    def _calculate_total_vehicles_by_type(self, pkms_year, tkms_year, year):
        """Calculate total vehicles for each main vehicle type.
        
        Implements calculations for:
        1. Simple vehicle types (Passenger Planes, Trains, High Speed Trains, Bikes)
        2. Trucks with LCV handling
        3. Freight planes (50% reduction)
        4. Various freight types (convert from Mega-tkm to Tera-tkm)
        5. Cars and buses (region-specific)
        6. Ships (special handling)
        
        Parameters
        ----------
        pkms_year : xr.DataArray
            Passenger kilometers for the year
        tkms_year : xr.DataArray
            Tonne kilometers for the year
        year : int
            Current year
            
        Returns
        -------
        pd.DataFrame
            Total vehicles by type and region
        """
        # Initialize result DataFrame with all modes and regions
        total_nr_vehicles = pd.DataFrame(
            0.0,
            index=[year],
            columns=pd.MultiIndex.from_product([all_modes, range(1, REGIONS + 3)],
                                               names=["Type", "Region"])
        )
        
        # 1. Simple vehicle types - no conversion needed
        for label in ["Passenger Planes", "Trains", "High Speed Trains", "Bikes"]:
            if label in pkms_year.coords.get("Type", []):
                # Use conversion factor and apply to pkms
                total_nr_vehicles[label] = self.conversion_factor_tkms[label].values * pkms_year.sel(Type=label).values
        
        # 2 & 3 & 4. Freight vehicles with special handling
        # Trucks are calculated differently because the IMAGE model does not account for LCV trucks
        if "Medium Freight Trucks" in tkms_year.coords.get("Type", []) and \
           "Heavy Freight Trucks" in tkms_year.coords.get("Type", []):
            trucks_total_tkm = tkms_year.sel(Type="Medium Freight Trucks") + tkms_year.sel(Type="Heavy Freight Trucks")
            trucks_LCV_tkm = trucks_total_tkm * LIGHT_COMMERCIAL_VEHICLE_SHARE
            MFT_percshare_tkm = tkms_year.sel(Type="Medium Freight Trucks") / trucks_total_tkm
            HFT_percshare_tkm = tkms_year.sel(Type="Heavy Freight Trucks") / trucks_total_tkm
            trucks_min_LCV = trucks_total_tkm - trucks_LCV_tkm
            trucks_MFT_tkm = trucks_min_LCV * MFT_percshare_tkm
            trucks_HFT_tkm = trucks_min_LCV * HFT_percshare_tkm
            
            # Apply conversion factors (these are already in tera tkm units due to conversion_factor)
            total_nr_vehicles["Medium Freight Trucks"] = (self.conversion_factor_tkms["Medium Freight Trucks"].values * 
                                                          trucks_MFT_tkm.values)
            total_nr_vehicles["Heavy Freight Trucks"] = (self.conversion_factor_tkms["Heavy Freight Trucks"].values * 
                                                         trucks_HFT_tkm.values)
            total_nr_vehicles["Light Commercial Vehicles"] = (self.conversion_factor_tkms["Light Commercial Vehicles"].values * 
                                                              trucks_LCV_tkm.values)
        
        # Freight planes - reduced by 50% (cargo on passenger planes)
        if "Freight Planes" in tkms_year.coords.get("Type", []):
            air_freight_tkms = tkms_year.sel(Type="Freight Planes") * 0.5
            total_nr_vehicles["Freight Planes"] = (self.conversion_factor_tkms["Freight Planes"].values / MEGA_TO_TERA * 
                                                    air_freight_tkms.values)
        
        # Other freight types needing conversion from Mega to Tera
        for vehicle_type in ["Freight Trains", "Inland Ships"]:
            if vehicle_type in tkms_year.coords.get("Type", []):
                total_nr_vehicles[vehicle_type] = (self.conversion_factor_tkms[vehicle_type].values / MEGA_TO_TERA * 
                                                    tkms_year.sel(Type=vehicle_type).values)
        
        # 5. Cars and buses (region-specific)
        if "Cars" in pkms_year.coords.get("Type", []) or "Regular Buses" in pkms_year.coords.get("Type", []):
            cars_buses = self._calculate_cars_buses(pkms_year, year)
            for vehicle_type, values in cars_buses.items():
                total_nr_vehicles[vehicle_type] = values
        
        # 6. Ships (special handling)
        if "international shipping" in tkms_year.coords.get("Type", []):
            ships = self._calculate_ships(tkms_year, year)
            for vehicle_type, values in ships.items():
                total_nr_vehicles[vehicle_type] = values
        
        return total_nr_vehicles
    
    def _calculate_cars_buses(self, pkms_year, year):
        """Calculate vehicles for cars and buses with region-specific handling.
        
        These vehicle types require individual region handling due to region-specific
        mileage/load data. The conversion is from passenger kilometers to vehicle counts
        using region-specific parameters.
        
        Parameters
        ----------
        pkms_year : xr.DataArray
            Passenger kilometers for the year with dimensions (Region, Type)
        year : int
            Current year
            
        Returns
        -------
        dict
            Dictionary mapping vehicle types to numpy arrays of vehicle counts
        """
        result = {}
        
        # Process Cars
        if "Cars" in pkms_year.coords.get("Type", []):
            car_pkms = pkms_year.sel(Type="Cars")
            # Convert passenger-km to vehicle numbers using conversion factor
            # The conversion factor should account for occupancy and mileage
            result["Cars"] = (self.conversion_factor_tkms["Cars"].values * car_pkms.values)
        
        # Process Regular Buses
        if "Regular Buses" in pkms_year.coords.get("Type", []):
            bus_pkms = pkms_year.sel(Type="Regular Buses")
            result["Regular Buses"] = (self.conversion_factor_tkms["Regular Buses"].values * bus_pkms.values)
        
        # Process Midi Buses
        if "Midi Buses" in pkms_year.coords.get("Type", []):
            midi_bus_pkms = pkms_year.sel(Type="Midi Buses")
            result["Midi Buses"] = (self.conversion_factor_tkms["Midi Buses"].values * midi_bus_pkms.values)
        
        return result
    
    def _calculate_ships(self, tkms_year, year):
        """Calculate vessels for international shipping with special handling.
        
        Ships have special calculation logic involving:
        - Different ship types (Small, Medium, Large, Very Large)
        - Ship capacity, loadfactor, and mileage characteristics
        - Share-based allocation of freight to ship types
        
        Parameters
        ----------
        tkms_year : xr.DataArray
            Tonne kilometers for the year with dimensions (Region, Type)
        year : int
            Current year
            
        Returns
        -------
        dict
            Dictionary mapping ship types to numpy arrays of vessel counts
        """
        result = {}
        
        # Get total international shipping demand
        if "international shipping" in tkms_year.coords.get("Type", []):
            total_shipping_tkm = tkms_year.sel(Type="international shipping")
            
            # Distribute across 4 ship types
            # These distributions should be based on capacity/loadfactor characteristics
            # For now, use equal distribution as placeholder - adjust based on actual ship data
            ship_types = ["Small Ships", "Medium Ships", "Large Ships", "Very Large Ships"]
            ship_fraction = 1.0 / len(ship_types)
            
            for ship_type in ship_types:
                # Allocate fraction of total tkms to each ship type
                ship_tkm = total_shipping_tkm * ship_fraction
                
                # Convert to vessel count using conversion factor
                if ship_type in self.conversion_factor_tkms.coords.get("Type", []):
                    result[ship_type] = (self.conversion_factor_tkms[ship_type].values / MEGA_TO_TERA * 
                                        ship_tkm.values)
                else:
                    # If no specific conversion factor, use generic approach
                    # Number of ships = tkms / (capacity * loadfactor * mileage)
                    result[ship_type] = ship_tkm.values / 100.0  # Placeholder denominator
        
        return result
    
    def _apply_vehicle_shares(self, total_vehicles, year):
        """Apply vehicle technology and fuel type shares to expand to subtypes.
        
        Converts aggregated vehicle counts to subcategories based on available
        technology and fuel type shares (e.g., Cars -> Cars-ICE, Cars-BEV, etc.).
        
        The shares represent the fraction of each vehicle type that uses a particular
        technology or fuel type in the given year.
        
        Parameters
        ----------
        total_vehicles : pd.DataFrame
            Total vehicles by main type and region, indexed by Type and Region
        year : int
            Current year
            
        Returns
        -------
        xr.DataArray
            Vehicles by type/subtype and region with dimensions (Type, Region)
        """
        # Get shares for the current year using label-based Time selection.
        year_key = self._normalize_year_key(year)

        if hasattr(self.vehicle_shares, "dims") and "Time" in self.vehicle_shares.dims:
            try:
                year_shares = self.vehicle_shares.sel(Time=year_key)
            except Exception:
                year_shares = self.vehicle_shares.sel(Time=year_key, method="nearest")
        else:
            year_shares = self.vehicle_shares[year_key]
        
        # Build output coordinates from model dimensions (fallback to share coords).
        subtypes = np.asarray(getattr(self, "Type", []))
        if subtypes.size == 0 and hasattr(year_shares, 'coords') and 'Type' in year_shares.coords:
            subtypes = year_shares.coords['Type'].values

        regions = np.asarray(getattr(self, "Region", []))
        if regions.size == 0 and hasattr(year_shares, 'coords') and 'Region' in year_shares.coords:
            regions = year_shares.coords['Region'].values
        
        # Create result dataframe with subtype structure
        subtype_vehicles = pd.DataFrame(
            0.0,
            index=[year_key],
            columns=pd.MultiIndex.from_product([subtypes, regions],
                                               names=["Type", "Region"])
        )
        
        # Allocate total vehicles to subtypes based on shares
        # Iterate over main vehicle types in total_vehicles
        for vehicle_type in total_vehicles.columns.get_level_values(0).unique():
            # Get count for this vehicle type across regions as a 1D row vector.
            type_totals = np.asarray(total_vehicles.loc[year_key, vehicle_type]).reshape(-1)
            
            # Get shares for this vehicle type if available
            if vehicle_type in year_shares.coords.get('Type', []):
                type_shares = year_shares.sel(Type=vehicle_type)
                
                # Allocate to each subtype based on share
                # type_shares should have dimensions (SubType, Region)
                if 'SubType' in type_shares.coords:
                    for region_idx, region in enumerate(regions):
                        if region_idx >= type_totals.size:
                            break
                        region_total = float(type_totals[region_idx])
                        if region_total > 0:
                            if 'Region' in type_shares.coords and region in type_shares.coords['Region'].values:
                                region_shares = type_shares.sel(Region=region)
                            else:
                                region_shares = type_shares.isel(Region=region_idx)
                            # Allocate total vehicles by fuel/technology share
                            for subtype, share_value in zip(region_shares.coords['SubType'].values, 
                                                            region_shares.values):
                                subtype_key = f"{vehicle_type} - {subtype}"
                                target_col = (subtype_key, region)
                                if target_col in subtype_vehicles.columns:
                                    subtype_vehicles.loc[year_key, target_col] = region_total * float(share_value)
            else:
                # If no shares are available, pass totals through to matching type/region columns.
                for region_idx, region in enumerate(regions):
                    if region_idx >= type_totals.size:
                        break
                    target_col = (vehicle_type, region)
                    if target_col in subtype_vehicles.columns:
                        subtype_vehicles.loc[year_key, target_col] = float(type_totals[region_idx])
        
        # Convert one-year row with MultiIndex columns to a proper 2D Type x Region matrix.
        row_series = subtype_vehicles.loc[year_key]
        full_index = pd.MultiIndex.from_product([subtypes, regions], names=["Type", "Region"])
        row_series = row_series.reindex(full_index, fill_value=0.0)
        matrix = row_series.unstack("Region")
        matrix = matrix.reindex(index=subtypes, columns=regions, fill_value=0.0)

        return xr.DataArray(
            matrix.to_numpy(dtype=float),
            dims=("Type", "Region"),
            coords={
                "Type": matrix.index.to_numpy(),
                "Region": matrix.columns.to_numpy(),
            },
        )
