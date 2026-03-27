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

from imagematerials.concepts import KnowledgeGraph, create_region_graph
from imagematerials.constants import IMAGE_REGIONS
from imagematerials.vehicles.constants import (
    END_YEAR,
    LIGHT_COMMERCIAL_VEHICLE_SHARE,
    MEGA_TO_TERA,
    PKMS_TO_VKMS,
    REGIONS,
    START_YEAR,
    all_modes,
)
from imagematerials.vehicles.shares_prism import get_vehicle_shares_prism
from imagematerials.vehicles.preprocessing.util import (
    xarray_conversion
)

REGION = prism.Dimension("Region")
STOCK_TYPE = prism.Dimension("Type")
TIME = prism.Dimension("Time")
UnitFlexibleStock = prism.DynamicUnit("my_unit_stock")


@prism.interface
class Stocks(prism.Model):
    """Vehicle stocks prism model for yearly calculations.
    
    Calculates vehicle stocks from yearly passenger and tonne kilometer demands,
    applying vehicle technology shares and conversion factors to determine the
    number of vehicles needed in each year and region.
    
    Attributes
    ----------
    Region : prism.Coords[REGION]
        Geographic regions
    Type : prism.Coords[STOCK_TYPE]
        Vehicle types (including subtypes with fuel/technology)
    Time : prism.Coords[TIME]
        Time steps for simulation
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
    
    # Dimensions
    Region: prism.Coords[REGION]
    Type: prism.Coords[STOCK_TYPE]
    Time: prism.Coords[TIME]
    
    # Inputs - Time-varying
    passengerkms: prism.TimeVariable
    tonekms: prism.TimeVariable
    vehicle_shares: prism.TimeVariable
    
    # Inputs - Static
    conversion_factor_tkms: xr.DataArray
    first_year_vehicle: pd.DataFrame
    market_share: xr.DataArray
    knowledge_graph: KnowledgeGraph
    set_unit_flexible: prism.VarUnit[UnitFlexibleStock]
    
    # Output
    stocks: prism.TimeVariable[REGION, STOCK_TYPE, UnitFlexibleStock] = prism.export()
    
    input_data: tuple[str] = (
        "passengerkms", "tonekms", "vehicle_shares", "conversion_factor_tkms", 
        "first_year_vehicle", "market_share", "knowledge_graph", "set_unit_flexible"
    )
    output_data: tuple[str] = ("stocks",)
    
    def compute_initial_values(self, time: prism.Timeline):
        """Initialize the stocks TimeVariable with zeros.
        
        Parameters
        ----------
        time : prism.Timeline
            The simulation timeline
        """
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
        unit = str(self.set_unit_flexible)
        
        # Get the km data for this year
        pkms_year = self.passengerkms[t].values if hasattr(self.passengerkms[t], 'values') else self.passengerkms[t]
        tkms_year = self.tonekms[t].values if hasattr(self.tonekms[t], 'values') else self.tonekms[t]
        
        # Calculate total vehicles for simple types and aggregated freight
        total_vehicles = self._calculate_vehicle_stocks(pkms_year, tkms_year, t)
        
        # Store result in stocks TimeVariable
        self.stocks[t].loc[:] = prism.Q_(total_vehicles, unit)
    
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
        # Get shares for the current year from TimeVariable
        year_shares = self.vehicle_shares[year]
        
        # Initialize result structure for subtypes
        # Determine available subtypes from shares
        if hasattr(year_shares, 'coords') and 'Type' in year_shares.coords:
            subtypes = year_shares.coords['Type'].values
            regions = year_shares.coords.get('Region', self.Region.values).values
        else:
            # Fallback if structure differs
            subtypes = self.Type.values
            regions = self.Region.values
        
        # Create result dataframe with subtype structure
        subtype_vehicles = pd.DataFrame(
            0.0,
            index=[year],
            columns=pd.MultiIndex.from_product([subtypes, regions],
                                               names=["Type", "Region"])
        )
        
        # Allocate total vehicles to subtypes based on shares
        # Iterate over main vehicle types in total_vehicles
        for vehicle_type in total_vehicles.columns.get_level_values(0).unique():
            # Get count for this vehicle type across regions
            type_totals = total_vehicles[vehicle_type]
            
            # Get shares for this vehicle type if available
            if vehicle_type in year_shares.coords.get('Type', []):
                type_shares = year_shares.sel(Type=vehicle_type)
                
                # Allocate to each subtype based on share
                # type_shares should have dimensions (SubType, Region)
                if 'SubType' in type_shares.coords:
                    for region in regions:
                        region_total = type_totals.get(region, 0.0)
                        if region_total > 0:
                            region_shares = type_shares.sel(Region=region)
                            # Allocate total vehicles by fuel/technology share
                            for subtype, share_value in zip(region_shares.coords['SubType'].values, 
                                                            region_shares.values):
                                subtype_key = f"{vehicle_type} - {subtype}"
                                if subtype_key in subtype_vehicles.columns:
                                    subtype_vehicles.loc[year, (subtype_key, region)] = region_total * float(share_value)
            else:
                # If no shares available for this type, keep it as is
                subtype_vehicles.loc[year, vehicle_type] = total_vehicles.loc[year, vehicle_type].values
        
        # Convert to xarray
        return xr.DataArray(
            subtype_vehicles.values,
            dims=("Type", "Region"),
            coords={
                "Type": subtype_vehicles.columns.get_level_values(0).unique(),
                "Region": subtype_vehicles.columns.get_level_values(1).unique()
            }
        )
