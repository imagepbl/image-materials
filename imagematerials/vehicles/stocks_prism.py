"""Vehicle stocks prism model for yearly stock calculations.

This module provides a prism.Model that calculates vehicle stocks from yearly
passenger and tonne kilometer demands, using the shares and conversion factors
determined during preprocessing.
"""

from imagematerials.factory import ModelFactory

import logging
import warnings
import xarray as xr
import numpy as np
import pandas as pd
import prism
from pathlib import Path
from pint.errors import UnitStrippedWarning

from imagematerials.concepts import KnowledgeGraph, create_region_graph
from imagematerials.constants import IMAGE_REGIONS, base_directory, base_directory_integration
from imagematerials.preprocessing import get_preprocessing_data
from imagematerials.util import read_climate_policy_config
from imagematerials.vehicles.constants import (
    END_YEAR,
    LIGHT_COMMERCIAL_VEHICLE_SHARE,
    MEGA_TO_TERA,
    PKMS_TO_VKMS,
    REGIONS,
    START_YEAR,
    all_modes, 
    LOAD_FACTOR
)
#from imagematerials.vehicles.preprocessing.main_prism import vehicles_preprocessing_integration
from imagematerials.vehicles.shares_prism import get_vehicle_shares_prism
from imagematerials.vehicles.preprocessing.util import (
    xarray_conversion
)

from imagematerials.vehicles.modelling_functions import convert_from_tkpm_or_pkm_to_nr_vehicles

REGION = prism.Dimension("Region", IMAGE_REGIONS)
STOCK_TYPE = prism.Dimension("Type")
MODE = prism.Dimension("Mode", all_modes)
TIME = prism.Dimension("Time")
UnitFlexibleStock = prism.DynamicUnit("my_unit_stock")

# inconvert_from_tkpm_or_pkm_to_nr_vehicles we downcast xarray to numpy 
# however we assign a unit in the same step to make it a prism array - therefore supress warning
warnings.filterwarnings(
    "ignore",
    message="The unit of the quantity is stripped when downcasting to ndarray.",
    category=UnitStrippedWarning,
)


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
    climate_policy_scenario_dir: Path
    # total_vehicles: prism.Array[REGION, MODE] = prism.export() # exported for debugging and testing
    stocks: prism.TimeVariable = prism.export()
    
    input_data: tuple[str] = (
        "passengerkms", "tonekms", "vehicle_shares", "conversion_factor_tkms", 
        "first_year_vehicle", "market_share", "knowledge_graph", "set_unit_flexible"
    )
    output_data: tuple[str] = ("stocks",)
    
    def compute_initial_values(self, 
                               time: prism.Timeline):
        """Initialize the stocks TimeVariable with zeros.
        
        Parameters
        ----------
        time : prism.Timeline
            The simulation timeline
        """

        # load "TIMER" model
        # from imagematerials.vehicles.timermodel import FakeTimerTransport

        # Load preprocessing of Vehicles data 
        vehicle_preprocessing = get_preprocessing_data(
            "vehicles", 
            base_directory_integration, # when run in .py file for debugginf this should be: base_directory!
            climate_policy_scenario_dir=self.climate_policy_scenario_dir,
            circular_economy_scenario_dirs=None,
            integration_preprocessing = True
        )

        # # initialize fake timer model to get passengerkms and tonekms data
        # self.timer_model = FakeTimerTransport(time,
        #                                       self.climate_policy_scenario_dir)
        
        # self.timer_model.compute_initial_values(time, 
        #                                          self.climate_policy_scenario_dir)

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

        # stocks used for mfa calculations
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
        self.first_year_vehicle = vehicle_data["first_year_vehicle"]
        self.market_share = vehicle_data["market_share"]
        self.weights = vehicle_data["weights"]
        # TODO: why are we not using time-dependet IMAGE load data where possible 
        self.load = vehicle_data["load"]
        self.loadfactor = vehicle_data["loadfactor"]
        self.mileages = vehicle_data["mileages"]
        self.kilometrage_cars = vehicle_data["kilometrage_cars"]
        self.kilometrage_midi_bus = vehicle_data["kilometrage_midi_bus"]
        self.kilometrage_bus = vehicle_data["kilometrage_bus"]

        # Vehicle shares may be absent in integration preprocessing; compute them on demand.
        if "vehicle_shares" in vehicle_data:
            self.vehicle_shares = vehicle_data["vehicle_shares"]
        elif "shares" in vehicle_data:
            self.vehicle_shares = vehicle_data["shares"]
        else:
            climate_policy_config = read_climate_policy_config(self.climate_policy_scenario_dir)
            self.vehicle_shares = get_vehicle_shares_prism(
                self.climate_policy_scenario_dir,
                climate_policy_config,
                self.knowledge_graph,
            )

        # dimension check?
        # compute historic tail
        # vhc_originl = ModelFactory.load_pkl("examples/model_results/test.pkl")
        # self.stocks_original = vhc_originl.vehicles.get("stocks")
    
    def compute_values(self, time: prism.Time, 
                       passenger_kms,
                       tonne_kms,
                       load_car):
        """Calculate vehicle stocks for the current timestep.
        
        Computes the number of vehicles needed to satisfy the yearly passenger
        and tonne kilometer demands, using conversion factors and vehicle shares.
        
        Parameters
        ----------
        time : prism.Time
            Current simulation time step
        """
        t = time.t
        print("Current time step:", t)
        # unit = str(self.set_unit_flexible)

        # run the "timer model" to get the latest passengerkms and tonekms data for this time step
        # self.timer_model.compute_values(time)
        # Directly transfer TIMER transport activity values into the vehicles stock model.
        self.passenger_kms = passenger_kms
        self.tonne_kms = tonne_kms
        self.load_car = load_car
        # self.total_vehicles: prism.Array[REGION, MODE, "count"] = prism.export()

        self.total_vehicles = self._calculate_total_vehicles_by_type(t)
        
        # # Apply vehicle shares to expand to subtypes (fuel/technology variants)
        # stocks_with_subtypes = self._apply_vehicle_shares(self.total_vehicles, t)
         
        # # Store result using label-based Time indexing (calendar year, not positional index).
        # self.stocks.loc[dict(Time=t)] = prism.Q_(self.total_vehicles, "count")
    
    
    def _calculate_total_vehicles_by_type(self, time: prism.Time):
        """
        Calculate total number of vehicles by type for a given year, based on passenger and tonne kilometers.
        """
        # load pkl model to assert
        
        # initiate xra.DataArray to hold total vehicles by type and region
        # convert unit from terra to base unit (pkm or tkm)
        self.passenger_kms = self.passenger_kms.pint.to("person*km")
        self.tonne_kms = self.tonne_kms.pint.to("tonne*km")

        # 1. Simple vehicle types ["Passenger Planes", "Trains", "High Speed Trains", "Bikes"]
        simple_types = ["Passenger Planes", "Trains", "High Speed Trains", "Bikes"]
        # unit handling maybe not the nicest way, but works for now, otherwise all need to be loaded as xr instead of pd
        nr_vehicles_simple = convert_from_tkpm_or_pkm_to_nr_vehicles(self.passenger_kms, 
                                                                     simple_types, 
                                                                     self.load, 
                                                                     self.loadfactor, 
                                                                     self.mileages, 
                                                                     "passenger", 
                                                                     time)
        # test to assert that the calculation of simple vehicle types matches the original model for this time step (within a reasonable tolerance)
        # assert self.stocks_original.sel(Type = simple_types).loc[int(prism.M_(time))].sum() == nr_vehicles_simple.sum(), f"Discrepancy in simple vehicle types calculation for time {time.t}: original total {self.stocks_original.sel(Type = simple_types, Time = time.t).sum().values}, new total {nr_vehicles_simple.sum().values}"
        
        # # 2 (trucks) & 3 (freight planes) & 4. (freight trains & inland ships) Freight vehicles with special handling
        # 2 Trucks are calculated differently because the IMAGE model does not account for LCV trucks
        trucks_total_tkm = (self.tonne_kms.sel(Type="Medium Freight Trucks") + self.tonne_kms.sel(Type="Heavy Freight Trucks"))
        # share of medium freight trucks tkm
        MFT_percshare_tkm = self.tonne_kms.sel(Type="Medium Freight Trucks") / trucks_total_tkm
        # share of heavy freight trucks tkm
        HFT_percshare_tkm = self.tonne_kms.sel(Type="Heavy Freight Trucks") / trucks_total_tkm
        # calcaulte the tkm for LCV, MFT, and HFT trucks based on shares
        trucks_LCV_tkm = trucks_total_tkm * LIGHT_COMMERCIAL_VEHICLE_SHARE

        trucks_min_LCV = trucks_total_tkm - trucks_LCV_tkm
        trucks_MFT_tkm = trucks_min_LCV * MFT_percshare_tkm
        trucks_HFT_tkm = trucks_min_LCV * HFT_percshare_tkm
        
        # 3. Freight planes - reduced by 50% (cargo on passenger planes)
        air_freight_tkms = self.tonne_kms.sel(Type="Freight Planes") * 0.5
        # 4. Freight trains & Inland Ships
        freight_trains_ships_tkms = self.tonne_kms.sel(Type=["Freight Trains", "Inland Ships"])

        # join to one DataArray and give Type coordinate the same name
        freight_vehicles_special_types = ["Light Commercial Vehicles", "Medium Freight Trucks", 
                                          "Heavy Freight Trucks", "Freight Planes", "Freight Trains", 
                                          "Inland Ships"]
        freight_vehicles_special_tkm = xr.concat([trucks_LCV_tkm.assign_coords(Type=freight_vehicles_special_types[0]).expand_dims("Type"),
                                trucks_MFT_tkm.assign_coords(Type=freight_vehicles_special_types[1]).expand_dims("Type"),
                                trucks_HFT_tkm.assign_coords(Type=freight_vehicles_special_types[2]).expand_dims("Type"),
                                air_freight_tkms.assign_coords(Type="Freight Planes").expand_dims("Type"),
                                freight_trains_ships_tkms], 
                                dim="Type")
        
        freight_vehicles_special_vehicles = convert_from_tkpm_or_pkm_to_nr_vehicles(freight_vehicles_special_tkm, 
                                                                                    freight_vehicles_special_types,
                                                                                    self.load, 
                                                                                    self.loadfactor, 
                                                                                    self.mileages,
                                                                                    "freight", 
                                                                                    time) 
        # 5. Cars and buses (region-specific), cars in brackets to not drop the dimension
        cars_pkms = self.passenger_kms.sel(Type=["Cars"])
         
        cars_vehicles = convert_from_tkpm_or_pkm_to_nr_vehicles(cars_pkms,
                                                            ["Cars"], 
                                                            self.load_car, 
                                                            LOAD_FACTOR, 
                                                            self.kilometrage_cars, 
                                                            "passenger", 
                                                            time)
   
        total = xr.concat([nr_vehicles_simple, freight_vehicles_special_vehicles, cars_vehicles], dim="Type")
        ####
        # Test, delete later
        # for vehicle in total.coords["Type"].values:
            # # skip these for now as they are still need to be split up in suptypes in the new calculation
            # # compare to 1e-6 to account for small discrepancies in data handling
            # if vehicle not in ["Light Commercial Vehicles", "Cars", "Medium Freight Trucks", "Heavy Freight Trucks", "Regular Buses", "Midi Buses"]:
                # compare_old = self.stocks_original.sel(Type = vehicle).loc[int(prism.M_(time))].sum()
                # compare_new = total.sel(Type=vehicle).sum()
                # assert abs(compare_new - compare_old) < 1e-6, f"Discrepancy in {vehicle} calculation for time {time}: original total {self.stocks_original.sel(Type = vehicle).loc[int(prism.M_(time))].sum().values}, new total {total.sel(Type=vehicle).sum().values}"
            # if vehicle in ["Cars"]:
                # # there are some minor differences, so for cars assert if higher than 100 cars
                # sub_types = ['Cars - BEV','Cars - FCV', 'Cars - HEV', 'Cars - ICE', 'Cars - PHEV','Cars - Trolley']
                # compare_old = self.stocks_original.sel(Type = sub_types).loc[int(prism.M_(time))].sum(["Type", "Region"])
                # compare_new = total.sel(Type="Cars").sum()
                # print("Discrepancy in Cars calculation for time {}: original total {}, new total {}".format(time, compare_new.values, compare_old.values))
                # assert abs(compare_new - compare_old) < 100, f"Discrepancy in {vehicle} calculation for time {time}: original total {self.stocks_original.sel(Type = sub_types).loc[int(prism.M_(time))].sum(["Type", "Region"]).values}, new total {total.sel(Type="Cars").sum().values}"

        return total

        
        

        
        # # 6. Ships (special handling)
        # if "international shipping" in tkms_year.coords.get("Type", []):
        #     ships = self._calculate_ships(tkms_year, year)
        #     for vehicle_type, values in ships.items():
        #         self.total_vehicles.loc[dict(Mode=vehicle_type)] = values
        
        