import prism
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

from imagematerials.constants import IMAGE_REGIONS, START_YEAR_HISTORIC_integration
from imagematerials.util import read_climate_policy_config
from imagematerials.vehicles.constants import pkms_label, tkms_label
from imagematerials.vehicles.preprocessing.util import (
    get_passengerkms,
    get_tonkms,
    normalize_year_value,
)
from imagematerials.vehicles.stocks_prism import VehicleStocks

REGION = prism.Dimension("Region", IMAGE_REGIONS)
STOCK_TYPE = prism.Dimension("Type")
PASSENGER_TYPE = prism.Dimension("PassengerType", pkms_label)
FREIGHT_TYPE = prism.Dimension("FreightType", tkms_label)
COHORT = prism.Dimension("Cohort")
TIME = prism.Dimension("Time")
MATERIAL_TYPE = prism.Dimension("material")

@prism.interface
class TimerModel(prism.Model):
    """Read TIMER transport activity data and convert it to prism.TimeVariable 
    (this will later be replaced by direct link to TIMER)."""

    climate_policy_scenario_dir: Path | None = None
    passenger_kms: prism.TimeVariable[REGION, PASSENGER_TYPE, "pkm"] = prism.export()
    tonne_kms: prism.TimeVariable[REGION, FREIGHT_TYPE, "tkm"] = prism.export()
    passenger_kms_all_xr: pd.DataFrame | None = None # loaded .out data
    tonne_kms_all_xr: pd.DataFrame | None = None # loaded .out data

    def compute_initial_values(self, timeline: prism.Timeline):
        scenario_dir = self.climate_policy_scenario_dir
        climate_policy_config = read_climate_policy_config(scenario_dir)
        passenger_kms_all = get_passengerkms(scenario_dir, climate_policy_config)
        tonne_kms_all = get_tonkms(scenario_dir, climate_policy_config)

        # convert the loaded data into xarray DataArrays for easier access during compute_values
        self.passenger_kms_all_xr = (passenger_kms_all.drop(index=[27, 28], level= "region") # drop empty and global 
                                     .rename_axis(index={"time": "Time", "region": "Region"}, columns="PassengerType")
                                     .stack()
                                     .rename("passenger_kms")
                                     .to_xarray())

        self.passenger_kms_all_xr = prism.Q_(self.passenger_kms_all_xr, "pkm")

        self.tonne_kms_all_xr = (tonne_kms_all.drop(index=[27, 28], level= "region") # drop empty and global 
                                 .rename_axis(index={"time": "Time", "region": "Region"}, columns="FreightType")
                                 .stack()
                                 .rename("tonne_kms")
                                 .to_xarray())

        self.tonne_kms_all_xr = prism.Q_(self.tonne_kms_all_xr, "tkm")
        
        # assign Region short names
        self.passenger_kms_all_xr = self.passenger_kms_all_xr.assign_coords(Region=IMAGE_REGIONS)
        self.tonne_kms_all_xr = self.tonne_kms_all_xr.assign_coords(Region=IMAGE_REGIONS)

        print(self.passenger_kms_all_xr)

    def compute_values(self, time: prism.Time):
        t, dt = time.t, time.dt
        # convert the xarray Data to prism TimeVariable for the current time step
        self.passenger_kms[t] = self.passenger_kms_all_xr.sel(Time=t)
        self.tonne_kms[t] = self.tonne_kms_all_xr.sel(Time=t)



@prism.interface
class TIMERMaterials(prism.Model):
    """
    """

    climate_policy_scenario_dir: Path | None = None
    _submodels: list | None = None
    timer_model: TimerModel | None = None

    # Dimensions - derived dynamically from VehicleStocks submodel
    # TODO ensure this follows the way of defining coordinates as desired by prism
    # Region: prism.Coords[REGION]
    # Type: prism.Coords[STOCK_TYPE]
    #Cohort: prism.Coords[COHORT]
    # Time: prism.Coords[TIME]

    # Exported variables
    #inflow_materials: prism.TimeVariable[REGION, STOCK_TYPE, MATERIAL_TYPE, "kg"] = prism.export()
    #outflow_by_cohort_materials: prism.TimeVariable[REGION, STOCK_TYPE, MATERIAL_TYPE, "kg"] = prism.export()
    
    def compute_initial_values(self, timeline: prism.Timeline):
        # Handle historic tail
        self.historic_tail_computed = False

        timeline_start = getattr(getattr(self, "timeline", None), "start", START_YEAR_HISTORIC_integration)
        self.complete_timeline = prism.Timeline(timeline_start, timeline.end, timeline.stepsize)
        # Initialize submodels eagerly so inspection in notebooks works after setup.
        self._submodels = []
        self.init_submodels(timeline)

    def init_submodels(self, timeline: prism.Timeline):
        if self._submodels is None:
            self._submodels = []

        self.timer_model = TimerModel(
            timeline,
            climate_policy_scenario_dir=self.climate_policy_scenario_dir,
        )
        self.timer_model.compute_initial_values(timeline)

        self.vehicles = VehicleStocks(
            timeline,
            climate_policy_scenario_dir=self.climate_policy_scenario_dir
        )
        self.vehicles.compute_initial_values(
            timeline,
            climate_policy_scenario_dir=self.climate_policy_scenario_dir
        )
        
        # TODO: find a better way to do this
        # Extract dimensions from the VehicleStocks submodel if not already set
        if not hasattr(self, 'Region') or self.Region is None:
            self.Region = self.vehicles.Region
        if not hasattr(self, 'Type') or self.Type is None:
            self.Type = self.vehicles.Type
        if not hasattr(self, 'Time') or self.Time is None:
            self.Time = self.vehicles.Time
        
        # Set compute_args for the submodel (empty for now, as VehicleStocks doesn't need dynamic inputs yet)
        self.vehicles.compute_args = {}
        self._submodels = [self.vehicles]


    def compute_values(
        self,
        time: prism.Time,
        passenger_kms=None,
        tonkms=None,
    ):
        if not hasattr(self, "complete_timeline"):
            timeline_start = getattr(getattr(self, "timeline", None), "start", START_YEAR_HISTORIC_integration)
            self.complete_timeline = prism.Timeline(
                timeline_start,
                self.timeline.end,
                self.timeline.stepsize,
            )

        if not hasattr(self, "historic_tail_computed"):
            self.historic_tail_computed = False

        if self._submodels is None or not self._submodels:
            self.init_submodels(self.complete_timeline)
        
        if not self.historic_tail_computed:
        # TODO: make a cache for the historic tail calculation
            for historic_time in self.complete_timeline:
                prism_time = prism.Time(self.complete_timeline.start,
                                        self.complete_timeline.end,
                                        self.complete_timeline.stepsize, historic_time)
                if historic_time >= time.t:
                    break
                self._compute_one_timestep(prism_time)
            self.historic_tail_computed = True

        self._compute_one_timestep(time)

    def _compute_one_timestep(self, time: prism.Time):
        print(f"Computing values for time {time.t}...")
        self.timer_model.compute_values(time)

        # Directly transfer TIMER transport activity values into the vehicles stock model.
        self.vehicles.passenger_kms = self.timer_model.passenger_kms
        self.vehicles.tonkms = self.timer_model.ton_kms
        self.vehicles.tone_kms = self.timer_model.tonne_kms

        self.vehicles.compute_values(time)

