import prism
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

from imagematerials.constants import IMAGE_REGIONS
from imagematerials.util import read_climate_policy_config
from imagematerials.vehicles.constants import pkms_label, tkms_label
from imagematerials.vehicles.preprocessing.util import (
    get_passengerkms,
    get_tonkms,

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
class FakeTimerTransport(prism.Model):
    """Read TIMER transport activity data and convert it to prism.TimeVariable 
    (this will later be replaced by direct link to TIMER)."""

    climate_policy_scenario_dir: Path | None = None
    passenger_kms: prism.Array[REGION, PASSENGER_TYPE, "pkm"] = prism.export()
    tonne_kms: prism.Array[REGION, FREIGHT_TYPE, "tkm"] = prism.export()
    passenger_kms_all_xr: pd.DataFrame | None = None # loaded .out data
    tonne_kms_all_xr: pd.DataFrame | None = None # loaded .out data

    def compute_initial_values(self, timeline: prism.Timeline, climate_policy_scenario_dir: Path | None = None):
        if climate_policy_scenario_dir is None:
            scenario_dir = self.climate_policy_scenario_dir
        else:
            scenario_dir = climate_policy_scenario_dir
            self.climate_policy_scenario_dir = scenario_dir

        if scenario_dir is None:
            raise ValueError("climate_policy_scenario_dir must be provided to FakeTimerTransport (got None)")

        climate_policy_config = read_climate_policy_config(scenario_dir)
        passenger_kms_all = get_passengerkms(scenario_dir, climate_policy_config)
        tonne_kms_all = get_tonkms(scenario_dir, climate_policy_config)

        # convert the loaded data into xarray DataArrays for easier access during compute_values
        self.passenger_kms_all_xr = (passenger_kms_all.drop(index=[27, 28], level= "region") # drop empty and global 
                                     .rename_axis(index={"time": "Time", "region": "Region"}, columns="PassengerType")
                                     .stack()
                                     .rename("passenger_kms")
                                     .to_xarray())

        self.passenger_kms_all_xr = prism.Q_(self.passenger_kms_all_xr, "terapkm") #TODO: check if this is right

        self.tonne_kms_all_xr = (tonne_kms_all.drop(index=[27, 28], level= "region") # drop empty and global 
                                 .rename_axis(index={"time": "Time", "region": "Region"}, columns="FreightType")
                                 .stack()
                                 .rename("tonne_kms")
                                 .to_xarray())

        self.tonne_kms_all_xr = prism.Q_(self.tonne_kms_all_xr, "teratkm") # TODO: check if this is right
        
        # assign Region short names
        self.passenger_kms_all_xr = self.passenger_kms_all_xr.assign_coords(Region=IMAGE_REGIONS)
        self.tonne_kms_all_xr = self.tonne_kms_all_xr.assign_coords(Region=IMAGE_REGIONS)

    def compute_values(self, time: prism.Time):
        t, dt = time.t, time.dt
        # convert the xarray Data to prism.Array for the current time step, this will be used in the VehicleStocks model

        self.passenger_kms = self.passenger_kms_all_xr.sel(Time=t)
        self.tonne_kms = self.tonne_kms_all_xr.sel(Time=t)