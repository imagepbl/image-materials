import prism
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

from imagematerials.constants import IMAGE_REGIONS
from imagematerials.read_mym import read_mym_df
from imagematerials.util import read_climate_policy_config
from imagematerials.vehicles.constants import pkms_label, tkms_label
from imagematerials.vehicles.preprocessing.util import (
    get_passengerkms,
    get_tonkms,

)
from imagematerials.vehicles.stocks_prism import VehicleStocks

REGION = prism.Dimension("Region", IMAGE_REGIONS)
STOCK_TYPE = prism.Dimension("Type")
PASSENGER_TYPE = prism.Dimension("Type", pkms_label)
FREIGHT_TYPE = prism.Dimension("Type", tkms_label)
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
    load_car_xr: pd.DataFrame | None = None # loaded .out data

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

        # load of cars is also given by TIMER (not load factor as previously named!!!)
        # TODO: why not use this for other modes?
        load_car: pd.DataFrame = read_mym_df(scenario_dir.joinpath(
            climate_policy_config['data_files']['transport']['passenger']['load'])).rename(columns={"DIM_1": "region"})

        # convert the loaded data into xarray DataArrays for easier access during compute_values
        self.passenger_kms_all_xr = (passenger_kms_all.drop(index=[27, 28], level= "region") # drop empty and global 
                                     .rename_axis(index={"time": "Time", "region": "Region"}, columns="Type")
                                     .stack()
                                     .rename("passenger_kms")
                                     .to_xarray())*1e12 # from tera to base unit (pkm)

        self.passenger_kms_all_xr = prism.Q_(self.passenger_kms_all_xr, "person*km")

        self.tonne_kms_all_xr = (tonne_kms_all.drop(index=[27, 28], level= "region") # drop empty and global 
                                 .rename_axis(index={"time": "Time", "region": "Region"}, columns="Type")
                                 .stack()
                                 .rename("tonne_kms")
                                 .to_xarray())*1e6 # from millions to base unit (tkm)

        self.tonne_kms_all_xr = prism.Q_(self.tonne_kms_all_xr, "tonne*km")

        # also load load for cars to xarray and assign region coordinates
        # 5 seems to be cars, taken from Sebastiaans code, but should be checked
        self.load_car_xr = load_car[["time", "region", 5]].rename(columns={"time": "Time", "region": "Region", 5: "load"}
                                                                              ).set_index(["Time", "Region"])["load"].to_xarray().transpose("Time", "Region")
        self.load_car_xr = prism.Q_(self.load_car_xr, "person/count")
        # assign Region short names
        self.passenger_kms_all_xr = self.passenger_kms_all_xr.assign_coords(Region=IMAGE_REGIONS)
        self.tonne_kms_all_xr = self.tonne_kms_all_xr.assign_coords(Region=IMAGE_REGIONS)
        self.load_car_xr = self.load_car_xr.assign_coords(Region=IMAGE_REGIONS)

    def compute_values(self, time: prism.Time):
        t, dt = time.t, time.dt
        # convert the xarray Data to prism.Array for the current time step, this will be used in the VehicleStocks model

        self.passenger_kms = self.passenger_kms_all_xr.sel(Time=t)
        self.tonne_kms = self.tonne_kms_all_xr.sel(Time=t)
        self.load_car = self.load_car_xr.sel(Time=t)