import prism
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

from imagematerials.constants import IMAGE_REGIONS
from imagematerials.read_mym import read_mym_df
from imagematerials.vehicles.constants import pkms_label, tkms_label
from imagematerials.vehicles.preprocessing.util import simulate_timer_data
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

        timer_data = simulate_timer_data(scenario_dir)
        self.passenger_kms_all_xr = timer_data["passenger_kms"]
        self.tonne_kms_all_xr = timer_data["tonne_kms"]
        self.load_car_xr = timer_data["load_car"]

        # initialize vehicles model 
        self.vehicles_model = VehicleStocks(timeline,
                                            self.climate_policy_scenario_dir)
        
        self.vehicles_model.compute_initial_values(timeline)
        
    def compute_values(self, time: prism.Time):
        t, dt = time.t, time.dt
        # convert the xarray Data to prism.Array for the current time step, this will be used in the VehicleStocks model

        self.passenger_kms = self.passenger_kms_all_xr.sel(Time=t)
        self.tonne_kms = self.tonne_kms_all_xr.sel(Time=t)
        self.load_car = self.load_car_xr.sel(Time=t)

        # self.timer_model.compute_values(time)
        self.vehicles_model.compute_values(time, 
                                           passenger_kms=self.passenger_kms, 
                                           tonne_kms=self.tonne_kms, 
                                           load_car=self.load_car)