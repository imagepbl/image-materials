import prism
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

from imagematerials.constants import IMAGE_REGIONS, START_YEAR_HISTORIC_integration

from imagematerials.vehicles.constants import pkms_label, tkms_label

from imagematerials.vehicles.stocks_prism import VehicleStocks
from imagematerials.vehicles.timertransport import FakeTimerTransport

REGION = prism.Dimension("Region", IMAGE_REGIONS)
STOCK_TYPE = prism.Dimension("Type")
PASSENGER_TYPE = prism.Dimension("Type", pkms_label)
FREIGHT_TYPE = prism.Dimension("Type", tkms_label)
COHORT = prism.Dimension("Cohort")
TIME = prism.Dimension("Time")
MATERIAL_TYPE = prism.Dimension("material")


@prism.interface
class TIMERMaterials(prism.Model):
    """
    Replaces the factory for now
    """
    climate_policy_scenario_dir: Path | None = None
    _submodels: list | None = None
    timer_model: FakeTimerTransport | None = None

    def compute_initial_values(self, timeline: prism.Timeline):
        # Handle historic tail
        self.historic_tail_computed = False
        # for now ignore historic tail in timeline (START_YEAR_HISTORIC_integration)
        self.complete_timeline = prism.Timeline(START_YEAR_HISTORIC_integration, timeline.end, timeline.stepsize)
    
    
    def init_submodels(self, timeline: prism.Timeline):
        # inside VehicleStocks we also call the FakeTimerTransport to load the data that we later get from TIMER
        climate_policy_scenario_dir = self.climate_policy_scenario_dir
        
        vehicles = VehicleStocks(timeline, 
                                 climate_policy_scenario_dir)
        vehicles.compute_initial_values(timeline)
        
        # Extract dimensions from the VehicleStocks submodel if not already set
        if not hasattr(self, 'Region') or self.Region is None:
            self.Region = vehicles.Region
        if not hasattr(self, 'Type') or self.Type is None:
            self.Type = vehicles.Type
        if not hasattr(self, 'Time') or self.Time is None:
            self.Time = vehicles.Time
        
        # Set compute_args for the submodel (empty for now, as VehicleStocks doesn't need dynamic inputs yet)
        vehicles.compute_args = {}
            
        self._submodels = [vehicles]

    def compute_values(
        self,
        time: prism.Time
        ):
        # if not self.historic_tail_computed:
        # # TODO: make a cache for the historic tail calculation
        #     for historic_time in self.complete_timeline:
        #         prism_time = prism.Time(self.complete_timeline.start,
        #                                 self.complete_timeline.end,
        #                                 self.complete_timeline.stepsize, historic_time)
        #         if historic_time >= time.t:
        #             break # break to move out of historic tail loop
        #         self._compute_one_timestep(prism_time)
        #     self.historic_tail_computed = True
        self._compute_one_timestep(time)
        
    def _compute_one_timestep(self, time: prism.Time):
        print(f"{time.t}", end="\r")
        for model in self._submodels:
            model.compute_values(time, **model.compute_args)

        # Interact with Stock model and Materials model from here
