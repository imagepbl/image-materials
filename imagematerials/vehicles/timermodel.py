import prism
import numpy as np

from imagematerials.constants import START_YEAR_HISTORIC_integration
from imagematerials.util import read_climate_policy_config
from imagematerials.vehicles.preprocessing.util import get_passengerkms, get_tonkms
from imagematerials.vehicles.stocks_prism import VehicleStocks
from pathlib import Path

REGION = prism.Dimension("Region")
STOCK_TYPE = prism.Dimension("Type")
COHORT = prism.Dimension("Cohort")
TIME = prism.Dimension("Time")
MATERIAL_TYPE = prism.Dimension("material")


@prism.interface
class TIMERMaterials(prism.Model):
    """
    """

    climate_policy_scenario_dir: Path | None = None
    _submodels: list | None = None

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
        self._initialize_default_transport_inputs()

    @staticmethod
    def _year_value(time_value):
        magnitude = getattr(time_value, "magnitude", None)
        if magnitude is None:
            magnitude = getattr(time_value, "m", None)
        if magnitude is None:
            magnitude = time_value

        magnitude_array = np.asarray(magnitude)
        if magnitude_array.size == 1:
            magnitude = magnitude_array.item()

        try:
            return int(magnitude)
        except (TypeError, ValueError):
            return magnitude

    def _set_vehicle_input_for_time(self, time_value, passenger_kms=None, tonkms=None):
        year_key = self._year_value(time_value)

        if not hasattr(self.vehicles, "passengerkms") or self.vehicles.passengerkms is None:
            self.vehicles.passengerkms = {}
        if not hasattr(self.vehicles, "tonekms") or self.vehicles.tonekms is None:
            self.vehicles.tonekms = {}

        if passenger_kms is not None:
            self.vehicles.passengerkms[year_key] = passenger_kms

        if tonkms is not None:
            self.vehicles.tonekms[year_key] = tonkms

    def _initialize_default_transport_inputs(self):
        if getattr(self.vehicles, "passengerkms", None) and getattr(self.vehicles, "tonekms", None):
            return

        scenario_dir = self.climate_policy_scenario_dir or Path("data", "raw", "image", "SSP2_baseline")
        climate_policy_config = read_climate_policy_config(scenario_dir)
        passenger_kms_all = get_passengerkms(scenario_dir, climate_policy_config)
        tonkms_all = get_tonkms(scenario_dir, climate_policy_config)

        for year in passenger_kms_all.index.get_level_values("time").unique():
            self._set_vehicle_input_for_time(
                year,
                passenger_kms=passenger_kms_all.loc[year],
                tonkms=tonkms_all.loc[year],
            )


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

        if passenger_kms is not None or tonkms is not None:
            self._set_vehicle_input_for_time(time.t, passenger_kms=passenger_kms, tonkms=tonkms)

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
        print(f"{time.t}", end="\r")
        for model in self._submodels:
            model.compute_values(time, **model.compute_args)

        # Interact with Stock model and Materials model from here
