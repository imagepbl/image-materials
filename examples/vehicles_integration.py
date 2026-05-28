from pathlib import Path
import warnings

import prism
from prism import Q_

from imagematerials.vehicles.timermodel import TIMERMaterials, TimerModel

# Define the complete timeline, including historic tail
time_start = Q_(1971, 'year')
time_end = Q_(2060, 'year')
simulation_start = Q_(1971, 'year')
step = Q_(1, 'year')
complete_timeline = prism.Timeline(time_start, time_end, step)
simulation_timeline = prism.Timeline(simulation_start, time_end, step)
path_climate_data = Path("data", "raw", "image", "SSP2_baseline")

tkm_pkm = TimerModel(complete_timeline, climate_policy_scenario_dir=path_climate_data)
tkm_pkm.simulate(complete_timeline)
tkm_pkm

# timer_vehicle_model = TIMERMaterials(complete_timeline, path_climate_data)

# timer_vehicle_model.simulate(simulation_timeline)