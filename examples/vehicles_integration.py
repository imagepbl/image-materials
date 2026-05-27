from pathlib import Path
import warnings

import prism
from prism import Q_

from imagematerials.vehicles.timermodel import TIMERMaterials

# Define the complete timeline, including historic tail
time_start = Q_(1960, 'year')
time_end = Q_(2060, 'year')
simulation_start = Q_(1970, 'year')
step = Q_(1, 'year')
complete_timeline = prism.Timeline(time_start, time_end, step)
simulation_timeline = prism.Timeline(simulation_start, time_end, step)
path_climate_data = Path("/data/raw/image/SSP2_baseline")

timer_vehicle_model = TIMERMaterials(complete_timeline, path_climate_data) # TODO: fix signature

timer_vehicle_model.simulate(simulation_timeline)