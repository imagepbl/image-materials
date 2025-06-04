import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from imagematerials.eol.constants import start_year, end_year, full_time, SCENARIO_SELECT


# define interpolation method
def interpolate_eol_rates(ds, start_year, end_year, min_value = 0, max_value = 1):              # collection, reuse, and recycling rate cannot be lower than zero or higher than 1

    reindexed = ds.reindex(time=full_time)
    interpolated = reindexed.interpolate_na(dim='Time', method='linear', fill_value='extrapolate')
    clipped = interpolated.clip(min=min_value, max = max_value)

    return clipped