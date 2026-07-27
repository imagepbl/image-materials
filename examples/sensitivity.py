#%%

from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import prism
import warnings
import numpy as np
import time  
  

from imagematerials.sensitivity_analysis.changedata import change_sector, ChangeAction, ChangeReplace
from imagematerials.factory import ModelFactory
from imagematerials.maintenance import Maintenance
from imagematerials.model import GenericMaterials, GenericStocks
from imagematerials.preprocessing import get_preprocessing_data

from imagematerials.sensitivity_analysis.monte_carlo import load_ranges_material_intensities, sample_material_intensities, load_ranges_lifetimes, sample_lifetimes

warnings.filterwarnings("ignore")

path_current = Path().resolve()
path_base = path_current.parent #.parent # base path of the project -> image-materials
path_base = Path(path_base, "data", "raw")

#%%

# Get the preprocessing data for the vehicles sector only once
vhc_sector = get_preprocessing_data(
    "vehicles", Path("..", "data", "raw"), 
    climate_policy_scenario_dir = Path("..", "data", "raw", "image", "SSP2_baseline"), 
    circular_economy_scenario_dirs = None
)

#%% lifeitime #####################################

ranges_lt = load_ranges_lifetimes(path_base / "vehicles" / "test_lifetimes_years.csv")
rng = np.random.default_rng(42)   # seed once for reproducibility
lt = sample_lifetimes(ranges_lt, rng=rng,keep_stats=["mean", "stdev"])
