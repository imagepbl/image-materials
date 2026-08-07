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

from imagematerials.sensitivity_analysis.monte_carlo import (
    sample_values, 
    load_material_intensities, 
    load_lifetimes,
    process_material_intensities,
    process_lifetimes
)

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

df_lifetimes = load_lifetimes(path_base / "vehicles" / "vehicles_lifetimes.csv")
rng = np.random.default_rng(42)   # seed once for reproducibility
print("df_lifetimes", df_lifetimes)

start = time.time()
all_output = {}
N = 2
for i in range(N):
    print(N)
    df_lifetimes_sampled = sample_values(df_lifetimes, rng=rng)
    print("df_lifetimes_sampled", df_lifetimes_sampled)
    df_lifetimes_processed = process_lifetimes(df_lifetimes_sampled, "vehicles")
    print("df_lifetimes_processed", df_lifetimes_processed)

    
