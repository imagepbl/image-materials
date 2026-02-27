#%%
import pandas as pd
import numpy as np
from pathlib import Path
import pint
import xarray as xr


from imagematerials.constants import IMAGE_REGIONS

path_current = Path().resolve()
path_base = path_current.parent.parent # base path of the project -> image-materials
path_data = Path(path_base, "data", "raw")

df_phs =  pd.read_csv(Path(path_data,"electricity","IHA_PSH_Capacity_data.csv"), index_col = [0])

