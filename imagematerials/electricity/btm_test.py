#%%
import pandas as pd
import numpy as np
from pathlib import Path
import pint
import xarray as xr
import matplotlib.pyplot as plt
import warnings
from pint.errors import UnitStrippedWarning

import prism
from imagematerials.read_mym import read_mym_df
from imagematerials.constants import IMAGE_REGIONS
from imagematerials.concepts import create_image_region_graph, create_electricity_graph, create_region_graph
from imagematerials.electricity.constants import (
    STANDARD_SCEN_EXTERNAL_DATA,
    YEAR_FIRST_GRID,
    EPG_TECHNOLOGIES,
    STD_LIFETIMES_ELECTR,
)
from imagematerials.electricity.utils import add_historic_stock, interpolate_xr

path_current = Path().resolve()
path_base = path_current.parent.parent # base path of the project -> image-materials
path_data = Path(path_base, "data", "raw")

year_start = 1970
year_end = 2100

####################################################################################################
#%% Load data

# 2. IMAGE/TIMER files -----------------------------------------
# Generation capacity (stock demand per generation technology) in MW peak capacity
gcap_data = read_mym_df(Path(path_data, "image", "SSP2_baseline", "EnergyServices", "GCap.out"))



# Transform to xarray #
knowledge_graph_region = create_image_region_graph() #create_region_graph()
knowledge_graph_electr = create_electricity_graph()

# Gcap ------
gcap_data = gcap_data.loc[~gcap_data['DIM_1'].isin([27,28])]  # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies
gcap_data = gcap_data.loc[gcap_data['time'].isin(range(year_start, year_end + 1)), ['time', 'DIM_1', *range(1, len(EPG_TECHNOLOGIES) + 1)]]  # only keep relevant years and technology columns
# Extract coordinate labels
years = sorted(gcap_data['time'].unique())
regions = sorted(gcap_data['DIM_1'].unique())
techs = list(range(1, len(EPG_TECHNOLOGIES)+1))
# Convert to 3D array: (Year, Region, Tech)
data_array = gcap_data[techs].to_numpy().reshape(len(years), len(regions), len(techs))
# Build xarray DataArray
gcap_xr = xr.DataArray(
    data_array,
    dims=('Time', 'Region', 'Type'),
    coords={
        'Time': years,
        'Region': [("region_" + str(r)) for r in regions],
        'Type': [str(r) for r in techs]
    },
    name='GenerationCapacity'
)
gcap_xr = prism.Q_(gcap_xr, "MW")
gcap_xr = knowledge_graph_region.rebroadcast_xarray(gcap_xr, output_coords=IMAGE_REGIONS, dim="Region") 
gcap_xr = knowledge_graph_electr.rebroadcast_xarray(gcap_xr, output_coords=EPG_TECHNOLOGIES, dim="Type")


# TIMER data only start in 1971, so we add a historic tail back to YEAR_FIRST_GRID=1921 #TODO to be adjusted
gcap_xr_interp = add_historic_stock(gcap_xr, YEAR_FIRST_GRID)









