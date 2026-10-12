from pathlib import Path
import pandas as pd

import prism
import pym

from imagematerials.model import (
    GenericStocks,
    MaterialIntensities,
    SharesInflowStocks,
    ElectricVehicleBatteries,
    EvBatteryLinkModule
)
from imagematerials.factory import ModelFactory
from imagematerials.preprocessing import get_preprocessing_data
from imagematerials.concepts import create_h2_graph
from imagematerials.util import resolve_circular_economy_scenario, convert_lifetime
from imagematerials.read_mym import read_mym_df, mymarray_to_xarray
from imagematerials.electricity.utils import (
    interpolate_xr, 
    add_historic_stock,
)
from imagematerials.electricity.constants import (
    STD_LIFETIMES_ELECTR,
    YEAR_FIRST_GRID
)
from imagematerials.hydrogen.constants import (
    YEAR_FIRST_H2_PRODUCTION
)


# TODO absolute path of file "preprocessing.py" ? current solution can differ depending on IDE used
path_current = Path().resolve()
path_base = path_current.parent #.parent # base path of the project -> image-materials
path_base = Path(path_base, "data", "raw")

year_start = YEAR_FIRST_GRID
year_end = 2100

#===================================================================================================
#%% Timer data
h2cap = mymarray_to_xarray(pym.read_mym(Path(path_base, "image", "SSP2_baseline", "EnergyServices", "H2Cap.out")),
                          dim_1="NRT", dim_2="HydrogenSourcesTotal")

h2cap = prism.Q_(h2cap, "GW")
h2cap = h2cap.rename({'t': 'time','NRT': 'Region','HydrogenSourcesTotal': 'Type'})
h2cap = h2cap.drop_sel(Type='total')

graph_h2 = create_h2_graph()
types = ["CBF", "CBF+CCS", "electrolysis"]
h2cap = graph_h2.aggregate_sum(h2cap, output_coords=types)

# TIMER data only start in 1971, so we add a historic tail back to YEAR_FIRST_GRID=1921 #TODO to be adjusted
h2cap_da_interp = interpolate_xr(h2cap, year_start, year_end) # fills the years before with the first value (which is 0), to align timeseries with other sectors (e.g. electricity)

#===================================================================================================
#%% external data

# material intensities (kg/MW) -----------------------------------------------------------------
# material compositions of hydrogen production technologies (kg/MW)
material_intensities = pd.read_csv(path_base / "hydrogen" / "standard_data" / "hydrogen_material_intensities_long.csv",
                        usecols=["time","technology","sub_technology","material","value"])
# create new column with sub-technology name if available, otherwise technology name
# material_intensities["Type"] = material_intensities["sub_technology"].fillna(material_intensities["technology"])
material_intensities["Type"] = (
        material_intensities["technology"] + "_" + material_intensities["sub_technology"]
    )
material_intensities_da = (
    material_intensities.set_index(["time", "Type", "material"])["value"]
    .to_xarray()
    .rename({"time": "Cohort"})
)
material_intensities_da = prism.Q_(material_intensities_da, "kg/MW")
material_intensities_da.name = "HydrogenMaterialIntensities"
# material_intensities_da = knowledge_graph_electr.rebroadcast_xarray(material_intensities_da, output_coords=EPG_SUB_TECHNOLOGIES, dim="Type")
material_intensities_da_interp = interpolate_xr(material_intensities_da, year_start, year_end)


# market shares (%) ----------------------------------------------------------------------------
market_shares = pd.read_csv(path_base / "hydrogen" / "standard_data" / "hydrogen_market_shares_long.csv",
                        usecols=["time","technology","sub_technology","value"])

market_shares["Type"] = (
            market_shares["technology"] + "_" + market_shares["sub_technology"]
        )
market_shares_da = (
    market_shares.set_index(["time", "Type"])["value"]
    .to_xarray()
    .rename({"time": "Cohort"})
)
market_shares_da = prism.Q_(market_shares_da, "dimensionless")
market_shares_da.name = "HydrogenMarketShares"

market_shares_da_interp = interpolate_xr(market_shares_da, year_start, year_end)

# lifetimes (yr) ----------------------------------------------------------------------------
# 
lifetimes = pd.read_csv(path_base / "hydrogen" / "standard_data" / "hydrogen_lifetimes.csv",
                        usecols=["time","technology","sub_technology","value"])
lifetimes["Type"] = (
            lifetimes["technology"] + "_" + lifetimes["sub_technology"]
        )
lifetimes_da = (
    lifetimes.set_index(["time", "Type"])["value"]
    .to_xarray()
    .rename({"time": "Cohort"})
)
lifetimes_da = lifetimes_da.expand_dims({"DistributionParams": ["mean", "stdev"]})
lifetimes_da.name = "GenerationLifetime"

lifetimes_da_interp = interpolate_xr(lifetimes_da, year_start, year_end) 
lifetimes_da_interp.loc[dict(DistributionParams="stdev")] = lifetimes_da_interp.loc[dict(DistributionParams="mean")] * STD_LIFETIMES_ELECTR

# The lifetimes are converted to the proper format for the model (dictionary with keys:distribution name, values:datarrays containing distribution parameters)
lifetimes_da_interp = convert_lifetime(lifetimes_da_interp)

#===================================================================================================
#%%  Prep_data File

# bring preprocessing data into a generic format for the model
prep_data = {}
prep_data["stocks"] = h2cap_da_interp
prep_data["lifetimes"] = lifetimes_da_interp
prep_data["shares"] = market_shares_da_interp
prep_data["material_intensities"] = material_intensities_da_interp
prep_data["knowledge_graph"] = graph_h2
prep_data["set_unit_flexible"] = str(prism.U_(prep_data["stocks"])) # prism.U_ gives the unit back