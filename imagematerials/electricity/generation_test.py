#%%

import pandas as pd
import numpy as np
from pathlib import Path
import pint
import xarray as xr
from importlib.resources import files

import prism
from imagematerials.read_mym import read_mym_df
from imagematerials.util import dataset_to_array, pandas_to_xarray, convert_lifetime
from imagematerials.concepts import create_electricity_graph, create_region_graph
from imagematerials.electricity.utils import (
    MNLogit,
    create_prep_data, 
    logistic, 
    quadratic,
    interpolate_xr, 
    add_historic_stock, 
    calculate_grid_growth, 
    calculate_fraction_underground,  
    normalize_selected_techs,
    calculate_storage_market_shares
)
from imagematerials.electricity.preprocessing.circular_economy_measures import (
    apply_ce_measures_to_elc_generation
)

from imagematerials.constants import IMAGE_REGIONS

from imagematerials.electricity.constants import (
    STANDARD_SCEN_EXTERNAL_DATA,
    YEAR_FIRST_GRID,
    SENS_ANALYSIS,
    EPG_TECHNOLOGIES,
    EPG_SUB_TECHNOLOGIES,
    STD_LIFETIMES_ELECTR,
    EV_BATTERIES,
    unit_mapping
)

# for ttesting, remove later:
year_start = YEAR_FIRST_GRID
year_end = 2100

path_current = Path().resolve()
path_base = path_current.parent.parent # base path of the project -> image-materials
path_base = Path(path_base, "data", "raw")
scenario = "SSP2_baseline"

path_image_output = Path(path_base, "image", scenario)
path_external_data_standard = Path(path_base, "electricity", "standard_data")
path_external_data_scenario = Path(path_base, "electricity", scenario)
# test if path_external_data_scenario exists and if not set to standard scenario
if not path_external_data_scenario.exists():
    path_external_data_scenario = Path(path_base, "electricity", STANDARD_SCEN_EXTERNAL_DATA)

# assert path_external_data_scenario.is_dir()

###########################################################################################################
# Read in files #

# 1. External Data --------------------------------------------- 

# lifetimes of Gcap tech (original data according to van Vuuren 2006, PhD Thesis)
lifetimes = pd.read_csv(path_external_data_standard / "generation_lifetimes.csv",
                         usecols=["Time","technology","sub_technology","value"])        

# material compositions of electricity generation tecnologies (kg/MW)
material_intensities = pd.read_csv(path_external_data_standard / "generation_material_intensities.csv",
                         usecols=["Time","technology","sub_technology","material","value"], comment="#")#.transpose()

# market shares
market_shares = pd.read_csv(path_external_data_standard / "generation_subtechnologies_market_shares.csv",
                         usecols=["Time","sub_technology","value"]) #"technology",

# 2. IMAGE/TIMER files -----------------------------------------
# Generation capacity (stock demand per generation technology) in MW peak capacity
gcap_data = read_mym_df(path_image_output / "EnergyServices" / 'GCap.out')
# gcap_data = read_mym_df(
#     climate_policy_config["config_file_path"] /
#     climate_policy_config["data_files"]['GCap']
# )


###########################################################################################################
#%% Transform to xarray #
knowledge_graph_region = create_region_graph()
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
    dims=('Time', 'Region', 'SuperType'),
    coords={
        'Time': years,
        'Region': [str(r) for r in regions],
        'SuperType': [str(r) for r in techs]
    },
    name='GCap'
)
gcap_xr = prism.Q_(gcap_xr, "MW")
gcap_xr = knowledge_graph_region.rebroadcast_xarray(gcap_xr, output_coords=IMAGE_REGIONS, dim="Region") 
gcap_xr = knowledge_graph_electr.rebroadcast_xarray(gcap_xr, output_coords=EPG_TECHNOLOGIES, dim="SuperType")
# gcap_xr = gcap_xr.assign_coords(SuperType=np.array(gcap_xr.SuperType.values, dtype=object)) # rebroadcast_xarray changes the type of the coordinates to numpy strings (np.str_), so convert back to python strings (str)

# material intensities (kg/MW) ---------------------------------------------------------------------
# create new column with sub-technology name if available, otherwise technology name
material_intensities['Type'] = material_intensities['sub_technology'].fillna(material_intensities['technology'])
material_intensities_da = (
    material_intensities.set_index(["Time", "Type", "material"])["value"]
    .to_xarray()
    .rename({"Time": "Cohort"})
)
material_intensities_da = prism.Q_(material_intensities_da, "kg/MW")
material_intensities_da.name = "GenerationMaterialIntensities"
material_intensities_da = knowledge_graph_electr.rebroadcast_xarray(material_intensities_da, output_coords=EPG_SUB_TECHNOLOGIES, dim="Type")

# market shares (%) --------------------------------------------------------------------------------
market_shares_da = (
    market_shares.set_index(["Time", "sub_technology"])["value"]
    .to_xarray()
    .rename({"sub_technology": "Type", "Time": "Cohort"})
)
market_shares_da = prism.Q_(market_shares_da, "dimensionless")
market_shares_da.name = "GenerationMarketShares"

# lifetimes (yr) -----------------------------------------------------------------------------------
# create new column with sub-technology name if available, otherwise technology name
lifetimes['Type'] = lifetimes['sub_technology'].fillna(lifetimes['technology'])
lifetimes_da = (
    lifetimes.set_index(["Time", "Type"])["value"]
    .to_xarray()
    .rename({"Time": "Cohort"})
)
lifetimes_da = lifetimes_da.expand_dims({"DistributionParams": ["mean", "stdev"]})
lifetimes_da.name = "StorageLifetime"

#%% test
tech_mix = [
    "SPV",
    "SPVR",
    "parabolic trough",
    "solar tower",
    "WON_GB-DFIG",
    "WON_GB-PMSG",
    "WON_DD-PMSG",
    "WON_DD-EESG",
    "WOFF_GB-DFIG",
    "WOFF_GB-PMSG",
    "WOFF_DD-PMSG",
    "WOFF_DD-EESG",
    "WAVE",
    "HYD",
    "OREN",
    "GEO",
    "H2P",
    "NUC",
    "FREE12",
    "ClST",
    "OlST",
    "NGOT",
    "BioST",
    "IGCC",
    "OlCC",
    "NGCC",
    "BioCC",
    "ClCS",
    "OlCS",
    "NGCS",
    "BioCS",
    "ClCHP",
    "OlCHP",
    "NGCHP",
    "BioCHP",
    "ClCHPCS",
    "OlCHPCS",
    "NGCHPCS",
    "BioCHPCS",
    "GeoCHP",
    "H2CHP"
]

market_shares_da_interp = interpolate_xr(market_shares_da, year_start, year_end)

test = knowledge_graph_electr.rebroadcast_xarray(gcap_xr, output_coords=tech_mix, dim="Type", shares = market_shares_da_interp.rename({"Cohort": "Time"}))


###########################################################################################################
#%% Interpolate #

# interpolate_xr: The lifetimes & material intensities are only given for specific years (2020 and 2050), so we linearly interpolate to get values for the years 2020-2050.
# The values before 2020 are kept constant at the 2020 level, and the values after 2050 are kept constant at the 2050 level.
lifetimes_da_interp = interpolate_xr(lifetimes_da, year_start, year_end)
lifetimes_da_interp.loc[dict(DistributionParams="stdev")] = lifetimes_da_interp.loc[dict(DistributionParams="mean")] * STD_LIFETIMES_ELECTR
material_intensities_da_interp = interpolate_xr(material_intensities_da, year_start, year_end)
market_shares_da_interp = interpolate_xr(market_shares_da, year_start, year_end)

# TIMER data only start in 1971, so we add a historic tail back to YEAR_FIRST_GRID=1921 #TODO to be adjusted
gcap_xr_interp = add_historic_stock(gcap_xr, year_start)


###########################################################################################################
# CE measures #

# Depending on circular economy scenario, apply different measures
# if circular_economy_config is not None:
#     material_intensities_da_interp, lifetimes_da_interp = apply_ce_measures_to_elc_generation(material_intensities_da_interp,
                                                                                            # lifetimes_da_interp,
                                                                                            # circular_economy_config)
    

###########################################################################################################
# Prep_data File #

# The lifetimes are converted to the proper format for the model (dictionary with keys:distribution name, values:datarrays containing distribution parameters)
lifetimes_da_interp = convert_lifetime(lifetimes_da_interp)

# bring preprocessing data into a generic format for the model
prep_data = {}
prep_data["lifetimes"] = lifetimes_da_interp
prep_data["stocks"] = gcap_xr_interp
prep_data["shares"] = market_shares_da_interp
prep_data["material_intensities"] = material_intensities_da_interp
prep_data["knowledge_graph"] = create_electricity_graph()
prep_data["set_unit_flexible"] = str(prism.U_(prep_data["stocks"])) # prism.U_ gives the unit back




