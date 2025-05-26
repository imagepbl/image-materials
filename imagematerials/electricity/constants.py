""""Global constants for the ELMA model."""
from pathlib import Path
import os
import pint

ureg = pint.UnitRegistry(force_ndarray_like=True)

# --- Settings & constants

# start year of historic IMAGE data
YEAR_START = 1971
# start year of the full model period (including stock-development from
# scratch, which needs to be the oldest year of any vehicle, all stock
# calculations are initiated in this year, so this has an effect on
# runtime)
# TODO: set FIRST_YEAR based on minimum value in data-files
YEAR_FIRST_GRID = 1926   # UK Electricity supply act - https://www.bbc.com/news/uk-politics-11619751   
YEAR_FIRST = 1807  # first_year_vehicle.values.min()
YEAR_END = 2060
YEAR_OUT = 2060 # year of output generation
YEAR_SWITCH = 1990  # year after which other batteries than lead-acid are allowed
REGIONS = 26


# Scenario settings ---------------------------------------------
SCEN = "SSP2"
# CP or 2D (Add "_RE" for Resource Efficiency)
VARIANT = "2D_RE"
# PROJECT = "mock_project"
# FOLDER = SCEN + "_" + VARIANT
# OUTPUT_FOLDER = base_dir.joinpath("..", "..", "output", PROJECT, FOLDER)


# Conversion factors ---------------------------------------------

MEGA_TO_TERA = 1_000_000  # 1 Tera = 1,000,000 Mega
PKMS_TO_VKMS = 1_000_000_000_000
TONNES_TO_KGS = 1000


# Vehicle related constants ---------------------------------------------

# reference loadfactor of cars in TIMER (the trp_trvl_Load.out file is
# relative to this BASE loadfcator (persons/car))
LOAD_FACTOR = 1.6

LIGHT_COMMERCIAL_VEHICLE_SHARE = 0.04 
# 0.04 is the fraction of the tkms driven by light commercial vehicles according to the IEA
BEV_CAPACITY_CURRENT  = 59.6    #kWh current battery capacity of full electric vehicles, see current_specs.xlsx
PHEV_CAPACITY_CURRENT = 11.2    #kWh current battery capacity of plugin electric vehicles, see current_specs.xlsx
# TODO: is this even used somewhere?


