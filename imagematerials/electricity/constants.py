""""Global constants for the ELMA model."""
from pathlib import Path
import os
import pint
import itertools

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
YEAR_END = 2060    # end year of the calculations
YEAR_OUT = 2060    # year of output generation = last year of reporting (in the output files) 
YEAR_LAST = 2060   # last year available in the IMAGE data-files (which are input to ELMA)
YEAR_SWITCH = 1990 # year after which other batteries than lead-acid are allowed
REGIONS = 26

COHORTS = 50


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


# General constants ---------------------------------------------

STD_LIFETIMES_ELECTR = 0.214              # standard deviation as a fraction of the mean lifetime applicable to energy equipment (Asset Management for Infrastructure Systems: Energy and Water, Balzer & Schorn 2015)
# TODO: different std for lines, transformers, generation, storage, etc.? scenario dependent?


# Electricity Generation related constants ---------------------------------------------

TECH_GEN = 34   # number of electricity generation technologies -> 33 technologies + 1 empty row

# Vehicle related constants ---------------------------------------------

TECH_VEHICLES = 25    # number of vehicle types 

# reference loadfactor of cars in TIMER (the trp_trvl_Load.out file is
# relative to this BASE loadfcator (persons/car))
LOAD_FACTOR = 1.6

LIGHT_COMMERCIAL_VEHICLE_SHARE = 0.04 
# 0.04 is the fraction of the tkms driven by light commercial vehicles according to the IEA
BEV_CAPACITY_CURRENT  = 59.6    #kWh current battery capacity of full electric vehicles, see current_specs.xlsx
PHEV_CAPACITY_CURRENT = 11.2    #kWh current battery capacity of plugin electric vehicles, see current_specs.xlsx
# TODO: is this even used somewhere?


# Storage related constants ---------------------------------------------

# TECH_STORAGE = ?

PHS_KG_PERKWH = 26.8   # kg per kWh storage capacity (as weight addition to existing hydro plants to make them pumped) 




# Visualization related ---------------------------------------------

# Generation technologies
technologies = [
    'Solar PV', 'Solar PV residential', 'CSP', 'Wind onshore', 'Wind offshore', 'Wave', 'Hydro', 
    'Other Renewables', 'Geothermal', 'Hydrogen power', 'Nuclear', '<EMPTY>', 'Conv. Coal', 
    'Conv. Oil', 'Conv. Natural Gas', 'Waste', 'IGCC', 'OGCC', 'NG CC', 'Biomass CC', 
    'Coal + CCS', 'Oil/Coal + CCS', 'Natural Gas + CCS', 'Biomass + CCS', 'CHP Coal', 
    'CHP Oil', 'CHP Natural Gas', 'CHP Biomass', 'CHP Coal + CCS', 'CHP Oil + CCS', 
    'CHP Natural Gas + CCS', 'CHP Biomass + CCS', 'CHP Geothermal', 'CHP Hydrogen'
]
# Define color and linestyle pools
colors = plt.get_cmap('tab20').colors  # 20 distinct colors
linestyles = ['-', '--', ':'] #'-.'
# Create a cycle of (color, linestyle) combinations
style_combinations = list(itertools.product(colors, linestyles))
assert len(technologies) <= len(style_combinations), "Not enough unique combinations for all technologies."
# Map technologies to (color, linestyle)
dict_gentech_styles = {tech: style_combinations[i] for i, tech in enumerate(technologies)}


