""""Global constants for the ELMA model."""
from pathlib import Path
import os
import pint
import itertools
import matplotlib.pyplot as plt

ureg = pint.UnitRegistry(force_ndarray_like=True)

# --- Settings & constants

YEAR_FIRST_GRID = 1926  # UK Electricity supply act - https://www.bbc.com/news/uk-politics-11619751   
YEAR_FIRST = 1807       # first_year_vehicle.values.min()
YEAR_SWITCH = 1990      # year after which other batteries than lead-acid are allowed
REGIONS = 26
COHORTS = 50


# Scenario settings ---------------------------------------------
STANDARD_SCEN_EXTERNAL_DATA = "SSP2_M_CP"
SCENARIO_DEFAULT = "SSP2_M_CP"

# Sensitivity Analysis - default, high_stor, high_grid
SENS_ANALYSIS = "default" 


# Conversion factors ---------------------------------------------

MEGA_TO_TERA = 1_000_000  # 1 Tera = 1,000,000 Mega
PKMS_TO_VKMS = 1_000_000_000_000
TONNES_TO_KGS = 1000


# General constants ---------------------------------------------

STD_LIFETIMES_ELECTR = 0.214  # standard deviation as a fraction of the mean lifetime applicable to energy equipment (Asset Management for Infrastructure Systems: Energy and Water, Balzer & Schorn 2015)
# TODO: different std for lines, transformers, generation, storage, etc.? scenario dependent?

# Define the units for each dimension
unit_mapping = {
    'time': ureg.year,
    'year': ureg.year,
    'Year': ureg.year,
    'yr': ureg.year,
    'kg': ureg.kilogram,
    '%': ureg.percent,
    't': ureg.tonne,
    'MW': ureg.megawatt, #added
    'GW': ureg.gigawatt, #added
    'MWh': ureg.megawatthour, #added
}

REGIONS_TIMER = [ # TODO: check if this is true
    "Canada",
    "USA",
    "Mexico",
    "Rest of Central America",
    "Brazil",
    "Rest of South America",
    "Northern Africa",
    "Western Africa",
    "Eastern Africa",
    "South Africa",
    "Western Europe",
    "Central Europe",
    "Turkey",
    "Ukraine +",
    "Asian-Stan",
    "Russia +",
    "Middle East",
    "India +",
    "Korea",
    "China +",
    "Southeastern Asia",
    "Indonesia +",
    "Japan",
    "Oceania",
    "Rest of South Asia",
    "Rest of Southern Africa"
]

#  #regions in the file kilometrage.csv are like this:
#  region_list  = [
#  'Canada',
#  'US',
#  'Mexico',
#  'Rest C.Am.',
#  'Brazil',
#  'Rest S.Am.',
#  'N.Africa',
#  'W.Africa',
#  'E.Africa',
#  'South Africa',
#  'W.Europe',
#  'C.Europe',
#  'Turkey',
#  'Ukraine',
#  'Stan',
#  'Russia',
#  'M.East',
#  'India',
#  'Korea',
#  'China',
#  'SE.Asia',
#  'Indonesia',
#  'Japan',
#  'Oceania',
#  'Rest S.Asia',
#  'Rest S.Africa']

# Electricity Generation related constants ---------------------------------------------

TECH_GEN = 34   # number of electricity generation technologies -> 33 technologies + 1 empty row

# names of generation technologies as in the input files (e.g. composition_generation.csv) from Sebastiaan - should be renamed in files to match TIMER names in the future
GEN_TYPES_SEBASTIAAN = ["Solar PV", "Solar PV residential", "CSP", "Wind onshore", "Wind offshore", 
                        "Wave", "Hydro", "Other Renewables", "Geothermal","Hydrogen power", "Nuclear","<EMPTY>", "Conv. Coal",
                        "Conv. Oil", "Conv. Natural Gas","Waste", "IGCC", "OGCC", "NG CC", "Biomass CC",
                        "Coal + CCS", "Oil/Coal + CCS", "Natural Gas + CCS", "Biomass + CCS",
                        "CHP Coal", "CHP Oil", "CHP Natural Gas", "CHP Biomass",
                        "CHP Coal + CCS", "CHP Oil + CCS", "CHP Natural Gas + CCS", "CHP Biomass + CCS", "CHP Geothermal", "CHP Hydrogen"]
# names of generation technologies as in TIMER model
EPG_TECHNOLOGIES = [
    "SPV",
    "SPVR",
    "CSP",
    "WON",
    "WOFF",
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

EPG_TECHNOLOGIES_VRE = [
    "SPV",
    # "SPVR", #?
    "WON",
    "WOFF",
    # "WAVE", #?
]


# Define mapping: technology -> category
DICT_GENTECH_TO_CATEGORY = {
    "Solar PV": 'Solar', 
    "Solar PV residential": 'Solar',
    "CSP": 'Solar', 
    "Wind onshore": 'Wind',
    "Wind offshore": 'Wind', 
    "Wave": 'Other Renewables',
    "Hydro": 'Other Renewables',
    "Other Renewables": 'Other Renewables',
    "Geothermal": 'Other Renewables',
    'Hydrogen power': 'Hydrogen',
    "Nuclear": 'Nuclear',
    "Conv. Coal": 'Fossil',
    "Conv. Oil": 'Fossil',
    "Conv. Natural Gas": 'Fossil',
    "Waste": 'Fossil',
    "IGCC": 'Fossil',
    "OGCC": 'Fossil',
    "NG CC": 'Fossil',
    "Biomass CC": 'Biomass',
    "Coal + CCS": 'Fossil + CCS',
    "Oil/Coal + CCS": 'Fossil + CCS',
    "Natural Gas + CCS": 'Fossil + CCS',
    "Biomass + CCS": 'Biomass',
    "CHP Coal": 'Fossil',
    "CHP Oil": 'Fossil',
    "CHP Natural Gas": 'Fossil',
    "CHP Biomass": 'Biomass',
    "CHP Geothermal": 'Other Renewables',
    "CHP Hydrogen": 'Hydrogen',
    "CHP Coal + CCS": 'Fossil + CCS',
    "CHP Oil + CCS": 'Fossil + CCS',
    "CHP Natural Gas + CCS": 'Fossil + CCS',
    "CHP Biomass + CCS": 'Biomass'
}

# Vehicle related constants ---------------------------------------------

TECH_VEHICLES = 25    # number of vehicle types 

# reference loadfactor of cars in TIMER (the trp_trvl_Load.out file is
# relative to this BASE loadfcator (persons/car))
LOAD_FACTOR = 1.6

LIGHT_COMMERCIAL_VEHICLE_SHARE = 0.04 # TODO: is this even used somewhere?
# 0.04 is the fraction of the tkms driven by light commercial vehicles according to the IEA
BEV_CAPACITY_CURRENT  = 59.6    #kWh current battery capacity of full electric vehicles, see current_specs.xlsx
PHEV_CAPACITY_CURRENT = 11.2    #kWh current battery capacity of plugin electric vehicles, see current_specs.xlsx



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
# Map technologies to (color, linestyle)
DICT_GENTECH_STYLES = {tech: style_combinations[i] for i, tech in enumerate(technologies)}

DICT_GEN_CATEGORY_COLORS = {
    'Solar':             "#FBBF09",
    'Wind':              "#4BABFF",
    'Biomass':           "#42DD88",
    'Other Renewables':  "#B6F795",
    'Hydrogen':          '#B9FAF8',
    'Nuclear':           "#B06106",
    'Fossil':            "#575354",
    'Fossil + CCS':      "#BBB8B9"
}


# Grid Storage technologies
technologies = [
    "Flywheel", "Compressed Air", "Hydrogen FC", "NiMH", "Deep-cycle Lead-Acid", "LMO",
    "NMC", "NCA", "LFP", "LTO", "Zinc-Bromide", "Vanadium Redox", "Sodium-Sulfur", "ZEBRA",
    "Lithium Sulfur", "Lithium Ceramic", "Lithium-air"
]
# Define color and linestyle pools
colors = plt.get_cmap('tab20').colors  # 20 distinct colors
linestyles = ['-', '--', ':'] #'-.'
# Create a cycle of (color, linestyle) combinations
style_combinations = list(itertools.product(colors, linestyles))
# Map technologies to (color, linestyle)
DICT_STOR_STYLES = {tech: style_combinations[i] for i, tech in enumerate(technologies)}

DICT_STOR_CATEGORY_COLORS_SEBASTIAAN = {
    'mechanical storage':              "#AC501A",
    'PHS':                             "#6F4126",
    'lithium batteries':               "#E95E0D",
    'molten salt and flow batteries':  "#F07C32",
    'other':                           '#F69B58'
}



DICT_MATERIALS_COLORS = {
    'Steel':     '#FF9B85',
    'Aluminium': '#B9FAF8',
    'Concrete':  '#AAF683',
    'Plastics':  '#60D394',
    'Glass':     '#EE6055',
    'Cu':        '#FB6376',
    'Nd':        '#B8D0EB',
    'Ta':        '#B298DC',
    'Co':        '#F669C0',
    'Pb':        '#6F2DBD',
    'Mn':        "#31E7E7",
    'Ni':        '#FCB1A6',
    'Other':     '#FFD97D'
}

DICT_GRID_COLORS = {
    #'Lines Overhead': '#FF9B85',
    #'Lines Underground': '#FFD97D',
    'Lines':        '#8cb369', #'#007f5f',
    'Transformers': '#f4a259', #'#aacc00',
    'Substations':  '#bc4b51' #'#55a630'
}

DICT_GRID_STYLES_1 = {
    'HV':                           ('#ef767a', '-'),
    'HV - Lines - Overhead':        ('#ef767a', '-'),
    'HV - Lines - Underground':     ('#ef767a', '--'),
    'HV - Transformers':            ('#ef767a', '-'),
    'HV - Substations':             ('#ef767a', '--'),

    'MV':                           ('#456990', '-'),
    'MV - Lines - Overhead':        ('#456990', '-'),
    'MV - Lines - Underground':     ('#456990', '--'),
    'MV - Transformers':            ('#456990', '-'),
    'MV - Substations':             ('#456990', '--'),

    'LV':                           ('#49beaa', '-'),
    'LV - Lines - Overhead':        ('#49beaa', '-'),
    'LV - Lines - Underground':     ('#49beaa', '--'),
    'LV - Transformers':            ('#49beaa', '-'),
    'LV - Substations':             ('#49beaa', '--')
}

DICT_GRID_STYLES_2 = {
    'HV':                           ('#ef767a', '-'),
    'HV - Lines - Overhead':        ('#f4845f', '-'),
    'HV - Lines - Underground':     ('#f4845f', '--'),
    'HV - Transformers':            ('#f7b267', '-'),
    'HV - Substations':             ('#f25c54', '--'),

    'MV':                           ('#456990', '-'),
    'MV - Lines - Overhead':        ('#0077b6', '-'),
    'MV - Lines - Underground':     ('#0077b6', '--'),
    'MV - Transformers':            ('#c0fdff', '-'),
    'MV - Substations':             ('#023e8a', '--'),

    'LV':                           ('#38b000', '-'), ##49beaa
    'LV - Lines - Overhead':        ('#70e000', '-'),
    'LV - Lines - Underground':     ('#70e000', '--'),
    'LV - Transformers':            ('#ccff33', '-'),
    'LV - Substations':             ('#007200', '--')
}

DICT_ELECTR_COLORS = {
    'Generation':   '#277da1',
    'Storage':      '#f9844a',
    'Transmission': '#90be6d'
}


