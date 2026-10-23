""""Global constants for the ELMA model."""
from pathlib import Path
import os
import pint
import itertools
import matplotlib.pyplot as plt

ureg = pint.UnitRegistry(force_ndarray_like=True)

# --- Settings & constants

YEAR_FIRST_GRID = 1926  # UK Electricity supply act - https://www.bbc.com/news/uk-politics-11619751   
YEAR_FIRST_VEHICLES = 1807       # first_year_vehicle.values.min()
YEAR_SWITCH = 1990      # year after which other batteries than lead-acid are allowed
REGIONS = 26
COHORTS = 50


# Scenario settings ---------------------------------------------
STANDARD_SCEN_EXTERNAL_DATA = "SSP2_baseline"
SCENARIO_DEFAULT = "SSP2_baseline"

# Sensitivity Analysis - default, high_stor, high_grid
SENS_ANALYSIS = "default" 


# Conversion factors ---------------------------------------------

MEGA_TO_TERA = 1_000_000  # 1 Tera = 1,000,000 Mega
PKMS_TO_VKMS = 1_000_000_000_000
TONNES_TO_KGS = 1000

####################################################################################################
# General constants

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

IHA_REGIONS = ["North & Central America", "South America", "East Asia & Pacific", "South & Central Asia", "Europe", "Africa"]

iha_region_map = {
    # North & Central America
    "CAN": "North & Central America", "USA": "North & Central America", "MEX": "North & Central America", "RCAM": "North & Central America",
    # South America
    "BRA": "South America", "RSAM": "South America",
    # East Asia & Pacific
    "INDO": "East Asia & Pacific", "CHN": "East Asia & Pacific", "KOR": "East Asia & Pacific", "JAP": "East Asia & Pacific", "SEAS": "East Asia & Pacific", "OCE": "East Asia & Pacific",
    # South & Central Asia
    "ME": "South & Central Asia", "RUS": "South & Central Asia", "STAN": "South & Central Asia", "INDIA": "South & Central Asia", "RSAS": "South & Central Asia",
    # Europe
    "WEU": "Europe", "CEU": "Europe", "TUR": "Europe", "UKR": "Europe",
    # Africa
    "NAF": "Africa", "WAF": "Africa", "SAF": "Africa", "RSAF": "Africa", "EAF": "Africa",
}

####################################################################################################
# Generation related constants 

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

EPG_SUB_TECHNOLOGIES = [
    "SPV_c-Si",
    "SPV_AsGa",
    "SPV_CdTe",
    "SPV_CIGS",
    "SPV_a-Si",
    "SPVR_c-Si",
    "SPVR_AsGa",
    "SPVR_CdTe",
    "SPVR_CIGS",
    "SPVR_a-Si",
    "CSP_parabolic trough",
    "CSP_solar tower",
    "WON_GB-DFIG",
    "WON_GB-PMSG",
    "WON_DD-PMSG",
    "WON_DD-EESG",
    "WOFF_GB-DFIG",
    "WOFF_GB-PMSG",
    "WOFF_DD-PMSG",
    # "WOFF_DD-EESG",
    "WAVE_tech1",
    "HYD_tech1",
    "OREN_tech1",
    "GEO_tech1",
    "H2P_tech1",
    "NUC_tech1",
    "FREE12_tech1",
    "ClST_tech1",
    "OlST_tech1",
    "NGOT_tech1",
    "BioST_tech1",
    "IGCC_tech1",
    "OlCC_tech1",
    "NGCC_tech1",
    "BioCC_tech1",
    "ClCS_tech1",
    "OlCS_tech1",
    "NGCS_tech1",
    "BioCS_tech1",
    "ClCHP_tech1",
    "OlCHP_tech1",
    "NGCHP_tech1",
    "BioCHP_tech1",
    "ClCHPCS_tech1",
    "OlCHPCS_tech1",
    "NGCHPCS_tech1",
    "BioCHPCS_tech1",
    "GeoCHP_tech1",
    "H2CHP_tech1"
]

EPG_TECHNOLOGIES_FINAL = [
    "SPV_c-Si",
    "SPV_AsGa",
    "SPV_CdTe",
    "SPV_CIGS",
    "SPV_a-Si",
    "SPVR_c-Si",
    "SPVR_AsGa",
    "SPVR_CdTe",
    "SPVR_CIGS",
    "SPVR_a-Si",
    "CSP_parabolic trough",
    "CSP_solar tower",
    "WON_GB-DFIG",
    "WON_GB-PMSG",
    "WON_DD-PMSG",
    "WON_DD-EESG",
    "WOFF_GB-DFIG",
    "WOFF_GB-PMSG",
    "WOFF_DD-PMSG",
    # "WOFF_DD-EESG",
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

EPG_SUBTECH_TO_TECH = {
    "WON_GB-DFIG": ["WON"],
    "WON_GB-PMSG": ["WON"],
    "WON_DD-PMSG": ["WON"],
    "WON_DD-EESG": ["WON"],
    "WOFF_GB-DFIG": ["WOFF"],
    "WOFF_GB-PMSG": ["WOFF"],
    "WOFF_DD-PMSG": ["WOFF"],
    # "WOFF_DD-EESG": ["WOFF"],
    "CSP_parabolic trough": ["CSP"],
    "CSP_solar tower": ["CSP"],
    "SPV_c-Si": ["SPV"],
    "SPV_AsGa": ["SPV"],
    "SPV_CdTe": ["SPV"],
    "SPV_CIGS": ["SPV"],
    "SPV_a-Si": ["SPV"],
    "SPVR_c-Si": ["SPVR"],
    "SPVR_AsGa": ["SPVR"],
    "SPVR_CdTe": ["SPVR"],
    "SPVR_CIGS": ["SPVR"],
    "SPVR_a-Si": ["SPVR"],
    "WAVE_tech1": ["WAVE"],
    "HYD_tech1": ["HYD"],
    "OREN_tech1": ["OREN"],
    "GEO_tech1": ["GEO"],
    "H2P_tech1": ["H2P"],
    "NUC_tech1": ["NUC"],
    "FREE12_tech1": ["FREE12"],
    "ClST_tech1": ["ClST"],
    "OlST_tech1": ["OlST"],
    "NGOT_tech1": ["NGOT"],
    "BioST_tech1": ["BioST"],
    "IGCC_tech1": ["IGCC"],
    "OlCC_tech1": ["OlCC"],
    "NGCC_tech1": ["NGCC"],
    "BioCC_tech1": ["BioCC"],
    "ClCS_tech1": ["ClCS"],
    "OlCS_tech1": ["OlCS"],
    "NGCS_tech1": ["NGCS"],
    "BioCS_tech1": ["BioCS"],
    "ClCHP_tech1": ["ClCHP"],
    "OlCHP_tech1": ["OlCHP"],
    "NGCHP_tech1": ["NGCHP"],
    "BioCHP_tech1": ["BioCHP"],
    "ClCHPCS_tech1": ["ClCHPCS"],
    "OlCHPCS_tech1": ["OlCHPCS"],
    "NGCHPCS_tech1": ["NGCHPCS"],
    "BioCHPCS_tech1": ["BioCHPCS"],
    "GeoCHP_tech1": ["GeoCHP"],
    "H2CHP_tech1": ["H2CHP"],
}

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

# EV_BATTERIES see definition in imagematerials/vehicles/constants.py


####################################################################################################
# Storage related constants 

# Pumped Hydropower Storage (PHS) related constants ------------------------------------------------
# growth factor for PHS capacity relative to growth in storage energy demand after 2060 in the high 
# PHS scenario (phs_high); a value of 0.5 means that if the storage energy demand grows by 10% from 
# one year to the next, the PHS capacity will grow by 5% in that year.
factor_phs_growth_rel_demand = 0.5 
# Assumption on average discharge duration of PHS plants in hours. The discharge duration can vary 
# widely in reality, typically ranging 6-24 h. This simplifying assumption is needed due to the lack
# of data on installed storage energy capacity of PHS plants (MWh). Since the stock model is done in 
# terms of energy capacity, we need to convert the available power capacity data (MW) to energy capacity.
# Note: important for us is not the total energy produced by the PHS plant/energy cycled in a year, 
# but the energy capacity of the reservoir, which determines how much energy can be stored at a given 
# time as well as the materials needed for the reservoir construction (size of the reservoir).
mean_discharge_duration = 12
flag_phs_scenario = "phs_low" # "phs_low" or "phs_high"

PHS_KG_PERKWH = 26.8   # kg per kWh storage capacity (as weight addition to existing hydro plants to make them pumped) 

# Behind-the-meter storage related constants -------------------------------------------------------
# ratio_btm_to_solar = 0.5 #0.5 MW power capacity per 1 MW Solar PV
ratio_btm_to_solar = 2 #2 MWh energy capacity per 1 MW Solar PV

TECH_STATIONARY_STORAGE = ["LFP", "Na-ion", "low-nickel", "high-nickel"] # "LFP", "NMC333", "NMC532", "NMC622", "NMC811", "NMC955", "Na-ion", "flow-ZnBr", "flow-vanadium", "lead-acid"

TECH_STATIONARY_STORAGE_ALL = ["LFP", "Na-ion", "low-nickel", "high-nickel", "PHS"]

dict_storage_tech_to_groups  = {
    "nickel battery":           ["NiMH"],
    "lead-acid battery":        ["Deep-cycle Lead-Acid"],
    "lithium ion battery":      ["LMO", "NMC", "NCA", "LFP", "LTO"],
    "flow battery":             ["Zinc-Bromide", "Vanadium Redox"],
    "molten-salt battery":      ["Sodium-Sulfur", "ZEBRA"],
    "lithium-metal battery":    ["Lithium Sulfur", "Lithium Ceramic", "Lithium-air"],
    "mechanical":               ["PHS", "Flywheel", "Compressed Air"],
    "fuel-cells":               ["Hydrogen FC"]
}


####################################################################################################
# Visualization related 

# IMAGE regions
# Generate 26 distinct colors by combining named color sets
tab20 = plt.cm.tab20.colors          # 20 colors
tab20b = plt.cm.tab20b.colors        # 20 more, pick 6
extra = tab20b[::3][:6]              # pick every 3rd to maximize difference
COLORS_IMAGE_REGIONS = list(tab20) + list(extra)   # 26 colors total


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


