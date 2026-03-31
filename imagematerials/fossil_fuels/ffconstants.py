from pathlib import Path
import os
import pint
import itertools
import matplotlib.pyplot as plt

ureg = pint.UnitRegistry(force_ndarray_like=True)

# --- Settings & constants
#first_production_year = {"coal": 1880, "oil": 1900, "gas": 1920}
#stabilisation_year = {"coal": 1920, "oil": 1970, "gas": 1970}
YEAR_FIRST_GRID =  1970 # change this to align with FUMA (it is the file i believe) 
#YEAR_FIRST = 1807       # first_year_vehicle.values.min()
#YEAR_SWITCH = 1990      # year after which other batteries than lead-acid are allowed
REGIONS = 26
COHORTS = 50 #check what this is


# Scenario settings ---------------------------------------------
#see about changing these to the correct ones (VLLO, ML, Lifetech, etc.)
STANDARD_SCEN_EXTERNAL_DATA = "SSP2_M_CP"
SCENARIO_DEFAULT = "SSP2_M_CP"

# Sensitivity Analysis - default, high_stor, high_grid
SENS_ANALYSIS = "default" 


# Conversion factors ---------------------------------------------

MEGA_TO_TERA = 1_000_000  # 1 Tera = 1,000,000 Mega
PKMS_TO_VKMS = 1_000_000_000_000
TONNES_TO_KGS = 1000


# General constants ---------------------------------------------

Standard_deviation_lifetime = 0.3 

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

TECH_GEN = 27   # number of fossil  technologies -> 26 technologies + 1 empty row

# # names of generation technologies as in the input files (e.g. composition_generation.csv) from Sebastiaan - should be renamed in files to match TIMER names in the future
# GEN_TYPES_FOSSIL = ["Coal", "Oil", "Gas"]

# names of fossil fuel types (check with TIMER model)
FF_TECHNOLOGIES = [
    "coal open cast",
    "coal underground",
    "oil onshore extraction",
    "oil offshore extraction",
    "gas onshore extraction",
    "gas offshore extraction",
    "coal preparation",
    "oil storage",
    "oil refinery",
    "gas processing",
    "coal truck",
    "coal train",
    "coal ocean ship",
    "coal inland ship",
    "oil truck",
    "oil train",
    "oil ocean ship",
    "oil inland ship",
    "gas truck",
    "gas train",
    "gas ocean ship",
    "gas inland ship"
    "oil crude composition pipeline",
    "oil product pipeline",
    "gas distribution pipeline",
    "gas transmission pipeline"
]

# Define mapping: technology -> category
DICT_GENTECH_TO_CATEGORY = {
    "coal open cast: coal",
    "coal underground: coal",
    "oil onshore extraction: oil",
    "oil offshore extraction: oil",
    "gas onshore extraction: gas",
    "gas offshore extraction: gas",
    "coal preparation: coal",
    "oil storage: oil",
    "oil refinery: oil",
    "gas processing: gas",
    "coal truck: coal",
    "coal train: coal",
    "coal ocean ship: coal",
    "coal inland ship: coal",
    "oil truck: oil",
    "oil train: oil",
    "oil ocean ship: oil",
    "oil inland ship: oil",
    "gas truck: gas",
    "gas train: gas",
    "gas ocean ship: gas",
    "gas inland ship: gas"
    "oil crude composition pipeline: oil",
    "oil product pipeline: oil",
    "gas distribution pipeline: gas",
    "gas transmission pipeline: gas"}

# Constant direct from FUMA
#primary_fuel_types_ff = ["coal", "conv oil", "unconv oil", "gas"]  # 1st 4

#final_fuel_types1 = [
   # "Solids",
   # "Liquids1",
    #"Sec gas",
    #"Hydrogen",
    #"Sec mod biomass",
    #"Sec heat",
    #"Sec trad biomass",
    #"Electricity",
    #"Liquids2",
#]
#final_fuel_types_ff1 = [
   # "Solids",
   # "Liquids1",
   # "Sec gas",
   # "Hydrogen",
    #"Sec heat",
    #"Electricity",
   # "Liquids2",
#]

#final_fuel_types2 = [
    #"Solids",
    #"Liquids1",
    #"Liquids2",
    #"Sec gas",
    #"Sec mod biomass",
    #"Sec trad biomass",
    #"Hydrogen",
    #"Sec heat",
    #"Electricity",
#]  # 1st 9, last one is total
#final_fuel_types_ff2 = [
   # "Solids",
    #"Liquids1",
   # "Liquids2",
    #"Sec gas",
#]  # 1st 9, last one is total

#sectors_ff = [
  #  "industry",
   # "transport",
   # "residential",
   # "services",
   # "other",
   # "non-energy",
   # "bunkers",
#]  # 1st 7, last one is total

# # Vehicle related constants ---------------------------------------------

# TECH_VEHICLES = 25    # number of vehicle types 

# # reference loadfactor of cars in TIMER (the trp_trvl_Load.out file is
# # relative to this BASE loadfcator (persons/car))
# LOAD_FACTOR = 1.6

# LIGHT_COMMERCIAL_VEHICLE_SHARE = 0.04 # TODO: is this even used somewhere?
# # 0.04 is the fraction of the tkms driven by light commercial vehicles according to the IEA
# BEV_CAPACITY_CURRENT  = 59.6    #kWh current battery capacity of full electric vehicles, see current_specs.xlsx
# PHEV_CAPACITY_CURRENT = 11.2    #kWh current battery capacity of plugin electric vehicles, see current_specs.xlsx



# Storage related constants ---------------------------------------------

# TECH_STORAGE = ?

#PHS_KG_PERKWH = 26.8   # kg per kWh storage capacity (as weight addition to existing hydro plants to make them pumped) 




# Visualization related ---------------------------------------------

# Generation technologies
technologies = [
    "coal open cast",
    "coal underground",
    "oil onshore extraction",
    "oil offshore extraction",
    "gas onshore extraction",
    "gas offshore extraction",
    "coal preparation",
    "oil storage",
    "oil refinery",
    "gas processing",
    "coal truck",
    "coal train",
    "coal ocean ship",
    "coal inland ship",
    "oil truck",
    "oil train",
    "oil ocean ship",
    "oil inland ship",
    "gas truck",
    "gas train",
    "gas ocean ship",
    "gas inland ship"
    "oil crude composition pipeline",
    "oil product pipeline",
    "gas distribution pipeline",
    "gas transmission pipeline"
]
# Define color and linestyle pools
colors = plt.get_cmap('tab20').colors  # 20 distinct colors
linestyles = ['-', '--', ':'] #'-.'
# Create a cycle of (color, linestyle) combinations
style_combinations = list(itertools.product(colors, linestyles))
# Map technologies to (color, linestyle)
DICT_GENTECH_STYLES = {tech: style_combinations[i] for i, tech in enumerate(technologies)}

DICT_GEN_CATEGORY_COLORS = {
    'Coal':             "#F9564F",
    'Oil':              "#5D536B",
    'Gas':              "#5C9EAD"
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

# DICT_GRID_COLORS = {
#     #'Lines Overhead': '#FF9B85',
#     #'Lines Underground': '#FFD97D',
#     'Lines':        '#8cb369', #'#007f5f',
#     'Transformers': '#f4a259', #'#aacc00',
#     'Substations':  '#bc4b51' #'#55a630'
# }

# DICT_GRID_STYLES_1 = {
#     'HV':                           ('#ef767a', '-'),
#     'HV - Lines - Overhead':        ('#ef767a', '-'),
#     'HV - Lines - Underground':     ('#ef767a', '--'),
#     'HV - Transformers':            ('#ef767a', '-'),
#     'HV - Substations':             ('#ef767a', '--'),

#     'MV':                           ('#456990', '-'),
#     'MV - Lines - Overhead':        ('#456990', '-'),
#     'MV - Lines - Underground':     ('#456990', '--'),
#     'MV - Transformers':            ('#456990', '-'),
#     'MV - Substations':             ('#456990', '--'),

#     'LV':                           ('#49beaa', '-'),
#     'LV - Lines - Overhead':        ('#49beaa', '-'),
#     'LV - Lines - Underground':     ('#49beaa', '--'),
#     'LV - Transformers':            ('#49beaa', '-'),
#     'LV - Substations':             ('#49beaa', '--')
# }

# DICT_GRID_STYLES_2 = {
#     'HV':                           ('#ef767a', '-'),
#     'HV - Lines - Overhead':        ('#f4845f', '-'),
#     'HV - Lines - Underground':     ('#f4845f', '--'),
#     'HV - Transformers':            ('#f7b267', '-'),
#     'HV - Substations':             ('#f25c54', '--'),

#     'MV':                           ('#456990', '-'),
#     'MV - Lines - Overhead':        ('#0077b6', '-'),
#     'MV - Lines - Underground':     ('#0077b6', '--'),
#     'MV - Transformers':            ('#c0fdff', '-'),
#     'MV - Substations':             ('#023e8a', '--'),

#     'LV':                           ('#38b000', '-'), ##49beaa
#     'LV - Lines - Overhead':        ('#70e000', '-'),
#     'LV - Lines - Underground':     ('#70e000', '--'),
#     'LV - Transformers':            ('#ccff33', '-'),
#     'LV - Substations':             ('#007200', '--')
# }

# DICT_ELECTR_COLORS = {
#     'Generation':   '#277da1',
#     'Storage':      '#f9844a',
#     'Transmission': '#90be6d'
# }


"""Module containing global constants
"""
import numpy as np
import prism
from prism import Q_

# Time
BASE_TIMELINE = prism.Timeline(
    start=Q_(1971, 'year'),
    end=Q_(2100, 'year'),
    stepsize=Q_(1, 'year'))
# End-point of calibration period (start year 1971) (some data up to 2007, but
# 2005 is guaranteed as complete dataset)
T_CALIB = Q_(2015, 'year')
# Last year of historic data for fossil fuel calibration.
T_CALIB_FOSSIL = Q_(2015, 'year')
# Time for transition to central learn matrix.
T_LEARN = Q_(2020, 'year')
T_STOP_LEARNING = Q_(2000, 'year')
# Year in which subsidy adjustment rate is fixed
T_SUBRATE = Q_(2015, 'year')
TRADE_YEAR = Q_(1971, 'year')
SUBTYPE_SEPARATOR = " - "


# Regions
IMAGE_REGIONS = [
    "CAN",
    "USA",
    "MEX",
    "RCAM",
    "BRA",
    "RSAM",
    "NAF",
    "WAF",
    "EAF",
    "SAF",
    "WEU",
    "CEU",
    "TUR",
    "UKR",
    "STAN",
    "RUS",
    "ME",
    "INDIA",
    "KOR",
    "CHN",
    "SEAS",
    "INDO",
    "JAP",
    "OCE",
    "RSAS",
    "RSAF"
]

Region = prism.Dimension('region', IMAGE_REGIONS + ["World"])  # World regions
ExtendedRegion = prism.Dimension('region', Region.coords + ["World2"])
ImportRegion = prism.Dimension("import_region", Region.coords)
ExportRegion = prism.Dimension("export_region", Region.coords)
ReducedRegion = prism.Dimension("region", IMAGE_REGIONS)
ReducedImportRegion = prism.Dimension("import_region", IMAGE_REGIONS)
ReducedExportRegion = prism.Dimension("export_region", IMAGE_REGIONS)


#def region_diag(
   # matrix: prism.Array[ExportRegion, ImportRegion],
  #  dim: str = "region"
#) -> prism.Array[Region]:
#    unit = matrix.pint.units if prism.USE_PINT else None
#    return prism.array(
#        np.diagonal(matrix),
#        dims={dim: Region.coords},
#        unit=unit)


# Energy carriers
EnergyCarrier = prism.Dimension("carrier", [
    "solid fuel",
    "liquid fuel",
    "gaseous fuel",
    "hydrogen",
    "modern biofuel",
    "secondary heat",
    "traditional biofuel",
    "electricity",
    "total"
])
PrimaryEnergyCarrier = prism.Dimension("carrier", [
    "coal",
    "conventional oil",
    "unconventional oil",
    "natural gas",
    "modern biofuel",
    "traditional biofuel",
    "nuclear",
    "renewables",
    "hydro-electricity"
])

# Resources
ConventionalCategory = prism.Dimension("category", [1, 2, 3, 4, 5, 6, 7])
UnconventionalCategory = prism.Dimension("category", [8, 9, 10, 11, 12])
Category = prism.Dimension(
    "category",
    [*ConventionalCategory.coords, *UnconventionalCategory.coords]
)  # Resource categories
CategoryType = prism.Dimension("category_type", ['conventional', 'unconventional'])

# Constants
#   Carbon content per fuel (coal, oil, gas and biomass) unit: kg-C/GJ
Fuel = prism.Dimension("fuel", ["coal", "oil", "gas", "modern biofuel"])
CARBON_CONTENT_FUEL_2 = prism.Array[Fuel, 'kg*C/GJ'](values=[25.5, 19.3, 15.3, 26])

# Calculations
EROICalculation = prism.Dimension("eroi_calculation", ['exogenous', 'endogenous'])

# Price deflator
DEFLATOR_WORLD = 1.12

# Tolerances
SAFE_MARGIN = 1.125  # Safety margin in investing in fossil fuel producing capacity.

EPS = 1.0e-6  # Small number used as tolerance.
if prism.USE_PINT:
    EPS_GJ = prism.Q_(EPS, 'GJ')
    EPS_PJ = prism.Q_(EPS, 'PJ')
    EPS_DOL_PER_GJ = prism.Q_(EPS, 'dol/GJ')
else:
    EPS_GJ = EPS
    EPS_PJ = EPS
    EPS_DOL_PER_GJ = EPS

# Unit conversions
if prism.USE_PINT:
    GJ = prism.Q_(1, 'GJ')
    TJ_TO_GJ = 1
    PJ_TO_GJ = 1
else:
    GJ = 1
    TJ_TO_GJ = 1e3
    PJ_TO_GJ = 1e6

# Energy carriers
_ENERGY_CARRIERS = [
    "Solid fuel",
    "Liquid fuel",
    "Gaseous fuel",
    "Hydrogen",
    "Modern Biofuel",
    "Secondary Heat",
    "Traditional BioFuel",
    "Electricity"
]
# Energy carriers in secondary fuel use
EnergyCarriers = prism.Dimension('energycarrier', _ENERGY_CARRIERS)

# Travel modes
_MODE_TRVL = [
    "Walk",
    "Bike",
    "Bus",
    "Train",
    "Car",
    "High-Speed Train",
    "Air"
]
TravelModes = prism.Dimension('mode', _MODE_TRVL)  # Travel modes
_MODE_FRGT = [
    "National Shipping",
    "Train",
    "Medium Truck",
    "Heavy Truck",
    "Air Cargo",
    "International Shipping",
    "Pipeline"
]
FreightModes = prism.Dimension('mode', _MODE_FRGT)  # Freight



