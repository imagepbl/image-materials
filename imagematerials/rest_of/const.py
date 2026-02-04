# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 14:36:58 2024

@author: Arp00003
"""

import pandas as pd
from pathlib import Path

#%%
# start year of analysis
start_year = 1971 
# end year of analysis. Can not be differentan 2100
end_year = 2100
# choose when switch from historic projection to scenario based projection
switch_year = 2100

#%%
# PATHS - set here your own data paths 

# INPUT
SCENARIO = "SSP2_CP"

path_input_data = Path(f"../../../data/raw/image/")
path_input_data_cons = "../../../data/raw/rest-of"

path_scenario_data_fossil = f"../../../data/raw/image/{SCENARIO}/EnergyFlows/"
path_scenario_data_water = f"../../../data/raw/image/{SCENARIO}/EnergyFlows/"

# Output
path_figures = "../figures_test"


#%%

def class_num_to_str(num: int):
    return f'class_ {num}'

numeric_region_map = {
    "CAN" : 'class_ 1',
    "USA": 'class_ 2',
    "MEX": 'class_ 3',
    "RCAM": 'class_ 4',
    "BRA": 'class_ 5',
    "RSAM": 'class_ 6',
    "NAF": 'class_ 7',
    "WAF": 'class_ 8',
    "EAF": 'class_ 9',
    "SAF": 'class_ 10',
    "WEU": 'class_ 11',
    "CEU": 'class_ 12',
    "TUR": 'class_ 13',
    "UKR": 'class_ 14',
    "STAN": 'class_ 15',
    "RUS": 'class_ 16',
    "ME": 'class_ 17',
    "INDIA": 'class_ 18',
    "KOR": 'class_ 19',
    "CHN": 'class_ 20',
    "SEAS": 'class_ 21',
    "INDO": 'class_ 22',
    "JAP": 'class_ 23',
    "OCE": 'class_ 24',
    "RSAS": 'class_ 25',
    "RSAF": 'class_ 26'
    }


# NRCT
CLASS_TO_REGION_DICT = {
    'class_ 1': 'Canada',
    'class_ 2': 'USA',
    'class_ 3': 'Mexico',
    'class_ 4': 'Rest C. America',
    'class_ 5': 'Brazil',
    'class_ 6': 'Rest S. America',
    'class_ 7': 'N. Africa',
    'class_ 8': 'W. Africa',
    'class_ 9': 'E. Africa',
    'class_ 10': 'South Africa',
    'class_ 11': 'W. Europe',
    'class_ 12': 'C. Europe',
    'class_ 13': 'Turkey',
    'class_ 14': 'Ukraine region',
    'class_ 15': 'Kazakhstan region',
    'class_ 16': 'Russia',
    'class_ 17': 'Middle East',
    'class_ 18': 'India',
    'class_ 19': 'Korea',
    'class_ 20': 'China',
    'class_ 21': 'SE. Asia',
    'class_ 22': 'Indonesia',
    'class_ 23': 'Japan',
    'class_ 24': 'Oceania',
    'class_ 25': 'Rest S. Asia',
    'class_ 26': 'Rest S. Africa',
    'class_ 27': 'Global'
}

all_regions = list(CLASS_TO_REGION_DICT.values())

REGION_TO_CLASS_DICT = {val: key for (key, val) in CLASS_TO_REGION_DICT.items()}

REGION_TO_CLASS_DICT_IMAGE_MAT = {
    'Canada': 'class_ 1',
    'US': 'class_ 2',
    'Mexico': 'class_ 3',
    'Rest C.Am.': 'class_ 4',
    'Brazil': 'class_ 5',
    'Rest S.Am.': 'class_ 6',
    'N.Africa': 'class_ 7',
    'W.Africa': 'class_ 8',
    'E.Africa': 'class_ 9',
    'South Africa': 'class_ 10',
    'W.Europe': 'class_ 11',
    'C.Europe': 'class_ 12',
    'Turkey': 'class_ 13',
    'Ukraine': 'class_ 14',
    'Stan': 'class_ 15',
    'Russia': 'class_ 16',
    'M.East': 'class_ 17',
    'India': 'class_ 18',
    'Korea': 'class_ 19',
    'China': 'class_ 20',
    'SE.Asia': 'class_ 21',
    'Indonesia': 'class_ 22',
    'Japan': 'class_ 23',
    'Oceania': 'class_ 24',
    'Rest S.Asia': 'class_ 25',
    'Rest S.Africa': 'class_ 26',
    'Global': 'class_ 27'}

REGION_TO_CLASS_DICT_IMAGE_MAT_NR = {
    'Canada': '1',
    'US': '2',
    'Mexico': '3',
    'Rest C.Am.': '4',
    'Brazil': '5',
    'Rest S.Am.': '6',
    'N.Africa': '7',
    'W.Africa': '8',
    'E.Africa': '9',
    'South Africa': '10',
    'W.Europe': '11',
    'C.Europe': '12',
    'Turkey': '13',
    'Ukraine': '14',
    'Stan': '15',
    'Russia': '16',
    'M.East': '17',
    'India': '18',
    'Korea': '19',
    'China': '20',
    'SE.Asia': '21',
    'Indonesia': '22',
    'Japan': '23',
    'Oceania': '24',
    'Rest S.Asia': '25',
    'Rest S.Africa': '26',
    'Global': 'class_ 27'}


COPPER_AVERAGE_REGIONS_TO_IMAGE = {
    'Other North America countries' : ['class_ 1', 'class_ 3', 'class_ 4'],
    'USA': ['class_ 2'], 
    'South America': ['class_ 5', 'class_ 6'],
    'Europe': ['class_ 11', 'class_ 12', 'class_ 13','class_ 14','class_ 15', 'class_ 16'],
    'Rest of Asia' : ['class_ 17', 'class_ 21', 'class_ 22', 'class_ 25'],
    'China' : ['class_ 20'],
    'Africa' : ['class_ 7', 'class_ 8', 'class_ 9', 'class_ 10', 'class_ 26'],
    'Oceania' : ['class_ 24'],
    'India': ['class_ 18'],
    'Japan' : ['class_ 23'],
    'Korea' : ['class_ 19'],
}

# Matching Regions to IMAGE Regions 

IAI_TO_IMAGE_CLASSES = {
    'Africa': [
        'class_ 7',  # N.Africa
        'class_ 8',  # W.Africa
        'class_ 9',  # E.Africa
        'class_ 10', # South Africa
        'class_ 26'  # Rest S.Africa
    ],
    'Asia (ex China)': [
        'class_ 17',  # M.East
        'class_ 18',  # India
        'class_ 19',  # Korea
        'class_ 21',  # SE.Asia
        'class_ 22',  # Indonesia
        'class_ 25'   # Rest S.Asia
    ],
    'Japan' : ['class_ 23'],  # Japan

    'China (Estimated)': [
        'class_ 20'  # China
    ],
    'Estimated Unreported to IAI': [
        'class_ 8',   # W.Africa (e.g. Cape Verde)
        'class_ 10',  # South Africa (e.g. ZA1)
        'class_ 26',  # Rest S.Africa (e.g. Réunion, Mayotte)
    ],
    'Gulf Cooperation Council': [
        'class_ 17'  # M.East
    ],
    'North America': [
        'class_ 1',  # Canada
        'class_ 2',  # US
        'class_ 3'   # Mexico
    ],
    'Oceania': [
        'class_ 24'  # Oceania
    ],
    'Russia & Eastern Europe': [
        'class_ 14',  # Ukraine
        'class_ 15',  # Stan
        'class_ 16'   # Russia
    ],
    'South America': [
        'class_ 4',  # Rest C.Am.
        'class_ 5',  # Brazil
        'class_ 6'   # Rest S.Am.
    ],
    'Western & Central Europe': [
        'class_ 11',  # W.Europe
        'class_ 12',  # C.Europe
        'class_ 13'   # Turkey
    ]
}


aluminium_regions = {
    'all_together' : ['Africa', 'Asia (ex China)', 
                    'Estimated Unreported to IAI', 'Gulf Cooperation Council',
                    'North America', 'Russia & Eastern Europe',
                    'South America', 'Western & Central Europe'],
    'china' : ['China (Estimated)'],
    'oceania' : ['Oceania'],
}

# get list of all IMAGE regions
all_regions_list = list(CLASS_TO_REGION_DICT.copy().values())

all_regions_list_class = list(REGION_TO_CLASS_DICT.values())


# SAND region grouping
# groups in IMAGE that are collectively being fitted for models because their similarity in consumption


# collect these above defined groups in a dictionary
SAND_GROUPING_REGIONS = {
    'Canada':  ['class_ 1'],
    'China':   ['class_ 20'],
    'Average': ['class_ 5', 'class_ 12', 'class_ 13', 'class_ 14','class_ 15', 
                'class_ 16', 'class_ 17', 'class_ 18', 'class_ 19', 
                'class_ 22', 'class_ 7', 'class_ 21'],
    'Lower':    ['class_ 3', 'class_ 4', 'class_ 6', 'class_ 9', 'class_ 10', 
                 'class_ 8', 'class_ 25', 'class_ 26'],
    'Japan':    ['class_ 23'],
    'High' : ['class_ 2', 'class_ 24', 'class_ 11']
    }

#%%

def get_key(val, my_dict):
   
    for key, value in my_dict.items():
        if val == value:
            return key
 
    return "key doesn't exist"
 

#%%

models_output_dict = {
    # 'log-log model' : 0,
    # 'semi-log model' : 1,
    # 'log-inverse model' : 2,
    # 'log-log-inverse model' : 3,
    # 'log-log-square model' : 4,
    'non-linear inverse model' : 0,
    'gompertz model' : 1,
    'logistic growth model' : 2,
    'limited growth model' : 3,
    'log gauss saturate model' : 4
    }

#%% parser function to write query function based on variable names of DIM1 and DIM2
# dictionaries can be used to adapt the script if changes occcur (add on categories, switch range, ...)

# parse like this: print(f'{parse_dim(1, "food")} and {parse_dim(2, "wheat")}')
def parse_dim(material_type: str, dim: int, *args): #args is tupel e.g. ('food',)
    #select either DIM1_dict or DIM2_dict, DIM is dependend on categroy (e.g. biomass) type
    category_dict = globals().get(f'DIM{dim}_{material_type}_dict', None) 
    assert category_dict is not None
    categories = [category_dict[category] for category in args] #collect categories
    return f'DIM_{dim} == {categories}'

#%% BIOMASS dictionaries

# NUFPT Types of use of food products + Total
DIM1_crops_dict = {
    'food': 1,
    'feed': 2,
    'other use' : 3, 
    'stock change': 4, # for animal products and crops stock change is empty
    'total': 5
}

# NFCT 
DIM2_crops_dict = {
    'wheat' : 1, 
    'rice' : 2,
    'maize' : 3,
    'tropical cereals' : 4,
    'other temperate cereals' : 5,
    'pulses' : 6,
    'soybeans': 7,
    'temperate oil crops' : 8,
    'tropical oil crops' : 9,
    'temperate roots & tubers' : 10,
    'tropical roots & tubers' : 11,
    'sugar crops' : 12,
    'oil, palm fruit' : 13,
    'vegetables & fruits' : 14,
    'other non-food luxury, spices' : 15,
    'plant based fibers' : 16,
    'total' : 17
}

# NGST grazing system
DIM1_tfeed_dict = {
    'intensive grazing system' : 1,
    'extensive grazing system' : 2,
    'total' : 3
}

# NFPT : Products used for feeds + total 
DIM2_tfeed_dict = {
    'food crops' : 1, 
    'animal products' : 2,
    'residues' : 3,
    'scavenging' : 4, 
    'grass & fodder' : 5,
    'total' : 6
}

#NAT: animal types + total
DIM3_tfeed_dict = {
    'non-dairy cattle' : 1,
    'dairy cattle' : 2,
    'pigs' : 3,
    'sheep & goats' : 4,
    'poultry' : 5,
    'total' : 6
}

#NWCT Wood products including Total
DIM1_wood_dict = {
    'pulpw. & particles' : 1,
    'sawlogs, veneer, other' : 2,
    'fuelw. & charcoal' : 3, 
    'total' : 4
}

# NAPT Animal products incl. Total
DIM2_animalproducts_dict = {
    'beef' : 1,
    'milk' : 2,
    'pork' : 3,
    'mutton &goat meat' : 4,
    'poultry & eggs' : 5,
    'total' : 6
}

# NBCT biofuel crops incl. total
DIM1_biofuelcrops_dict = {
    'sugar cane' : 1, # belongs to crop type sugar crops
    'maize' : 2,
    'woody biofuels' : 3,
    'non woody biofuels' : 4, # in LPJmL grassy biofuel
    'total' : 5
}

#%% FOSSIL dictionaries

# PRIM + 4: TPES extra (with 4 additional primary energy supply
DIM2_primary_dict = {
    'coal' : 1,
    'oil' : 2,
    'nan' : 3,
    'natural gas' : 4,
    'modern biofuel' : 5,
    'traditional biofuel' : 6,
    'nuclear' : 7,
    'solar thermal' : 8,
    'solar PV' : 9,
    'wind' : 10,
    'other renewables' : 11,
    'hydro electricity' : 12,
    'total' : 13
    }

# Primary to Secondary
# Primary
DIM1_primsec_dict = {
    1: 'coal', 
    2: 'conventional oil', 
    3: 'unconventional oil', 
    4: 'natural gasses', 
    5: 'modern biomass', # in TIMER: modern biomass
    6: 'traditional biomass', # in TIMER: traditional biomass
    7: 'nuclear', 
    8: 'solar thermal', 
    9: 'solar PV', 
    10: 'wind', 
    11: 'other renewables'}

DIM1_primsec_reversed_dict = {v: k for k, v in DIM1_primsec_dict.items()}

# Secondary adapted
DIM2_primsec_dict = {
    1: 'solids',            # solid fossils
    2: 'liquids',           # liquid fossils
    3: 'gasses',            # in TIMER: gasses fossils
    4: 'hydrogen',          # in TIMER: hydrogen 
    5: 'liquids',           # in TIMER: modern biofuel
    6: 'secondary heat', 
    7: 'liquids',           # in TIMER: traditional biofuel
    8: 'electricity', 
    9: 'non-energy'}         # in TIMER was: non-energy (i.e. feedstock) --> most comes from liquids and oil 

# Seconday original as in TIMER
DIM2_primsec_reversed_dict = {
    'solid fossils': 1,            
    'liquid fossils': 2,           
    'gasses fossils': 3,        
    'hydrogen' : 4,         
    'modern biofuel': 5,  
    'secondary heat' : 6, 
    'traditional biofuel' : 7,
    'electricity' : 8, 
    'non-energy' : 9}         # in TIMER was: non-energy (i.e. feedstock) --> most comes from liquids and oil 

#TODO: if sth goes to non-energy: split up in energy carrier (DIM3 for sec to final)
# TODO: if 


# Secondary to Final
# NECS9T number secondary energy carriers  incl total
DIM3_seconden_dict = {
    1 : 'solids',               # in TIMER: coal
    2 : 'liquids',              # in TIMER: heavy oil
    3 : 'liquids',              # in TIMER: light oil
    4 : 'gasses',       # in TIMER: natural gas
    5 : 'liquids',              # in TIMER: modern biofuel
    6 : 'liquids',              # in TIMER: traditional biofuel
    7 : 'gasses',             # in TIMER: hydrogen
    8 : 'secondary heat',
    9 : 'electricity',
    10 : 'total'
    }

# reversed and original as in TIMER
DIM3_seconden_reversed_dict = {
    'coal' : 1,               # in TIMER: coal
    'heavy oil' : 2,            # in TIMER: heavy oil
    'light oil' : 3,          # in TIMER: light oil
    'natural gas' : 4,       # in TIMER: natural gas
    'modern biofuel' : 5,          # in TIMER: modern biofuel
    'traditional biofuel' : 6,              # in TIMER: traditional biofuel
    'hydrogen' : 7,             # in TIMER: hydrogen
    'secondary heat' : 8,
    'electricity' : 9,
    'total' : 10
    }

DIM3_seconden_fossils_dict = {
    1 : 'solids',               # in TIMER: coal
    2 : 'liquids',              # in TIMER: heavy oil
    3 : 'liquids',              # in TIMER: light oil
    4 : 'gasses',       # in TIMER: natural gas
    5 : 'modern biofuel',              # in TIMER: modern biofuel
    6 : 'traditional biofuel',              # in TIMER: traditional biofuel
    7 : 'hydrogen',             # in TIMER: hydrogen
    8 : 'secondary heat',
    9 : 'electricity',
    10 : 'total'
    }

# S sectors final energy
DIM2_sectors_dict = {
    1 : 'industry',
    2 : 'transport (no bunkers)',
    3 : 'residential',
    4 : 'service',
    5 : 'other TFC',
    6 : 'non-energy',        
    7 : 'bunkers',
    8 : 'total'
    }


#%% weight conversions energy

# https://apps.cer-rec.gc.ca/Conversion/conversion-tables.aspx?GoCTemplateCulture=en-CA unit conversion

#%% IRP MF4Plus material categories

CAT_NON_METALLIC_MINERALS = [
    'Structural clays', 
    'Gypsum', 
    'Limestone',
    'Sand gravel and crushed rock for construction',
    'Chemical minerals n.e.c.',
    'Industrial minerals n.e.c',
    'Ornamental or building stone',
    'Salt', 'Industrial sand and gravel',
    'Other non-metallic minerals n.e.c.',
    'Specialty clays',
    'Dolomite',
    'Fertilizer minerals n.e.c.',
    'Chalk'
    ]

CAT_METALS = [
    'Chromium ores concentrates and compounds',
    'Copper ores concentrates and compounds',
    'Gold ores concentrates and compounds',
    'Iron ores concentrates and compounds',
    'Manganese ores concentrates and compounds',
    'Other metal ores concentrates and compounds nec. including mixed',
    'Bauxite and other aluminium ores concentrates and compounds',
    'Nickel ores concentrates and compounds',
    'Lead ores concentrates and compounds',
    'Silver ores concentrates and compounds',
    'Tin ores concentrates and compounds',
    'Uranium ores concentrates and compounds',
    'Zinc ores concentrates and compounds',
    'Platinum group metal ores concentrates and compounds',
    'Titanium ores concentrates and compounds',
    'Aluminium metals alloys and manufactures',
    'Chromium metals alloys and manufactures',
    'Copper metals alloys and manufactures',
    'Gold metals alloys and manufactures',
    'Iron metals alloys and manufactures',
    'Lead metals alloys and manufactures',
    'Magnesium metals alloys and manufactures',
    'Manganese metals alloys and manufactures',
    'Nickel metals alloys and manufactures',
    'Platinum group metal metals alloys and manufactures',
    'Silver metals alloys and manufactures',
    'Simple undefined or mixed processed metals nec',
    'Tin metals alloys and manufactures',
    'Titanium metals alloys and manufactures',
    'Uranium metals alloys and manufactures',
    'Zinc metals alloys and manufactures'
    ]

CAT_FOSSILS = [
    'Crude oil',
    'Other Bituminous Coal',
    'Natural gas',
    'Natural gas liquids',
    'Lignite (brown coal)',
    'Oil shale and tar sands',
    'Peat',
    'Anthracite',
    'Coking Coal',
    'Other Sub-Bituminous Coal'
    ]

CAT_BIOMASS = [
    'Timber (Industrial roundwood)',
    'Wild fish catch',
    'Wood fuel and other extraction',
    'Grazed biomass',
    'Cereals n.e.c.',
    'Other crop residues (sugar and fodder beet leaves etc)',
    'Fibres',
    'Fruits',
    'Nuts',
    'Oil bearing crops',
    'Pulses',
    'Rice',
    'Roots and tubers',
    'Spice - beverage - pharmaceutical crops',
    'Straw',
    'Sugar crops',
    'Vegetables',
    'Wheat',
    'All other aquatic animals',
    'Tobacco',
    'Aquatic plants',
    'Other crops n.e.c',
    'Fodder crops (including biomass harvest from grassland)'
    ]
           
CAT_PRODUCTS_FROM_BIOMASS = [
    'Products mainly from biomass nec.',
    'Dairy products birds eggs and honey',
    'Live animals other than in 1.4.',
    'Meat and meat preparations',
    'Other products from animals (animal fibres skins furs leather etc.)'
    ]

# The following categories also exist:
# Excavated earthen materials (including soil) nec Excavated earthen materials (including soil) nec
# Mixed and complex products nec. Mixed / complex products nec.
# Waste for final treatment and disposal Waste for final treatment and disposal


# Use relative paths
path_image_materials = Path("../image-materials")
path_image_scenarios = Path("image")

raw_data = path_image_materials / "data/raw"
path_rest_of_data = raw_data / "rest-of"

scenario_base_path = Path("scenario_config")

scenarios = ["SSP2_M_CP", "SSP2_VLLO", "SSP2_VLLO_LifeTech"]
scenario_list = {"base":("SSP2_M_CP", None),
                 "climate":("SSP2_VLLO", None),
                 "narrow":("SSP2_VLLO_LifeTech", ["narrow"])}

scenario_name_resource_model = "SSP2_M_CP"

cement_in_concrete_factor = 0.12
sand_in_cement_conversion = 0.17 #(silica)
sand_in_concrete = cement_in_concrete_factor*sand_in_cement_conversion
sand_in_glass_conversion = 0.7


R5_mapping = {
    "OECD & EU" : ["1", "12", "23", "24", "13", "2", "11"],
    "Reforming Economies" : ["16", "15", "14"],
    "Asia" : ["20", "18", "22", "19", "25", "21"],
    "Middle East & Africa" : ["9", "17", "7", "26", "10", "8"],
    "Latin America" : ["5", "3", "4", "6"]
}

REGION_TO_CLASS_DICT_IMAGE_MAT = {
    'Canada': '1',
    'US': '2',
    'Mexico': '3',
    'Rest C.Am.': '4',
    'Brazil': '5',
    'Rest S.Am.': '6',
    'N.Africa': '7',
    'W.Africa': '8',
    'E.Africa': '9',
    'South Africa': '10',
    'W.Europe': '11',
    'C.Europe': '12',
    'Turkey': '13',
    'Ukraine': '14',
    'Stan': '15',
    'Russia': '16',
    'M.East': '17',
    'India': '18',
    'Korea': '19',
    'China': '20',
    'SE.Asia': '21',
    'Indonesia': '22',
    'Japan': '23',
    'Oceania': '24',
    'Rest S.Asia': '25',
    'Rest S.Africa': '26'}

# Reverse mapping: class number to region name
CLASS_TO_REGION_DICT_IMAGE_MAT = {v: k for k, v in REGION_TO_CLASS_DICT_IMAGE_MAT.items()}

R5_to_IAI = {
    "OECD & EU" : ['Japan', 'North America', 'Western & Central Europe'],
    "Reforming Economies" : ['Russia & Eastern Europe'],
    "Asia" : ['Asia (ex China)', 'China (Estimated)'],
    "Middle East & Africa" : ['Africa', 'Estimated Unreported to IAI', 'Gulf Cooperation Council'],
    "Latin America" : ['South America']
    }


