from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import os, re, yaml

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

YAML_DIR =  Path("../data/raw/reporting/yaml")

# Map IAMC family → YAML filename
FAMILY_YAML = {
    "Final Material Demand": "final_material_demand.yaml",
    "Final Product Demand":  "final-product-demand.yaml",
    "Product Stock":        "product-stock.yaml",
    "Stock Retirement":     "stock-retirement.yaml",
    "Material Stock":        "material-stock.yaml",
    "Material Outflow":      "material-outflow.yaml",
    "Material Losses":       "material-losses.yaml",
    "Scrap":                 "scrap.yaml",
}

# -----------------------------------------------------------------------------
# Core dimensions
# -----------------------------------------------------------------------------

CORE_DIMS: Dict[str, str] = {"time": "Time", "region": "Region"}

# -----------------------------------------------------------------------------
# Scenarios (IAMC label → model key)
# -----------------------------------------------------------------------------
SCENARIO_MAP: Dict[str, str] = {
    "SSP2": "base",
    "SSP2_narrow_act": "narrow_activity",
    "SSP2_narrow_prod": "narrow_product",    
    "SSP2_narrow": "narrow",
    "SSP2_slow": "slow",
    "SSP2_close": "close",
    "SSP2_narrow_slow_close": "narrow_slow_close",

    "SSP2_26_tax": "base_26",
    "SSP2_narrow_act_26_tax": "narrow_activity_26",
    "SSP2_narrow_prod_26_tax": "narrow_product_26",
    "SSP2_narrow_26_tax": "narrow_26",
    "SSP2_slow_26_tax": "slow_26",
    "SSP2_close_26_tax": "close_26",
    "SSP2_narrow_slow_close_26_tax": "narrow_slow_close_26",

    "SSP2_19_tax": "base_19",
    "SSP2_narrow_slow_close_19_tax": "narrow_slow_close_19",
}

# -----------------------------------------------------------------------------
# Helpers to load templates/units from YAML (keeps names editable in YAML)
# -----------------------------------------------------------------------------
def _load_family_templates(family: str) -> Tuple[List[str], Dict[str, str]]:
    """
    Return (templates, units_per_template) from a variable-family YAML file.

    YAML format expected (list of items):
      - "<IAMC template>":
          unit: "<unit string>"
    """
    family_name = FAMILY_YAML.get(family)
    if not family_name:
        return [], {}
    path = YAML_DIR / family_name
    if not path.exists():
        return [], {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    templates: List[str] = []
    units: Dict[str, str] = {}
    for item in data:
        if not isinstance(item, dict) or not item:
            continue
        tpl = next(iter(item.keys()))
        meta = item[tpl] if isinstance(item[tpl], dict) else {}
        templates.append(tpl)
        units[tpl] = meta.get("unit", "")
    return templates, units

# -----------------------------------------------------------------------------
# IAMC variable families (model_var points to the model key per sector module)
# -----------------------------------------------------------------------------
IAMC_VAR_SPECS: Dict[str, Dict] = {
    "Final Product Demand": {
        "model_var": "inflow",
        "templates": _load_family_templates("Final Product Demand")[0],
        "units_per_template": _load_family_templates("Final Product Demand")[1],
    },
    "Product Stock": {
        "model_var": "stocks",
        "templates": _load_family_templates("Product Stock")[0],
        "units_per_template": _load_family_templates("Product Stock")[1],
    },
    "Stock Retirement": {
        "model_var": "outflow",
        "templates": _load_family_templates("Stock Retirement")[0],
        "units_per_template": _load_family_templates("Stock Retirement")[1],
    },
    "Final Material Demand": {
        "model_var": "inflow_materials",
        "templates": _load_family_templates("Final Material Demand")[0],
        "units_per_template": _load_family_templates("Final Material Demand")[1],
    },
    "Material Stock": {
        "model_var": "stock_by_cohort_materials",  
        "templates": _load_family_templates("Material Stock")[0],
        "units_per_template": _load_family_templates("Material Stock")[1],
    },
    "Material Outflow": {
        "model_var": "outflow_by_cohort_materials",
        "templates": _load_family_templates("Material Outflow")[0],
        "units_per_template": _load_family_templates("Material Outflow")[1],
    },
    "Scrap": {
        "model_var": "sum_outflow",
        "templates": _load_family_templates("Scrap")[0],
        "units_per_template": _load_family_templates("Scrap")[1],
    },
    "Material Losses": {
        "model_var": "losses_materials",
        "templates": _load_family_templates("Material Losses")[0],
        "units_per_template": _load_family_templates("Material Losses")[1],
    },

}

# -----------------------------------------------------------------------------
# Knowledge graphs
# -----------------------------------------------------------------------------
from imagematerials.concepts import create_vehicle_graph, create_building_graph, create_electricity_graph, create_region_graph, Node

kgraph_v = create_vehicle_graph()
kgraph_b = create_building_graph()
kgraph_e = create_electricity_graph()

kgraph_region = create_region_graph()

# Vehicles knowledge graph
# road

car_types = ['Cars - BEV', 'Cars - FCV', 'Cars - HEV','Cars - ICE', 'Cars - PHEV', 'Cars - Trolley']
bus_types = ['Regular Buses - PHEV', 'Regular Buses - Trolley','Regular Buses - BEV', 'Regular Buses - FCV','Regular Buses - HEV', 'Regular Buses - ICE',
            'Midi Buses - BEV','Midi Buses - FCV', 'Midi Buses - HEV', 'Midi Buses - ICE','Midi Buses - PHEV', 'Midi Buses - Trolley'
                                                                  ]
truck_types = [ 'Heavy Freight Trucks - BEV', 'Heavy Freight Trucks - FCV','Heavy Freight Trucks - HEV', 'Heavy Freight Trucks - ICE','Heavy Freight Trucks - PHEV', 'Heavy Freight Trucks - Trolley',
            'Medium Freight Trucks - BEV', 'Medium Freight Trucks - FCV','Medium Freight Trucks - HEV', 'Medium Freight Trucks - ICE','Medium Freight Trucks - PHEV', 'Medium Freight Trucks - Trolley',
            'Light Commercial Vehicles - BEV','Light Commercial Vehicles - FCV','Light Commercial Vehicles - HEV','Light Commercial Vehicles - ICE','Light Commercial Vehicles - PHEV','Light Commercial Vehicles - Trolley'
            ]

kgraph_v.add(Node("Road", inherits_from="Vehicles"))
kgraph_v.add(Node("Road|Cars", inherits_from="Road"))
kgraph_v.add(Node("Road|Buses", inherits_from="Road"))
kgraph_v.add(Node("Road|Trucks", inherits_from="Road"))
kgraph_v.add(Node("Road|Bicycles", inherits_from="Road"))

for ct in car_types: kgraph_v[ct].inherits_from = "Road|Cars"
for bt in bus_types: kgraph_v[bt].inherits_from = "Road|Buses"
for tt in truck_types: kgraph_v[tt].inherits_from = "Road|Trucks"
kgraph_v["Bikes"].synonyms.append("Road|Bicycles")

# rail
pass_rail_types = ["Trains", "High Speed Trains"]
kgraph_v.add(Node("Rail", inherits_from="Vehicles"))
kgraph_v.add(Node("Rail|Passenger", inherits_from="Rail"))
kgraph_v.add(Node("Rail|Cargo", inherits_from="Rail"))
kgraph_v["Freight Trains"].synonyms.append("Rail|Cargo")
kgraph_v["Freight Trains"].synonyms.append("Rail|Cargo Trains")
kgraph_v["Freight Trains"].inherits_from = "Rail"
for pt in pass_rail_types: kgraph_v[pt].inherits_from = "Rail|Passenger"
kgraph_v["Rail|Passenger"].synonyms.append("Rail|Passenger Trains")

# air
kgraph_v.add(Node("Air", inherits_from="Vehicles"))
kgraph_v.add(Node("Air|Passenger", inherits_from="Air"))
kgraph_v.add(Node("Air|Cargo", inherits_from="Air"))
kgraph_v["Freight Planes"].synonyms.append("Air|Cargo")
kgraph_v["Freight Planes"].synonyms.append("Air|Cargo Planes")
kgraph_v["Passenger Planes"].synonyms.append("Air|Passenger")
kgraph_v["Passenger Planes"].synonyms.append("Air|Passenger Planes")

# water
kgraph_v.add(Node("Water", inherits_from="Vehicles"))
kgraph_v.add(Node("Water|Ships", inherits_from="Water"))
for st in ["Small Ships", "Medium Ships", "Large Ships", "Very Large Ships", "Inland Ships"]:
    kgraph_v[st].inherits_from = "Water|Ships"

# Buildings knowledge graph
# residential
kgraph_b.add(Node("Residential", inherits_from="Buildings"))
kgraph_b.add(Node("Residential|Detached Houses", inherits_from="Residential"))
kgraph_b.add(Node("Residential|Semi-Detached Houses", inherits_from="Residential"))
kgraph_b.add(Node("Residential|Apartments", inherits_from="Residential"))
kgraph_b.add(Node("Residential|Residential High-Rise", inherits_from="Residential"))

for detached_types in ["Detached - Urban", "Detached - Rural"]:
    kgraph_b[detached_types].inherits_from = "Residential|Detached Houses"
for semidetached_types in ["Semi-detached - Urban", "Semi-detached - Rural"]:
    kgraph_b[semidetached_types].inherits_from = "Residential|Semi-Detached Houses"
for apartment_types in ["Appartment - Urban", "Appartment - Rural"]:
    kgraph_b[apartment_types].inherits_from = "Residential|Apartments"
for highrise_types in ["High-rise - Urban", "High-rise - Rural"]:
    kgraph_b[highrise_types].inherits_from = "Residential|Residential High-Rise"

# commercial
kgraph_b.add(Node("Commercial", inherits_from="Buildings"))
kgraph_b.add(Node("Commercial|Retail", inherits_from="Commercial"))
kgraph_b.add(Node("Commercial|Hotels", inherits_from="Commercial"))
kgraph_b.add(Node("Commercial|Offices", inherits_from="Commercial"))
kgraph_b.add(Node("Commercial|Public Buildings", inherits_from="Commercial"))
kgraph_b["Retail+"].synonyms.append("Commercial|Retail")
kgraph_b["Hotels+"].synonyms.append("Commercial|Hotels")
kgraph_b["Office"].synonyms.append("Commercial|Offices")
kgraph_b["Govt+"].synonyms.append("Commercial|Public Buildings")

# Electricity knowledge graph
#kgraph_e.add(Node("Generation", inherits_from="Electricity"))
kgraph_e.add(Node("Generation|Nuclear", inherits_from="Generation"))
kgraph_e.add(Node("Generation|Solar", inherits_from="Generation"))
kgraph_e.add(Node("Generation|Wind", inherits_from="Generation"))
kgraph_e.add(Node("Generation|Gas", inherits_from="Generation"))
kgraph_e.add(Node("Generation|Coal", inherits_from="Generation"))
kgraph_e.add(Node("Generation|Biomass", inherits_from="Generation"))
kgraph_e.add(Node("Generation|Hydro", inherits_from="Generation"))
kgraph_e.add(Node("Generation|Other", inherits_from="Generation"))
kgraph_e.add(Node("Transmission and Distribution", inherits_from="Electricity"))
kgraph_e.add(Node("Transmission and Distribution|Lines", inherits_from="Transmission and Distribution"))
kgraph_e.add(Node("Transmission and Distribution|Transformers", inherits_from="Transmission and Distribution"))
kgraph_e.add(Node("Transmission and Distribution|Substations", inherits_from="Transmission and Distribution"))

#kgraph_e.add("Storage", inherits_from="Electricity"))
kgraph_e.add(Node("Storage|PHS", inherits_from="Storage"))
kgraph_e.add(Node("Storage|Other", inherits_from="Storage"))

# generation types
for solar_types in ["SPV", "SPVR", "CSP"]:
    kgraph_e[solar_types].inherits_from = "Generation|Solar"

for wind_types in ["WON", "WOFF"]:
    kgraph_e[wind_types].inherits_from = "Generation|Wind"

for biomass_types in ["BioST","BioCC","BioCS","BioCHP","BioCHPCS" ]:
    kgraph_e[biomass_types].inherits_from = "Generation|Biomass"

for nuclear_types in ["NUC"]:                            
    kgraph_e[nuclear_types].inherits_from = "Generation|Nuclear"

for hydro_types in ["HYD"]:
    kgraph_e[hydro_types].inherits_from = "Generation|Hydro"

for gas_types in ['NGOT',"NGCC","NGCS","NGCHP","NGCHPCS"]:
    kgraph_e[gas_types].inherits_from = "Generation|Gas"

for coal_types in ["ClST", "IGCC","ClCS","ClCHP","ClCHPCS"]:
    kgraph_e[coal_types].inherits_from = "Generation|Coal"

for other_types in ["WAVE","OREN","GEO","H2P","OlST", "OlCC","OlCS", "OlCHP","OlCHPCS","GeoCHP", "H2CHP", "FREE12"]:   # added FREE12 here bcs otherwise it crashes and it should be 0 anyway - TODO: improve this...
    kgraph_e[other_types].inherits_from = "Generation|Other"

# transmission and distribution types
for line_types in ['HV - Lines - Overhead', 'HV - Lines - Underground','MV - Lines - Overhead', 'MV - Lines - Underground',
 'LV - Lines - Overhead', 'LV - Lines - Underground']:
    kgraph_e[line_types].inherits_from = "Transmission and Distribution|Lines"

for transformer_types in ['HV - Transformers', 'MV - Transformers', 'LV - Transformers']:
    kgraph_e[transformer_types].inherits_from = "Transmission and Distribution|Transformers"
    
for substation_types in ['HV - Substations', 'MV - Substations','LV - Substations']:
    kgraph_e[substation_types].inherits_from = "Transmission and Distribution|Substations"

# storage types
for PHS_type in ['PHS']:
    kgraph_e[PHS_type].inherits_from = "Storage|PHS"

# for V2G_type in ['V2G']:                                  # remove the hasthag once V2G is included.
#     kgraph_e[PHS_type].inherits_from = "Storage|V2G"

for mechanical_storage_types in ['Compressed Air', 'Flywheel']:
    kgraph_e[mechanical_storage_types].inherits_from = "Storage|Other"

for lithium_battery_types in ['LFP', 'LMO', 'LTO', 'Lithium Ceramic', 'Lithium Sulfur', 'Lithium-air', 'NCA', 'NMC']:
    kgraph_e[lithium_battery_types].inherits_from = "Storage|Other"

# for molten_salt_and_flow_battery_types in []:
#     kgraph_e[molten_salt_and_flow_battery_types].inherits_from = "Molten Salt and Flow Battery"

for other_types in ['Deep-cycle Lead-Acid','Hydrogen FC', 'NiMH','Sodium-Sulfur', 'Vanadium Redox' ,'ZEBRA', 'Zinc-Bromide']: 
    kgraph_e[other_types].inherits_from = "Storage|Other"


TAG_WHITELISTS ={
   # =========================
    # BUILDINGS
    # =========================
    "Building Types": [
        # totals
        "Residential",
        "Commercial",

        # residential breakdown
        "Residential|Detached Houses",
        "Residential|Semi-Detached Houses",
        "Residential|Apartments",
        "Residential|Residential High-Rise",
        "Residential|Informal",

        # commercial breakdown
        "Commercial|Retail",
        "Commercial|Hotels",
        "Commercial|Offices",
        "Commercial|Public Buildings",
        "Commercial|Other",
    ],

    # =========================
    # VEHICLES
    # =========================
    "Vehicle Types": [
        # road
    
        "Road|Cars",
        "Road|Buses",
        "Road|Trucks",
        "Road|Bicycles",

        # rail

        "Rail|Passenger",
        "Rail|Cargo",

        # air
        "Air|Passenger",
        "Air|Cargo",

        # water
        "Water|Ships",
    ],
   # =========================
    # ELECTRICITY
    # =========================
    "Electricity Types": [
        "Generation",
        "Generation|Solar",
        "Generation|Wind",
        "Generation|Gas",
        "Generation|Coal",
        "Generation|Biomass",
        "Generation|Hydro",
        "Generation|Nuclear",
        "Generation|Other",
        "Transmission and Distribution",
        "Storage",
    ]
} 


# -----------------------------------------------------------------------------
# Renaming materials
# -----------------------------------------------------------------------------
MATERIAL_NAME_MAP: Dict[str, str] = {
    "aluminium": "Aluminum",
    "wood": "Construction Wood",
    'brick': "Brick",
    'concrete':"Concrete",
    'cement': "Cement",
    'copper': "Copper", 
    'glass': "Glass",        
    'plastics': "Chemicals|Plastics",
    'rubber': "Rubber", 
    'steel': "Steel", 

}

RAW_MATERIALS_KEEP = {"steel", "concrete", "copper", "aluminium", "brick", "wood", "plastics", "glass"}

# -----------------------------------------------------------------------------
# Demand sector for eol
# -----------------------------------------------------------------------------
# Group Type -> single IAMC label (Types are summed within each group)
EOL_DEMAND_SECTOR_GROUPS = {
    "Transportation": ["passenger", "freight"],
    "Buildings|Residential": ["urban", "rural"],
    "Buildings|Commercial": ["commercial"],
    "Buildings": ["urban", "rural", "commercial"],
    "Electricity": ["generation", "grid", "storage"], 
}

# -----------------------------------------------------------------------------
# list f
# -----------------------------------------------------------------------------
__all__ = [
    "YAML_DIR",
    "FAMILY_YAML",
    "kgraph_v",
    "kgraph_b",
    "kgraph_region",
    "CORE_DIMS",
    "SCENARIO_MAP",
    "IAMC_VAR_SPECS",
    "MATERIAL_NAME_MAP",
    "TAG_WHITELISTS",
]
