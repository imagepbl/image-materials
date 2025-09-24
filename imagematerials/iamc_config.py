from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import os, re, yaml

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

YAML_DIR =  Path("../data/raw/circular_economy_scenarios/reporting/yaml")

# Map IAMC family → YAML filename
FAMILY_YAML = {
    "Final Material Demand": "final_material_demand.yaml",
    "Final Product Demand":  "final-product-demand.yaml",
    "Product Stock": "product-stock.yaml",
    "Stock Retirement": "stock-retirement.yaml",
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
    "SSP2_narrow": "narrow",
    "SSP2_slow": "slow",
    "SSP2_close": "close",
    "SSP2_narrow_slow_close": "narrow_slow_close",

    "SSP2_19": "base",
    "SSP2_narrow_19": "narrow",
    "SSP2_slow_19": "slow",
    "SSP2_close_19": "close",
    "SSP2_narrow_slow_close_19": "narrow_slow_close",

    "SSP2_26": "base",
    "SSP2_narrow_slow_close_26": "narrow_slow_close",
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
    fname = FAMILY_YAML.get(family)
    if not fname:
        return [], {}
    path = YAML_DIR / fname
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
        "model_var": "stock_by_cohort",
        "templates": _load_family_templates("Product Stock")[0],
        "units_per_template": _load_family_templates("Product Stock")[1],
    },
    "Stock Retirement": {
        "model_var": "outflow_by_cohort",
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
        "model_var": "recyclable_materials",
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
from imagematerials.concepts import create_vehicle_graph, create_building_graph,create_region_graph, Node

kgraph_v = create_vehicle_graph()
kgraph_b = create_building_graph()

kgraph_region = create_region_graph()

# ------- expanding knowledge graphs
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

# ------- Buildings KG enrichments
# residential
kgraph_b.add(Node("Residential", inherits_from="Buildings"))
kgraph_b.add(Node("Residential|Detached Houses", inherits_from="Residential"))
kgraph_b.add(Node("Residential|Semi-Detached Houses", inherits_from="Residential"))
kgraph_b.add(Node("Residential|Apartments", inherits_from="Residential"))
kgraph_b.add(Node("Residential|Residential High-Rise", inherits_from="Residential"))

for dt in ["Detached - Urban", "Detached - Rural"]:
    kgraph_b[dt].inherits_from = "Residential|Detached Houses"
for sdt in ["Semi-detached - Urban", "Semi-detached - Rural"]:
    kgraph_b[sdt].inherits_from = "Residential|Semi-Detached Houses"
for at in ["Appartment - Urban", "Appartment - Rural"]:
    kgraph_b[at].inherits_from = "Residential|Apartments"
for ht in ["High-rise - Urban", "High-rise - Rural"]:
    kgraph_b[ht].inherits_from = "Residential|Residential High-Rise"

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

# -----------------------------------------------------------------------------
# Simple name canonicalization for materials (no KG yet)
# -----------------------------------------------------------------------------
MATERIAL_NAME_MAP: Dict[str, str] = {
    "Aluminium": "Aluminum",
    "Wood": "Construction Wood",

}

# -----------------------------------------------------------------------------
# Demand sector for eol
# -----------------------------------------------------------------------------
# Group Type -> single IAMC label (we'll sum the Types within each group)
EOL_DEMAND_SECTOR_GROUPS = {
    "Transportation": ["passenger", "freight"],
    "Buildings|Residential": ["urban", "rural"],
    "Buildings|Commercial": ["commercial"],
    "Buildings": ["urban", "rural", "commercial"],
    "Electricity": ["generation", "grid", "storage"], 
}

# -----------------------------------------------------------------------------
# Routing: requested model variable name → IAMC family
# -----------------------------------------------------------------------------
ROUTE_MODELVAR_TO_IAMC: Dict[str, str] = {
    "inflow_materials": "Final Material Demand",
    "inflow": "Final Product Demand",
    "stock_by_cohort": "Product Stock",
    "outflow_by_cohort": "Stock Retirement",
    "stock_by_cohort_materials": "Material Stock",
    "stocks": "Material Stock",
    "outflow_by_cohort_materials": "Material Outflow",
    "outflow_materials": "Material Outflow",
    "losses_materials": "Material Losses",
    "recyclable_materials": "Scrap",
}

# -----------------------------------------------------------------------------
# Sector bindings
# -----------------------------------------------------------------------------
SECTOR_MODULES: Dict[str, Dict] = {
    "vehicles":  {"module_attr": "vehicles",  "placeholders": ["Vehicles"]},
    "buildings": {"module_attr": "buildings", "placeholders": ["Building Types"]},
    "eol":       {"module_attr": "eol",       "placeholders": []},
}

# -----------------------------------------------------------------------------
# Placeholder → DataArray dim + mapping per IAMC family
# Only keep what you use now; others can be added later.
# format: "path" (e.g., "Road|Cars") or "leaf" (e.g., "Cars")
# -----------------------------------------------------------------------------
# TAG_DIM_SPECS: Dict[str, Dict[str, Dict]] = {
#     "Final Product Demand": {
#         "Vehicles": {
#             "dim": "Type",
#             "kg": kgraph_v,
#             "to": "Vehicles",
#             "format": "path",
#         },
#         "Building Types": {
#             "dim": "Type",
#             "kg": kgraph_b,
#             "to": "Building Types",
#             "format": "path",
#         },
#         "Engineered Material": {
#             "dim": "material",
#             "kg": None,
#             "to": "Engineered Material",
#             "format": "leaf",
# },
    
#     },

#     "Final Material Demand": {},
#     "Product Stock": {},
#     "Stock Retirement": {},
#     "Material Stock": {},
#     "Material Outflow": {},
#     "Material Losses": {},
#     "Scrap": {},
# }

# -----------------------------------------------------------------------------
# Optional canonicalization after mapping
# -----------------------------------------------------------------------------
#TAG_RENAMES: Dict[str, Dict[str, str]] = {
    # "Building Types": {"Appartment": "Apartment"},
    # "Vehicles": {"Bikes": "Road|Bicycles"},
#}

# -----------------------------------------------------------------------------
# Optional: lightweight validation (will print NOTES for missing placeholder maps)
# -----------------------------------------------------------------------------
# def validate_config() -> None:
#     if not YAML_DIR.exists():
#         print(f"[iamc_config] WARNING: YAML_DIR does not exist: {YAML_DIR}")

#     for mv, fam in ROUTE_MODELVAR_TO_IAMC.items():
#         if fam not in IAMC_VAR_SPECS:
#             print(f"[iamc_config] WARNING: family {fam!r} (from {mv!r}) missing in IAMC_VAR_SPECS")
#             continue
#         if not IAMC_VAR_SPECS[fam].get("model_var"):
#             print(f"[iamc_config] WARNING: IAMC_VAR_SPECS[{fam!r}] missing model_var")

#     # Check placeholders in templates exist in TAG_DIM_SPECS
#     for fam, spec in IAMC_VAR_SPECS.items():
#         fam_tags = TAG_DIM_SPECS.get(fam, {})
#         for tpl in spec.get("templates", []):
#             phs = re.findall(r"\{([^}]+)\}", tpl)
#             for ph in phs:
#                 if ph not in fam_tags:
#                     print(f"[iamc_config] NOTE: placeholder {{{ph}}} in family {fam!r} has no TAG_DIM_SPECS entry")

# try:
#     validate_config()
# except Exception as _e:
#     print(f"[iamc_config] Validation error (non-fatal): {_e}")

# -----------------------------------------------------------------------------
# Public API
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
    "ROUTE_MODELVAR_TO_IAMC",
    "SECTOR_MODULES",
    "TAG_DIM_SPECS",
    "TAG_RENAMES",
    "MATERIAL_NAME_MAP",
    "validate_config",
]
