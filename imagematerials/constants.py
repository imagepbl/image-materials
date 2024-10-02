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
_IMAGE_REGIONS = [
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
Region = prism.NewDim('region', _IMAGE_REGIONS + ["World"])  # World regions
ExtendedRegion = prism.NewDim('region', Region.coords + ["World2"])
ImportRegion = prism.NewDim("import_region", Region.coords)
ExportRegion = prism.NewDim("export_region", Region.coords)
ReducedRegion = prism.NewDim("region", _IMAGE_REGIONS)
ReducedImportRegion = prism.NewDim("import_region", _IMAGE_REGIONS)
ReducedExportRegion = prism.NewDim("export_region", _IMAGE_REGIONS)


def region_diag(
    matrix: prism.Array[ExportRegion, ImportRegion],
    dim: str = "region"
) -> prism.Array[Region]:
    unit = matrix.pint.units if prism.USE_PINT else None
    return prism.array(
        np.diagonal(matrix),
        dims={dim: Region.coords},
        unit=unit)


# Energy carriers
EnergyCarrier = prism.NewDim("carrier", [
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
PrimaryEnergyCarrier = prism.NewDim("carrier", [
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
ConventionalCategory = prism.NewDim("category", [1, 2, 3, 4, 5, 6, 7])
UnconventionalCategory = prism.NewDim("category", [8, 9, 10, 11, 12])
Category = prism.NewDim(
    "category",
    [*ConventionalCategory.coords, *UnconventionalCategory.coords]
)  # Resource categories
CategoryType = prism.NewDim("category_type", ['conventional', 'unconventional'])

# Constants
#   Carbon content per fuel (coal, oil, gas and biomass) unit: kg-C/GJ
Fuel = prism.NewDim("fuel", ["coal", "oil", "gas", "modern biofuel"])
CARBON_CONTENT_FUEL_2 = prism.Array[Fuel, 'kg*C/GJ'](values=[25.5, 19.3, 15.3, 26])

# Calculations
EROICalculation = prism.NewDim("eroi_calculation", ['exogenous', 'endogenous'])

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
EnergyCarriers = prism.NewDim('energycarrier', _ENERGY_CARRIERS)

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
TravelModes = prism.NewDim('mode', _MODE_TRVL)  # Travel modes
_MODE_FRGT = [
    "National Shipping",
    "Train",
    "Medium Truck",
    "Heavy Truck",
    "Air Cargo",
    "International Shipping",
    "Pipeline"
]
FreightModes = prism.NewDim('mode', _MODE_FRGT)  # Freight
