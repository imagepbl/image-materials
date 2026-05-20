"""Module containing the constants used for the buildings sector.

This module will probably be (partly) deprecated at some point in favor of configuration
files and/or default values.
"""

import xarray as xr
import numpy as np
from imagematerials.concepts import KnowledgeGraph, Node

# Set general constants
REGIONS = 26         # 26 IMAGE regions
#building_types = 4  # 4 building types: detached, semi-detached, appartments & high-rise 
#area = 2            # 2 areas: rural & urban
#materials = 7       # 6 materials: Steel, Cement, Concrete, Wood, Copper, Aluminium, Glass
INFLATION = 1.2423  # gdp/cap inflation correction between 2005 (IMAGE data) & 2016 (commercial calibration) according to https://www.bls.gov/data/inflation_calculator.htm
START_YEAR = 1971   # starting year of IMAGE data files
END_YEAR = 2100     # year for which the output is generated (e.g. choose 2050 for shorter runtime & smaller filesize)
HIST_YEAR = 1721    # historick stock-tail is pre-caluculated from this year onward
#switch_year = 2019  # year that the data on building type split (of the stock) ends
YEARS = range(START_YEAR-1, END_YEAR + 1)
ALL_YEARS = list(range(HIST_YEAR, END_YEAR+1))
REGIONS_RANGE = range(1, REGIONS+1)

urban_share_1820 = 0.072 #https://www2.census.gov/programs-surveys/decennial/1990/tables/cph-2/table-4.pdf

# Set Flags for sensitivity analysis
FLAG_ALPHA  = 0     # switch for the sensitivity analysis on alpha, if 1 the maximum alpha is 10% above the maximum found in the data
FLAG_EXPDEC = 0     # switch to choose between Gompertz and Exponential Decay function for commercial floorspace demand (0 = Gompertz, 1 = Expdec)
FLAG_NORMAL = 0     # switch to choose between Weibull and Normal lifetime distributions (0 = Weibull, 1 = Normal)
flag_Mean   = 0     # switch to choose between material intensity settings (0 = regular regional, 1 = mean, 2 = high, 3 = low, 4 = median)

# scenario selection
base_scenario    = "SSP2"
scenario_variant = "CP"    # CP = Current Policies, 2D = 2-Degree Climate Policy, RE indicates additional resource efficiency assumptions

# scenario assumptions
if scenario_variant == "CP_RE" or scenario_variant == "2D_RE":
    LOWCOMM = 0.7   # multiplier on commercial floorspace for resource efficience scenario (adjusts the maximum per cap commercial floorspace) transitioning between 2020 and 2060
else:
    LOWCOMM = 1

GOMPERTZ_EXPDEC = (25.601, 28.431, 0.0415)

# Initialize minimum values
MINIMUM_COM = {
    "Office": 25,
    "Retail+": 25,
    "Hotels+": 25,
    "Govt+": 25
}

area_labels = {
    1: "Total",
    2: "Urban",
    3: "Rural",
    4: "Urban Q1",
    5: "Urban Q2",
    6: "Urban Q3",
    7: "Urban Q4",
    8: "Urban Q5",
    9: "Rural Q1",
    10: "Rural Q2",
    11: "Rural Q3",
    12: "Rural Q4",
    13: "Rural Q5",
}

urban_q_areas = ["Urban Q1", "Urban Q2", "Urban Q3", "Urban Q4", "Urban Q5"]
rural_q_areas = ["Rural Q1", "Rural Q2", "Rural Q3", "Rural Q4", "Rural Q5"]
quintiles_generic = ["Q1", "Q2", "Q3", "Q4", "Q5"]

commercial_types = ["Office", "Retail+", "Hotels+", "Govt+"]


# Re-broadcast Area: Urban/Rural -> quintiles using a local Area knowledge graph
area_graph = KnowledgeGraph(
    Node("Urban"),
    *[Node(q, inherits_from="Urban") for q in urban_q_areas],
    Node("Rural"),
    *[Node(q, inherits_from="Rural") for q in rural_q_areas],
)