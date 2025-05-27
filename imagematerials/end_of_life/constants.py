import warnings
from pathlib import Path
import xarray as xr
import numpy as np
import prism

# general

start_year  = 1971
end_year    = 2060
full_time = np.arange(start_year, end_year + 1)

# scenario
base_scenario    = "SSP2"
scenario_variant = "2D_RE"    # CP = Current Policies, 2D = 2-Degree Climate Policy, RE indicates additional resource efficiency assumptions
SCENARIO_SELECT = base_scenario + "_" + scenario_variant

# End-of-Life types
EolTypes = prism.Dimension("eol", [
    "reusable",
    "recyclable",
    "losses",
    "surplus losses"
]
)