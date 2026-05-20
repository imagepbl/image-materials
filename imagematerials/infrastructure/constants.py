"""Constants shared across infrastructure preprocessing and simulation."""

# Default scenario selection (matches imagematerials.electricity.constants pattern).
SCEN = "SSP2"

# Simulation horizon (used by infrastructure.py CLI).
YEAR_START = 1971   # first year of the simulation
YEAR_END = 2100     # last year of the simulation
YEAR_OUT = 2100     # last year of reporting

# v5 line 2192: first constructed highway — start of modern transportation era.
# Stocks are extended back to this year to give the dynamic stock model a tail.
FIRST_YEAR_GRID = 1911

# Maps IMAGE region column names used in some input files to the canonical
# region names used throughout the rest of the preprocessing.
COLUMN_MAPPING = {
    "China region": "China",
    "Indonesia region": "Indonesia Region",
    "Southeastern Asia": "South Eastern Asia",
    "Korea region": "Korea",
    "Russia region": "Russia Region",
}

# Lifetime road class map from the original v5 script
# (infra_lifetime_weibull.xlsx). Higher class = more durable. Urban roads
# get higher classes than rural equivalents. Used by both the material
# intensity and lifetime computations.
LIFETIME_RC_MAP = {
    "urban_paved":   {"motorway": 7, "primary": 6, "secondary": 5, "tertiary": 4,
                      "cycle": 1, "informal": 1, "local": 3, "pedestrian": 1},
    "urban_unpaved": {"motorway": 7, "primary": 6, "secondary": 5, "tertiary": 4,
                      "cycle": 1, "informal": 1, "local": 3, "pedestrian": 1},
    "rural_paved":   {"motorway": 6, "primary": 4, "secondary": 3, "tertiary": 2,
                      "cycle": 1, "informal": 1, "local": 1, "pedestrian": 1},
    "rural_unpaved": {"motorway": 6, "primary": 4, "secondary": 3, "tertiary": 2,
                      "cycle": 1, "informal": 1, "local": 1, "pedestrian": 1},
}
