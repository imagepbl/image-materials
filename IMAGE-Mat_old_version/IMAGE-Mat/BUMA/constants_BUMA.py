# Set general constants
regions = 26        # 26 IMAGE regions
building_types = 4  # 4 building types: detached, semi-detached, appartments & high-rise 
area = 2            # 2 areas: rural & urban
materials = 7       # 6 materials: Steel, Cement, Concrete, Wood, Copper, Aluminium, Glass
inflation = 1.2423  # gdp/cap inflation correction between 2005 (IMAGE data) & 2016 (commercial calibration) according to https://www.bls.gov/data/inflation_calculator.htm
start_year = 1971   # starting year of IMAGE data files
end_year = 2060     # year for which the output is generated (e.g. choose 2050 for shorter runtime & smaller filesize)
hist_year = 1721    # historick stock-tail is pre-caluculated from this year onward
switch_year = 2019  # year that the data on building type split (of the stock) ends
year_list_sva   = [*range(1970, 2025), *range(2030, 2101, 5)]
years = range(start_year, end_year + 1)
regions_range = range(1, regions+1)

# Set Flags for sensitivity analysis
flag_alpha  = 0     # switch for the sensitivity analysis on alpha, if 1 the maximum alpha is 10% above the maximum found in the data
flag_ExpDec = 0     # switch to choose between Gompertz and Exponential Decay function for commercial floorspace demand (0 = Gompertz, 1 = Expdec)
flag_Normal = 0     # switch to choose between Weibull and Normal lifetime distributions (0 = Weibull, 1 = Normal)
flag_Mean   = 0     # switch to choose between material intensity settings (0 = regular regional, 1 = mean, 2 = high, 3 = low, 4 = median)

# scenario selection
base_scenario    = "SSP2"
scenario_variant = "2D_RE"    # CP = Current Policies, 2D = 2-Degree Climate Policy, RE indicates additional resource efficiency assumptions

# scenario assumptions
if scenario_variant == "CP_RE" or scenario_variant == "2D_RE":
    lowComm = 0.7   # multiplier on commercial floorspace for resource efficience scenario (adjusts the maximum per cap commercial floorspace) transitioning between 2020 and 2060
else:
    lowComm = 1

#%%Load files & arrange tables ----------------------------------------------------

scenario_select = base_scenario + "_" + scenario_variant

if flag_Mean == 0:
    file_addition = ''
elif flag_Mean == 1:
    file_addition = '_mean'
elif flag_Mean ==2:
    file_addition = '_high'
elif flag_Mean ==3:
    file_addition = '_low'
else:
    file_addition = '_median'