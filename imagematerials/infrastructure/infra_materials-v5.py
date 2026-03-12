# -*- coding: utf-8 -*-
"""
@author: m.van.engelenburg@cml.leidenuniv.nl

Adapted from: deetman@cml.leidenuniv.nl
"""
# define imports, counters & settings
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from past.builtins import execfile
import math

YOUR_DIR = 'C:\\Academic\\Chapter2\\2. dMFA\\ELMA'
os.chdir(YOUR_DIR)

datapath = YOUR_DIR + '\\grid_data\\'

execfile('read_mym.py')         # module to read in mym (IMAGE output) files
from read_mym import read_mym_df
idx = pd.IndexSlice             # needed for slicing multi-index dataframes (pandas)
scenario = "SSP2"
variant  = "BL"                 # switch to "BL" for Baseline or '450' for 2-degree
sa_settings = "Feb_breakdown"         # settings for the sensitivity analysis (default, high_grid, dynamic_MI)
path = scenario + "\\" +  scenario + "_" + variant + "\\"

startyear = 1971
endyear = 2100
outyear = 2100                  # last year of reporting (in the output files)
years = endyear - startyear  + 1
regions = 26


###With declining inputs (population, gdp, hdi or other inputs) road are not directly being demolished. There is a time lag effect, or roads are not demolished at all. By turning on or off the following function we decouple the input indicator from the direct decrease in roads.
ROAD_DECLINE_FUNCTION = 1
RAIL_HISTOR_FUNCTION = 1



#%% Read in files --------------------------------------------------------------------------
#Load in a file to load in the specific IMAGE region
region_load_in = pd.read_csv(datapath + 'region_load.csv', index_col=0, names=None).transpose()
# Set the first column as the index, but keep it as a column
region_list = list(region_load_in.columns.values)

###Load in calibration data###
data_2024 = pd.read_excel('C:\\Academic\\Chapter2\\2. dMFA\\TRIPI-fut\\2024-data.xlsx', index_col=0)  # Load Urban Area data


load_share = read_mym_df('C:\\Academic\\Chapter2\\Scenarios\\'+scenario+'\\EnergyServices\\trp_trvl_Load.out')  

updated_vkm_pkm = read_mym_df('C:\\Academic\\Chapter2\\Scenarios\\'+scenario+'\\EnergyServices\\trp_trvl_pkm.out')    
updated_vkm_tkm = read_mym_df('C:\\Academic\\Chapter2\\Scenarios\\'+scenario+'\\EnergyServices\\trp_frgt_Tkm.out') 
gdp_cap = read_mym_df('C:\\Academic\\Chapter2\\Scenarios\\'+scenario+'\\Socioeconomic\\gdp_pc.scn')
urban_pop_df = read_mym_df('C:\\Academic\\Chapter2\\Scenarios\\'+scenario+'\\Socioeconomic\\URBPOPTOT.out')   
rural_pop_df = read_mym_df('C:\\Academic\\Chapter2\\Scenarios\\'+scenario+'\\Socioeconomic\\RURPOPTOT.out')

###EXTERNAL SOURCES###
hdi = pd.read_excel('C:\\Academic\\Chapter2\\2. dMFA\\TRIPI-fut\\HDI-kedi-et-al-2024-SSP2.xlsx', index_col=0)  # Load HDI data
area = pd.read_excel('C:\\Academic\\Chapter2\\2. dMFA\\TRIPI-fut\\urban-area_ssp4_dense.xlsx', index_col=0)  # Load Urban Area data

###MATERIAL INPUT FILES###
materials_shares = pd.read_excel('C:\\Academic\\Chapter2\\2. dMFA\\Formulas\\Materials-paving.xlsx', index_col=0)  # Load road surface shares
# Iterate through the groupings by 'Road_type', 'Region_type', and 'Paving_type'
for road_type, road_group in materials_shares.groupby('Road_type'):
    for region_type, region_group in road_group.groupby('Region_type'):
        for paving_type, paving_group in region_group.groupby('Paving_type'):
            # Create the variable name based on the first 3 letters of each grouping, prefixed by "m_"
            var_name = f"m_{road_type[:3]}_{region_type[:3]}_{paving_type[:3]}"
            
            # Drop the first three columns in the grouped data
            paving_group = paving_group.iloc[:, 3:]
            
            # Modify index names to characters after the last underscore in each index
            paving_group.index = paving_group.index.str.split('_').str[-1]
            
            # Assign the modified dataframe to a variable in the global scope
            globals()[var_name] = paving_group
mat_int = pd.read_excel('C:\\Academic\\Chapter2\\2. dMFA\\TRIPI-fut\\material-intensity-roads.xlsx', index_col=0)  # Load material intensity per road type and region
#material intensity bridges and tunnels
file_path = 'C:\\Academic\\Chapter2\\2. dMFA\\TRIPI-fut\\material-intensity-bridtun.xlsx'
sheets = pd.read_excel(file_path, sheet_name=None, index_col=0)
###ROAD TYPE DETERMINATION THROUGH HDI- LOAD IN FILES###
urban_road_types = pd.read_excel('C:\\Academic\\Chapter2\\2. dMFA\\TRIPI-fut\\urban-type-distribution.xlsx', index_col=0)  # Load HDI - area data
rural_road_types = pd.read_excel('C:\\Academic\\Chapter2\\2. dMFA\\TRIPI-fut\\rural-type-distribution.xlsx', index_col=0)  # Load HDI - area data

####ROAD PAVING SHARES DETERMINATION THROUGH GDP - LOAD IN FILES###
urban_paving_types = pd.read_excel('C:\\Academic\\Chapter2\\2. dMFA\\TRIPI-fut\\urban-paving-distribution.xlsx', index_col=0)  # Load HDI - area data
rural_paving_types = pd.read_excel('C:\\Academic\\Chapter2\\2. dMFA\\TRIPI-fut\\rural-paving-distribution.xlsx', index_col=0)  # Load HDI - area data

####RAIL INPUTS - HISTORIC LENGTHS AND GROWTH RATES
rail_lengths_df = pd.read_excel('C:\\Academic\\Chapter2\\2. dMFA\\TRIPI-fut\\rail_historic_length.xlsx', index_col=0)
r_values_df = pd.read_excel('C:\\Academic\\Chapter2\\2. dMFA\\TRIPI-fut\\rail_growth_rates.xlsx', index_col=0)
###RAIL INPUTS - ELECTRIFICATION LENGTH AND GROWTH RATES
rail_elec_lengths_df = pd.read_excel('C:\\Academic\\Chapter2\\2. dMFA\\TRIPI-fut\\rail_elec_length.xlsx', index_col=0)
r_elec_values_df = pd.read_excel('C:\\Academic\\Chapter2\\2. dMFA\\TRIPI-fut\\rail_elec_growth.xlsx', index_col=0)

###LIFETIMES AND MATERIAL INTENSITY FILES###
if sa_settings in ["IMAGE_jan_2026_CP_all_base", "IMAGE_jan_2026_CP_slow_base","SSP2_CP_all_19_tax","SSP2_CP_all_26_tax","SSP2_CP_slow_26_tax"]:
    lifetime_grid_elements = pd.read_csv(datapath + 'operational_lifetime_slow.csv', index_col=0)              # Average lifetime in years
else:
    lifetime_grid_elements = pd.read_csv(datapath + 'operational_lifetime.csv', index_col=0)              # Average lifetime in years
    
#lifetime_grid_elements = pd.read_csv(datapath + 'operational_lifetime.csv', index_col=0)              # Average lifetime in years
materials_infra = pd.read_excel(datapath + 'Materials_infra.xlsx', index_col=[0,1])                        # Material intensity of grid lines specific material content for Hv, Mv & Lv lines, & specific for underground vs. aboveground lines. (kg/km)
material_list = list(materials_infra.columns.values)

materials_infra_bt = pd.read_excel(datapath + 'Materials_infra_bt.xlsx', index_col=[0,1,2])                        # Material intensity of grid lines specific material content for Hv, Mv & Lv lines, & specific for underground vs. aboveground lines. (kg/km)
WeibullParamsDataFrame = pd.read_excel(datapath + 'infra_lifetime_weibull.xlsx', index_col=0)              # Average lifetime in years



#%% Preprocessing input variables --------------------------------------------------------------------------
##VKM##
#set multi-index based on the frist two columns
updated_vkm_pkm.set_index(['time'], inplace=True)
updated_vkm_tkm.set_index(['time'], inplace=True)

tkms_label  = ['regions','Inland ships', 'Cargo Trains', 'medium truck', 'heavy truck', 'Cargo Planes', 'Ships', 'empty', 'total']
pkms_label  = ['regions','walking', 'Bikes', 'Buses', 'Trains', 'Cars', 'HST', 'Planes', 'total']

# insert column descriptions
updated_vkm_tkm.columns = tkms_label
updated_vkm_pkm.columns = pkms_label

# Step 1: Sum up 'medium truck' and 'heavy truck' into 'Trucks'
updated_vkm_tkm['Trucks'] = updated_vkm_tkm['medium truck'] + updated_vkm_tkm['heavy truck']

# Step 2: Remove 'medium truck' and 'heavy truck' columns
updated_vkm_tkm = updated_vkm_tkm.drop(columns=['medium truck', 'heavy truck'])

# Step 3: Remove 'empty' and 'total' columns from tkms_df
updated_vkm_tkm = updated_vkm_tkm.drop(columns=['empty', 'total'])

# Step 4: Remove 'walking' and 'total' columns from pkms_df
updated_vkm_pkm = updated_vkm_pkm.drop(columns=['walking', 'total'])

# Exclude the 'regions' column when performing calculations
updated_vkm_tkm.iloc[:, 1:] = updated_vkm_tkm.iloc[:, 1:] * 1000000
updated_vkm_pkm.iloc[:, 1:] = updated_vkm_pkm.iloc[:, 1:] * 1000000000000

# Combine tkms_df and pkms_df into a new dataframe called vkm
vkm = pd.concat([updated_vkm_pkm, updated_vkm_tkm], axis=1)

# Drop the duplicate 'regions' column while keeping one
vkm = vkm.loc[:, ~vkm.columns.duplicated()]

# Rename the index from 'time' to 'years'
vkm = vkm.rename_axis("years")

vkm = vkm[~vkm["regions"].isin([27, 28])]

#Conversion of pkm and tkm to vkm following load and capacity factors from Table 7.1 of Deetman (2021)
vkm['Buses_vkm'] = vkm['Buses'] * 0.06 / 23 * 0.43 + vkm['Buses'] * 0.94 /57*0.43
vkm['Cars_vkm'] = vkm['Cars'] / (4 * 0.45)
vkm['Bicycle_vkm'] = vkm['Bikes']
vkm['Trucks_vkm'] = vkm['Trucks'] *0.04 / 0.74  + vkm['Trucks'] * 0.48 /7.95 + vkm['Trucks'] * 0.48 /14

##+ vkm['Bicycle_vkm'] excluded for now, as this might be the true driver of road construction. Use later for cycling area
#####RAIL_VKM#####
vkm['Trains_VKM'] = vkm['Trains'] /400  + (vkm['Cargo Trains'] /4165 * 0.45)
vkm['HST'] = vkm['HST'] /472 
vkm_rail = vkm[['Trains_VKM','regions']]
vkm_HST = vkm[['HST','regions']]
vkm_rail_pivoted = vkm_rail.pivot(columns='regions', values='Trains_VKM')
vkm_rail_HST_pivoted = vkm_HST.pivot(columns='regions', values='HST')
vkm_rail_pivoted.columns = region_list
vkm_rail_HST_pivoted.columns = region_list

vkm['total_vkm'] = vkm['Buses_vkm']  + vkm['Cars_vkm']   + vkm['Trucks_vkm']
# Keep only 'total_vkm' and 'region' columns
vkm = vkm[['total_vkm', 'regions']]
# Pivot the DataFrame to make 'region' values into columns
vkm_pivoted = vkm.pivot(columns='regions', values='total_vkm')
vkm_pivoted.columns = region_list

##GDP and POPULATION##
# Drop the last two columns from each dataframe
gdp_cap = gdp_cap.iloc[:, :-2]
urban_pop_df = urban_pop_df.iloc[:, :-2]
rural_pop_df = rural_pop_df.iloc[:, :-2]

# Now assign region names
gdp_cap.columns = region_list
urban_pop_df.columns = region_list
rural_pop_df.columns = region_list
# Rename the index from 'time' to 'years'
gdp_cap = gdp_cap.rename_axis("years")
urban_pop_df = urban_pop_df.rename_axis("years")
rural_pop_df = rural_pop_df.rename_axis("years")

gdp_cap_new = gdp_cap
# Define the mapping of old column names to new column names
column_mapping = {
    'China region': 'China',
    'Indonesia region': 'Indonesia Region',
    'Southeastern Asia': 'South Eastern Asia',
    'Korea region': 'Korea',
    'Russia region': 'Russia Region'
}
rural_pop_df = rural_pop_df.rename(columns=column_mapping)
urban_pop_df = urban_pop_df.rename(columns=column_mapping)
urban_population = urban_pop_df.sort_index(axis=1)
rural_population = rural_pop_df.sort_index(axis=1)
###HDI###
# Load in IMAGE-level HDI prediction - Currently only SSP2 from Kedi et al. (2024)
hdi = hdi.reset_index()
hdi = hdi.drop(columns=['Area', 'ISO3', 'ISOCode'])

IMAGE_hdi = hdi.groupby('IMAGE-region').mean()

# Step 1: Reindex the DataFrame to include all years from 1970 to 2100
all_years = list(range(1970, 2101))
IMAGE_hdi = IMAGE_hdi.reindex(columns=all_years)

# Step 2: Interpolate missing values linearly across rows
IMAGE_hdi = IMAGE_hdi.interpolate(method='linear', axis=1)
# Transpose the DataFrame
IMAGE_hdi_transposed = IMAGE_hdi.T
###############Change Column naming for HDI ######################
IMAGE_hdi_transposed = IMAGE_hdi_transposed.rename(columns=column_mapping)
IMAGE_hdi_transposed = IMAGE_hdi_transposed.drop(index=1970)


###LAND AND URBAN AREAS###
#############Load in urban and rural areas############
area = area.reset_index()
area = area.drop(columns=['Cntry_name', 'eFUA_name', 'GRIP-region','R10'])

IMAGE_total_area = area.groupby('IMAGE-region')['FUA_total_area'].sum()

area_clean = area.dropna()
IMAGE_urban_share = area_clean.groupby('IMAGE-region').sum()
IMAGE_urban_share = IMAGE_urban_share.drop(columns=['FUA_area', 'FUA_urban', 'FUA_p_2015'])
IMAGE_urban_share = IMAGE_urban_share.div(IMAGE_urban_share['FUA_total_area'], axis=0)
IMAGE_urban_share = IMAGE_urban_share.drop(columns=['FUA_total_area'])

all_years = list(range(1870, 2101))
IMAGE_urban_share = IMAGE_urban_share.reindex(columns=all_years)
IMAGE_urban_share = IMAGE_urban_share.interpolate(method='linear', axis=1)

###SMOOTHING OF INPUT AREA DATA - SOURCE INPUT AREA MIXED INPUTS - SMOOTHING THE TOTAL PROJECTION
# Select the relevant years
years_to_smooth = range(1950, 2101)
IMAGE_urban_share_subset = IMAGE_urban_share[years_to_smooth].copy()
# Apply moving average smoothing
window_avg_size = 20  # You can adjust the window size as needed
smoothed_data = IMAGE_urban_share_subset.apply(lambda col: col.rolling(window=window_avg_size, center=True, min_periods=1).mean(), axis=1)

# Reinsert the smoothed data back into the original DataFrame
IMAGE_urban_share[smoothed_data.columns] = smoothed_data
###END OF SMOOTHING FUNCTION

IMAGE_rural_share = 1 - IMAGE_urban_share

# Multiply all columns of IMAGE_urban_share with the IMAGE_total_area Series
IMAGE_urban_area = IMAGE_urban_share.mul(IMAGE_total_area, axis=0)
IMAGE_rural_area = IMAGE_rural_share.mul(IMAGE_total_area, axis=0)

################Calculate urban and rural population density#######################
# Filter to keep only the columns (years) from 1971 to 2060
IMAGE_urban_area_filtered = IMAGE_urban_area.loc[:, 1971:2100]
IMAGE_rural_area_filtered = IMAGE_rural_area.loc[:, 1971:2100]
# Transpose the filtered DataFrames
IMAGE_urban_area_transposed = IMAGE_urban_area_filtered.T
IMAGE_rural_area_transposed = IMAGE_rural_area_filtered.T

# Rename the columns in IMAGE_urban_area
IMAGE_urban_area_transposed = IMAGE_urban_area_transposed.rename(columns=column_mapping)
IMAGE_rural_area_transposed = IMAGE_rural_area_transposed.rename(columns=column_mapping)
IMAGE_urban_area_transposed = IMAGE_urban_area_transposed.sort_index(axis=1)
IMAGE_rural_area_transposed = IMAGE_rural_area_transposed.sort_index(axis=1)

urban_pop_density = urban_population* 1000000 / IMAGE_urban_area_transposed
rural_pop_density = rural_population* 1000000 / IMAGE_rural_area_transposed
pop_density = ((urban_population+rural_population) * 1000000) / (IMAGE_urban_area_transposed+IMAGE_rural_area_transposed)

# ############Load in GDP and split in urban and rural fractions#############
rural_share = rural_population / (rural_population+ urban_population)
urban_share = 1 - rural_share

gdp_total = gdp_cap_new * ((rural_population + urban_population) * 1000000)

# Define the regression function for GDP share urban - rural. See supplementary information Table --- Current Excel file C:\Academic\Chapter2\2. dMFA\Formulas\region_pkm_tkm
def gdp_share_regression(x):
    return 1.0267 * x + 0.0412
# Apply the regression function to all values in urban_share
# gdp_urban_share = urban_share.applymap(gdp_share_regression)

# # Cap values at a maximum of 0.98
# gdp_urban_share = gdp_urban_share.clip(upper=0.995)

# gdp_rural_share = 1 - gdp_urban_share

# gdp_urban_total = gdp_total * gdp_urban_share
# gdp_rural_total = gdp_total * gdp_rural_share

# gdp_urban_cap = gdp_urban_total / (urban_population * 1000000)
# gdp_rural_cap = gdp_rural_total / (rural_population * 1000000)

# Assuming gdp_total, urban_share, gdp_share_regression, 
# urban_population, and rural_population are defined DataFrames.

gdp_urban_share = urban_share.applymap(gdp_share_regression)
gdp_urban_share = gdp_urban_share.clip(upper=0.995)
gdp_rural_share = 1 - gdp_urban_share

# Population in millions (as used in your original calculation)
# Ensure no division by zero if population is 0
pop_u_mil = (urban_population * 1000000).replace(0, 1e-9)
pop_r_mil = (rural_population * 1000000).replace(0, 1e-9)

# Calculate the initial (uncorrected) allocations
initial_gdp_urban_total = gdp_total * gdp_urban_share
initial_gdp_rural_total = gdp_total * gdp_rural_share

# Calculate initial (uncorrected) per-capita values
initial_gdp_urban_cap = initial_gdp_urban_total / pop_u_mil
initial_gdp_rural_cap = initial_gdp_rural_total / pop_r_mil

## 1. 📈 Define Floors and Minimum Required GDP
print("Calculating GDP floors...")

# Use cummax() for a "high-water mark" peak. 
# This calculates the peak *up to that year* for each region.
urban_floor_cap = (initial_gdp_urban_cap.cummax() * 0.75).fillna(0)
rural_floor_cap = (initial_gdp_rural_cap.cummax() * 0.75).fillna(0)

# Calculate the minimum *total* GDP each sector *must* have to meet its floor
min_gdp_urban = urban_floor_cap * pop_u_mil
min_gdp_rural = rural_floor_cap * pop_r_mil

## 2. ⚖️ Identify "Solvable" vs. "Unsolvable" Scenarios
# The "unsolvable" scenario is when the *sum of minimums*
# is more than the *total available* GDP for that year/region.
total_required = min_gdp_urban + min_gdp_rural
is_unsolvable = total_required > gdp_total

## 3. 🛠️ Calculate Allocations for Both Scenarios

# --- Scenario A: "Solvable" (total_required <= gdp_total) ---
# We have enough GDP to satisfy both floors.
# We just need to enforce the boundaries by clipping the *initial* allocation.

# The new urban GDP must be *at least* its minimum...
min_allowed_urban_gdp = min_gdp_urban
# ...and *at most* what's left over after giving rural its minimum.
max_allowed_urban_gdp = gdp_total - min_gdp_rural

# Clip the initial allocation. This "nudges" the values to meet the floors
# while staying within the total GDP constraint.
# This clip is safe because in this scenario, min_allowed <= max_allowed.
urban_solvable = initial_gdp_urban_total.clip(
    lower=min_allowed_urban_gdp, 
    upper=max_allowed_urban_gdp
)

# --- Scenario B: "Unsolvable" (total_required > gdp_total) ---
# We CANNOT satisfy both floors.
# The "fair" solution is to split the *entire* gdp_total
# proportionally based on their minimum requirements.
# (Adding 1e-9 to avoid division by zero if both minimums are 0)
urban_share_of_required = min_gdp_urban / (total_required + 1e-9)
urban_unsolvable = gdp_total * urban_share_of_required

## 4. 🗺️ Combine Scenarios to Get Final Allocations
# Use np.where to pick the result from Scenario A or B for each cell.
# np.where(condition, value_if_true, value_if_false)
final_gdp_urban_total = np.where(
    is_unsolvable,
    urban_unsolvable,  # Use proportional split
    urban_solvable     # Use clipped value
)
# Convert back to DataFrame
final_gdp_urban_total = pd.DataFrame(
    final_gdp_urban_total, 
    index=gdp_total.index, 
    columns=gdp_total.columns
)

# Rural *always* gets what's left over. This ensures the total is preserved.
final_gdp_rural_total = gdp_total - final_gdp_urban_total

## 5. ✅ Final Recalculation
# These are your final, corrected values.
gdp_urban_total = final_gdp_urban_total
gdp_rural_total = final_gdp_rural_total

gdp_urban_cap = gdp_urban_total / pop_u_mil
gdp_rural_cap = gdp_rural_total / pop_r_mil

print("Redistribution complete.")
print("Final Urban GDP per Capita (head):")
print(gdp_urban_cap.head())
print("\nFinal Rural GDP per Capita (head):")
print(gdp_rural_cap.head())


#################Calculat urban & rural VKM###########################3
def vkm_urban_share_regression(x):
    return 0.1153 * x + 1637.4
# Apply the regression function to all values in urban_share
vkm_urban_cap = gdp_urban_cap.applymap(vkm_urban_share_regression)
vkm_urban_tot = vkm_urban_cap * urban_population * 1000000

def vkm_rural_share_regression(x):
    return 0.5603 * x + 78.578
# Apply the regression function to all values in urban_share
vkm_rural_cap = gdp_rural_cap.applymap(vkm_rural_share_regression)
vkm_rural_tot = vkm_rural_cap * rural_population * 1000000

vkm_urban_share = vkm_urban_tot / (vkm_urban_tot+vkm_rural_tot)
vkm_rural_share = 1 - vkm_urban_share

#Proof Tomer that this is the right approach
vkm_urban_IMAGE_cap = vkm_pivoted * vkm_urban_share / (urban_population * 1000000)
vkm_rural_IMAGE_cap = vkm_pivoted * vkm_rural_share / (rural_population * 1000000)

######################Sort columns by alphabet####################
# Sort the columns of the DataFrame alphabetically

sorted_IMAGE_hdi = IMAGE_hdi_transposed.sort_index(axis=1)
sorted_IMAGE_hdi = sorted_IMAGE_hdi.loc[sorted_IMAGE_hdi.index <= 2100]

sorted_urban_population = urban_population.sort_index(axis=1)
sorted_rural_population = rural_population.sort_index(axis=1)

sorted_urban_pop_density = urban_pop_density.sort_index(axis=1)
sorted_rural_pop_density = rural_pop_density.sort_index(axis=1)
sorted_pop_density = pop_density.sort_index(axis=1)

sorted_gdp_urban_cap = gdp_urban_cap.sort_index(axis=1)
sorted_gdp_rural_cap = gdp_rural_cap.sort_index(axis=1)




sorted_vkm_urban_cap = vkm_urban_IMAGE_cap.sort_index(axis=1)
sorted_vkm_rural_cap = vkm_rural_IMAGE_cap.sort_index(axis=1)

sorted_vkm_rail_tot = vkm_rail_pivoted.sort_index(axis=1)
sorted_vkm_HST_rail_tot = vkm_rail_HST_pivoted.sort_index(axis=1)

sorted_vkm_rail_cap = sorted_vkm_rail_tot / ((sorted_urban_population+sorted_rural_population * 1000000))

#%% Preprocessing infrastructure physical attributes --------------------------------------------------------------------------
#####################Road Area calculation###################
##########BACKCASTING - FORECASTING BASED ON 2024 DATA#######################
def rural_roads(x1, x2):
    # Calculate y based on the provided formula
    y = 0.000317871 + (-6.55403e-5 * np.log(x1)) + (x2 * 5.52978e-9)
    return y

def urban_roads(x1, x2, x3):
    # Calculate y based on the provided formula
    y = 0.000118208 + (-1.37056e-5 * np.log(x1)) + (x2 * 2.56944e-9) + (x3 * 3.30002e-10)
    return y

# Known data for 2024
actual_rural_road_area_2024 = data_2024.loc['rural_roads']  # Insert actual value for rural area in 2024
actual_urban_road_area_2024 = data_2024.loc['urban_roads']  # Insert actual value for urban area in 2024

sorted_rural_pop_density_2024 = sorted_rural_pop_density.loc[2024]
sorted_gdp_rural_cap_2024 = sorted_gdp_rural_cap.loc[2024]
sorted_rural_population_2024 = sorted_rural_population.loc[2024]

sorted_urban_pop_density_2024 = sorted_urban_pop_density.loc[2024]
sorted_vkm_urban_cap_2024 = sorted_vkm_urban_cap.loc[2024]
sorted_gdp_urban_cap_2024 = sorted_gdp_urban_cap.loc[2024]
sorted_urban_population_2024 = sorted_urban_population.loc[2024]

rural_road_cap = rural_roads(sorted_rural_pop_density,sorted_gdp_rural_cap)
urban_road_cap = urban_roads(sorted_urban_pop_density,sorted_vkm_urban_cap,sorted_gdp_urban_cap)

actual_rural_road_cap_2024 = actual_rural_road_area_2024 / (sorted_rural_population_2024*1000000)
actual_urban_road_cap_2024 = actual_urban_road_area_2024 / (sorted_urban_population_2024*1000000)

min_positive_values = rural_road_cap.apply(lambda row: row[row > 0].nsmallest(3).mean() if (row > 0).sum() >= 3 else row[row > 0].mean(),axis=1)

#min_positive_values = rural_road_cap.apply(lambda row: row[row > 0].min(), axis=1)
############SMOOTHING OF THE COLUMN VALUES ######################3
from scipy.ndimage import gaussian_filter1d

# Define the standard deviation for the Gaussian filter
std_dev = 5 # Adjust standard deviation for desired smoothness

# Apply Gaussian filter directly on the Series data
smoothed_min_positive_values = pd.Series(
    gaussian_filter1d(min_positive_values, sigma=std_dev), 
    index=min_positive_values.index
)

# Update the original Series with the smoothed values
min_positive_values = smoothed_min_positive_values

# Check for negative values and replace entire column where needed
for col in rural_road_cap.columns:
    if (rural_road_cap[col] < 0).any():  # Only process columns with negative values
        rural_road_cap[col] = min_positive_values  # Apply smoothed minimum positive value to entire column
# Step 2: Find the minimum positive value in each row of `rural_road_cap`
# This returns a Series where each value is the minimum positive value in the corresponding row
min_positive_values = urban_road_cap.apply(lambda row: row[row > 0].nsmallest(3).mean() if (row > 0).sum() >= 3 else row[row > 0].mean(),axis=1)
# Define the standard deviation for the Gaussian filter
# Apply Gaussian filter directly on the Series data
smoothed_min_positive_values = pd.Series(
    gaussian_filter1d(min_positive_values, sigma=std_dev), 
    index=min_positive_values.index
)
# Update the original Series with the smoothed values
min_positive_values = smoothed_min_positive_values

# Check for negative values and replace entire column where needed
for col in urban_road_cap.columns:
    if (urban_road_cap[col] < 0).any():  # Only process columns with negative values
        urban_road_cap[col] = min_positive_values  # Apply smoothed minimum positive value to entire column
############SMOOTHING OF THE COLUMN VALUES ######################3
# Applying Gaussian filter along each row, with a standard deviation controlling smoothness
std_dev = 1  # Adjust standard deviation for smoothness
# Apply Gaussian filter to each row, keeping index and columns
smoothed_urban_road_cap = urban_road_cap.apply(
    lambda col: pd.Series(gaussian_filter1d(col, sigma=std_dev), index=col.index),
    axis=0
)

# Set the original index and column names
smoothed_urban_road_cap.index = urban_road_cap.index
smoothed_urban_road_cap.columns = urban_road_cap.columns

urban_road_cap = smoothed_urban_road_cap

predicted_rural_road_cap_2024 = rural_road_cap.loc[2024]
predicted_urban_road_cap_2024 = urban_road_cap.loc[2024]

# Calculate adjustment factor based on 2024 actual vs. predicted
rural_adjustment_factor = actual_rural_road_cap_2024 / predicted_rural_road_cap_2024
urban_adjustment_factor = actual_urban_road_cap_2024 / predicted_urban_road_cap_2024

rural_road_cap_adjusted = rural_road_cap * rural_adjustment_factor
urban_road_cap_adjusted = urban_road_cap * urban_adjustment_factor

rural_road_area = rural_road_cap_adjusted * (sorted_rural_population*1000000)
urban_road_area = urban_road_cap_adjusted * (sorted_urban_population*1000000)

####Slowing Decline in road area###
# Check if ROAD_DECLINE_FUNCTION is enabled (1)
if ROAD_DECLINE_FUNCTION == 1:

    # Create a new DataFrame to store the adjusted values
    adjusted_rural_road_area = rural_road_area.copy()
    
    # Iterate through each column
    for column in rural_road_area.columns:
        for i in range(len(rural_road_area) - 1):  # Iterate through all years except the last
            if i + 1 < len(rural_road_area):  # Ensure we can check 10 years ahead
                current_value = adjusted_rural_road_area.iloc[i, adjusted_rural_road_area.columns.get_loc(column)]
                future_value = adjusted_rural_road_area.iloc[i + 1, adjusted_rural_road_area.columns.get_loc(column)]
                
                # Check if there's a decline over the next 5 years
                if future_value < current_value:
                    # Calculate the decline based on the difference between current and next year
                    next_year_value = adjusted_rural_road_area.iloc[i + 1, adjusted_rural_road_area.columns.get_loc(column)]
                    decline_amount = (current_value - next_year_value)
                    
                    # Adjust the current value (i) in the adjusted DataFrame
                    adjusted_rural_road_area.iloc[i + 1 , adjusted_rural_road_area.columns.get_loc(column)] = current_value - decline_amount * 0

    # Create a new DataFrame to store the adjusted values
    adjusted_urban_road_area = urban_road_area.copy()
    
    # Iterate through each column
    for column in urban_road_area.columns:
        for i in range(len(urban_road_area) - 1):  # Iterate through all years except the last
            if i + 1 < len(urban_road_area):  # Ensure we can check 10 years ahead
                current_value = adjusted_urban_road_area.iloc[i, adjusted_urban_road_area.columns.get_loc(column)]
                future_value = adjusted_urban_road_area.iloc[i + 1, adjusted_urban_road_area.columns.get_loc(column)]
                
                # Check if there's a decline over the next 5 years
                if future_value < current_value:
                    # Calculate the decline based on the difference between current and next year
                    next_year_value = adjusted_urban_road_area.iloc[i + 1, adjusted_urban_road_area.columns.get_loc(column)]
                    decline_amount = (current_value - next_year_value)
                    
                    # Adjust the current value (i) in the adjusted DataFrame
                    adjusted_urban_road_area.iloc[i+1, adjusted_urban_road_area.columns.get_loc(column)] = current_value - decline_amount * 0

    print("Road decline function is active. Roads will have a lagging effect to declining inputs.")

    obsolete_road_rural_area = adjusted_rural_road_area - rural_road_area
    obsolete_road_urban_area = adjusted_urban_road_area - urban_road_area    

    def apply_obsolete_rule(series):
        """
        Applies the "obsolete" rule to a pandas Series (a single column).
        It identifies the last continuous block of non-zero values at the end of the series.
        If this block is preceded by a zero, then all values from the beginning
        up to and including that preceding zero are set to 0.
        If the series ends with a zero, and there's a non-zero block before it,
        then the values up to the LAST zero *before* that final non-zero block are zeroed out.
        If there are no zeros, or if the series is continuously non-zero from the start
        to the last non-zero value, no change is made.
        """
        processed_series = series.copy()
        n = len(series)
    
        # 1. Find the start of the last continuous non-zero block from the end
        last_non_zero_block_start_idx = -1
        for i in range(n - 1, -1, -1):
            if series.iloc[i] != 0:
                last_non_zero_block_start_idx = i
            else:
                # If we hit a zero, and we've already found a non-zero block,
                # then this zero is the one preceding that block.
                if last_non_zero_block_start_idx != -1:
                    break # Found the first zero before the last non-zero block
        
        # Case 1: Series is all zeros or ends with a zero and nothing to save
        if last_non_zero_block_start_idx == -1: # Series is all zeros
            # If the series was all zeros, we might still want to return all zeros
            # or leave it as is, depending on the implicit meaning.
            # Given "retain values if no last 0", an all-zero series has no "last 0"
            # that precedes a non-zero block. So it should probably remain as is.
            # However, the original examples imply a zeroing out if a zero is present.
            # Let's say if it's all zeros, it remains all zeros.
            if (series == 0).all():
                return series # Already all zeros, nothing to change
            else: # This scenario should not happen if last_non_zero_block_start_idx == -1 unless series is all zeros
                return series # Should not be reached based on logic above
    
        # 2. Find the last zero *before* this last non-zero block
        zero_to_zero_out_up_to_idx = -1
        for i in range(last_non_zero_block_start_idx - 1, -1, -1):
            if series.iloc[i] == 0:
                zero_to_zero_out_up_to_idx = i
                break # Found the relevant zero, stop
                
        # 3. Apply zeroing if a relevant zero was found
        if zero_to_zero_out_up_to_idx != -1:
            processed_series.iloc[:zero_to_zero_out_up_to_idx + 1] = 0
        # Else: no relevant zero found (i.e., the series was continuously non-zero
        # from the start up to the last non-zero block), so no change.        
        return processed_series
    
    def process_obsolete_dataframes():
        """
        Iterates through all DataFrames in the global scope that have 'obsolete_' in their name,
        and applies the apply_obsolete_rule to each of their columns (vertically).
        """
        global_vars = globals()
        
        for var_name, var_value in global_vars.items():
            if isinstance(var_value, pd.DataFrame) and 'obsolete_' in var_name:
                print(f"Processing DataFrame: {var_name}")
                # Apply the rule to each column of the DataFrame.
                # This loop ensures the 'apply_obsolete_rule' (which operates vertically on a Series)
                # is called for each column independently.
                for col in var_value.columns:
                    print(f"  Applying rule vertically to column: '{col}'")
                    var_value[col] = apply_obsolete_rule(var_value[col])
                print(f"Finished processing {var_name}.\n")
    
    process_obsolete_dataframes()

    def fix_obsolete_columns(df):
        for col in df.columns:
            if df[col].iloc[-1] == 0 and not (df[col] == 0).all():
                df[col] = 0  # Replace entire column with 0
        
    fix_obsolete_columns(obsolete_road_rural_area)
    fix_obsolete_columns(obsolete_road_urban_area)
 
    adjusted_rural_road_area = adjusted_rural_road_area - obsolete_road_rural_area
    adjusted_urban_road_area = adjusted_urban_road_area - obsolete_road_urban_area
 
else:
    # Code if the function is inactive
    print("Road decline function is inactive. Roads follow normal logic.")
    adjusted_rural_road_area = rural_road_area
    adjusted_urban_road_area = urban_road_area

################Parking area calculation#######################Intercept issue - negative values#######
def rural_parking(x):
    return 0.0137 * x - 0.000000008
# Apply the regression function to all values in rural parking share
rural_parking_cap = rural_road_cap.applymap(rural_parking)
#rural_parking_cap = rural_parking_cap.where(rural_parking_cap >= 0, 0)

def urban_parking(x):
    return 0.1143 * x - 0.000000002
# Apply the regression function to all values in urban_share
urban_parking_cap = urban_road_cap.applymap(urban_parking)
#urban_parking_cap = urban_parking_cap.where(urban_parking_cap >= 0, 0)

# Known data for 2024
actual_rural_parking_area_2024 = data_2024.loc['rural_parking_area']  # Insert actual value for rural area in 2024
actual_urban_parking_area_2024 = data_2024.loc['urban_parking']  # Insert actual value for urban area in 2024

predicted_rural_parking_cap_2024 = rural_parking_cap.loc[2024]
predicted_urban_parking_cap_2024 = urban_parking_cap.loc[2024]
predicted_rural_parking_area_2024 = predicted_rural_parking_cap_2024 * (sorted_rural_population_2024 * 1000000)
predicted_urban_parking_area_2024 = predicted_urban_parking_cap_2024 * (sorted_urban_population_2024 * 1000000)

# Calculate adjustment factor based on 2024 actual vs. predicted
rural_parking_adjustment_factor = actual_rural_parking_area_2024 / predicted_rural_parking_area_2024
urban_parking_adjustment_factor = actual_urban_parking_area_2024 / predicted_urban_parking_area_2024

rural_parking_area = rural_parking_cap * (sorted_rural_population*1000000) * rural_parking_adjustment_factor / 1000000
urban_parking_area = urban_parking_cap * (sorted_urban_population*1000000) * urban_parking_adjustment_factor / 1000000

#####################PAVING SHARE#################################
def rural_paving(x):
    return 0.1244 * math.log(x) - 0.789
rural_paving_share = sorted_gdp_rural_cap.applymap(rural_paving)
rural_unpaved_share = 1 - rural_paving_share

def urban_paving(x):
    return 0.151* math.log(x) - 0.7656
urban_paving_share = sorted_gdp_urban_cap.applymap(urban_paving)
urban_unpaved_share = 1 - urban_paving_share

rural_paved_parking_area = rural_parking_area * rural_paving_share
rural_unpaved_parking_area = rural_parking_area * rural_unpaved_share

urban_paved_parking_area = urban_parking_area * urban_paving_share
urban_unpaved_parking_area = urban_parking_area * urban_unpaved_share
######################ROAD AREAS DISTRIBUTION########################

# Create a DataFrame from the data
urban_road_types_extend = pd.DataFrame(urban_road_types)

# New HDI range from 0.00 to 1.00 in 0.01 increments
new_HDI = np.arange(0.00, 1.01, 0.01)

# Interpolate each column based on the new HDI range
urban_road_types_extend = urban_road_types_extend.reindex(new_HDI).interpolate(method="linear")


# Step 1: Define new bins and labels that match the finer HDI increments
# We'll use 0.1 intervals for the bins, but you can adjust to even smaller increments if needed
bins = np.arange(0, 1.01, 0.01)  # Creates bins [0.0-0.1, 0.1-0.2, ..., 0.9-1.0]
labels = [f"{round(bins[i], 1)}-{round(bins[i+1] - 0.01, 2)}" for i in range(len(bins) - 1)]

# Step 2: Assign Urban_HDI_Range based on new bins and labels
urban_road_types_extend["Urban_HDI_Range"] = pd.cut(urban_road_types_extend.index, bins=bins, labels=labels, right=False)

# Step 3: Map sorted_IMAGE_hdi HDI values to the corresponding range
# Sample sorted_IMAGE_hdi DataFrame (replace with actual data if available)
sorted_IMAGE_hdi_ranges = sorted_IMAGE_hdi.apply(lambda col: pd.cut(col, bins=bins, labels=labels, right=True))


# Step 3: Create separate DataFrames for each '_share' column in urban_road_types
share_dfs = {}

# Merge each HDI range in sorted_IMAGE_hdi_ranges with the corresponding share data in urban_road_types
for column in urban_road_types_extend.columns:
    if '_share' in column:
        # Merge each share column by the HDI range
        share_dfs[column] = sorted_IMAGE_hdi_ranges.applymap(lambda x: urban_road_types_extend.loc[urban_road_types_extend['Urban_HDI_Range'] == x, column].values[0])
# Dictionary to store the area-adjusted DataFrames
area_adjusted_dfs = {}
obsolete_adjusted_dfs = {}

# Multiply each `_share` DataFrame in `share_dfs` by `urban_road_area`
for column, share_df in share_dfs.items():
    # Multiply element-wise by `urban_road_area`
    area_adjusted_dfs[column] = share_df * adjusted_urban_road_area
    obsolete_adjusted_dfs[column] = share_df * obsolete_road_urban_area

u_cycling_area = area_adjusted_dfs['cycle_share']
u_informal_area = area_adjusted_dfs['informal_share']
u_local_area = area_adjusted_dfs['local_share']
u_motorway_area = area_adjusted_dfs['motorway_share']
u_pedestrian_area = area_adjusted_dfs['pedestrian_share']
u_primary_area = area_adjusted_dfs['primary_share']
u_secondary_area = area_adjusted_dfs['secondary_share']
u_tertiary_area = area_adjusted_dfs['tertiary_share']   

u_cycling_obsolete = obsolete_adjusted_dfs['cycle_share']
u_informal_obsolete = obsolete_adjusted_dfs['informal_share']
u_local_obsolete = obsolete_adjusted_dfs['local_share']
u_motorway_obsolete = obsolete_adjusted_dfs['motorway_share']
u_pedestrian_obsolete = obsolete_adjusted_dfs['pedestrian_share']
u_primary_obsolete = obsolete_adjusted_dfs['primary_share']
u_secondary_obsolete = obsolete_adjusted_dfs['secondary_share']
u_tertiary_obsolete = obsolete_adjusted_dfs['tertiary_share']   

#######RURAL
# Create a DataFrame from the data
rural_road_types_extend = pd.DataFrame(rural_road_types)

# Interpolate each column based on the new HDI range
rural_road_types_extend = rural_road_types_extend.reindex(new_HDI).interpolate(method="linear")
# Step 1: Define new bins and labels that match the finer HDI increments
# We'll use 0.1 intervals for the bins, but you can adjust to even smaller increments if needed
bins = np.arange(0, 1.01, 0.01)  # Creates bins [0.0-0.1, 0.1-0.2, ..., 0.9-1.0]
labels = [f"{round(bins[i], 1)}-{round(bins[i+1] - 0.01, 2)}" for i in range(len(bins) - 1)]
# Step 2: Assign Urban_HDI_Range based on new bins and labels
rural_road_types_extend["Rural_HDI_Range"] = pd.cut(rural_road_types_extend.index, bins=bins, labels=labels, right=False)

# Step 3: Map sorted_IMAGE_hdi HDI values to the corresponding range
# Sample sorted_IMAGE_hdi DataFrame (replace with actual data if available)
sorted_IMAGE_hdi_ranges = sorted_IMAGE_hdi.apply(lambda col: pd.cut(col, bins=bins, labels=labels, right=True))

# Step 4: Create separate DataFrames for each '_share' column in rural_road_types
share_dfs = {}

# Merge each HDI range in sorted_IMAGE_hdi_ranges with the corresponding share data in rural_road_types
for column in rural_road_types_extend.columns:
    if '_share' in column:
        # Merge each share column by the HDI range
        share_dfs[column] = sorted_IMAGE_hdi_ranges.applymap(lambda x: rural_road_types_extend.loc[rural_road_types_extend['Rural_HDI_Range'] == x, column].values[0])
# Dictionary to store the area-adjusted DataFrames
area_rur_adjusted_dfs = {}
obsolete_rur_adjusted_dfs = {}

# Multiply each `_share` DataFrame in `share_dfs` by `urban_road_area`
for column, share_df in share_dfs.items():
    # Multiply element-wise by `urban_road_area`
    area_rur_adjusted_dfs[column] = share_df * adjusted_rural_road_area
    obsolete_rur_adjusted_dfs[column] = share_df * obsolete_road_rural_area

r_cycling_area = area_rur_adjusted_dfs['cycle_share']
r_informal_area = area_rur_adjusted_dfs['informal_share']
r_local_area = area_rur_adjusted_dfs['local_share']
r_motorway_area = area_rur_adjusted_dfs['motorway_share']
r_pedestrian_area = area_rur_adjusted_dfs['pedestrian_share']
r_primary_area = area_rur_adjusted_dfs['primary_share']
r_secondary_area = area_rur_adjusted_dfs['secondary_share']
r_tertiary_area = area_rur_adjusted_dfs['tertiary_share']  

r_cycling_obsolete = obsolete_rur_adjusted_dfs['cycle_share']
r_informal_obsolete = obsolete_rur_adjusted_dfs['informal_share']
r_local_obsolete = obsolete_rur_adjusted_dfs['local_share']
r_motorway_obsolete = obsolete_rur_adjusted_dfs['motorway_share']
r_pedestrian_obsolete = obsolete_rur_adjusted_dfs['pedestrian_share']
r_primary_obsolete = obsolete_rur_adjusted_dfs['primary_share']
r_secondary_obsolete = obsolete_rur_adjusted_dfs['secondary_share']
r_tertiary_obsolete = obsolete_rur_adjusted_dfs['tertiary_share']   
###############################################################
############################Paving shares#######################
# Create a DataFrame from the data
urban_paving_types_extend = pd.DataFrame(urban_paving_types)

# New HDI range from 0.00 to 1.00 in 0.01 increments
new_GDP = np.arange(0, 150000, 1000)

# Interpolate each column based on the new HDI range
urban_paving_types_extend = urban_paving_types_extend.reindex(new_GDP).interpolate(method="linear")
# Step 1: Define new bins and labels that match the finer HDI increments
# We'll use 0.1 intervals for the bins, but you can adjust to even smaller increments if needed
bins = np.arange(0, 150000, 1000)  # Creates bins [0.0-0.1, 0.1-0.2, ..., 0.9-1.0]
labels = [f"{round(bins[i], 1)}-{round(bins[i+1] - 1000, 2)}" for i in range(len(bins) - 1)]

# Step 2: Assign Urban_HDI_Range based on new bins and labels
urban_paving_types_extend["Urban_GDP_range"] = pd.cut(urban_paving_types_extend.index, bins=bins, labels=labels, right=False)

# Apply cummax() to all columns
sorted_gdp_urban_cap_max = sorted_gdp_urban_cap.cummax()

# Step 3: Map sorted_IMAGE_hdi HDI values to the corresponding range
# Sample sorted_IMAGE_hdi DataFrame (replace with actual data if available)
sorted_gdp_urban_cap_ranges = sorted_gdp_urban_cap.apply(lambda col: pd.cut(col, bins=bins, labels=labels, right=True))

# Step 3: Create separate DataFrames for each '_share' column in urban_road_types
paving_shares = {}
# Merge each HDI range in sorted_IMAGE_hdi_ranges with the corresponding share data in urban_road_types
for column in urban_paving_types_extend.columns:
    if 'Paved' in column:
        # Merge each share column by the HDI range
        paving_shares[column] = sorted_gdp_urban_cap_ranges.applymap(lambda x: urban_paving_types_extend.loc[urban_paving_types_extend['Urban_GDP_range'] == x, column].values[0])

# New dictionary to store the multiplied DataFrames
paving_shares_unpaved = {}

urban_paved_road_area = {}
urban_unpaved_road_area = {}

# Iterate over each DataFrame in the paving_shares dictionary
for column, share_df in paving_shares.items():
    # Compute the unpaved shares
    paving_shares_unpaved[column] = 1 - share_df
      
# Create a DataFrame from the data
rural_paving_types_extend = pd.DataFrame(rural_paving_types)

# New HDI range from 0.00 to 1.00 in 0.01 increments
new_GDP = np.arange(0, 200000, 1000)

# Interpolate each column based on the new HDI range
rural_paving_types_extend = rural_paving_types_extend.reindex(new_GDP).interpolate(method="linear")

# Step 1: Define new bins and labels that match the finer HDI increments
# We'll use 0.1 intervals for the bins, but you can adjust to even smaller increments if needed
bins = np.arange(0, 200000, 1000)  # Creates bins [0.0-0.1, 0.1-0.2, ..., 0.9-1.0]
labels = [f"{round(bins[i], 1)}-{round(bins[i+1] - 1000, 2)}" for i in range(len(bins) - 1)]

# Step 2: Assign Urban_HDI_Range based on new bins and labels
rural_paving_types_extend["Rural_GDP_range"] = pd.cut(rural_paving_types_extend.index, bins=bins, labels=labels, right=False)

# Apply cummax() to all columns
sorted_gdp_rural_cap_max = sorted_gdp_rural_cap.cummax()
# Step 3: Map sorted_IMAGE_hdi HDI values to the corresponding range
# Sample sorted_IMAGE_hdi DataFrame (replace with actual data if available)
sorted_gdp_rural_cap_ranges = sorted_gdp_rural_cap_max.apply(lambda col: pd.cut(col, bins=bins, labels=labels, right=True))

# Step 3: Create separate DataFrames for each '_share' column in urban_road_types
paving_shares_rural = {}
# Merge each HDI range in sorted_IMAGE_hdi_ranges with the corresponding share data in urban_road_types
for column in rural_paving_types_extend.columns:
    if 'Paved' in column:
        # Merge each share column by the HDI range
        paving_shares_rural[column] = sorted_gdp_rural_cap_ranges.applymap(lambda x: rural_paving_types_extend.loc[rural_paving_types_extend['Rural_GDP_range'] == x, column].values[0])
       
# New dictionary to store the multiplied DataFrames
paving_shares_rural_unpaved = {}

# Iterate over each DataFrame in the paving_shares dictionary
for column, share_df in paving_shares_rural.items():
    # Compute the unpaved shares
    paving_shares_rural_unpaved[column] = 1 - share_df

# Calculations for paved areas
u_area_paved_local = u_local_area * paving_shares["Paved share local"]
u_area_paved_cycle = u_cycling_area * paving_shares["Paved share cycle"]
u_area_paved_informal = u_informal_area * paving_shares["Paved share informal"]
u_area_paved_motorway = u_motorway_area * paving_shares["Paved share motorway"]
u_area_paved_pedestrian = u_pedestrian_area * paving_shares["Paved share pedestrian"]
u_area_paved_primary = u_primary_area * paving_shares["Paved share primary"]
u_area_paved_secondary = u_secondary_area * paving_shares["Paved share secondary"]
u_area_paved_tertiary = u_tertiary_area * paving_shares["Paved share tertiary"]

# Calculations for unpaved areas
u_area_unpaved_local = u_local_area * paving_shares_unpaved["Paved share local"]
u_area_unpaved_cycle = u_cycling_area * paving_shares_unpaved["Paved share cycle"]
u_area_unpaved_informal = u_informal_area * paving_shares_unpaved["Paved share informal"]
u_area_unpaved_motorway = u_motorway_area * paving_shares_unpaved["Paved share motorway"]
u_area_unpaved_pedestrian = u_pedestrian_area * paving_shares_unpaved["Paved share pedestrian"]
u_area_unpaved_primary = u_primary_area * paving_shares_unpaved["Paved share primary"]
u_area_unpaved_secondary = u_secondary_area * paving_shares_unpaved["Paved share secondary"]
u_area_unpaved_tertiary = u_tertiary_area * paving_shares_unpaved["Paved share tertiary"]

# Calculations for paved areas
r_area_paved_local = r_local_area * paving_shares_rural["Paved share local"]
r_area_paved_cycle = r_cycling_area * paving_shares_rural["Paved share cycle"]
r_area_paved_informal = r_informal_area * paving_shares_rural["Paved share informal"]
r_area_paved_motorway = r_motorway_area * paving_shares_rural["Paved share motorway"]
r_area_paved_pedestrian = r_pedestrian_area * paving_shares_rural["Paved share pedestrian"]
r_area_paved_primary = r_primary_area * paving_shares_rural["Paved share primary"]
r_area_paved_secondary = r_secondary_area * paving_shares_rural["Paved share secondary"]
r_area_paved_tertiary = r_tertiary_area * paving_shares_rural["Paved share tertiary"]

# Calculations for unpaved areas
r_area_unpaved_local = r_local_area * paving_shares_rural_unpaved["Paved share local"]
r_area_unpaved_cycle = r_cycling_area * paving_shares_rural_unpaved["Paved share cycle"]
r_area_unpaved_informal = r_informal_area * paving_shares_rural_unpaved["Paved share informal"]
r_area_unpaved_motorway = r_motorway_area * paving_shares_rural_unpaved["Paved share motorway"]
r_area_unpaved_pedestrian = r_pedestrian_area * paving_shares_rural_unpaved["Paved share pedestrian"]
r_area_unpaved_primary = r_primary_area * paving_shares_rural_unpaved["Paved share primary"]
r_area_unpaved_secondary = r_secondary_area * paving_shares_rural_unpaved["Paved share secondary"]
r_area_unpaved_tertiary = r_tertiary_area * paving_shares_rural_unpaved["Paved share tertiary"]

# Calculations for paved areas
u_area_paved_obsolete_local = u_local_obsolete *  paving_shares["Paved share local"]
u_area_paved_obsolete_cycle = u_cycling_obsolete *  paving_shares["Paved share cycle"]
u_area_paved_obsolete_informal = u_informal_obsolete *  paving_shares["Paved share informal"]
u_area_paved_obsolete_motorway = u_motorway_obsolete *  paving_shares["Paved share motorway"]
u_area_paved_obsolete_pedestrian = u_pedestrian_obsolete *  paving_shares["Paved share pedestrian"]
u_area_paved_obsolete_primary = u_primary_obsolete *  paving_shares["Paved share primary"]
u_area_paved_obsolete_secondary = u_secondary_obsolete *  paving_shares["Paved share secondary"]
u_area_paved_obsolete_tertiary = u_tertiary_obsolete *  paving_shares["Paved share tertiary"]

# Calculations for unpaved areas
u_area_unpaved_obsolete_local = u_local_obsolete *  paving_shares_unpaved["Paved share local"]
u_area_unpaved_obsolete_cycle = u_cycling_obsolete *  paving_shares_unpaved["Paved share cycle"]
u_area_unpaved_obsolete_informal = u_informal_obsolete *  paving_shares_unpaved["Paved share informal"]
u_area_unpaved_obsolete_motorway = u_motorway_obsolete *  paving_shares_unpaved["Paved share motorway"]
u_area_unpaved_obsolete_pedestrian = u_pedestrian_obsolete *  paving_shares_unpaved["Paved share pedestrian"]
u_area_unpaved_obsolete_primary = u_primary_obsolete *  paving_shares_unpaved["Paved share primary"]
u_area_unpaved_obsolete_secondary = u_secondary_obsolete *  paving_shares_unpaved["Paved share secondary"]
u_area_unpaved_obsolete_tertiary = u_tertiary_obsolete *  paving_shares_unpaved["Paved share tertiary"]

# Calculations for paved areas
r_area_paved_obsolete_local = r_local_obsolete *  paving_shares_rural["Paved share local"]
r_area_paved_obsolete_cycle = r_cycling_obsolete *  paving_shares_rural["Paved share cycle"]
r_area_paved_obsolete_informal = r_informal_obsolete *  paving_shares_rural["Paved share informal"]
r_area_paved_obsolete_motorway = r_motorway_obsolete *  paving_shares_rural["Paved share motorway"]
r_area_paved_obsolete_pedestrian = r_pedestrian_obsolete *  paving_shares_rural["Paved share pedestrian"]
r_area_paved_obsolete_primary = r_primary_obsolete *  paving_shares_rural["Paved share primary"]
r_area_paved_obsolete_secondary = r_secondary_obsolete *  paving_shares_rural["Paved share secondary"]
r_area_paved_obsolete_tertiary = r_tertiary_obsolete *  paving_shares_rural["Paved share tertiary"]

# Calculations for unpaved areas
r_area_unpaved_obsolete_local = r_local_obsolete *  paving_shares_rural_unpaved["Paved share local"]
r_area_unpaved_obsolete_cycle = r_cycling_obsolete *  paving_shares_rural_unpaved["Paved share cycle"]
r_area_unpaved_obsolete_informal = r_informal_obsolete *  paving_shares_rural_unpaved["Paved share informal"]
r_area_unpaved_obsolete_motorway = r_motorway_obsolete *  paving_shares_rural_unpaved["Paved share motorway"]
r_area_unpaved_obsolete_pedestrian = r_pedestrian_obsolete *  paving_shares_rural_unpaved["Paved share pedestrian"]
r_area_unpaved_obsolete_primary = r_primary_obsolete *  paving_shares_rural_unpaved["Paved share primary"]
r_area_unpaved_obsolete_secondary = r_secondary_obsolete *  paving_shares_rural_unpaved["Paved share secondary"]
r_area_unpaved_obsolete_tertiary = r_tertiary_obsolete *  paving_shares_rural_unpaved["Paved share tertiary"]

# List of DataFrames to smooth
dataframes = [
    # Urban - Paved
    u_area_paved_local, u_area_paved_cycle, u_area_paved_informal,
    u_area_paved_motorway, u_area_paved_pedestrian, u_area_paved_primary,
    u_area_paved_secondary, u_area_paved_tertiary,

    # Urban - Unpaved
    u_area_unpaved_local, u_area_unpaved_cycle, u_area_unpaved_informal,
    u_area_unpaved_motorway, u_area_unpaved_pedestrian, u_area_unpaved_primary,
    u_area_unpaved_secondary, u_area_unpaved_tertiary,

    # Rural - Paved
    r_area_paved_local, r_area_paved_cycle, r_area_paved_informal,
    r_area_paved_motorway, r_area_paved_pedestrian, r_area_paved_primary,
    r_area_paved_secondary, r_area_paved_tertiary,

    # Rural - Unpaved
    r_area_unpaved_local, r_area_unpaved_cycle, r_area_unpaved_informal,
    r_area_unpaved_motorway, r_area_unpaved_pedestrian, r_area_unpaved_primary,
    r_area_unpaved_secondary, r_area_unpaved_tertiary,

    # Urban - Paved (Obsolete)
    u_area_paved_obsolete_local, u_area_paved_obsolete_cycle, u_area_paved_obsolete_informal,
    u_area_paved_obsolete_motorway, u_area_paved_obsolete_pedestrian, u_area_paved_obsolete_primary,
    u_area_paved_obsolete_secondary, u_area_paved_obsolete_tertiary,

    # Urban - Unpaved (Obsolete)
    u_area_unpaved_obsolete_local, u_area_unpaved_obsolete_cycle, u_area_unpaved_obsolete_informal,
    u_area_unpaved_obsolete_motorway, u_area_unpaved_obsolete_pedestrian, u_area_unpaved_obsolete_primary,
    u_area_unpaved_obsolete_secondary, u_area_unpaved_obsolete_tertiary,

    # Rural - Paved (Obsolete)
    r_area_paved_obsolete_local, r_area_paved_obsolete_cycle, r_area_paved_obsolete_informal,
    r_area_paved_obsolete_motorway, r_area_paved_obsolete_pedestrian, r_area_paved_obsolete_primary,
    r_area_paved_obsolete_secondary, r_area_paved_obsolete_tertiary,

    # Rural - Unpaved (Obsolete)
    r_area_unpaved_obsolete_local, r_area_unpaved_obsolete_cycle, r_area_unpaved_obsolete_informal,
    r_area_unpaved_obsolete_motorway, r_area_unpaved_obsolete_pedestrian, r_area_unpaved_obsolete_primary,
    r_area_unpaved_obsolete_secondary, r_area_unpaved_obsolete_tertiary
]

# Apply smoothing to each DataFrame
for df in dataframes:
    for col in df.columns:
        original_data = df[col].copy()
        df[col] = df[col].rolling(window=5, center=True).mean()
        # Fill NaN values at the edges with the original data
        df[col] = df[col].fillna(original_data)

##################RAIL - URban rail###################
def urban_rail(hdi, urban_pop_density):
    # Calculate y based on the provided formula
    y = 2.718**-2.048095 * (sorted_IMAGE_hdi ** 8.0777) / (sorted_urban_pop_density ** 0.5885)
    return y

urban_rail_cap = urban_rail(sorted_IMAGE_hdi,sorted_urban_pop_density)

actual_urban_rail_2024 = data_2024.loc['urban_rail']  # Insert actual value for rural area in 2024

predicted_urban_rail_cap_2024 = urban_rail_cap.loc[2024]
predicted_urban_rail_length_2024 = predicted_urban_rail_cap_2024 * (sorted_urban_population_2024 * 1000000)

# Calculate adjustment factor based on 2024 actual vs. predicted
urban_rail_adjustment_factor = actual_urban_rail_2024 / predicted_urban_rail_length_2024

urban_rail_length = urban_rail_cap * (sorted_urban_population*1000000) * urban_rail_adjustment_factor

##########RAIL THROUGH CARRYING CAPCITY#############


##########RAIL THROUGH CARRYING CAPCITY#############
##########RAIL THROUGH DIRECT VKM LINKAGE#############
##########RAIL THROUGH DIRECT VKM LINKAGE & DYNAMIC ELASTICITY#############
import numpy as np
import pandas as pd

# Make row 2020 the same as row 2019
sorted_vkm_rail_tot.loc[2020] = sorted_vkm_rail_tot.loc[2019]
# Make row 2021 the same as row 2022
sorted_vkm_rail_tot.loc[2021] = sorted_vkm_rail_tot.loc[2022]

sorted_vkm_HST_rail_tot.loc[2020] = sorted_vkm_HST_rail_tot.loc[2019]
# Make row 2021 the same as row 2022
sorted_vkm_HST_rail_tot.loc[2021] = sorted_vkm_HST_rail_tot.loc[2022]

# Ensure VKM is transposed so regions are index and years are columns
vkm_data = sorted_vkm_rail_tot.T
vkm_data.columns = vkm_data.columns.astype(str)

# --- 1. System Parameters ---
MAX_ANNUAL_CHANGE_PERCENT = 0.03
LOOKAHEAD_YEARS = 10
DECLINE_LAG = 5
DECLINE_WINDOW = 10

# Dynamic Elasticity Parameters
MIN_ELASTICITY = 0.1  # Mature / Stagnant phase
MAX_ELASTICITY = 0.6  # Rapid Expansion phase
GROWTH_MIDPOINT = 0.03 # 3% annual growth triggers the 50/50 transition
STEEPNESS = 200 # How suddenly the network switches from mature to expanding

def get_dynamic_elasticity(growth_rate):
    """
    Returns an elasticity value between 0.1 and 0.9 based on the anticipated growth rate.
    Uses a smooth S-curve (Logistic function).
    """
    return MIN_ELASTICITY + (MAX_ELASTICITY - MIN_ELASTICITY) / (1 + np.exp(-STEEPNESS * (growth_rate - GROWTH_MIDPOINT)))

#DETERMINE SHARES OF HIGH SPEED RAIL
HST_share = sorted_vkm_HST_rail_tot / sorted_vkm_rail_tot
HST_share = HST_share.fillna(0)

rail_lengths_df.columns = rail_lengths_df.columns.astype(str)

# Initialize the rollover backlog (tracks unmet construction demand)
backlog = pd.Series(0.0, index=rail_lengths_df.index)


# --- 2. FORECASTING (2024 -> 2100) WITH ANTICIPATION & ROLLOVER ---
print("Forecasting 2024-2100 with dynamic elasticity, rollovers, and sticky decline...")

# Change range from 2100 to 2101 so the loop processes the year 2100
for year in range(2024, 2101): 
    year_str = str(year)
    prev_year_str = str(year - 1)
    L_prev = rail_lengths_df[prev_year_str]
    
    # --- A. GROWTH MECHANIC (Lookahead) ---
    future_year = min(year + LOOKAHEAD_YEARS, 2100)
    years_ahead = max(1, future_year - year)
    
    # If your vkm_data ONLY goes to 2099, this will throw a KeyError for 2100.
    # Assuming VKM data reaches 2100:
    VKM_curr = vkm_data[year_str]
    VKM_future = vkm_data[str(future_year)]
    
    # Calculate anticipated annualized growth over the next decade
    vkm_growth_rate = ((VKM_future - VKM_curr) / VKM_curr.replace(0, np.nan)) / years_ahead
    vkm_growth_rate = vkm_growth_rate.fillna(0)
    
    # Determine the regional elasticity based on the future phase of growth
    current_elasticity = get_dynamic_elasticity(vkm_growth_rate)
    
    # Isolate positive growth
    active_growth = np.clip(vkm_growth_rate, 0, None)
    
    # If demand is flat/declining, cancel unbuilt historical projects
    backlog = np.where(active_growth > 0, backlog, 0)
    backlog = pd.Series(backlog, index=L_prev.index)
    
    # Calculate total desired track using our dynamic elasticity
    desired_new_track = (L_prev * active_growth * current_elasticity) + backlog
    
    # Calculate Cap and Actual Build
    max_build_allowed = L_prev * MAX_ANNUAL_CHANGE_PERCENT
    actual_build = np.minimum(desired_new_track, max_build_allowed)
    
    # Save unmet demand into the backlog for next year
    backlog = desired_new_track - actual_build
    

    # --- B. DECLINE MECHANIC (Lookback) ---
    min_vkm_year = int(vkm_data.columns.min())
    past_lag_year = max(year - DECLINE_LAG, min_vkm_year)
    past_window_start = max(past_lag_year - DECLINE_WINDOW, min_vkm_year)
    
    VKM_past_lag = vkm_data[str(past_lag_year)]
    VKM_past_start = vkm_data[str(past_window_start)]
    
    # Did VKM experience a net drop over the 10 year window leading up to 5 years ago?
    is_sustained_decline = VKM_past_lag < VKM_past_start
    
    years_in_decline = past_lag_year - past_window_start
    if years_in_decline > 0:
        decline_rate = (VKM_past_lag - VKM_past_start) / VKM_past_start.replace(0, np.nan) / years_in_decline
    else:
        decline_rate = pd.Series(0, index=VKM_past_lag.index)
    decline_rate = decline_rate.fillna(0)
    
    # Decline only happens if we are NOT currently growing, AND the past shows sustained drop
    # We use MIN_ELASTICITY (0.1) for decline because dismantling infrastructure is sticky
    active_decline = np.where((active_growth == 0) & is_sustained_decline, decline_rate, 0)
    actual_decline_amount = L_prev * active_decline * MIN_ELASTICITY
    
    # Cap the decline speed for safety
    min_decline_allowed = -L_prev * MAX_ANNUAL_CHANGE_PERCENT
    actual_decline_capped = np.maximum(actual_decline_amount, min_decline_allowed)
    
    # --- C. FINAL UPDATE ---
    rail_lengths_df[year_str] = L_prev + actual_build + actual_decline_capped


# --- 3. HINDCASTING (1995 -> 1971) ---
print("Hindcasting 1995-1971 using historical dynamic elasticity...")
# Safely stop the loop so it calculates 1971 as prev_year_str
for year in range(1995, 1971, -1):
    year_str = str(year)
    prev_year_str = str(year - 1)
    
    L_t = rail_lengths_df[year_str]
    VKM_t = vkm_data[year_str]
    VKM_prev = vkm_data[prev_year_str]
    
    # Historical Growth Signal (Year-over-Year)
    vkm_growth_rate = (VKM_t - VKM_prev) / VKM_prev.replace(0, np.nan)
    vkm_growth_rate = vkm_growth_rate.fillna(0)
    
    # Calculate what the elasticity would have been based on that growth
    historical_elasticity = get_dynamic_elasticity(vkm_growth_rate)
    
    # Reverse calculate
    combined_rate = vkm_growth_rate * historical_elasticity
    L_prev_calculated = L_t / (1 + combined_rate)
    
    growth_amount = L_t - L_prev_calculated
    
    # Cap
    max_change_allowed = L_t * MAX_ANNUAL_CHANGE_PERCENT
    capped_growth = np.clip(growth_amount, -max_change_allowed, max_change_allowed)

    rail_lengths_df[prev_year_str] = L_t - capped_growth

# --- MISSING YEAR FIX: Extrapolate 1970 to match dataframe sizes ---
# Since we lack VKM data for 1970 to run the loop, we use the physical 
# growth rate between 1971 and 1972 as a proxy to step back one more year.
L_1972 = rail_lengths_df['1972']
L_1971 = rail_lengths_df['1971']

# Calculate the physical rail growth rate that occurred from 1971 to 1972
rail_growth_71_72 = (L_1972 - L_1971) / L_1971.replace(0, np.nan)
rail_growth_71_72 = rail_growth_71_72.fillna(0)

# Reverse calculate 1970 and append it to the DataFrame
rail_lengths_df['1970'] = L_1971 / (1 + rail_growth_71_72)


# --- 4. Final Result Processing ---
final_columns = sorted(rail_lengths_df.columns, key=int)
final_df = rail_lengths_df[final_columns]

rail_length = final_df.T
rail_length = rail_length.iloc[1:]
rail_length.index = pd.Index(pd.to_numeric(rail_length.index, errors='coerce').astype('Int64'))

#BACKCASTING HST FOR A NUMBER OF REGIONS DUE TO EVIDENCE OF HSR BEFORE IMAGE DATA
def backcast_linear(df, region, anchor_year, start_year=None, min_value=0.01):
    out = df.sort_index().copy()
    if isinstance(out.index, pd.DatetimeIndex):
        out.index = out.index.year
    if anchor_year not in out.index:
        raise KeyError(f"{anchor_year} not in index for {region}")
    if start_year is None:
        start_year = out.index.min()
    y = out[region].astype(float)
    post_mask = (out.index >= anchor_year) & y.notna()
    x_post = out.index[post_mask].values.astype(float)
    y_post = y[post_mask].values
    if len(x_post) < 2:
        nxt = y.loc[y.index > anchor_year].dropna()
        if len(nxt) == 0:
            raise ValueError(f"Need at least one value after {anchor_year} to estimate slope for {region}")
        x_post = np.array([anchor_year, float(nxt.index[0])])
        y_post = np.array([y.loc[anchor_year], float(nxt.iloc[0])])
    slope, intercept = np.polyfit(x_post, y_post, 1)
    anchor_val = y.loc[anchor_year]
    pre_years = out.index[(out.index < anchor_year) & (out.index >= start_year)].astype(float)
    if pre_years.size:
        backcast_vals = anchor_val - slope * (anchor_year - pre_years)
        backcast_vals = np.clip(backcast_vals, min_value, None)
        out.loc[pre_years.astype(int), region] = backcast_vals
    return out

# Backcast Japan from 2010 backwards to earliest year
HST_share_adj = backcast_linear(HST_share, region='Japan', anchor_year=2010)

# Backcast Germany from 2015 backwards, but only starting at 2000
HST_share_adj = backcast_linear(HST_share_adj, region='Western Europe', anchor_year=2010, start_year=1980)
HST_share_adj = backcast_linear(HST_share_adj, region='Korea', anchor_year=2010, start_year=1990)
HST_share_adj = backcast_linear(HST_share_adj, region='China', anchor_year=2010, start_year=2000)

HST_length = rail_length * HST_share_adj
# Apply a 5-year rolling mean to smooth the data
HST_length = HST_length.rolling(window=10, min_periods=3, center=True).mean()

###WORK IN PROGRESS OBSOLETE FUNCTIONS####

RAIL_DECLINE_FUNCTION = 1
if RAIL_DECLINE_FUNCTION == 1:
    adjusted_rail_length = rail_length.copy()
    
    for column in rail_length.columns:
        for i in range(len(rail_length) - 1): 
            if i + 1 < len(rail_length): 
                current_value = adjusted_rail_length.iloc[i, adjusted_rail_length.columns.get_loc(column)]
                future_value = adjusted_rail_length.iloc[i + 1, adjusted_rail_length.columns.get_loc(column)]
                
                if future_value < current_value:
                    next_year_value = adjusted_rail_length.iloc[i + 1, adjusted_rail_length.columns.get_loc(column)]
                    decline_amount = (current_value - next_year_value)
                    adjusted_rail_length.iloc[i + 1 , adjusted_rail_length.columns.get_loc(column)] = current_value - decline_amount * 0

    print("Road decline function is active. Roads will have a lagging effect to declining inputs.")

    obsolete_rail_length = adjusted_rail_length - rail_length
    
    adjusted_HST_length = HST_length.copy()
    
    for column in rail_length.columns:
        for i in range(len(rail_length) - 1): 
            if i + 1 < len(rail_length): 
                current_value = adjusted_HST_length.iloc[i, adjusted_HST_length.columns.get_loc(column)]
                future_value = adjusted_HST_length.iloc[i + 1, adjusted_HST_length.columns.get_loc(column)]
                
                if future_value < current_value:
                    next_year_value = adjusted_HST_length.iloc[i + 1, adjusted_HST_length.columns.get_loc(column)]
                    decline_amount = (current_value - next_year_value)
                    adjusted_HST_length.iloc[i + 1 , adjusted_HST_length.columns.get_loc(column)] = current_value - decline_amount * 0

    def apply_obsolete_rule(series):
        processed_series = series.copy()
        n = len(series)
        last_non_zero_block_start_idx = -1
        for i in range(n - 1, -1, -1):
            if series.iloc[i] != 0:
                last_non_zero_block_start_idx = i
            else:
                if last_non_zero_block_start_idx != -1:
                    break 
        
        if last_non_zero_block_start_idx == -1: 
            if (series == 0).all():
                return series 
            else: 
                return series 
    
        zero_to_zero_out_up_to_idx = -1
        for i in range(last_non_zero_block_start_idx - 1, -1, -1):
            if series.iloc[i] == 0:
                zero_to_zero_out_up_to_idx = i
                break 
                
        if zero_to_zero_out_up_to_idx != -1:
            processed_series.iloc[:zero_to_zero_out_up_to_idx + 1] = 0
        
        return processed_series
    
    def process_obsolete_dataframes():
        global_vars = globals()
        
        for var_name, var_value in global_vars.items():
            if isinstance(var_value, pd.DataFrame) and 'obsolete_' in var_name:
                print(f"Processing DataFrame: {var_name}")
                for col in var_value.columns:
                    print(f"  Applying rule vertically to column: '{col}'")
                    var_value[col] = apply_obsolete_rule(var_value[col])
                print(f"Finished processing {var_name}.\n")
    
    process_obsolete_dataframes()

    def fix_obsolete_columns(df):
        for col in df.columns:
            if df[col].iloc[-1] == 0 and not (df[col] == 0).all():
                df[col] = 0
        
    fix_obsolete_columns(obsolete_rail_length)

    adjusted_rail_length = adjusted_rail_length - obsolete_rail_length
 
else:
    print("Road decline function is inactive. Roads follow normal logic.")
    adjusted_rail_length = rail_length


bridge_rail_2024 = data_2024.loc['rail_bridge_share'] 
tunnel_rail_2024 = data_2024.loc['rail_tunnel_share'] 

rail_length = adjusted_rail_length
rail_obsolete = obsolete_rail_length

HST_length = adjusted_HST_length

bridge_rail = adjusted_rail_length * bridge_rail_2024
tunnel_rail = adjusted_rail_length * tunnel_rail_2024

bridge_rail_obsolete = obsolete_rail_length * bridge_rail_2024
tunnel_rail_obsolete = obsolete_rail_length * tunnel_rail_2024
#####################Bridges & Tunnels -- Roads ###########################
actual_rural_bridges_2024 = data_2024.loc['rural_bridges']  # Insert actual value for rail length in 2024
actual_rural_tunnels_2024 = data_2024.loc['rural_tunnels']  # Insert actual value for rail length in 2024
actual_urban_bridges_2024 = data_2024.loc['urban_bridges']  # Insert actual value for rail length in 2024
actual_urban_tunnels_2024 = data_2024.loc['urban_tunnels']  # Insert actual value for rail length in 2024


#########Ratio of bridges and tunnels to current trends of road area
actual_rur_brid_ratio = actual_rural_bridges_2024 / actual_rural_road_area_2024
actual_rur_tun_ratio = actual_rural_tunnels_2024 / actual_rural_road_area_2024
actual_urb_brid_ratio = actual_urban_bridges_2024 / actual_urban_road_area_2024
actual_urb_tun_ratio = actual_urban_tunnels_2024 / actual_urban_road_area_2024

rural_road_bridges = adjusted_rural_road_area * actual_rur_brid_ratio
rural_road_tunnels = adjusted_rural_road_area * actual_rur_tun_ratio
urban_road_bridges = adjusted_urban_road_area * actual_urb_brid_ratio
urban_road_tunnels = adjusted_urban_road_area * actual_urb_tun_ratio

obsolete_road_bridges_rural = obsolete_road_rural_area * actual_rur_brid_ratio
obsolete_road_tunnels_rural = obsolete_road_rural_area * actual_rur_tun_ratio
obsolete_road_bridges_urban = obsolete_road_urban_area * actual_urb_brid_ratio
obsolete_road_tunnels_urban = obsolete_road_urban_area * actual_urb_tun_ratio

# Extract sheets into separate variables
rur_brid_share = sheets['rur-bridge-share']  # Replace 'Sheet1' with actual sheet names
rur_tun_share = sheets['rur-tunnel-share']
urb_brid_share = sheets['urb-bridge-share']
urb_tun_share = sheets['urb-tunnel-share']

# Define the mapping of old index names to new index names
index_mapping = {
    'China region': 'China',
    'Indonesia region': 'Indonesia Region',
    'Southeastern Asia': 'South Eastern Asia',
    'Korea region': 'Korea',
    'Russia region': 'Russia Region'
}

# For China the share of motorway for rural motorway tunnels and bridges are outlying. Here we used the shares from the Korea region
# This is done because China is also the region with the most missing data, so it is likely that motorway are overrepresented
# Rename the indices in each dataframe
rur_brid_share = rur_brid_share.rename(index=index_mapping)
rur_tun_share = rur_tun_share.rename(index=index_mapping)
urb_brid_share = urb_brid_share.rename(index=index_mapping)
urb_tun_share = urb_tun_share.rename(index=index_mapping)

# Define empty dictionary to store the resulting DataFrames
resulting_dataframes = {}

# Process rural bridges with rural bridge shares
for bridge_type in rur_brid_share.columns:
    # Extract the multiplier for the current bridge type, making sure it's a DataFrame
    # and aligning its regions (index) with the columns in `rural_road_bridges`
    multiplier = rur_brid_share[[bridge_type]].transpose()
    
    # Ensure `multiplier` has regions as columns to match `rural_road_bridges`
    multiplier.columns = rur_brid_share.index  # Set the columns to region names

    # Perform element-wise multiplication with alignment on the region columns
    resulting_dataframes[f"rural_road_bridges_{bridge_type}"] = rural_road_bridges.mul(multiplier.iloc[0], axis=1)

# Process rural tunnels with rural tunnel shares
for tunnel_type in rur_tun_share.columns:
    multiplier = rur_tun_share[[tunnel_type]].transpose()
    multiplier.columns = rur_tun_share.index
    resulting_dataframes[f"rural_road_tunnels_{tunnel_type}"] = rural_road_tunnels.mul(multiplier.iloc[0], axis=1)

# Process urban bridges with urban bridge shares
for bridge_type in urb_brid_share.columns:
    multiplier = urb_brid_share[[bridge_type]].transpose()
    multiplier.columns = urb_brid_share.index
    resulting_dataframes[f"urban_road_bridges_{bridge_type}"] = urban_road_bridges.mul(multiplier.iloc[0], axis=1)

# Process urban tunnels with urban tunnel shares
for tunnel_type in urb_tun_share.columns:
    multiplier = urb_tun_share[[tunnel_type]].transpose()
    multiplier.columns = urb_tun_share.index
    resulting_dataframes[f"urban_road_tunnels_{tunnel_type}"] = urban_road_tunnels.mul(multiplier.iloc[0], axis=1)

# Process rural bridges with rural bridge shares
for bridge_type in rur_brid_share.columns:
    # Extract the multiplier for the current bridge type, making sure it's a DataFrame
    # and aligning its regions (index) with the columns in `rural_road_bridges`
    multiplier = rur_brid_share[[bridge_type]].transpose()
    
    # Ensure `multiplier` has regions as columns to match `rural_road_bridges`
    multiplier.columns = rur_brid_share.index  # Set the columns to region names

    # Perform element-wise multiplication with alignment on the region columns
    resulting_dataframes[f"obsolete_rural_road_bridges_{bridge_type}"] = obsolete_road_bridges_rural.mul(multiplier.iloc[0], axis=1)

# Process rural tunnels with rural tunnel shares
for tunnel_type in rur_tun_share.columns:
    multiplier = rur_tun_share[[tunnel_type]].transpose()
    multiplier.columns = rur_tun_share.index
    resulting_dataframes[f"obsolete_rural_road_tunnels_{tunnel_type}"] = obsolete_road_tunnels_rural.mul(multiplier.iloc[0], axis=1)

# Process urban bridges with urban bridge shares
for bridge_type in urb_brid_share.columns:
    multiplier = urb_brid_share[[bridge_type]].transpose()
    multiplier.columns = urb_brid_share.index
    resulting_dataframes[f"obsolete_urban_road_bridges_{bridge_type}"] = obsolete_road_bridges_urban.mul(multiplier.iloc[0], axis=1)

# Process urban tunnels with urban tunnel shares
for tunnel_type in urb_tun_share.columns:
    multiplier = urb_tun_share[[tunnel_type]].transpose()
    multiplier.columns = urb_tun_share.index
    resulting_dataframes[f"obsolete_urban_road_tunnels_{tunnel_type}"] = obsolete_road_tunnels_urban.mul(multiplier.iloc[0], axis=1)

# At this point, `resulting_dataframes` contains the 32 new DataFrames.
###############MATERIALS-Intensity calculation####################

# Create a function to multiply an area DataFrame by a material DataFrame
def multiply_area_with_material(area_df, material_df):
    # Initialize an empty list to store the resulting DataFrames
    result_frames = []
    
    # Perform the row-wise multiplication with broadcasting
    for index_value, m_row in material_df.iterrows():
        # Multiply all rows of area_df by the current row of material_df
        multiplied_rows = area_df.copy()
        
        # Multiply the numeric columns of area_df with the material row
        for col in multiplied_rows.select_dtypes(include=['number']).columns:
            multiplied_rows[col] *= m_row[col]  # Use col directly since it matches

        # Add the current index value of material_df as a new column
        multiplied_rows['material_index'] = index_value
        
        # Append to the list of results
        result_frames.append(multiplied_rows)

    # Concatenate all the DataFrames in the list into a single DataFrame
    result = pd.concat(result_frames, ignore_index=True)

    # Reset index to ensure a clean index for the resulting DataFrame
    result.reset_index(drop=True, inplace=True)

    # Set the index of the result DataFrame to match the original area_df index
    original_index = area_df.index.tolist() * material_df.shape[0]  # Repeat the index for the number of material rows
    result.index = original_index

    # Rename the last column to 'material'
    result.rename(columns={result.columns[-1]: 'material'}, inplace=True)    
    return result

# Define mappings of area DataFrames to material DataFrames
area_material_mapping = {
    'u_area_paved_local': m_loc_urb_pav,
    'u_area_paved_cycle': m_cyc_urb_pav,
    'u_area_paved_informal': m_inf_urb_pav,
    'u_area_paved_motorway': m_mot_urb_pav,
    'u_area_paved_pedestrian': m_ped_urb_pav,
    'u_area_paved_primary': m_pri_urb_pav,
    'u_area_paved_secondary': m_sec_urb_pav,
    'u_area_paved_tertiary': m_ter_urb_pav,
    
    'u_area_unpaved_local': m_loc_urb_unp,
    'u_area_unpaved_cycle': m_cyc_urb_unp,
    'u_area_unpaved_informal': m_inf_urb_unp,
    'u_area_unpaved_pedestrian': m_ped_urb_unp,
    'u_area_unpaved_primary': m_pri_urb_unp,
    'u_area_unpaved_secondary': m_sec_urb_unp,
    'u_area_unpaved_tertiary': m_ter_urb_unp,

    'r_area_paved_local': m_loc_rur_pav,
    'r_area_paved_cycle': m_cyc_rur_pav,
    'r_area_paved_informal': m_inf_rur_pav,
    'r_area_paved_motorway': m_mot_rur_pav,
    'r_area_paved_pedestrian': m_ped_rur_pav,
    'r_area_paved_primary': m_pri_rur_pav,
    'r_area_paved_secondary': m_sec_rur_pav,
    'r_area_paved_tertiary': m_ter_rur_pav,    

    'r_area_unpaved_local': m_loc_rur_unp,
    'r_area_unpaved_cycle': m_cyc_rur_unp,
    'r_area_unpaved_informal': m_inf_rur_unp,
    'r_area_unpaved_pedestrian': m_ped_rur_unp,
    'r_area_unpaved_primary': m_pri_rur_unp,
    'r_area_unpaved_secondary': m_sec_rur_unp,
    'r_area_unpaved_tertiary': m_ter_rur_unp,
}


# Store the results for each area DataFrame
final_results = {}

# Iterate through the area-material mappings
for area_name, material_df in area_material_mapping.items():
    # Get the DataFrame by the area name dynamically (ensure they exist)
    area_df = globals()[area_name]  # This assumes you have variables with these names
    # Multiply and store the result
    final_results[area_name] = multiply_area_with_material(area_df, material_df)

###############CALCULATING MATERIAL INTENSITIES##################################
from collections import defaultdict

# Dictionary to hold DataFrames categorized by material
material_dict = defaultdict(dict)
# Loop through each DataFrame in the original dictionary
for name, final_results in final_results.items():
    # Get unique material values in the current DataFrame
    unique_materials = final_results['material'].unique()
  
    # For each unique material, filter rows with that material and store in the respective dictionary
    for material in unique_materials:
        # Filter rows by material and drop the 'material' column
        filtered_df = final_results[final_results['material'] == material].drop(columns=['material']).copy()
        material_dict[material][name] = filtered_df
        
# material_dict now contains dictionaries for each unique material, 
# where each dictionary has DataFrames containing only that material.
# Define road class mapping based on key names
road_class_map = {
    'motorway': 1,
    'primary': 2,
    'secondary': 3,
    'tertiary': 4,
    'cycle': 5,
    'informal': 5,
    'local': 5,
    'pedestrian': 5
}

#################TRIAL and ERROR#######################
# Define a dictionary to map each material to the result dictionary keys
# Dictionary to hold the final results
result_dict = {}

# Define the materials and their corresponding dictionaries
materials = [
    'asphalt', 'brick', 'concrete', 'metal', 'stone', 'wood'
]
# Define a dictionary to map each material to the result dictionary keys
material_result_keys = {
    'asphalt': 'asphalt_multiplied',
    'brick': 'brick_multiplied',
    'concrete': 'concrete_multiplied',
    'metal': 'metal_multiplied',
    'stone': 'stone_multiplied',
    'wood': 'wood_multiplied'
}
# Initialize result dictionaries for each material
for material in materials:
    result_dict[material_result_keys[material]] = {}

# Access the "pavingstone" dictionary within material_dict for the special calculations
pavingstone_dict = material_dict.get('pavingstone', {})

# Define the factors for pavingstone contributions to each material
#Factors taken from TRIPI (2024)
pavingstone_factors = {
    'brick': 0.0381,
    'stone': 0.2705,
    'concrete': 0.6913
}

# Process each material normally
for material in materials:
    material_calc = material_dict.get(material, {})
    # Initialize a sub-dictionary for the current material's multiplied results
    multiplied_dict = result_dict[material_result_keys[material]]

    # Loop through each key (road class) in the current material dictionary
    for full_key, material_soc in material_calc.items():
        # Extract the relevant part (road class) from the key name
        road_class_key = full_key.split('_')[-1]  # Get the last part of the key
        
        # Determine the road class from the extracted part using the road_class_map
        road_class = road_class_map.get(road_class_key)
        
        if road_class is None:
            print(f"No matching road class found for key '{full_key}'. Skipping.")
            continue
        
        # Check if road_class exists in the index of mat_int
        if road_class not in mat_int.index:
            print(f"No data found in mat_int for road class {road_class}.")
            continue
        
        # Retrieve the row corresponding to road_class from the index
        mat_int_row = mat_int.loc[road_class]
        
        # Filter the row to keep only the values related to the current material
        if 'material' in mat_int.columns:
            filtered_row = mat_int_row[mat_int_row['material'] == material]
        else:
            print("Material column not found in mat_int_row.")
            continue
        
        # Check if we still have rows after filtering
        if filtered_row.empty:
            print(f"No {material} material found for road class {road_class}. Skipping.")
            continue
        
        # Drop the 'material' column from the filtered row
        filtered_row = filtered_row.drop(columns='material')  # Drop 'material' and keep the rest
        
        # Get the values for multiplication, filling any remaining NaN values with 0
        mat_int_row_values = filtered_row.values

        # Multiply each column in `material_df` by the values in `filtered_row`
        multiplied_df = material_soc.multiply(mat_int_row_values, axis=1)
        
        # Store the multiplied DataFrame in the dictionary for the current material
        multiplied_dict[full_key] = multiplied_df

# Special handling for pavingstone and adding its contributions to brick, stone, and concrete
for full_key, pavingstone_df in pavingstone_dict.items():
    # Extract the road class key
    road_class_key = full_key.split('_')[-1]
    road_class = road_class_map.get(road_class_key)

    if road_class is None or road_class not in mat_int.index:
        print(f"No matching road class found for key '{full_key}' or road class missing in mat_int. Skipping.")
        continue

    # Loop through each material and its factor for the pavingstone calculations
    for material, factor in pavingstone_factors.items():
        # Get the dictionary where this result should be added
        material_result_key = material_result_keys[material]
        material_multiplied_dict = result_dict[material_result_key]

        # Multiply the initial pavingstone values by the factor
        pavingstone_initial_multiplied_df = pavingstone_df * factor
        
        # Retrieve the row from mat_int for the specified material and road class
        material_row = mat_int.loc[road_class]
        material_filtered_row = material_row[material_row['material'] == material] if 'material' in mat_int.columns else None

        if material_filtered_row is None or material_filtered_row.empty:
            print(f"No {material} material found for road class {road_class} in pavingstone calculation. Filling with zeros.")
            material_values = pd.DataFrame(0, index=[road_class], columns=mat_int.columns.drop('material')).values
        else:
            material_values = material_filtered_row.drop(columns='material').fillna(0).values

        # Multiply the pavingstone initial result by the material values
        pavingstone_contribution_df = pavingstone_initial_multiplied_df.multiply(material_values, axis=1)
        
        # Add to the existing results in the dictionary
        if full_key in material_multiplied_dict:
            material_multiplied_dict[full_key] += pavingstone_contribution_df
        else:
            material_multiplied_dict[full_key] = pavingstone_contribution_df

# Dictionary to hold the final results
result_dict_aggregate = {}

# Define the materials and their corresponding dictionaries
materials = [
    'asphalt', 'brick', 'concrete', 'metal', 'stone', 'wood'
]

# Define a dictionary to map each material to the result dictionary keys
material_result_keys = {
    'asphalt': 'asphalt_multiplied',
    'brick': 'bricks_multiplied',
    'concrete': 'concrete_multiplied',
    'metal': 'metal_multiplied',
    'stone': 'stone_multiplied',
    'wood': 'wood_multiplied'
}

# Initialize result dictionaries for each material
for material in materials:
    result_dict_aggregate[material_result_keys[material]] = {}

# Define aggregate materials and initialize their result storage
aggregate_materials = {
    'asphalt': 'aggregate_asphalt',
    'brick': 'aggregate_bricks',
    'concrete': 'aggregate_concrete',
    'pavingstone': 'aggregate_pavingstone',
    'stone': 'aggregate_stone'
}

# Initialize an empty dictionary to store aggregate results
aggregate_results_dict = {}

# Loop through each aggregate material and perform the multiplication
for aggregate_material, result_key in aggregate_materials.items():
    material_calc = material_dict.get(aggregate_material, {})
    
    # Initialize a sub-dictionary for the current aggregate material's multiplied results
    aggregate_results_dict[result_key] = {}

    # Loop through each key (road class) in the current aggregate material dictionary
    for full_key, material_agg in material_calc.items():
        # Extract the relevant part (road class) from the key name
        road_class_key = full_key.split('_')[-1]
        road_class = road_class_map.get(road_class_key)
        
        if road_class is None or road_class not in mat_int.index:
            print(f"No matching road class found for key '{full_key}' or road class missing in mat_int. Skipping.")
            continue
        
        # Retrieve the row from mat_int for the current aggregate material
        mat_int_row = mat_int.loc[road_class]
        
        # Filter for the aggregate material's row in mat_int
        if 'material' in mat_int.columns:
            aggregate_row = mat_int_row[mat_int_row['material'] == "aggregate"]
        else:
            print("Material column not found in mat_int_row.")
            continue
        
        # Ensure we have rows after filtering; if empty, fill with zeros
        if aggregate_row.empty:
            print(f"No {aggregate_material} material found for road class {road_class}. Filling with zeros.")
            aggregate_values = pd.DataFrame(0, index=[road_class], columns=mat_int.columns.drop('material')).values
        else:
            aggregate_values = aggregate_row.drop(columns='material').fillna(0).values

        # Perform the multiplication for each column in `material_df`
        multiplied_df = material_agg.multiply(aggregate_values, axis=1)
        
        # Store the multiplied DataFrame in the dictionary for the aggregate material
        aggregate_results_dict[result_key][full_key] = multiplied_df

        # Initialize a dictionary to store the summed DataFrames for each `full_key`
        summed_aggregate_results = {}
        
        # Find all unique `full_key` names across all dictionaries
        all_full_keys = set(key for d in aggregate_results_dict.values() for key in d.keys())
        
        # Iterate over each unique `full_key`
        for full_key in all_full_keys:
            # Start with an empty DataFrame initialized to zeros. 
            # We will adjust the index and columns once we find the first occurrence of `full_key`.
            summed_df = None
            
            # Sum each DataFrame corresponding to the current `full_key` across all dictionaries
            for result_key, nested_dict in aggregate_results_dict.items():
                if full_key in nested_dict:
                    # Initialize `summed_df` with the shape of the first available DataFrame if not already done
                    if summed_df is None:
                        summed_df = pd.DataFrame(0, index=nested_dict[full_key].index, columns=nested_dict[full_key].columns)
                    
                    # Add the DataFrame to the sum
                    summed_df += nested_dict[full_key]
            
            # Only store if we found at least one DataFrame for the current `full_key`
            if summed_df is not None:
                summed_aggregate_results[full_key] = summed_df

# Store the aggregated results in result_dict under 'aggregates_combined'
result_dict['aggregate_multiplied'] = summed_aggregate_results

#################CONTINUE CODING HERE ##############
# Initialize a list to store each melted DataFrame with necessary identifiers
df_list = []
final_result_df = []

# Iterate over each top-level material key in result_dict (e.g., 'aggregate_multiplied', 'other_material')
for material_key, material_data in result_dict.items():
    
    # Iterate over each second-level key and DataFrame within each material key
    for second_key, df in material_data.items():
        # Reset the index to make 'year' a column
        df = df.reset_index()
        
        # Melt the DataFrame to convert regions (columns) into rows with 'region' and 'value'
        melted_df = df.melt(id_vars=['index'], var_name='region', value_name='value')
        melted_df = melted_df.rename(columns={'index': 'year'})
        
        # Add columns for material and the second layer key
        melted_df['material'] = material_key.split('_')[0]  # Assuming material name is part before "_"
        melted_df['second_key'] = second_key  # Use the second-level dictionary key
        
        # Append the processed DataFrame to the list
        df_list.append(melted_df)

# Concatenate all melted DataFrames
combined_paved = pd.concat(df_list)
# Ensure there are no duplicate entries for each combination of `year`, `region`, and `second_key`

# Set the index and pivot the DataFrame
combined_pivot = combined_paved.pivot_table(
    index=['year', 'region', 'second_key'],  # Set these columns as the index
    columns='material',                     # Use unique values in `material` as columns
    values='value',                         # Use `value` column as data for the new columns
    aggfunc='first'                         # Use the first non-null value in case of duplicates
)

# Pivot to make 'material' a column, keeping 'year', 'region', and 'second_key' as a MultiIndex
final_result_df = combined_paved.pivot_table(index=['year', 'region', 'second_key'], columns='material', values='value')

#################COMBINE MATERIALS AND CALCULATE MATERIAL INTENSITY#############
# Initialize an empty dictionary to store each divided DataFrame with the second_key as the key
divided_df_dict = {}

# Iterate over each unique `second_key` in `final_result_df`
for second_key in final_result_df.index.get_level_values('second_key').unique():
    # Select the subset of `final_result_df` that matches the current `second_key`
    subset_df = final_result_df.xs(second_key, level='second_key')
    
    # Try to access the matching DataFrame from the current namespace using globals()
    if second_key in globals():
        # Retrieve the individual DataFrame
        matching_df = globals()[second_key]
        
        # Reshape the individual DataFrame to match the structure of subset_df
        matching_df = matching_df.reset_index()
        
        # Melt to convert regions from columns to rows with 'region' and 'value' columns
        matching_melted = matching_df.melt(id_vars=['index'], var_name='region', value_name='value')
        matching_melted = matching_melted.rename(columns={'index': 'year'})
        
        # Set MultiIndex to match subset_df's structure: ['year', 'region']
        matching_melted = matching_melted.set_index(['year', 'region'])
        
        # Reindex matching_melted to match subset_df's index structure
        matching_melted = matching_melted.reindex(subset_df.index).fillna(1)
        
        # Perform element-wise division on 'value' column of matching_melted
        divided_subset = subset_df.divide(matching_melted['value'], axis=0)
        
        # Add the 'second_key' level back into the index
        divided_subset['second_key'] = second_key
        divided_subset = divided_subset.set_index('second_key', append=True)
        
        # Store the divided DataFrame with `second_key` as the dictionary key
        divided_df_dict[second_key] = divided_subset
    else:
        print(f"No matching DataFrame variable found for second_key: {second_key}")

# Concatenate all divided subsets back into a single DataFrame
final_divided_df = pd.concat(divided_df_dict).sort_index()
# Remove the "Index" level from the multi-index DataFrame
final_divided_df = final_divided_df.reset_index(level=0, drop=True)

###############ADD UNPAVED INTENSITY####################
#######MAYEB  - UNPVAED##############
# Initialize an empty dictionary to store aggregate results
# Dictionary to hold the final results
result_dict_unpaved = {}
# Dictionary to hold the final results


# Define the materials and their corresponding dictionaries
materials = [
    'unpaved - construction'
]

# Define a dictionary to map each material to the result dictionary keys
material_result_keys = {
    'unpaved - construction': 'aggregate_multiplied',

}

# Initialize result dictionaries for each material
for material in materials:
    result_dict_unpaved[material_result_keys[material]] = {}

# Define aggregate materials and initialize their result storage
aggregate_materials = {
    'unpaved - construction': 'aggregate_multiplied',
}


aggregate_unpaved_results_dict = {}

# Loop through each aggregate material and perform the multiplication
for aggregate_material, result_key in aggregate_materials.items():
    material_calc = material_dict.get(aggregate_material, {})
    
    # Initialize a sub-dictionary for the current aggregate material's multiplied results
    aggregate_unpaved_results_dict[result_key] = {}

    # Loop through each key (road class) in the current aggregate material dictionary
    for full_key, material_df in material_calc.items():
        # Extract the relevant part (road class) from the key name
        road_class_key = full_key.split('_')[-1]
        road_class = road_class_map.get(road_class_key)
        
        if road_class is None or road_class not in mat_int.index:
            print(f"No matching road class found for key '{full_key}' or road class missing in mat_int. Skipping.")
            continue
        
        # Retrieve the row from mat_int for the current aggregate material
        mat_int_row = mat_int.loc[road_class]
        
        # Filter for the aggregate material's row in mat_int
        if 'material' in mat_int.columns:
            aggregate_row = mat_int_row[mat_int_row['material'] == "aggregate"]
        else:
            print("Material column not found in mat_int_row.")
            continue
        
        # Ensure we have rows after filtering; if empty, fill with zeros
        if aggregate_row.empty:
            print(f"No {aggregate_material} material found for road class {road_class}. Filling with zeros.")
            aggregate_values = pd.DataFrame(0, index=[road_class], columns=mat_int.columns.drop('material')).values
        else:
            aggregate_values = aggregate_row.drop(columns='material').fillna(0).values

        # Perform the multiplication for each column in `material_df`
        multiplied_df = material_df.multiply(aggregate_values, axis=1)
        
        # Store the multiplied DataFrame in the dictionary for the aggregate material
        aggregate_unpaved_results_dict[result_key][full_key] = multiplied_df

        # Initialize a dictionary to store the summed DataFrames for each `full_key`
        summed_unpaved_aggregate_results = {}
        
        # Find all unique `full_key` names across all dictionaries
        all_full_keys = set(key for d in aggregate_unpaved_results_dict.values() for key in d.keys())
        
        # Iterate over each unique `full_key`
        for full_key in all_full_keys:
            # Start with an empty DataFrame initialized to zeros. 
            # We will adjust the index and columns once we find the first occurrence of `full_key`.
            summed_df = None
            
            # Sum each DataFrame corresponding to the current `full_key` across all dictionaries
            for result_key, nested_dict in aggregate_unpaved_results_dict.items():
                if full_key in nested_dict:
                    # Initialize `summed_df` with the shape of the first available DataFrame if not already done
                    if summed_df is None:
                        summed_df = pd.DataFrame(0, index=nested_dict[full_key].index, columns=nested_dict[full_key].columns)
                    
                    # Add the DataFrame to the sum
                    summed_df += nested_dict[full_key]
            
            # Only store if we found at least one DataFrame for the current `full_key`
            if summed_df is not None:
                summed_unpaved_aggregate_results[full_key] = summed_df
# Store the aggregated results in result_dict under 'aggregates_combined'
result_dict_unpaved['aggregate_multiplied'] = summed_unpaved_aggregate_results


# Initialize a list to store each melted DataFrame with necessary identifiers
df_list = []

# Iterate over each top-level material key in result_dict (e.g., 'aggregate_multiplied', 'other_material')
for material_key, material_data in result_dict_unpaved.items():
    
    # Iterate over each second-level key and DataFrame within each material key
    for second_key, df in material_data.items():
        # Reset the index to make 'year' a column
        df = df.reset_index()
        
        # Melt the DataFrame to convert regions (columns) into rows with 'region' and 'value'
        melted_df = df.melt(id_vars=['index'], var_name='region', value_name='value')
        melted_df = melted_df.rename(columns={'index': 'year'})
        
        # Add columns for material and the second layer key
        melted_df['material'] = material_key.split('_')[0]  # Assuming material name is part before "_"
        melted_df['second_key'] = second_key  # Use the second-level dictionary key
        
        # Append the processed DataFrame to the list
        df_list.append(melted_df)

# Concatenate all melted DataFrames
combined_unp_df = pd.concat(df_list)

# Pivot to make 'material' a column, keeping 'year', 'region', and 'second_key' as a MultiIndex
final_result__unpaved_df = combined_unp_df.pivot_table(index=['year', 'region', 'second_key'], columns='material', values='value')


# Initialize an empty dictionary to store each divided DataFrame with the second_key as the key
divided_df_unpaved_dict = {}

# Iterate over each unique `second_key` in `final_result_df`
for second_key in final_result__unpaved_df.index.get_level_values('second_key').unique():
    # Select the subset of `final_result_df` that matches the current `second_key`
    subset_df = final_result__unpaved_df.xs(second_key, level='second_key')
    
    # Try to access the matching DataFrame from the current namespace using globals()
    if second_key in globals():
        # Retrieve the individual DataFrame
        matching_df = globals()[second_key]
        
        # Reshape the individual DataFrame to match the structure of subset_df
        matching_df = matching_df.reset_index()
        
        # Melt to convert regions from columns to rows with 'region' and 'value' columns
        matching_melted = matching_df.melt(id_vars=['index'], var_name='region', value_name='value')
        matching_melted = matching_melted.rename(columns={'index': 'year'})
        
        # Set MultiIndex to match subset_df's structure: ['year', 'region']
        matching_melted = matching_melted.set_index(['year', 'region'])
        
        # Reindex matching_melted to match subset_df's index structure
        matching_melted = matching_melted.reindex(subset_df.index).fillna(1)
        
        # Perform element-wise division on 'value' column of matching_melted
        divided_subset = subset_df.divide(matching_melted['value'], axis=0)
        
        # Add the 'second_key' level back into the index
        divided_subset['second_key'] = second_key
        divided_subset = divided_subset.set_index('second_key', append=True)
        
        # Store the divided DataFrame with `second_key` as the dictionary key
        divided_df_unpaved_dict[second_key] = divided_subset
    else:
        print(f"No matching DataFrame variable found for second_key: {second_key}")

# Concatenate all DataFrames from the dictionary into a single large DataFrame
combined_unpaved_df = pd.concat(divided_df_unpaved_dict.values(), axis=0)

# Concatenate final_divided_df and combined_unpaved_df along rows
combined_mi = pd.concat([final_divided_df, combined_unpaved_df], axis=0)

#################Share of aggregates that are assumed to be permanent###############
###ASSUMED 80% OF AGGREGATES###
permanent_aggregates = combined_mi['aggregate'] * 0.8
permanent_aggregates = permanent_aggregates * 1000000

combined_mi['aggregate'] = combined_mi['aggregate'] * 0.2


def create_new_variables_with_sums(permanent_aggregates):
    # Dictionaries to store new and difference DataFrames
    new_variables = {}
    difference_variables = {}
    
    # Iterate over the MultiIndex and values of the Series
    for (year, region, second_key), value in permanent_aggregates.items():
        try:
            # Dynamically get the original variable by its name
            variable_df = globals()[second_key]

            # Check if we already created a new DataFrame for this variable
            if second_key not in new_variables:
                # Copy the original DataFrame to avoid modifying the original
                new_variables[second_key] = variable_df.copy()

            # Multiply the matching value in the new DataFrame
            new_variables[second_key].loc[year, region] *= value
        except KeyError:
            # Handle cases where the variable or key is missing
            print(f"Skipping {second_key} for year {year}, region {region}: variable or key not found.")
    
    # Adjust values and calculate differences
    for key, df in new_variables.items():
        # Sort by year for proper adjustments
        df_sorted = df.sort_index()

        # Iterate over columns to adjust for higher previous year values
        for region in df_sorted.columns:
            for year in df_sorted.index[1:]:  # Skip the first year
                prev_year = year - 1
                if df_sorted.loc[prev_year, region] > df_sorted.loc[year, region]:
                    df_sorted.loc[year, region] = df_sorted.loc[prev_year, region]

        # Add "permanent_aggregates" as a new index level
        df_sorted['index'] = 'aggregate'
        df_sorted = df_sorted.set_index('index', append=True)

        # Store the adjusted DataFrame
        globals()[f"new_{key}"] = df_sorted
        
        # Calculate differences
        difference_df = df_sorted.diff().clip(lower=0)
        difference_df.iloc[0] = df_sorted.iloc[0]  # Set the first year to match the original values

        # Add "permanent_aggregates" index to the difference dataframe
        difference_df['index'] = 'aggregate'
        difference_df = difference_df.set_index('index', append=True)

        # Store the difference DataFrame
        globals()[f"diff_{key}"] = difference_df
        difference_variables[key] = difference_df

    # Ensure NaN values are treated as 0 before summation
    new_total_u = sum(df.fillna(0) for key, df in new_variables.items() if 'u_area' in key)
    new_total_r = sum(df.fillna(0) for key, df in new_variables.items() if 'r_area' in key)
    diff_total_u = sum(df.fillna(0) for key, df in difference_variables.items() if 'u_area' in key)
    diff_total_r = sum(df.fillna(0) for key, df in difference_variables.items() if 'r_area' in key)

    # Store the totals as global variables
    globals()['new_total_u'] = new_total_u
    globals()['new_total_r'] = new_total_r
    globals()['diff_total_u'] = diff_total_u
    globals()['diff_total_r'] = diff_total_r

    return new_variables, difference_variables, new_total_u, new_total_r, diff_total_u, diff_total_r

# Apply the function
new_variables, difference_variables, new_total_u, new_total_r, diff_total_u, diff_total_r = create_new_variables_with_sums(permanent_aggregates)

# Add "permanent_aggregates" as a third index for new_total_u and new_total_r
for df_name in ['new_total_u', 'new_total_r']:
    df = globals()[df_name]
    if 'index' not in df.columns:  # If "index" is not already added
        df['index'] = 'aggregate'
    df.set_index('index', append=True, inplace=True)
    globals()[df_name] = df  # Update the global variable

# Remove the third index for diff_total_u and diff_total_r
for df_name in ['diff_total_u', 'diff_total_r']:
    df = globals()[df_name]
    df.reset_index(level=2, drop=True, inplace=True)  # Remove the third index
    globals()[df_name] = df  # Update the global variable
    
    

#%% Inflow Outflow calculations 
from dynamic_stock_model import DynamicStockModel as DSM

#################CHANGE DYNAMICS TO MATCH REGRESSION
first_year_grid = 1911          # The first constructed highway - marking the start of the modern transportation era - https://www.independent.co.uk/travel/europe/the-world-s-first-motorway-piero-puricelli-s-masterpiece-is-the-focus-of-an-unlikely-pilgrimage-a6840816.html
stdev_mult = 5              # standard deviation as a fraction of the mean lifetime applicable to transport equipment (Asset Management for Infrastructure Systems: Energy and Water, Balzer & Schorn 2015)

#############################################
# Interpolate material intensities (dynamic content from 1924 to 2100, based on data files)
index = pd.MultiIndex.from_product([list(range(first_year_grid,endyear+1)), list(materials_infra.index.levels[1])])
materials_infra_interpol = pd.DataFrame(index=index, columns=materials_infra.columns)

for cat in list(materials_infra.index.levels[1]):
   materials_infra_1st   = materials_infra.loc[idx[materials_infra.index[0][0], cat],:]
   materials_infra_interpol.loc[idx[first_year_grid ,cat],:] = materials_infra_1st                # set the first year (1924) values to the first available values in the dataset (for the year 2000) 
   materials_infra_interpol.loc[idx[materials_infra.index.levels[0].min(),cat],:] = materials_infra.loc[idx[materials_infra.index.levels[0].min(),cat],:]                # set the middle year (2000) values to the first available values in the dataset (for the year 2000) 
   materials_infra_interpol.loc[idx[materials_infra.index.levels[0].max(),cat],:] = materials_infra.loc[idx[materials_infra.index.levels[0].max(),cat],:]                # set the last year (2100) values to the last available values in the dataset (for the year 2100) 
   materials_infra_interpol.loc[idx[:,cat],:] = materials_infra_interpol.loc[idx[:,cat],:].astype('float32').reindex(list(range(first_year_grid,endyear+1)), level=0).interpolate()


################################

# In order to calculate inflow & outflow smoothly (without peaks for the initial years), we calculate a historic tail to the stock, by adding a 0 value for first year of operation (=1924), then interpolate values towards 1971
def stock_tail(grid_stock):
    # Ensure index is Int64Index
    grid_stock.index = pd.Index(pd.to_numeric(grid_stock.index, errors='coerce').astype('Int64'))

    # Ensure all values are numeric
    grid_stock = grid_stock.apply(pd.to_numeric, errors='coerce')

    zero_value = [0 for _ in range(0, regions)]

    # Set all regions to 0 in the year of initial operation
    grid_stock.loc[first_year_grid] = zero_value  

    # Reindex and interpolate
    grid_stock_new = (
        grid_stock
        .reindex(list(range(first_year_grid, endyear + 1)))
        .interpolate()
    )

    return grid_stock_new

urban_cycle_paved = stock_tail(u_area_paved_cycle)
urban_informal_paved = stock_tail(u_area_paved_informal)
urban_local_paved = stock_tail(u_area_paved_local) 
urban_motorway_paved = stock_tail(u_area_paved_motorway) 
urban_pedestrian_paved = stock_tail(u_area_paved_pedestrian) 
urban_primary_paved = stock_tail(u_area_paved_primary) 
urban_secondary_paved = stock_tail(u_area_paved_secondary)            
urban_tertiary_paved = stock_tail(u_area_paved_tertiary)               
urban_parking_paved = stock_tail(urban_paved_parking_area)               

urban_cycle_unpaved = stock_tail(u_area_unpaved_cycle)
urban_informal_unpaved = stock_tail(u_area_unpaved_informal)
urban_local_unpaved = stock_tail(u_area_unpaved_local) 
urban_motorway_unpaved = stock_tail(u_area_unpaved_motorway) 
urban_pedestrian_unpaved = stock_tail(u_area_unpaved_pedestrian) 
urban_primary_unpaved = stock_tail(u_area_unpaved_primary) 
urban_secondary_unpaved = stock_tail(u_area_unpaved_secondary)            
urban_tertiary_unpaved = stock_tail(u_area_unpaved_tertiary)               
urban_parking_unpaved = stock_tail(urban_unpaved_parking_area)    

rural_cycle_paved = stock_tail(r_area_paved_cycle)
rural_informal_paved = stock_tail(r_area_paved_informal)
rural_local_paved = stock_tail(r_area_paved_local) 
rural_motorway_paved = stock_tail(r_area_paved_motorway) 
rural_pedestrian_paved = stock_tail(r_area_paved_pedestrian) 
rural_primary_paved = stock_tail(r_area_paved_primary) 
rural_secondary_paved = stock_tail(r_area_paved_secondary)            
rural_tertiary_paved = stock_tail(r_area_paved_tertiary)               
rural_parking_paved = stock_tail(rural_paved_parking_area)      

rural_cycle_unpaved = stock_tail(r_area_unpaved_cycle)
rural_informal_unpaved = stock_tail(r_area_unpaved_informal)
rural_local_unpaved = stock_tail(r_area_unpaved_local) 
rural_motorway_unpaved = stock_tail(r_area_unpaved_motorway) 
rural_pedestrian_unpaved = stock_tail(r_area_unpaved_pedestrian) 
rural_primary_unpaved = stock_tail(r_area_unpaved_primary) 
rural_secondary_unpaved = stock_tail(r_area_unpaved_secondary)            
rural_tertiary_unpaved = stock_tail(r_area_unpaved_tertiary)               
rural_parking_unpaved = stock_tail(rural_unpaved_parking_area)   


urban_cycle_paved_obsolete = stock_tail(u_area_paved_obsolete_cycle)
urban_informal_paved_obsolete = stock_tail(u_area_paved_obsolete_informal)
urban_local_paved_obsolete = stock_tail(u_area_paved_obsolete_local) 
urban_motorway_paved_obsolete = stock_tail(u_area_paved_obsolete_motorway) 
urban_pedestrian_paved_obsolete = stock_tail(u_area_paved_obsolete_pedestrian) 
urban_primary_paved_obsolete = stock_tail(u_area_paved_obsolete_primary) 
urban_secondary_paved_obsolete = stock_tail(u_area_paved_obsolete_secondary)            
urban_tertiary_paved_obsolete = stock_tail(u_area_paved_obsolete_tertiary)               
#urban_parking_paved_obsolete = stock_tail(urban_paved_obsolete_parking_area)               

urban_cycle_unpaved_obsolete = stock_tail(u_area_unpaved_obsolete_cycle)
urban_informal_unpaved_obsolete = stock_tail(u_area_unpaved_obsolete_informal)
urban_local_unpaved_obsolete = stock_tail(u_area_unpaved_obsolete_local) 
urban_motorway_unpaved_obsolete = stock_tail(u_area_unpaved_obsolete_motorway) 
urban_pedestrian_unpaved_obsolete = stock_tail(u_area_unpaved_obsolete_pedestrian) 
urban_primary_unpaved_obsolete = stock_tail(u_area_unpaved_obsolete_primary) 
urban_secondary_unpaved_obsolete = stock_tail(u_area_unpaved_obsolete_secondary)            
urban_tertiary_unpaved_obsolete = stock_tail(u_area_unpaved_obsolete_tertiary)               
#urban_parking_unpaved_obsolete = stock_tail(urban_unpaved_obsolete_parking_area)    

rural_cycle_paved_obsolete = stock_tail(r_area_paved_obsolete_cycle)
rural_informal_paved_obsolete = stock_tail(r_area_paved_obsolete_informal)
rural_local_paved_obsolete = stock_tail(r_area_paved_obsolete_local) 
rural_motorway_paved_obsolete = stock_tail(r_area_paved_obsolete_motorway) 
rural_pedestrian_paved_obsolete = stock_tail(r_area_paved_obsolete_pedestrian) 
rural_primary_paved_obsolete = stock_tail(r_area_paved_obsolete_primary) 
rural_secondary_paved_obsolete = stock_tail(r_area_paved_obsolete_secondary)            
rural_tertiary_paved_obsolete = stock_tail(r_area_paved_obsolete_tertiary)               
#rural_parking_paved_obsolete = stock_tail(rural_paved_obsolete_parking_area)      

rural_cycle_unpaved_obsolete = stock_tail(r_area_unpaved_obsolete_cycle)
rural_informal_unpaved_obsolete = stock_tail(r_area_unpaved_obsolete_informal)
rural_local_unpaved_obsolete = stock_tail(r_area_unpaved_obsolete_local) 
rural_motorway_unpaved_obsolete = stock_tail(r_area_unpaved_obsolete_motorway) 
rural_pedestrian_unpaved_obsolete = stock_tail(r_area_unpaved_obsolete_pedestrian) 
rural_primary_unpaved_obsolete = stock_tail(r_area_unpaved_obsolete_primary) 
rural_secondary_unpaved_obsolete = stock_tail(r_area_unpaved_obsolete_secondary)            
rural_tertiary_unpaved_obsolete = stock_tail(r_area_unpaved_obsolete_tertiary)               
#rural_parking_unpaved_obsolete = stock_tail(rural_unpaved_obsolete_parking_area)   

# List of dataframes
urban_paved_dfs = [
    urban_cycle_paved, urban_informal_paved, urban_local_paved, 
    urban_motorway_paved, urban_pedestrian_paved, urban_primary_paved, 
    urban_secondary_paved, urban_tertiary_paved, urban_parking_paved,
    urban_cycle_unpaved, urban_informal_unpaved, urban_local_unpaved, 
    urban_motorway_unpaved, urban_pedestrian_unpaved, urban_primary_unpaved, 
    urban_secondary_unpaved, urban_tertiary_unpaved, urban_parking_unpaved
]

rural_paved_dfs = [
    rural_cycle_paved, rural_informal_paved, rural_local_paved, 
    rural_motorway_paved, rural_pedestrian_paved, rural_primary_paved, 
    rural_secondary_paved, rural_tertiary_paved, rural_parking_paved,
    rural_cycle_unpaved, rural_informal_unpaved, rural_local_unpaved, 
    rural_motorway_unpaved, rural_pedestrian_unpaved, rural_primary_unpaved, 
    rural_secondary_unpaved, rural_tertiary_unpaved, rural_parking_unpaved
]

# Apply the transformation to all dataframes
for df in urban_paved_dfs + rural_paved_dfs:
    df[df < 0] = 0

total_urbanrail_length = stock_tail(urban_rail_length)
total_rail_length = stock_tail(rail_length)
total_rail_obsolete = stock_tail(rail_obsolete)
total_HST_length = stock_tail(HST_length)

total_rail_bridges = stock_tail(bridge_rail)
total_rail_tunnels = stock_tail(tunnel_rail)

total_rail_bridges_obsolete = stock_tail(bridge_rail_obsolete)
total_rail_tunnels_obsolete = stock_tail(tunnel_rail_obsolete)


urban_bridges_road = stock_tail(urban_road_bridges)
urban_tunnels_road = stock_tail(urban_road_tunnels)
rural_bridges_road = stock_tail(rural_road_bridges)
rural_tunnels_road = stock_tail(rural_road_tunnels)

##NEW ADJUSTMENT - SIMPLER CODE - DATAFRAMES IN DICTIONARY
# Define the function to create a historic tail
def stock_tail_bt (grid_stock, regions, first_year_grid, endyear):
    # Create a list of zeros for each region
    zero_value = [0] * regions
    # Add the first year of operation to the dataframe with zero values for each region
    grid_stock.loc[first_year_grid] = zero_value
    # Reindex to include all years up to the end year and interpolate missing values
    grid_stock_new = grid_stock.reindex(range(first_year_grid, endyear + 1)).interpolate()
    return grid_stock_new

# Apply stock_tail function to each DataFrame in resulting_dataframes
processed_dataframes = {name: stock_tail_bt(df, regions, first_year_grid, endyear)
                        for name, df in resulting_dataframes.items()}


# Filter out rows where the 'year' column has the value 1924
combined_mi = combined_mi[combined_mi.index.get_level_values('year') != 1924]
combined_mi = combined_mi * 1000000

# Step 1: Concatenate `combined_mi` and `materials_infra_bt` along the rows
combined_mi_bt = pd.concat([combined_mi, materials_infra_bt])
# Fill NaN values with 0 after concatenation
combined_mi_bt = combined_mi_bt.fillna(0)

# Add the 'brick' column to the 'bricks' column
combined_mi_bt['bricks'] += combined_mi_bt['brick']
# Drop the original 'brick' column since it has been combined into 'bricks'
combined_mi_bt = combined_mi_bt.drop(columns=['brick'])

# Ensure 'steel' column exists and initialize it with 0 if it does not
combined_mi_bt['steel'] = combined_mi_bt.get('steel', 0) + combined_mi_bt.get('metal', 0)
# Add the 'steel' column to the 'metal' column
combined_mi_bt['steel'] += combined_mi_bt['metal']
# Drop the original 'metal' column since it has been combined into 'metal'
combined_mi_bt = combined_mi_bt.drop(columns=['metal'])

#############################################
# Interpolate material intensities (dynamic content from 1924 to 2100, based on data files)

# Define `idx` for multi-indexing
idx = pd.IndexSlice

# Create the extended MultiIndex covering all years from first_year_grid to endyear
full_index = pd.MultiIndex.from_product(
    [list(range(first_year_grid, endyear + 1)), 
     list(combined_mi_bt.index.levels[1]), 
     list(combined_mi_bt.index.levels[2])],
    names=combined_mi_bt.index.names
)

# Reindex `combined_mi` to match the full index range
materials_roads_interpol = combined_mi_bt.reindex(full_index)

# Interpolate missing values over the year range at level 0
# This will linearly interpolate any missing values in each group defined by levels 1 and 2
materials_roads_interpol = materials_roads_interpol.groupby(level=[1, 2]).apply(
    lambda group: group.interpolate(method='linear', limit_direction='both')
)


# Convert all non-missing, non-zero values to 1; missing and zero values to 0
materials_roads_interpol_service = materials_roads_interpol.notna() & (materials_roads_interpol != 0)
materials_roads_interpol_service = materials_roads_interpol_service.astype(int)

# Keep only the first column
materials_roads_interpol_service = materials_roads_interpol_service.iloc[:, :1]

# Set all values in the first column to 1
materials_roads_interpol_service.iloc[:, 0] = 1

# Convert all non-missing, non-zero values to 1; missing and zero values to 0
materials_infra_interpol_service = materials_infra_interpol.notna() & (materials_infra_interpol != 0)
materials_infra_interpol_service = materials_infra_interpol_service.astype(int)

# Keep only the first column
materials_infra_interpol_service = materials_infra_interpol_service.iloc[:, :1]

# Set all values in the first column to 1
materials_infra_interpol_service.iloc[:, 0] = 1

def create_dynamic_parameter_series(start_value, target_value, full_time_index, start_year=2025, end_year=2050):
    """
    Generates a Pandas Series with parameter values that change linearly over a defined period.

    Args:
        start_value (float): The initial parameter value.
        target_value (float): The parameter value to be reached by the end_year.
        full_time_index (pd.Index or list): The complete time index for the analysis.
        start_year (int): The year the linear change begins.
        end_year (int): The year the linear change ends.

    Returns:
        pd.Series: A Series of parameter values indexed by year.
    """
    # Create a series with the initial value for all years
    dynamic_series = pd.Series(start_value, index=full_time_index)
    
    # Define the years over which the interpolation will occur
    interpolation_years = np.arange(start_year, end_year + 1)
    
    # Generate linearly spaced values from the start to the target value
    interpolated_values = np.linspace(start_value, target_value, len(interpolation_years))
    
    # Assign the interpolated values to the corresponding years in the series
    dynamic_series.loc[interpolation_years] = interpolated_values
    
    # Set all years after the end_year to the target value
    dynamic_series.loc[dynamic_series.index > end_year] = target_value
    
    return dynamic_series

# Function in which the stock-driven DSM is applied to return (the moving average of the) materials in the inflow & outflow for all regions
# material calculations are done in the same function as the dynamic stock calculations to be lean on memory (stock- & outlfow-by-cohort are large dataframes, which do nod need to be stored in this way)
def inflow_outflow(stock, lifetime_tuple, material_intensity):

    initial_year = stock.first_valid_index()
    outflow_mat  = pd.DataFrame(index=pd.MultiIndex.from_product([range(startyear,outyear+1), material_intensity.columns]), columns=stock.columns)
    inflow_mat   = pd.DataFrame(index=pd.MultiIndex.from_product([range(startyear,outyear+1), material_intensity.columns]), columns=stock.columns)   
    stock_mat    = pd.DataFrame(index=pd.MultiIndex.from_product([range(startyear,outyear+1), material_intensity.columns]), columns=stock.columns)
    out_oc_mat   = pd.DataFrame(index=pd.MultiIndex.from_product([range(first_year_grid,endyear+1), material_intensity.columns]), columns=stock.columns)
    out_sc_mat   = pd.DataFrame(index=pd.MultiIndex.from_product([range(first_year_grid,endyear+1), material_intensity.columns]), columns=stock.columns)
    out_in_mat   = pd.DataFrame(index=pd.MultiIndex.from_product([range(first_year_grid,endyear+1), material_intensity.columns]), columns=stock.columns)

    # define mean & standard deviation
    start_life, target_life = lifetime_tuple
    years = stock.index
    
    mean_list = []
    for year in years:
        if year < 2025:
            mean = start_life
        elif 2025 <= year <= 2050:
            mean = start_life + (target_life - start_life) * ((year - 2025) / (2050 - 2025))
        else:
            mean = target_life
        mean_list.append(mean)
    
    stdev_list = [mean * stdev_mult for mean in mean_list]

    for region in list(stock.columns):
        # define and run the DSM                                                                                            # list with the fixed (=mean) lifetime of grid elements, given for every timestep (1924-2100), needed for the DSM as it allows to change lifetime for different cohort (even though we keep it constant)
        DSMforward = DSM(t = np.arange(0,len(stock[region]),1), s=np.array(stock[region]), lt = {'Type': 'FoldedNormal', 'Mean': np.array(mean_list), 'StdDev': np.array(stdev_list)})  # definition of the DSM based on a folded normal distribution
        out_sc, out_oc, out_i = DSMforward.compute_stock_driven_model(NegativeInflowCorrect = True)                                                                 # run the DSM, to give 3 outputs: stock_by_cohort, outflow_by_cohort & inflow_per_year
    
        #convert to pandas df before multiplication with material intensity
        index=list(range(first_year_grid, endyear+1))
        out_sc_pd = pd.DataFrame(out_sc, index=index,  columns=index)
        out_oc_pd = pd.DataFrame(out_oc, index=index,  columns=index)
        out_in_pd = pd.DataFrame(out_i,  index=index)
        
        # sum the outflow & stock by cohort (using cohort specific material intensities)
        for material in list(material_intensity.columns):    
           out_oc_mat.loc[idx[:,material],region] = out_oc_pd.mul(material_intensity.loc[:,material], axis=1).sum(axis=1).to_numpy()
           out_sc_mat.loc[idx[:,material],region] = out_sc_pd.mul(material_intensity.loc[:,material], axis=1).sum(axis=1).to_numpy() 
           out_in_mat.loc[idx[:,material],region] = out_in_pd.mul(material_intensity.loc[:,material], axis=0).to_numpy()                
# Return only 1971-2100 values (Raw data, no smoothing)
           outflow_mat.loc[idx[:,material],region] = pd.Series(out_oc_mat.loc[idx[2:,material],region].astype('float64').values, index=list(range(initial_year,2101))).loc[list(range(1971,2101))].to_numpy()
           inflow_mat.loc[idx[:,material],region]  = pd.Series(out_in_mat.loc[idx[2:,material],region].astype('float64').values, index=list(range(initial_year,2101))).loc[list(range(1971,2101))].to_numpy()
           stock_mat.loc[idx[:,material],region]   = pd.Series(out_sc_mat.loc[idx[2:,material],region].astype('float64').values, index=list(range(initial_year,2101))).loc[list(range(1971,2101))].to_numpy()    

    return inflow_mat, outflow_mat, stock_mat


rail_in, rail_out, rail_stock = inflow_outflow(total_rail_length, lifetime_grid_elements.loc['rail',['start','end']].values, materials_infra_interpol.loc[idx[:,'standard_rail'],:].droplevel(1)) 
rail_bridge_in, rail_bridge_out, rail_bridge_stock = inflow_outflow(total_rail_bridges, lifetime_grid_elements.loc['rail_bridge',['start','end']].values, materials_infra_interpol.loc[idx[:,'rail_bridges'],:].droplevel(1)) 
rail_tunnel_in, rail_tunnel_out, rail_tunnel_stock = inflow_outflow(total_rail_tunnels, lifetime_grid_elements.loc['rail_tunnel',['start','end']].values, materials_infra_interpol.loc[idx[:,'rail_tunnels'],:].droplevel(1)) 

HST_in, HST_out, HST_stock = inflow_outflow(total_HST_length, lifetime_grid_elements.loc['rail',['start','end']].values, materials_infra_interpol.loc[idx[:,'highspeed_rail'],:].droplevel(1)) 

if RAIL_HISTOR_FUNCTION == 1:
    rail_obs_in, rail_obs_out, rail_obs_stock = inflow_outflow(total_rail_obsolete, lifetime_grid_elements.loc['obs_rail',['start','end']].values, materials_infra_interpol.loc[idx[:,'standard_rail'],:].droplevel(1)) 
    rail_brid_obs_in, rail_brid_obs_out, rail_brid_obs_stock = inflow_outflow(total_rail_bridges_obsolete, lifetime_grid_elements.loc['obs_rail',['start','end']].values, materials_infra_interpol.loc[idx[:,'rail_bridges'],:].droplevel(1)) 
    rail_tunn_obs_in, rail_tunn_obs_out, rail_tunn_obs_stock = inflow_outflow(total_rail_tunnels_obsolete, lifetime_grid_elements.loc['obs_rail',['start','end']].values, materials_infra_interpol.loc[idx[:,'rail_tunnels'],:].droplevel(1)) 
   
    
# Updated rail outflows
    rail_out = (rail_out.fillna(0) - rail_obs_in.fillna(0)).clip(lower=0)
    rail_bridge_out = (rail_bridge_out.fillna(0) - rail_brid_obs_in.fillna(0)).clip(lower=0)    
    rail_tunnel_out = (rail_tunnel_out.fillna(0) - rail_tunn_obs_in.fillna(0)).clip(lower=0) 


urban_rail_in, urban_rail_out, urban_rail_stock = inflow_outflow(total_urbanrail_length, lifetime_grid_elements.loc['urban_rail',['start','end']].values, materials_infra_interpol.loc[idx[:,'urban_rail'],:].droplevel(1))              

#####################Conversion to Weibull mixed lifetimes


def inflow_outflow_weibull(stock, shape_series, scale_series, material_intensity):
    """
    Calculates material inflows, outflows, and stocks based on a Weibull lifetime distribution.
    This version is robust to mismatches in time indices between the dynamic parameters
    and the stock data by using reindexing.
    """
    import numpy as np
    import pandas as pd

    # Initialize output DataFrames (assuming startyear, outyear, etc. are defined elsewhere)
    initial_year = stock.first_valid_index()
    outflow_mat = pd.DataFrame(index=pd.MultiIndex.from_product([range(startyear, outyear + 1), material_intensity.columns]), columns=stock.columns)
    inflow_mat = pd.DataFrame(index=pd.MultiIndex.from_product([range(startyear, outyear + 1), material_intensity.columns]), columns=stock.columns)
    stock_mat = pd.DataFrame(index=pd.MultiIndex.from_product([range(startyear, outyear + 1), material_intensity.columns]), columns=stock.columns)
    out_oc_mat = pd.DataFrame(index=pd.MultiIndex.from_product([range(first_year_grid, endyear + 1), material_intensity.columns]), columns=stock.columns)
    out_sc_mat = pd.DataFrame(index=pd.MultiIndex.from_product([range(first_year_grid, endyear + 1), material_intensity.columns]), columns=stock.columns)
    out_in_mat = pd.DataFrame(index=pd.MultiIndex.from_product([range(first_year_grid, endyear + 1), material_intensity.columns]), columns=stock.columns)

    # Matching the region from `stock.columns` to `material_intensity`
    for region in stock.columns:
        print(f"Processing region: {region}")

        # --- MODIFICATION START ---
        # Get the target index from the current stock data being processed.
        target_index = stock[region].index

        # Safely align the parameter series to the stock's index.
        # .reindex() will create a new series with the target_index.
        # method='pad' (or 'ffill') fills missing values with the last known value.
        # .bfill() is added to fill any values at the very beginning that 'pad' might miss.
        aligned_shape_series = shape_series.reindex(target_index, method='pad').bfill()
        aligned_scale_series = scale_series.reindex(target_index, method='pad').bfill()
        
        # Now convert to numpy arrays, assured that they have the correct length and no key errors.
        aligned_shape_values = aligned_shape_series.to_numpy()
        aligned_scale_values = aligned_scale_series.to_numpy()
        # --- MODIFICATION END ---
        
        DSMforward = DSM(
            t=np.arange(0, len(stock[region]), 1),
            s=np.array(stock[region]),
            # Pass the aligned arrays for both Scale and Shape.
            lt={'Type': 'Weibull', 'Scale': aligned_scale_values, 'Shape': aligned_shape_values}
        )

        out_sc, out_oc, out_i = DSMforward.compute_stock_driven_model(NegativeInflowCorrect=True)

        # ... (the rest of your function remains the same)
        index = list(range(first_year_grid, endyear+1))
        out_sc_pd = pd.DataFrame(out_sc, index=index, columns=index)
        out_oc_pd = pd.DataFrame(out_oc, index=index, columns=index)
        out_in_pd = pd.DataFrame(out_i, index=index)
        
        for material in material_intensity.columns:
            try:
                for region_name in material_intensity.index.get_level_values(1).unique():
                    if region_name == region:
                        aligned_material_intensity = material_intensity.loc[idx[:, region_name], material].droplevel(1)
                        out_oc_mat.loc[idx[:, material], region_name] = out_oc_pd.mul(aligned_material_intensity, axis=1).sum(axis=1).to_numpy()
                        out_sc_mat.loc[idx[:, material], region_name] = out_sc_pd.mul(aligned_material_intensity, axis=1).sum(axis=1).to_numpy()
                        out_in_mat.loc[idx[:, material], region_name] = out_in_pd.mul(aligned_material_intensity, axis=0).to_numpy()
# Return raw data (no smoothing)
                        outflow_mat.loc[idx[:,material],region] = pd.Series(out_oc_mat.loc[idx[2:,material],region].astype('float64').values, index=list(range(initial_year,2101))).loc[list(range(1971,2101))].to_numpy()
                        inflow_mat.loc[idx[:,material],region]  = pd.Series(out_in_mat.loc[idx[2:,material],region].astype('float64').values, index=list(range(initial_year,2101))).loc[list(range(1971,2101))].to_numpy()
                        stock_mat.loc[idx[:,material],region]   = pd.Series(out_sc_mat.loc[idx[2:,material],region].astype('float64').values, index=list(range(initial_year,2101))).loc[list(range(1971,2101))].to_numpy()
            except (KeyError, IndexError) as e:
                print(f"{type(e).__name__}: {e}")

    return inflow_mat, outflow_mat, stock_mat

# --- Use dictionaries to store initial parameters ---
scale_values = {}
shape_values = {}
for i in range(1, 11):  # For Road Classes 1 to 8
    scale_values[f'rc{i}'] = WeibullParamsDataFrame.loc[f'Road Class {i}', 'scale']
    shape_values[f'rc{i}'] = WeibullParamsDataFrame.loc[f'Road Class {i}', 'shape']

for i in range(1, 10):  # For Bridge Classes 1 to 7
    scale_values[f'bc{i}'] = WeibullParamsDataFrame.loc[f'Bridge Class {i}', 'scale']
    shape_values[f'bc{i}'] = WeibullParamsDataFrame.loc[f'Bridge Class {i}', 'shape']


# --- Generate the time-varying scale and shape parameters ---

# Define the full time range of your model (replace with your actual variables)
# For example: full_time_index = range(1900, 2101)
full_time_index = range(startyear, outyear + 1) 

# Dictionaries to hold the new dynamic Series for scale and shape
dynamic_scale_rc = {}
dynamic_shape_rc = {}

if sa_settings in ["IMAGE_jan_2026_CP_all_base", "IMAGE_jan_2026_CP_slow_base"]:
    # Define the target mapping: rc(i) becomes rc(i-2) in 2050
    target_map = {2:1, 3:1, 4:2, 5:3, 6:4, 7:5, 8:6, 9:7, 10:8}
else:
    target_map = {2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10}


for i in range(1, 11):
    road_class = f'rc{i}'
    
    # Get starting values for scale and shape
    start_scale = scale_values[road_class]
    start_shape = shape_values[road_class]
    
    # Determine the target values based on the "2 levels down" rule
    if i in target_map:
        target_class = f'rc{target_map[i]}'
        target_scale = scale_values[target_class]
        target_shape = shape_values[target_class]
    else:
        # For rc1 and rc2, values stay the same
        target_scale = start_scale
        target_shape = start_shape
        
    # Create and store the dynamic series for BOTH scale and shape
    dynamic_scale_rc[road_class] = create_dynamic_parameter_series(start_scale, target_scale, full_time_index)
    dynamic_shape_rc[road_class] = create_dynamic_parameter_series(start_shape, target_shape, full_time_index)

# --- Dictionaries to hold the new dynamic Series for BRIDGE classes ---
dynamic_scale_bc = {}
dynamic_shape_bc = {}

# Define the target mapping for bridges, if any.
# For example: bc(i) becomes bc(i-2).
# If bc7 is the highest and doesn't change, its target will be itself.
if sa_settings in ["IMAGE_jan_2026_CP_all_base", "IMAGE_jan_2026_CP_slow_base"]:
    # Define the target mapping: rc(i) becomes rc(i-2) in 2050
    target_map_bc = {2:1,3: 1, 4: 2, 5: 3, 6: 4, 7: 5,8:6,9:7}
else:
    target_map_bc = {2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9}

# Loop through all your bridge classes (e.g., 1 to 7)
for i in range(1, 10):
    bridge_class = f'bc{i}'
    
    # Get starting values from your initial parameter dictionaries
    start_scale = scale_values[bridge_class]
    start_shape = shape_values[bridge_class]
    
    # Determine the target value
    if i in target_map_bc:
        target_class = f'bc{target_map_bc[i]}'
        target_scale = scale_values[target_class]
        target_shape = shape_values[target_class]
    else:
        # For classes with no lower target (e.g., bc1, bc2), values stay the same
        target_scale = start_scale
        target_shape = start_shape
        
    # Create and store the dynamic series for BOTH scale and shape
    # Assumes 'full_time_index' is already defined
    dynamic_scale_bc[bridge_class] = create_dynamic_parameter_series(start_scale, target_scale, full_time_index)
    dynamic_shape_bc[bridge_class] = create_dynamic_parameter_series(start_shape, target_shape, full_time_index)

# for i in range(1, 11):  # from 2 to 7
#     globals()[f'scale_value_rc{i}'] = WeibullParamsDataFrame.loc[f'Road Class {i}', 'scale']
#     globals()[f'shape_value_rc{i}'] = WeibullParamsDataFrame.loc[f'Road Class {i}', 'shape']
    
# for i in range(1, 10):  # from 2 to 7
#     globals()[f'scale_value_bc{i}'] = WeibullParamsDataFrame.loc[f'Bridge Class {i}', 'scale']
#     globals()[f'shape_value_bc{i}'] = WeibullParamsDataFrame.loc[f'Bridge Class {i}', 'shape']
###############CHECK FOR SERVICES################################

# --- Section: Urban Paved Surfaces ---
urban_cycle_in, urban_cycle_out, urban_cycle_stock = inflow_outflow_weibull(urban_cycle_paved, dynamic_shape_rc['rc1'], dynamic_scale_rc['rc1'], materials_roads_interpol.loc[idx[:,:,'u_area_paved_cycle'],:].droplevel(2))
urban_informal_in, urban_informal_out, urban_informal_stock = inflow_outflow_weibull(urban_informal_paved, dynamic_shape_rc['rc1'], dynamic_scale_rc['rc1'], materials_roads_interpol.loc[idx[:,:,'u_area_paved_informal'],:].droplevel(2))
urban_local_in, urban_local_out, urban_local_stock = inflow_outflow_weibull(urban_local_paved, dynamic_shape_rc['rc3'], dynamic_scale_rc['rc3'], materials_roads_interpol.loc[idx[:,:,'u_area_paved_local'],:].droplevel(2))
urban_motorway_in, urban_motorway_out, urban_motorway_stock = inflow_outflow_weibull(urban_motorway_paved, dynamic_shape_rc['rc7'], dynamic_scale_rc['rc7'], materials_roads_interpol.loc[idx[:,:,'u_area_paved_motorway'],:].droplevel(2))
urban_pedestrian_in, urban_pedestrian_out, urban_pedestrian_stock = inflow_outflow_weibull(urban_pedestrian_paved, dynamic_shape_rc['rc1'], dynamic_scale_rc['rc1'], materials_roads_interpol.loc[idx[:,:,'u_area_paved_pedestrian'],:].droplevel(2))
urban_primary_in, urban_primary_out, urban_primary_stock = inflow_outflow_weibull(urban_primary_paved, dynamic_shape_rc['rc6'], dynamic_scale_rc['rc6'], materials_roads_interpol.loc[idx[:,:,'u_area_paved_primary'],:].droplevel(2))
urban_secondary_in, urban_secondary_out, urban_secondary_stock = inflow_outflow_weibull(urban_secondary_paved, dynamic_shape_rc['rc5'], dynamic_scale_rc['rc5'], materials_roads_interpol.loc[idx[:,:,'u_area_paved_secondary'],:].droplevel(2))
urban_tertiary_in, urban_tertiary_out, urban_tertiary_stock = inflow_outflow_weibull(urban_tertiary_paved, dynamic_shape_rc['rc4'], dynamic_scale_rc['rc4'], materials_roads_interpol.loc[idx[:,:,'u_area_paved_tertiary'],:].droplevel(2))
urban_parking_in, urban_parking_out, urban_parking_stock = inflow_outflow_weibull(urban_parking_paved, dynamic_shape_rc['rc3'], dynamic_scale_rc['rc3'], materials_roads_interpol.loc[idx[:,:,'u_area_paved_local'],:].droplevel(2))

# --- Section: Urban Unpaved Surfaces ---
urban_unp_cycle_in, urban_unp_cycle_out, urban_unp_cycle_stock = inflow_outflow_weibull(urban_cycle_unpaved, dynamic_shape_rc['rc1'], dynamic_scale_rc['rc1'], materials_roads_interpol.loc[idx[:,:,'u_area_unpaved_cycle'],:].droplevel(2))
urban_unp_informal_in, urban_unp_informal_out, urban_unp_informal_stock = inflow_outflow_weibull(urban_informal_unpaved, dynamic_shape_rc['rc1'], dynamic_scale_rc['rc1'], materials_roads_interpol.loc[idx[:,:,'u_area_unpaved_informal'],:].droplevel(2))
urban_unp_local_in, urban_unp_local_out, urban_unp_local_stock = inflow_outflow_weibull(urban_local_unpaved, dynamic_shape_rc['rc3'], dynamic_scale_rc['rc3'], materials_roads_interpol.loc[idx[:,:,'u_area_unpaved_local'],:].droplevel(2))
urban_unp_pedestrian_in, urban_unp_pedestrian_out, urban_unp_pedestrian_stock = inflow_outflow_weibull(urban_pedestrian_unpaved, dynamic_shape_rc['rc1'], dynamic_scale_rc['rc1'], materials_roads_interpol.loc[idx[:,:,'u_area_unpaved_pedestrian'],:].droplevel(2))
urban_unp_primary_in, urban_unp_primary_out, urban_unp_primary_stock = inflow_outflow_weibull(urban_primary_unpaved, dynamic_shape_rc['rc6'], dynamic_scale_rc['rc6'], materials_roads_interpol.loc[idx[:,:,'u_area_unpaved_primary'],:].droplevel(2))
urban_unp_secondary_in, urban_unp_secondary_out, urban_unp_secondary_stock = inflow_outflow_weibull(urban_secondary_unpaved, dynamic_shape_rc['rc5'], dynamic_scale_rc['rc5'], materials_roads_interpol.loc[idx[:,:,'u_area_unpaved_secondary'],:].droplevel(2))
urban_unp_tertiary_in, urban_unp_tertiary_out, urban_unp_tertiary_stock = inflow_outflow_weibull(urban_tertiary_unpaved, dynamic_shape_rc['rc4'], dynamic_scale_rc['rc4'], materials_roads_interpol.loc[idx[:,:,'u_area_unpaved_tertiary'],:].droplevel(2))
urban_unp_parking_in, urban_unp_parking_out, urban_unp_parking_stock = inflow_outflow_weibull(urban_parking_unpaved, dynamic_shape_rc['rc3'], dynamic_scale_rc['rc3'], materials_roads_interpol.loc[idx[:,:,'u_area_unpaved_local'],:].droplevel(2))

# --- Section: Rural Paved Surfaces ---
rural_cycle_in, rural_cycle_out, rural_cycle_stock = inflow_outflow_weibull(rural_cycle_paved, dynamic_shape_rc['rc1'], dynamic_scale_rc['rc1'], materials_roads_interpol.loc[idx[:,:,'r_area_paved_cycle'],:].droplevel(2))
rural_informal_in, rural_informal_out, rural_informal_stock = inflow_outflow_weibull(rural_informal_paved, dynamic_shape_rc['rc1'], dynamic_scale_rc['rc1'], materials_roads_interpol.loc[idx[:,:,'r_area_paved_informal'],:].droplevel(2))
rural_local_in, rural_local_out, rural_local_stock = inflow_outflow_weibull(rural_local_paved, dynamic_shape_rc['rc1'], dynamic_scale_rc['rc1'], materials_roads_interpol.loc[idx[:,:,'r_area_paved_local'],:].droplevel(2))
rural_motorway_in, rural_motorway_out, rural_motorway_stock = inflow_outflow_weibull(rural_motorway_paved, dynamic_shape_rc['rc6'], dynamic_scale_rc['rc6'], materials_roads_interpol.loc[idx[:,:,'r_area_paved_motorway'],:].droplevel(2))
rural_pedestrian_in, rural_pedestrian_out, rural_pedestrian_stock = inflow_outflow_weibull(rural_pedestrian_paved, dynamic_shape_rc['rc1'], dynamic_scale_rc['rc1'], materials_roads_interpol.loc[idx[:,:,'r_area_paved_pedestrian'],:].droplevel(2))
rural_primary_in, rural_primary_out, rural_primary_stock = inflow_outflow_weibull(rural_primary_paved, dynamic_shape_rc['rc4'], dynamic_scale_rc['rc4'], materials_roads_interpol.loc[idx[:,:,'r_area_paved_primary'],:].droplevel(2))
rural_secondary_in, rural_secondary_out, rural_secondary_stock = inflow_outflow_weibull(rural_secondary_paved, dynamic_shape_rc['rc3'], dynamic_scale_rc['rc3'], materials_roads_interpol.loc[idx[:,:,'r_area_paved_secondary'],:].droplevel(2))
rural_tertiary_in, rural_tertiary_out, rural_tertiary_stock = inflow_outflow_weibull(rural_tertiary_paved, dynamic_shape_rc['rc2'], dynamic_scale_rc['rc2'], materials_roads_interpol.loc[idx[:,:,'r_area_paved_tertiary'],:].droplevel(2))
rural_parking_in, rural_parking_out, rural_parking_stock = inflow_outflow_weibull(rural_parking_paved, dynamic_shape_rc['rc2'], dynamic_scale_rc['rc2'], materials_roads_interpol.loc[idx[:,:,'r_area_paved_local'],:].droplevel(2))

# --- Section: Rural Unpaved Surfaces ---
rural_unp_cycle_in, rural_unp_cycle_out, rural_unp_cycle_stock = inflow_outflow_weibull(rural_cycle_unpaved, dynamic_shape_rc['rc1'], dynamic_scale_rc['rc1'], materials_roads_interpol.loc[idx[:,:,'r_area_unpaved_cycle'],:].droplevel(2))
rural_unp_informal_in, rural_unp_informal_out, rural_unp_informal_stock = inflow_outflow_weibull(rural_informal_unpaved, dynamic_shape_rc['rc1'], dynamic_scale_rc['rc1'], materials_roads_interpol.loc[idx[:,:,'r_area_unpaved_informal'],:].droplevel(2))
rural_unp_local_in, rural_unp_local_out, rural_unp_local_stock = inflow_outflow_weibull(rural_local_unpaved, dynamic_shape_rc['rc1'], dynamic_scale_rc['rc1'], materials_roads_interpol.loc[idx[:,:,'r_area_unpaved_local'],:].droplevel(2))
rural_unp_pedestrian_in, rural_unp_pedestrian_out, rural_unp_pedestrian_stock = inflow_outflow_weibull(rural_pedestrian_unpaved, dynamic_shape_rc['rc1'], dynamic_scale_rc['rc1'], materials_roads_interpol.loc[idx[:,:,'r_area_unpaved_pedestrian'],:].droplevel(2))
rural_unp_primary_in, rural_unp_primary_out, rural_unp_primary_stock = inflow_outflow_weibull(rural_primary_unpaved, dynamic_shape_rc['rc4'], dynamic_scale_rc['rc4'], materials_roads_interpol.loc[idx[:,:,'r_area_unpaved_primary'],:].droplevel(2))
rural_unp_secondary_in, rural_unp_secondary_out, rural_unp_secondary_stock = inflow_outflow_weibull(rural_secondary_unpaved, dynamic_shape_rc['rc3'], dynamic_scale_rc['rc3'], materials_roads_interpol.loc[idx[:,:,'r_area_unpaved_secondary'],:].droplevel(2))
rural_unp_tertiary_in, rural_unp_tertiary_out, rural_unp_tertiary_stock = inflow_outflow_weibull(rural_tertiary_unpaved, dynamic_shape_rc['rc2'], dynamic_scale_rc['rc2'], materials_roads_interpol.loc[idx[:,:,'r_area_unpaved_tertiary'],:].droplevel(2))
rural_unp_parking_in, rural_unp_parking_out, rural_unp_parking_stock = inflow_outflow_weibull(rural_parking_unpaved, dynamic_shape_rc['rc2'], dynamic_scale_rc['rc2'], materials_roads_interpol.loc[idx[:,:,'r_area_unpaved_local'],:].droplevel(2))

# --- Section: Obsolete Urban Paved Surfaces ---
obsolete_urban_cycle_in, obsolete_urban_cycle_out, obsolete_urban_cycle_stock = inflow_outflow_weibull(urban_cycle_paved_obsolete, dynamic_shape_rc['rc10'], dynamic_scale_rc['rc10'], materials_roads_interpol.loc[idx[:,:,'u_area_paved_cycle'],:].droplevel(2))
obsolete_urban_informal_in, obsolete_urban_informal_out, obsolete_urban_informal_stock = inflow_outflow_weibull(urban_informal_paved_obsolete, dynamic_shape_rc['rc10'], dynamic_scale_rc['rc10'], materials_roads_interpol.loc[idx[:,:,'u_area_paved_informal'],:].droplevel(2))
obsolete_urban_local_in, obsolete_urban_local_out, obsolete_urban_local_stock = inflow_outflow_weibull(urban_local_paved_obsolete, dynamic_shape_rc['rc10'], dynamic_scale_rc['rc10'], materials_roads_interpol.loc[idx[:,:,'u_area_paved_local'],:].droplevel(2))
obsolete_urban_motorway_in, obsolete_urban_motorway_out, obsolete_urban_motorway_stock = inflow_outflow_weibull(urban_motorway_paved_obsolete, dynamic_shape_rc['rc10'], dynamic_scale_rc['rc10'], materials_roads_interpol.loc[idx[:,:,'u_area_paved_motorway'],:].droplevel(2))
obsolete_urban_pedestrian_in, obsolete_urban_pedestrian_out, obsolete_urban_pedestrian_stock = inflow_outflow_weibull(urban_pedestrian_paved_obsolete, dynamic_shape_rc['rc10'], dynamic_scale_rc['rc10'], materials_roads_interpol.loc[idx[:,:,'u_area_paved_pedestrian'],:].droplevel(2))
obsolete_urban_primary_in, obsolete_urban_primary_out, obsolete_urban_primary_stock = inflow_outflow_weibull(urban_primary_paved_obsolete, dynamic_shape_rc['rc10'], dynamic_scale_rc['rc10'], materials_roads_interpol.loc[idx[:,:,'u_area_paved_primary'],:].droplevel(2))
obsolete_urban_secondary_in, obsolete_urban_secondary_out, obsolete_urban_secondary_stock = inflow_outflow_weibull(urban_secondary_paved_obsolete, dynamic_shape_rc['rc10'], dynamic_scale_rc['rc10'], materials_roads_interpol.loc[idx[:,:,'u_area_paved_secondary'],:].droplevel(2))
obsolete_urban_tertiary_in, obsolete_urban_tertiary_out, obsolete_urban_tertiary_stock = inflow_outflow_weibull(urban_tertiary_paved_obsolete, dynamic_shape_rc['rc10'], dynamic_scale_rc['rc10'], materials_roads_interpol.loc[idx[:,:,'u_area_paved_tertiary'],:].droplevel(2))
#obsolete_urban_parking_in, obsolete_urban_parking_out, obsolete_urban_parking_stock = inflow_outflow_weibull(urban_parking_paved_obsolete, dynamic_shape_rc['rc10'], dynamic_scale_rc['rc10'], materials_roads_interpol.loc[idx[:,:,'u_area_paved_local'],:].droplevel(2))

# --- Section: Obsolete Urban Unpaved Surfaces ---
obsolete_urban_unp_cycle_in, obsolete_urban_unp_cycle_out, obsolete_urban_unp_cycle_stock = inflow_outflow_weibull(urban_cycle_unpaved_obsolete, dynamic_shape_rc['rc10'], dynamic_scale_rc['rc10'], materials_roads_interpol.loc[idx[:,:,'u_area_unpaved_cycle'],:].droplevel(2))
obsolete_urban_unp_informal_in, obsolete_urban_unp_informal_out, obsolete_urban_unp_informal_stock = inflow_outflow_weibull(urban_informal_unpaved_obsolete, dynamic_shape_rc['rc10'], dynamic_scale_rc['rc10'], materials_roads_interpol.loc[idx[:,:,'u_area_unpaved_informal'],:].droplevel(2))
obsolete_urban_unp_local_in, obsolete_urban_unp_local_out, obsolete_urban_unp_local_stock = inflow_outflow_weibull(urban_local_unpaved_obsolete, dynamic_shape_rc['rc10'], dynamic_scale_rc['rc10'], materials_roads_interpol.loc[idx[:,:,'u_area_unpaved_local'],:].droplevel(2))
obsolete_urban_unp_pedestrian_in, obsolete_urban_unp_pedestrian_out, obsolete_urban_unp_pedestrian_stock = inflow_outflow_weibull(urban_pedestrian_unpaved_obsolete, dynamic_shape_rc['rc10'], dynamic_scale_rc['rc10'], materials_roads_interpol.loc[idx[:,:,'u_area_unpaved_pedestrian'],:].droplevel(2))
obsolete_urban_unp_primary_in, obsolete_urban_unp_primary_out, obsolete_urban_unp_primary_stock = inflow_outflow_weibull(urban_primary_unpaved_obsolete, dynamic_shape_rc['rc10'], dynamic_scale_rc['rc10'], materials_roads_interpol.loc[idx[:,:,'u_area_unpaved_primary'],:].droplevel(2))
obsolete_urban_unp_secondary_in, obsolete_urban_unp_secondary_out, obsolete_urban_unp_secondary_stock = inflow_outflow_weibull(urban_secondary_unpaved_obsolete, dynamic_shape_rc['rc10'], dynamic_scale_rc['rc10'], materials_roads_interpol.loc[idx[:,:,'u_area_unpaved_secondary'],:].droplevel(2))
obsolete_urban_unp_tertiary_in, obsolete_urban_unp_tertiary_out, obsolete_urban_unp_tertiary_stock = inflow_outflow_weibull(urban_tertiary_unpaved_obsolete, dynamic_shape_rc['rc10'], dynamic_scale_rc['rc10'], materials_roads_interpol.loc[idx[:,:,'u_area_unpaved_tertiary'],:].droplevel(2))
#obsolete_urban_unp_parking_in, obsolete_urban_unp_parking_out, obsolete_urban_unp_parking_stock = inflow_outflow_weibull(urban_parking_unpaved_obsolete, dynamic_shape_rc['rc10'], dynamic_scale_rc['rc10'], materials_roads_interpol.loc[idx[:,:,'u_area_unpaved_local'],:].droplevel(2))

# --- Section: Obsolete Rural Paved Surfaces ---
obsolete_rural_cycle_in, obsolete_rural_cycle_out, obsolete_rural_cycle_stock = inflow_outflow_weibull(rural_cycle_paved_obsolete, dynamic_shape_rc['rc10'], dynamic_scale_rc['rc10'], materials_roads_interpol.loc[idx[:,:,'r_area_paved_cycle'],:].droplevel(2))
obsolete_rural_informal_in, obsolete_rural_informal_out, obsolete_rural_informal_stock = inflow_outflow_weibull(rural_informal_paved_obsolete, dynamic_shape_rc['rc10'], dynamic_scale_rc['rc10'], materials_roads_interpol.loc[idx[:,:,'r_area_paved_informal'],:].droplevel(2))
obsolete_rural_local_in, obsolete_rural_local_out, obsolete_rural_local_stock = inflow_outflow_weibull(rural_local_paved_obsolete, dynamic_shape_rc['rc10'], dynamic_scale_rc['rc10'], materials_roads_interpol.loc[idx[:,:,'r_area_paved_local'],:].droplevel(2))
obsolete_rural_motorway_in, obsolete_rural_motorway_out, obsolete_rural_motorway_stock = inflow_outflow_weibull(rural_motorway_paved_obsolete, dynamic_shape_rc['rc10'], dynamic_scale_rc['rc10'], materials_roads_interpol.loc[idx[:,:,'r_area_paved_motorway'],:].droplevel(2))
obsolete_rural_pedestrian_in, obsolete_rural_pedestrian_out, obsolete_rural_pedestrian_stock = inflow_outflow_weibull(rural_pedestrian_paved_obsolete, dynamic_shape_rc['rc10'], dynamic_scale_rc['rc10'], materials_roads_interpol.loc[idx[:,:,'r_area_paved_pedestrian'],:].droplevel(2))
obsolete_rural_primary_in, obsolete_rural_primary_out, obsolete_rural_primary_stock = inflow_outflow_weibull(rural_primary_paved_obsolete, dynamic_shape_rc['rc10'], dynamic_scale_rc['rc10'], materials_roads_interpol.loc[idx[:,:,'r_area_paved_primary'],:].droplevel(2))
obsolete_rural_secondary_in, obsolete_rural_secondary_out, obsolete_rural_secondary_stock = inflow_outflow_weibull(rural_secondary_paved_obsolete, dynamic_shape_rc['rc10'], dynamic_scale_rc['rc10'], materials_roads_interpol.loc[idx[:,:,'r_area_paved_secondary'],:].droplevel(2))
obsolete_rural_tertiary_in, obsolete_rural_tertiary_out, obsolete_rural_tertiary_stock = inflow_outflow_weibull(rural_tertiary_paved_obsolete, dynamic_shape_rc['rc10'], dynamic_scale_rc['rc10'], materials_roads_interpol.loc[idx[:,:,'r_area_paved_tertiary'],:].droplevel(2))
#obsolete_rural_parking_in, obsolete_rural_parking_out, obsolete_rural_parking_stock = inflow_outflow_weibull(rural_parking_paved_obsolete, dynamic_shape_rc['rc10'], dynamic_scale_rc['rc10'], materials_roads_interpol.loc[idx[:,:,'r_area_paved_local'],:].droplevel(2))

# --- Section: Obsolete Rural Unpaved Surfaces ---
obsolete_rural_unp_cycle_in, obsolete_rural_unp_cycle_out, obsolete_rural_unp_cycle_stock = inflow_outflow_weibull(rural_cycle_unpaved_obsolete, dynamic_shape_rc['rc10'], dynamic_scale_rc['rc10'], materials_roads_interpol.loc[idx[:,:,'r_area_unpaved_cycle'],:].droplevel(2))
obsolete_rural_unp_informal_in, obsolete_rural_unp_informal_out, obsolete_rural_unp_informal_stock = inflow_outflow_weibull(rural_informal_unpaved_obsolete, dynamic_shape_rc['rc10'], dynamic_scale_rc['rc10'], materials_roads_interpol.loc[idx[:,:,'r_area_unpaved_informal'],:].droplevel(2))
obsolete_rural_unp_local_in, obsolete_rural_unp_local_out, obsolete_rural_unp_local_stock = inflow_outflow_weibull(rural_local_unpaved_obsolete, dynamic_shape_rc['rc10'], dynamic_scale_rc['rc10'], materials_roads_interpol.loc[idx[:,:,'r_area_unpaved_local'],:].droplevel(2))
obsolete_rural_unp_pedestrian_in, obsolete_rural_unp_pedestrian_out, obsolete_rural_unp_pedestrian_stock = inflow_outflow_weibull(rural_pedestrian_unpaved_obsolete, dynamic_shape_rc['rc10'], dynamic_scale_rc['rc10'], materials_roads_interpol.loc[idx[:,:,'r_area_unpaved_pedestrian'],:].droplevel(2))
obsolete_rural_unp_primary_in, obsolete_rural_unp_primary_out, obsolete_rural_unp_primary_stock = inflow_outflow_weibull(rural_primary_unpaved_obsolete, dynamic_shape_rc['rc10'], dynamic_scale_rc['rc10'], materials_roads_interpol.loc[idx[:,:,'r_area_unpaved_primary'],:].droplevel(2))
obsolete_rural_unp_secondary_in, obsolete_rural_unp_secondary_out, obsolete_rural_unp_secondary_stock = inflow_outflow_weibull(rural_secondary_unpaved_obsolete, dynamic_shape_rc['rc10'], dynamic_scale_rc['rc10'], materials_roads_interpol.loc[idx[:,:,'r_area_unpaved_secondary'],:].droplevel(2))
obsolete_rural_unp_tertiary_in, obsolete_rural_unp_tertiary_out, obsolete_rural_unp_tertiary_stock = inflow_outflow_weibull(rural_tertiary_unpaved_obsolete, dynamic_shape_rc['rc10'], dynamic_scale_rc['rc10'], materials_roads_interpol.loc[idx[:,:,'r_area_unpaved_tertiary'],:].droplevel(2))
#obsolete_rural_unp_parking_in, obsolete_rural_unp_parking_out, obsolete_rural_unp_parking_stock = inflow_outflow_weibull(rural_parking_unpaved_obsolete, dynamic_shape_rc['rc10'], dynamic_scale_rc['rc10'], materials_roads_interpol.loc[idx[:,:,'r_area_unpaved_local'],:].droplevel(2))

# ==========================================
# MASS BALANCE SUBTRACTIONS (WITH SAFEGUARDS)
# ==========================================

# Urban Paved Roads
urban_cycle_out = (urban_cycle_out.fillna(0) - obsolete_urban_cycle_in.fillna(0)).clip(lower=0)
urban_informal_out = (urban_informal_out.fillna(0) - obsolete_urban_informal_in.fillna(0)).clip(lower=0)
urban_local_out = (urban_local_out.fillna(0) - obsolete_urban_local_in.fillna(0)).clip(lower=0)
urban_motorway_out = (urban_motorway_out.fillna(0) - obsolete_urban_motorway_in.fillna(0)).clip(lower=0)
urban_pedestrian_out = (urban_pedestrian_out.fillna(0) - obsolete_urban_pedestrian_in.fillna(0)).clip(lower=0)
urban_primary_out = (urban_primary_out.fillna(0) - obsolete_urban_primary_in.fillna(0)).clip(lower=0)
urban_secondary_out = (urban_secondary_out.fillna(0) - obsolete_urban_secondary_in.fillna(0)).clip(lower=0)
urban_tertiary_out = (urban_tertiary_out.fillna(0) - obsolete_urban_tertiary_in.fillna(0)).clip(lower=0)

# Urban Unpaved Roads
urban_unp_cycle_out = (urban_unp_cycle_out.fillna(0) - obsolete_urban_unp_cycle_in.fillna(0)).clip(lower=0)
urban_unp_informal_out = (urban_unp_informal_out.fillna(0) - obsolete_urban_unp_informal_in.fillna(0)).clip(lower=0)
urban_unp_local_out = (urban_unp_local_out.fillna(0) - obsolete_urban_unp_local_in.fillna(0)).clip(lower=0)
urban_unp_pedestrian_out = (urban_unp_pedestrian_out.fillna(0) - obsolete_urban_unp_pedestrian_in.fillna(0)).clip(lower=0)
urban_unp_primary_out = (urban_unp_primary_out.fillna(0) - obsolete_urban_unp_primary_in.fillna(0)).clip(lower=0)
urban_unp_secondary_out = (urban_unp_secondary_out.fillna(0) - obsolete_urban_unp_secondary_in.fillna(0)).clip(lower=0)
urban_unp_tertiary_out = (urban_unp_tertiary_out.fillna(0) - obsolete_urban_unp_tertiary_in.fillna(0)).clip(lower=0)

# Rural Paved Roads
rural_cycle_out = (rural_cycle_out.fillna(0) - obsolete_rural_cycle_in.fillna(0)).clip(lower=0)
rural_informal_out = (rural_informal_out.fillna(0) - obsolete_rural_informal_in.fillna(0)).clip(lower=0)
rural_local_out = (rural_local_out.fillna(0) - obsolete_rural_local_in.fillna(0)).clip(lower=0)
rural_motorway_out = (rural_motorway_out.fillna(0) - obsolete_rural_motorway_in.fillna(0)).clip(lower=0)
rural_pedestrian_out = (rural_pedestrian_out.fillna(0) - obsolete_rural_pedestrian_in.fillna(0)).clip(lower=0)
rural_primary_out = (rural_primary_out.fillna(0) - obsolete_rural_primary_in.fillna(0)).clip(lower=0)
rural_secondary_out = (rural_secondary_out.fillna(0) - obsolete_rural_secondary_in.fillna(0)).clip(lower=0)
rural_tertiary_out = (rural_tertiary_out.fillna(0) - obsolete_rural_tertiary_in.fillna(0)).clip(lower=0)

# Rural Unpaved Roads
rural_unp_cycle_out = (rural_unp_cycle_out.fillna(0) - obsolete_rural_unp_cycle_in.fillna(0)).clip(lower=0)
rural_unp_informal_out = (rural_unp_informal_out.fillna(0) - obsolete_rural_unp_informal_in.fillna(0)).clip(lower=0)
rural_unp_local_out = (rural_unp_local_out.fillna(0) - obsolete_rural_unp_local_in.fillna(0)).clip(lower=0)
rural_unp_pedestrian_out = (rural_unp_pedestrian_out.fillna(0) - obsolete_rural_unp_pedestrian_in.fillna(0)).clip(lower=0)
rural_unp_primary_out = (rural_unp_primary_out.fillna(0) - obsolete_rural_unp_primary_in.fillna(0)).clip(lower=0)
rural_unp_secondary_out = (rural_unp_secondary_out.fillna(0) - obsolete_rural_unp_secondary_in.fillna(0)).clip(lower=0)
rural_unp_tertiary_out = (rural_unp_tertiary_out.fillna(0) - obsolete_rural_unp_tertiary_in.fillna(0)).clip(lower=0)

# Extract each DataFrame from the dictionary and assign it to a variable with the same name
for name, df in processed_dataframes.items():
    globals()[name] = df

# --- Section: Urban Bridges ---
urban_b_cycle_in, urban_b_cycle_out, urban_b_cycle_stock = inflow_outflow_weibull(urban_road_bridges_bike_b, dynamic_shape_bc['bc1'], dynamic_scale_bc['bc1'], materials_roads_interpol.loc[idx[:,:,'road_bridges_cycle'],:].droplevel(2))
urban_b_informal_in, urban_b_informal_out, urban_b_informal_stock = inflow_outflow_weibull(urban_road_bridges_informal_b, dynamic_shape_bc['bc1'], dynamic_scale_bc['bc1'], materials_roads_interpol.loc[idx[:,:,'road_bridges_informal'],:].droplevel(2))
urban_b_local_in, urban_b_local_out, urban_b_local_stock = inflow_outflow_weibull(urban_road_bridges_local_b, dynamic_shape_bc['bc3'], dynamic_scale_bc['bc3'], materials_roads_interpol.loc[idx[:,:,'road_bridges_local'],:].droplevel(2))
urban_b_motorway_in, urban_b_motorway_out, urban_b_motorway_stock = inflow_outflow_weibull(urban_road_bridges_motorway_b, dynamic_shape_bc['bc6'], dynamic_scale_bc['bc6'], materials_roads_interpol.loc[idx[:,:,'road_bridges_motorway'],:].droplevel(2))
urban_b_pedestrian_in, urban_b_pedestrian_out, urban_b_pedestrian_stock = inflow_outflow_weibull(urban_road_bridges_pedestrian_b, dynamic_shape_bc['bc1'], dynamic_scale_bc['bc1'], materials_roads_interpol.loc[idx[:,:,'road_bridges_pedestrian'],:].droplevel(2))
urban_b_primary_in, urban_b_primary_out, urban_b_primary_stock = inflow_outflow_weibull(urban_road_bridges_primary_b, dynamic_shape_bc['bc5'], dynamic_scale_bc['bc5'], materials_roads_interpol.loc[idx[:,:,'road_bridges_primary'],:].droplevel(2))
urban_b_secondary_in, urban_b_secondary_out, urban_b_secondary_stock = inflow_outflow_weibull(urban_road_bridges_secondary_b, dynamic_shape_bc['bc4'], dynamic_scale_bc['bc4'], materials_roads_interpol.loc[idx[:,:,'road_bridges_secondary'],:].droplevel(2))
urban_b_tertiary_in, urban_b_tertiary_out, urban_b_tertiary_stock = inflow_outflow_weibull(urban_road_bridges_tertiary_b, dynamic_shape_bc['bc3'], dynamic_scale_bc['bc3'], materials_roads_interpol.loc[idx[:,:,'road_bridges_tertiary'],:].droplevel(2))

# --- Section: Urban Tunnels ---
urban_t_cycle_in, urban_t_cycle_out, urban_t_cycle_stock = inflow_outflow_weibull(urban_road_tunnels_cycle_t, dynamic_shape_bc['bc1'], dynamic_scale_bc['bc1'], materials_roads_interpol.loc[idx[:,:,'road_tunnels_cycle'],:].droplevel(2))
urban_t_informal_in, urban_t_informal_out, urban_t_informal_stock = inflow_outflow_weibull(urban_road_tunnels_informal_t, dynamic_shape_bc['bc1'], dynamic_scale_bc['bc1'], materials_roads_interpol.loc[idx[:,:,'road_tunnels_informal'],:].droplevel(2))
urban_t_local_in, urban_t_local_out, urban_t_local_stock = inflow_outflow_weibull(urban_road_tunnels_local_t, dynamic_shape_bc['bc3'], dynamic_scale_bc['bc3'], materials_roads_interpol.loc[idx[:,:,'road_tunnels_local'],:].droplevel(2))
urban_t_motorway_in, urban_t_motorway_out, urban_t_motorway_stock = inflow_outflow_weibull(urban_road_tunnels_motorway_t, dynamic_shape_bc['bc6'], dynamic_scale_bc['bc6'], materials_roads_interpol.loc[idx[:,:,'road_tunnels_motorway'],:].droplevel(2))
urban_t_pedestrian_in, urban_t_pedestrian_out, urban_t_pedestrian_stock = inflow_outflow_weibull(urban_road_tunnels_pedestrian_t, dynamic_shape_bc['bc1'], dynamic_scale_bc['bc1'], materials_roads_interpol.loc[idx[:,:,'road_tunnels_pedestrian'],:].droplevel(2))
urban_t_primary_in, urban_t_primary_out, urban_t_primary_stock = inflow_outflow_weibull(urban_road_tunnels_primary_t, dynamic_shape_bc['bc5'], dynamic_scale_bc['bc5'], materials_roads_interpol.loc[idx[:,:,'road_tunnels_primary'],:].droplevel(2))
urban_t_secondary_in, urban_t_secondary_out, urban_t_secondary_stock = inflow_outflow_weibull(urban_road_tunnels_secondary_t, dynamic_shape_bc['bc4'], dynamic_scale_bc['bc4'], materials_roads_interpol.loc[idx[:,:,'road_tunnels_secondary'],:].droplevel(2))
urban_t_tertiary_in, urban_t_tertiary_out, urban_t_tertiary_stock = inflow_outflow_weibull(urban_road_tunnels_tertiary_t, dynamic_shape_bc['bc3'], dynamic_scale_bc['bc3'], materials_roads_interpol.loc[idx[:,:,'road_tunnels_tertiary'],:].droplevel(2))

# --- Section: Rural Bridges ---
rural_b_cycle_in, rural_b_cycle_out, rural_b_cycle_stock = inflow_outflow_weibull(rural_road_bridges_bike_b, dynamic_shape_bc['bc1'], dynamic_scale_bc['bc1'], materials_roads_interpol.loc[idx[:,:,'road_bridges_cycle'],:].droplevel(2))
rural_b_informal_in, rural_b_informal_out, rural_b_informal_stock = inflow_outflow_weibull(rural_road_bridges_informal_b, dynamic_shape_bc['bc1'], dynamic_scale_bc['bc1'], materials_roads_interpol.loc[idx[:,:,'road_bridges_informal'],:].droplevel(2))
rural_b_local_in, rural_b_local_out, rural_b_local_stock = inflow_outflow_weibull(rural_road_bridges_local_b, dynamic_shape_bc['bc1'], dynamic_scale_bc['bc1'], materials_roads_interpol.loc[idx[:,:,'road_bridges_local'],:].droplevel(2))
rural_b_motorway_in, rural_b_motorway_out, rural_b_motorway_stock = inflow_outflow_weibull(rural_road_bridges_motorway_b, dynamic_shape_bc['bc4'], dynamic_scale_bc['bc4'], materials_roads_interpol.loc[idx[:,:,'road_bridges_motorway'],:].droplevel(2))
rural_b_pedestrian_in, rural_b_pedestrian_out, rural_b_pedestrian_stock = inflow_outflow_weibull(rural_road_bridges_pedestrian_b, dynamic_shape_bc['bc1'], dynamic_scale_bc['bc1'], materials_roads_interpol.loc[idx[:,:,'road_bridges_pedestrian'],:].droplevel(2))
rural_b_primary_in, rural_b_primary_out, rural_b_primary_stock = inflow_outflow_weibull(rural_road_bridges_primary_b, dynamic_shape_bc['bc3'], dynamic_scale_bc['bc3'], materials_roads_interpol.loc[idx[:,:,'road_bridges_primary'],:].droplevel(2))
rural_b_secondary_in, rural_b_secondary_out, rural_b_secondary_stock = inflow_outflow_weibull(rural_road_bridges_secondary_b, dynamic_shape_bc['bc2'], dynamic_scale_bc['bc2'], materials_roads_interpol.loc[idx[:,:,'road_bridges_secondary'],:].droplevel(2))
rural_b_tertiary_in, rural_b_tertiary_out, rural_b_tertiary_stock = inflow_outflow_weibull(rural_road_bridges_tertiary_b, dynamic_shape_bc['bc2'], dynamic_scale_bc['bc2'], materials_roads_interpol.loc[idx[:,:,'road_bridges_tertiary'],:].droplevel(2))

# --- Section: Rural Tunnels ---
rural_t_cycle_in, rural_t_cycle_out, rural_t_cycle_stock = inflow_outflow_weibull(rural_road_tunnels_cycle_t, dynamic_shape_bc['bc1'], dynamic_scale_bc['bc1'], materials_roads_interpol.loc[idx[:,:,'road_tunnels_cycle'],:].droplevel(2))
rural_t_informal_in, rural_t_informal_out, rural_t_informal_stock = inflow_outflow_weibull(rural_road_tunnels_informal_t, dynamic_shape_bc['bc1'], dynamic_scale_bc['bc1'], materials_roads_interpol.loc[idx[:,:,'road_tunnels_informal'],:].droplevel(2))
rural_t_local_in, rural_t_local_out, rural_t_local_stock = inflow_outflow_weibull(rural_road_tunnels_local_t, dynamic_shape_bc['bc1'], dynamic_scale_bc['bc1'], materials_roads_interpol.loc[idx[:,:,'road_tunnels_local'],:].droplevel(2))
rural_t_motorway_in, rural_t_motorway_out, rural_t_motorway_stock = inflow_outflow_weibull(rural_road_tunnels_motorway_t, dynamic_shape_bc['bc4'], dynamic_scale_bc['bc4'], materials_roads_interpol.loc[idx[:,:,'road_tunnels_motorway'],:].droplevel(2))
rural_t_pedestrian_in, rural_t_pedestrian_out, rural_t_pedestrian_stock = inflow_outflow_weibull(rural_road_tunnels_pedestrian_t, dynamic_shape_bc['bc1'], dynamic_scale_bc['bc1'], materials_roads_interpol.loc[idx[:,:,'road_tunnels_pedestrian'],:].droplevel(2))
rural_t_primary_in, rural_t_primary_out, rural_t_primary_stock = inflow_outflow_weibull(rural_road_tunnels_primary_t, dynamic_shape_bc['bc3'], dynamic_scale_bc['bc3'], materials_roads_interpol.loc[idx[:,:,'road_tunnels_primary'],:].droplevel(2))
rural_t_secondary_in, rural_t_secondary_out, rural_t_secondary_stock = inflow_outflow_weibull(rural_road_tunnels_secondary_t, dynamic_shape_bc['bc2'], dynamic_scale_bc['bc2'], materials_roads_interpol.loc[idx[:,:,'road_tunnels_secondary'],:].droplevel(2))
rural_t_tertiary_in, rural_t_tertiary_out, rural_t_tertiary_stock = inflow_outflow_weibull(rural_road_tunnels_tertiary_t, dynamic_shape_bc['bc2'], dynamic_scale_bc['bc2'], materials_roads_interpol.loc[idx[:,:,'road_tunnels_tertiary'],:].droplevel(2))

# --- Section: Obsolete Urban Bridges ---
obsolete_urban_b_cycle_in, obsolete_urban_b_cycle_out, obsolete_urban_b_cycle_stock = inflow_outflow_weibull(obsolete_urban_road_bridges_bike_b, dynamic_shape_bc['bc9'], dynamic_scale_bc['bc9'], materials_roads_interpol.loc[idx[:,:,'road_bridges_cycle'],:].droplevel(2))
obsolete_urban_b_informal_in, obsolete_urban_b_informal_out, obsolete_urban_b_informal_stock = inflow_outflow_weibull(obsolete_urban_road_bridges_informal_b, dynamic_shape_bc['bc9'], dynamic_scale_bc['bc9'], materials_roads_interpol.loc[idx[:,:,'road_bridges_informal'],:].droplevel(2))
obsolete_urban_b_local_in, obsolete_urban_b_local_out, obsolete_urban_b_local_stock = inflow_outflow_weibull(obsolete_urban_road_bridges_local_b, dynamic_shape_bc['bc9'], dynamic_scale_bc['bc9'], materials_roads_interpol.loc[idx[:,:,'road_bridges_local'],:].droplevel(2))
obsolete_urban_b_motorway_in, obsolete_urban_b_motorway_out, obsolete_urban_b_motorway_stock = inflow_outflow_weibull(obsolete_urban_road_bridges_motorway_b, dynamic_shape_bc['bc9'], dynamic_scale_bc['bc9'], materials_roads_interpol.loc[idx[:,:,'road_bridges_motorway'],:].droplevel(2))
obsolete_urban_b_pedestrian_in, obsolete_urban_b_pedestrian_out, obsolete_urban_b_pedestrian_stock = inflow_outflow_weibull(obsolete_urban_road_bridges_pedestrian_b, dynamic_shape_bc['bc9'], dynamic_scale_bc['bc9'], materials_roads_interpol.loc[idx[:,:,'road_bridges_pedestrian'],:].droplevel(2))
obsolete_urban_b_primary_in, obsolete_urban_b_primary_out, obsolete_urban_b_primary_stock = inflow_outflow_weibull(obsolete_urban_road_bridges_primary_b, dynamic_shape_bc['bc9'], dynamic_scale_bc['bc9'], materials_roads_interpol.loc[idx[:,:,'road_bridges_primary'],:].droplevel(2))
obsolete_urban_b_secondary_in, obsolete_urban_b_secondary_out, obsolete_urban_b_secondary_stock = inflow_outflow_weibull(obsolete_urban_road_bridges_secondary_b, dynamic_shape_bc['bc9'], dynamic_scale_bc['bc9'], materials_roads_interpol.loc[idx[:,:,'road_bridges_secondary'],:].droplevel(2))
obsolete_urban_b_tertiary_in, obsolete_urban_b_tertiary_out, obsolete_urban_b_tertiary_stock = inflow_outflow_weibull(obsolete_urban_road_bridges_tertiary_b, dynamic_shape_bc['bc9'], dynamic_scale_bc['bc9'], materials_roads_interpol.loc[idx[:,:,'road_bridges_tertiary'],:].droplevel(2))

# --- Section: Obsolete Urban Tunnels ---
obsolete_urban_t_cycle_in, obsolete_urban_t_cycle_out, obsolete_urban_t_cycle_stock = inflow_outflow_weibull(obsolete_urban_road_tunnels_cycle_t, dynamic_shape_bc['bc9'], dynamic_scale_bc['bc9'], materials_roads_interpol.loc[idx[:,:,'road_tunnels_cycle'],:].droplevel(2))
obsolete_urban_t_informal_in, obsolete_urban_t_informal_out, obsolete_urban_t_informal_stock = inflow_outflow_weibull(obsolete_urban_road_tunnels_informal_t, dynamic_shape_bc['bc9'], dynamic_scale_bc['bc9'], materials_roads_interpol.loc[idx[:,:,'road_tunnels_informal'],:].droplevel(2))
obsolete_urban_t_local_in, obsolete_urban_t_local_out, obsolete_urban_t_local_stock = inflow_outflow_weibull(obsolete_urban_road_tunnels_local_t, dynamic_shape_bc['bc9'], dynamic_scale_bc['bc9'], materials_roads_interpol.loc[idx[:,:,'road_tunnels_local'],:].droplevel(2))
obsolete_urban_t_motorway_in, obsolete_urban_t_motorway_out, obsolete_urban_t_motorway_stock = inflow_outflow_weibull(obsolete_urban_road_tunnels_motorway_t, dynamic_shape_bc['bc9'], dynamic_scale_bc['bc9'], materials_roads_interpol.loc[idx[:,:,'road_tunnels_motorway'],:].droplevel(2))
obsolete_urban_t_pedestrian_in, obsolete_urban_t_pedestrian_out, obsolete_urban_t_pedestrian_stock = inflow_outflow_weibull(obsolete_urban_road_tunnels_pedestrian_t, dynamic_shape_bc['bc9'], dynamic_scale_bc['bc9'], materials_roads_interpol.loc[idx[:,:,'road_tunnels_pedestrian'],:].droplevel(2))
obsolete_urban_t_primary_in, obsolete_urban_t_primary_out, obsolete_urban_t_primary_stock = inflow_outflow_weibull(obsolete_urban_road_tunnels_primary_t, dynamic_shape_bc['bc9'], dynamic_scale_bc['bc9'], materials_roads_interpol.loc[idx[:,:,'road_tunnels_primary'],:].droplevel(2))
obsolete_urban_t_secondary_in, obsolete_urban_t_secondary_out, obsolete_urban_t_secondary_stock = inflow_outflow_weibull(obsolete_urban_road_tunnels_secondary_t, dynamic_shape_bc['bc9'], dynamic_scale_bc['bc9'], materials_roads_interpol.loc[idx[:,:,'road_tunnels_secondary'],:].droplevel(2))
obsolete_urban_t_tertiary_in, obsolete_urban_t_tertiary_out, obsolete_urban_t_tertiary_stock = inflow_outflow_weibull(obsolete_urban_road_tunnels_tertiary_t, dynamic_shape_bc['bc9'], dynamic_scale_bc['bc9'], materials_roads_interpol.loc[idx[:,:,'road_tunnels_tertiary'],:].droplevel(2))

# --- Section: Obsolete Rural Bridges ---
obsolete_rural_b_cycle_in, obsolete_rural_b_cycle_out, obsolete_rural_b_cycle_stock = inflow_outflow_weibull(obsolete_rural_road_bridges_bike_b, dynamic_shape_bc['bc9'], dynamic_scale_bc['bc9'], materials_roads_interpol.loc[idx[:,:,'road_bridges_cycle'],:].droplevel(2))
obsolete_rural_b_informal_in, obsolete_rural_b_informal_out, obsolete_rural_b_informal_stock = inflow_outflow_weibull(obsolete_rural_road_bridges_informal_b, dynamic_shape_bc['bc9'], dynamic_scale_bc['bc9'], materials_roads_interpol.loc[idx[:,:,'road_bridges_informal'],:].droplevel(2))
obsolete_rural_b_local_in, obsolete_rural_b_local_out, obsolete_rural_b_local_stock = inflow_outflow_weibull(obsolete_rural_road_bridges_local_b, dynamic_shape_bc['bc9'], dynamic_scale_bc['bc9'], materials_roads_interpol.loc[idx[:,:,'road_bridges_local'],:].droplevel(2))
obsolete_rural_b_motorway_in, obsolete_rural_b_motorway_out, obsolete_rural_b_motorway_stock = inflow_outflow_weibull(obsolete_rural_road_bridges_motorway_b, dynamic_shape_bc['bc9'], dynamic_scale_bc['bc9'], materials_roads_interpol.loc[idx[:,:,'road_bridges_motorway'],:].droplevel(2))
obsolete_rural_b_pedestrian_in, obsolete_rural_b_pedestrian_out, obsolete_rural_b_pedestrian_stock = inflow_outflow_weibull(obsolete_rural_road_bridges_pedestrian_b, dynamic_shape_bc['bc9'], dynamic_scale_bc['bc9'], materials_roads_interpol.loc[idx[:,:,'road_bridges_pedestrian'],:].droplevel(2))
obsolete_rural_b_primary_in, obsolete_rural_b_primary_out, obsolete_rural_b_primary_stock = inflow_outflow_weibull(obsolete_rural_road_bridges_primary_b, dynamic_shape_bc['bc9'], dynamic_scale_bc['bc9'], materials_roads_interpol.loc[idx[:,:,'road_bridges_primary'],:].droplevel(2))
obsolete_rural_b_secondary_in, obsolete_rural_b_secondary_out, obsolete_rural_b_secondary_stock = inflow_outflow_weibull(obsolete_rural_road_bridges_secondary_b, dynamic_shape_bc['bc9'], dynamic_scale_bc['bc9'], materials_roads_interpol.loc[idx[:,:,'road_bridges_secondary'],:].droplevel(2))
obsolete_rural_b_tertiary_in, obsolete_rural_b_tertiary_out, obsolete_rural_b_tertiary_stock = inflow_outflow_weibull(obsolete_rural_road_bridges_tertiary_b, dynamic_shape_bc['bc9'], dynamic_scale_bc['bc9'], materials_roads_interpol.loc[idx[:,:,'road_bridges_tertiary'],:].droplevel(2))

# --- Section: Obsolete Rural Tunnels ---
obsolete_rural_t_cycle_in, obsolete_rural_t_cycle_out, obsolete_rural_t_cycle_stock = inflow_outflow_weibull(obsolete_rural_road_tunnels_cycle_t, dynamic_shape_bc['bc9'], dynamic_scale_bc['bc9'], materials_roads_interpol.loc[idx[:,:,'road_tunnels_cycle'],:].droplevel(2))
obsolete_rural_t_informal_in, obsolete_rural_t_informal_out, obsolete_rural_t_informal_stock = inflow_outflow_weibull(obsolete_rural_road_tunnels_informal_t, dynamic_shape_bc['bc9'], dynamic_scale_bc['bc9'], materials_roads_interpol.loc[idx[:,:,'road_tunnels_informal'],:].droplevel(2))
obsolete_rural_t_local_in, obsolete_rural_t_local_out, obsolete_rural_t_local_stock = inflow_outflow_weibull(obsolete_rural_road_tunnels_local_t, dynamic_shape_bc['bc9'], dynamic_scale_bc['bc9'], materials_roads_interpol.loc[idx[:,:,'road_tunnels_local'],:].droplevel(2))
obsolete_rural_t_motorway_in, obsolete_rural_t_motorway_out, obsolete_rural_t_motorway_stock = inflow_outflow_weibull(obsolete_rural_road_tunnels_motorway_t, dynamic_shape_bc['bc9'], dynamic_scale_bc['bc9'], materials_roads_interpol.loc[idx[:,:,'road_tunnels_motorway'],:].droplevel(2))
obsolete_rural_t_pedestrian_in, obsolete_rural_t_pedestrian_out, obsolete_rural_t_pedestrian_stock = inflow_outflow_weibull(obsolete_rural_road_tunnels_pedestrian_t, dynamic_shape_bc['bc9'], dynamic_scale_bc['bc9'], materials_roads_interpol.loc[idx[:,:,'road_tunnels_pedestrian'],:].droplevel(2))
obsolete_rural_t_primary_in, obsolete_rural_t_primary_out, obsolete_rural_t_primary_stock = inflow_outflow_weibull(obsolete_rural_road_tunnels_primary_t, dynamic_shape_bc['bc9'], dynamic_scale_bc['bc9'], materials_roads_interpol.loc[idx[:,:,'road_tunnels_primary'],:].droplevel(2))
obsolete_rural_t_secondary_in, obsolete_rural_t_secondary_out, obsolete_rural_t_secondary_stock = inflow_outflow_weibull(obsolete_rural_road_tunnels_secondary_t, dynamic_shape_bc['bc9'], dynamic_scale_bc['bc9'], materials_roads_interpol.loc[idx[:,:,'road_tunnels_secondary'],:].droplevel(2))
obsolete_rural_t_tertiary_in, obsolete_rural_t_tertiary_out, obsolete_rural_t_tertiary_stock = inflow_outflow_weibull(obsolete_rural_road_tunnels_tertiary_t, dynamic_shape_bc['bc9'], dynamic_scale_bc['bc9'], materials_roads_interpol.loc[idx[:,:,'road_tunnels_tertiary'],:].droplevel(2))

# Urban Bridges
urban_b_cycle_out = (urban_b_cycle_out.fillna(0) - obsolete_urban_b_cycle_in.fillna(0)).clip(lower=0)
urban_b_informal_out = (urban_b_informal_out.fillna(0) - obsolete_urban_b_informal_in.fillna(0)).clip(lower=0)
urban_b_local_out = (urban_b_local_out.fillna(0) - obsolete_urban_b_local_in.fillna(0)).clip(lower=0)
urban_b_motorway_out = (urban_b_motorway_out.fillna(0) - obsolete_urban_b_motorway_in.fillna(0)).clip(lower=0)
urban_b_pedestrian_out = (urban_b_pedestrian_out.fillna(0) - obsolete_urban_b_pedestrian_in.fillna(0)).clip(lower=0) # Typo fixed here
urban_b_primary_out = (urban_b_primary_out.fillna(0) - obsolete_urban_b_primary_in.fillna(0)).clip(lower=0) # Typo fixed here
urban_b_secondary_out = (urban_b_secondary_out.fillna(0) - obsolete_urban_b_secondary_in.fillna(0)).clip(lower=0)
urban_b_tertiary_out = (urban_b_tertiary_out.fillna(0) - obsolete_urban_b_tertiary_in.fillna(0)).clip(lower=0)

# Urban Tunnels
urban_t_cycle_out = (urban_t_cycle_out.fillna(0) - obsolete_urban_t_cycle_in.fillna(0)).clip(lower=0)
urban_t_informal_out = (urban_t_informal_out.fillna(0) - obsolete_urban_t_informal_in.fillna(0)).clip(lower=0)
urban_t_local_out = (urban_t_local_out.fillna(0) - obsolete_urban_t_local_in.fillna(0)).clip(lower=0)
urban_t_motorway_out = (urban_t_motorway_out.fillna(0) - obsolete_urban_t_motorway_in.fillna(0)).clip(lower=0)
urban_t_pedestrian_out = (urban_t_pedestrian_out.fillna(0) - obsolete_urban_t_pedestrian_in.fillna(0)).clip(lower=0)
urban_t_primary_out = (urban_t_primary_out.fillna(0) - obsolete_urban_t_primary_in.fillna(0)).clip(lower=0)
urban_t_secondary_out = (urban_t_secondary_out.fillna(0) - obsolete_urban_t_secondary_in.fillna(0)).clip(lower=0)
urban_t_tertiary_out = (urban_t_tertiary_out.fillna(0) - obsolete_urban_t_tertiary_in.fillna(0)).clip(lower=0)

# Rural Bridges
rural_b_cycle_out = (rural_b_cycle_out.fillna(0) - obsolete_rural_b_cycle_in.fillna(0)).clip(lower=0)
rural_b_informal_out = (rural_b_informal_out.fillna(0) - obsolete_rural_b_informal_in.fillna(0)).clip(lower=0)
rural_b_local_out = (rural_b_local_out.fillna(0) - obsolete_rural_b_local_in.fillna(0)).clip(lower=0)
rural_b_motorway_out = (rural_b_motorway_out.fillna(0) - obsolete_rural_b_motorway_in.fillna(0)).clip(lower=0)
rural_b_pedestrian_out = (rural_b_pedestrian_out.fillna(0) - obsolete_rural_b_pedestrian_in.fillna(0)).clip(lower=0)
rural_b_primary_out = (rural_b_primary_out.fillna(0) - obsolete_rural_b_primary_in.fillna(0)).clip(lower=0)
rural_b_secondary_out = (rural_b_secondary_out.fillna(0) - obsolete_rural_b_secondary_in.fillna(0)).clip(lower=0)
rural_b_tertiary_out = (rural_b_tertiary_out.fillna(0) - obsolete_rural_b_tertiary_in.fillna(0)).clip(lower=0)

# Rural Tunnels
rural_t_cycle_out = (rural_t_cycle_out.fillna(0) - obsolete_rural_t_cycle_in.fillna(0)).clip(lower=0)
rural_t_informal_out = (rural_t_informal_out.fillna(0) - obsolete_rural_t_informal_in.fillna(0)).clip(lower=0)
rural_t_local_out = (rural_t_local_out.fillna(0) - obsolete_rural_t_local_in.fillna(0)).clip(lower=0)
rural_t_motorway_out = (rural_t_motorway_out.fillna(0) - obsolete_rural_t_motorway_in.fillna(0)).clip(lower=0)
rural_t_pedestrian_out = (rural_t_pedestrian_out.fillna(0) - obsolete_rural_t_pedestrian_in.fillna(0)).clip(lower=0)
rural_t_primary_out = (rural_t_primary_out.fillna(0) - obsolete_rural_t_primary_in.fillna(0)).clip(lower=0)
rural_t_secondary_out = (rural_t_secondary_out.fillna(0) - obsolete_rural_t_secondary_in.fillna(0)).clip(lower=0)
rural_t_tertiary_out = (rural_t_tertiary_out.fillna(0) - obsolete_rural_t_tertiary_in.fillna(0)).clip(lower=0)


def align_and_update_with_layout(source_df, target_layout_df, aggregates_key="aggregate"):
    """
    Align the source DataFrame to match the layout (index and columns) of the target layout DataFrame,
    but only update values for aggregates.
    
    Args:
        source_df (pd.DataFrame): The DataFrame containing the aggregate values to be updated.
        target_layout_df (pd.DataFrame): The DataFrame whose layout (index and columns) will be matched.
        aggregates_key (str): The key corresponding to the aggregate rows in the source DataFrame.
        
    Returns:
        pd.DataFrame: A DataFrame matching the layout of `target_layout_df` with updated aggregate values.
    """
    # Create a DataFrame with the same layout as the target
    aligned_df = target_layout_df.copy()
    aligned_df[:] = 0  # Initialize with zeros

    # Identify rows in the source that match the aggregates key
    aggregate_rows = source_df[source_df.index.get_level_values(1) == aggregates_key]
    
    # Update values in the aligned DataFrame where the layout matches
    for year, region in aggregate_rows.index:
        if (year, aggregates_key) in aligned_df.index:  # Match by layout
            aligned_df.loc[(year, aggregates_key)] = aggregate_rows.loc[(year, region)]
    
    return aligned_df

# Apply the function to create aligned DataFrames
new_total_u = align_and_update_with_layout(new_total_u, urban_cycle_stock)
new_total_r = align_and_update_with_layout(new_total_r, urban_cycle_stock)
diff_total_u = align_and_update_with_layout(diff_total_u, urban_cycle_stock)
diff_total_r = align_and_update_with_layout(diff_total_r, urban_cycle_stock)

# Store the updated DataFrames back to globals (optional)
globals()['new_total_u'] = new_total_u
globals()['new_total_r'] = new_total_r
globals()['diff_total_u'] = diff_total_u
globals()['diff_total_r'] = diff_total_r

# Define the options for each part of the variable names
options = {
    'prefix': ['new', 'diff'],
    'region': ['u', 'r'],
    'area': ['paved', 'unpaved'],
    'road_type': ['cycle', 'informal', 'local', 'motorway', 'pedestrian', 'primary', 'secondary', 'tertiary','parking']
}

# Iterate over all combinations of options
for prefix in options['prefix']:
    for region in options['region']:
        for area in options['area']:
            # Skip motorway for unpaved area
            if area == 'unpaved' and 'motorway' in options['road_type']:
                road_types = [rt for rt in options['road_type'] if rt != 'motorway']
            else:
                road_types = options['road_type']

            for road_type in road_types:
                # Construct the variable name
                var_name = f"{prefix}_{region}_area_{area}_{road_type}"

                # Check if the variable exists in globals()
                if var_name in globals():
                    # Get the DataFrame
                    df = globals()[var_name]

                    # Remove the last level of the index if the prefix is 'diff'
                    if prefix == 'diff':
                        df.index = df.index.droplevel(-1)

                    # Apply the function
                    aligned_df = align_and_update_with_layout(df, urban_cycle_stock)

                    # Store the updated DataFrame back to globals
                    globals()[var_name] = aligned_df
                else:
                    print(f"Variable {var_name} not found in globals().")

#%% prepare output variables
sorted_region_list = sorted(region_list)
material_list = list(materials_roads_interpol.columns.values)

# overall container of stock, inflow & outflow of materials (in kgs)
items = ['all_urban','all_rural','urban_parking','rural_parking','urban_paved_roads','urban_unpaved_roads','rural_paved_roads','rural_unpaved_roads','urban_roads','rural_roads', 'urban_bridges','urban_tunnels','rural_bridges','rural_tunnels','total_bridges','total_tunnels','total_rail','total_urban_rail','total_tram','hibernating_brid_tun', 'hibernating_roads','hibernating_rail','rail_bridge','rail_tunnel','total_HST']
flow =  ['stock', 'inflow', 'outflow']
columns = pd.MultiIndex.from_product([sorted_region_list, items], names=['regions', 'elements'])
index = pd.MultiIndex.from_product([flow, list(range(startyear,2101)), material_list], names=['flow', 'years', 'materials'])
infra_materials = pd.DataFrame(index=index, columns=columns)

items = ['roads','bridges','tunnels','rail','parking','obsolete']
flow =  ['stock', 'inflow', 'outflow']
columns = pd.MultiIndex.from_product([sorted_region_list, items], names=['regions', 'elements'])
index = pd.MultiIndex.from_product([flow, list(range(startyear,2101)), material_list], names=['flow', 'years', 'materials'])
infra_materials_scenario = pd.DataFrame(index=index, columns=columns)

items = ['roads','bridges','tunnels','rail','parking']
flow =  ['stock', 'inflow', 'outflow']
columns = pd.MultiIndex.from_product([sorted_region_list, items], names=['regions', 'elements'])
index = pd.MultiIndex.from_product([flow, list(range(1971,2101))], names=['flow', 'years'])
infra_materials_physical = pd.DataFrame(index=index, columns=columns)

# Define the year range
year_range = slice(1971, 2100)

# Filter each variable to only include the desired years
adjusted_rural_road_area = adjusted_rural_road_area.loc[year_range]
adjusted_urban_road_area = adjusted_urban_road_area.loc[year_range]
adjusted_rural_road_bridges = rural_road_bridges.loc[year_range]
adjusted_urban_road_bridges = urban_road_bridges.loc[year_range]
adjusted_rural_road_tunnels = rural_road_tunnels.loc[year_range]
adjusted_urban_road_tunnels = urban_road_tunnels.loc[year_range]
total_rail_length = total_rail_length.loc[year_range]
total_HST_length = total_HST_length.loc[year_range]

total_rail_bridges = total_rail_bridges.loc[year_range]
total_rail_tunnels = total_rail_tunnels.loc[year_range]

total_urbanrail_length = total_urbanrail_length.loc[year_range]
adjusted_rural_parking_area = rural_parking_area.loc[year_range]
adjusted_urban_parking_area = urban_parking_area.loc[year_range]

items_detail = [
    # Urban Paved Roads by Category
    'urban_cycle_paved', 'urban_informal_paved', 'urban_local_paved', 'urban_motorway_paved',
    'urban_pedestrian_paved', 'urban_primary_paved', 'urban_secondary_paved', 'urban_tertiary_paved', 'urban_parking_paved',

    # Urban Unpaved Roads by Category
    'urban_cycle_unpaved', 'urban_informal_unpaved', 'urban_local_unpaved', 'urban_pedestrian_unpaved',
    'urban_primary_unpaved', 'urban_secondary_unpaved', 'urban_tertiary_unpaved', 'urban_parking_unpaved',

    # Rural Paved Roads by Category
    'rural_cycle_paved', 'rural_informal_paved', 'rural_local_paved', 'rural_motorway_paved',
    'rural_pedestrian_paved', 'rural_primary_paved', 'rural_secondary_paved', 'rural_tertiary_paved', 'rural_parking_paved',

    # Rural Unpaved Roads by Category
    'rural_cycle_unpaved', 'rural_informal_unpaved', 'rural_local_unpaved', 'rural_pedestrian_unpaved',
    'rural_primary_unpaved', 'rural_secondary_unpaved', 'rural_tertiary_unpaved', 'rural_parking_unpaved',

    # Urban Bridges
    'urban_bridge_cycle', 'urban_bridge_informal', 'urban_bridge_local', 'urban_bridge_motorway',
    'urban_bridge_pedestrian', 'urban_bridge_primary', 'urban_bridge_secondary', 'urban_bridge_tertiary',

    # Urban Tunnels
    'urban_tunnel_cycle', 'urban_tunnel_informal', 'urban_tunnel_local', 'urban_tunnel_motorway',
    'urban_tunnel_pedestrian', 'urban_tunnel_primary', 'urban_tunnel_secondary', 'urban_tunnel_tertiary',

    # Rural Bridges
    'rural_bridge_cycle', 'rural_bridge_informal', 'rural_bridge_local', 'rural_bridge_motorway',
    'rural_bridge_pedestrian', 'rural_bridge_primary', 'rural_bridge_secondary', 'rural_bridge_tertiary',

    # Rural Tunnels
    'rural_tunnel_cycle', 'rural_tunnel_informal', 'rural_tunnel_local', 'rural_tunnel_motorway',
    'rural_tunnel_pedestrian', 'rural_tunnel_primary', 'rural_tunnel_secondary', 'rural_tunnel_tertiary',

    # Rail Categories
    'total_rail', 'total_urban_rail', 'total_tram', 'rail_bridge','rail_tunnel', 'total_HST',
    
    # obsolete categories
    'hibernating_rail','hibernating_rail_bridge','hibernating_rail_tunnel',',hibernating_brid_tun', 'hibernating_roads', 'obsolete_urban_bridges','obsolete_urban_tunnel','obsolete_rural_bridges','obsolete_rural_tunnel','obsolete_urban_paved_roads', 'obsolete_urban_unpaved_roads', 'obsolete_rural_paved_roads', 'obsolete_rural_unpaved_roads'
]
# List of flows
flow_detail = ['stock', 'inflow', 'outflow']
# Create the multi-index for columns
columns = pd.MultiIndex.from_product([sorted_region_list, items_detail], names=['regions', 'elements'])

# Create the multi-index for index (years from startyear to 2060, flows, materials)
index = pd.MultiIndex.from_product(
    [flow_detail, list(range(startyear, 2101)), material_list], 
    names=['flow', 'years', 'materials']
)

# Create the DataFrame using the multi-indexes
infra_materials_detail = pd.DataFrame(index=index, columns=columns)

infra_materials.loc[idx['stock', :,:],idx[:,'urban_paved_roads']]          =  urban_cycle_stock.to_numpy() + urban_informal_stock.to_numpy() + urban_local_stock.to_numpy() + urban_motorway_stock.to_numpy() + urban_pedestrian_stock.to_numpy() + urban_primary_stock.to_numpy() + urban_secondary_stock.to_numpy() + urban_tertiary_stock.to_numpy()+ new_u_area_paved_cycle.to_numpy()+ new_u_area_paved_informal.to_numpy()+ new_u_area_paved_local.to_numpy()+ new_u_area_paved_motorway.to_numpy()+ new_u_area_paved_pedestrian.to_numpy()+ new_u_area_paved_primary.to_numpy()+ new_u_area_paved_secondary.to_numpy()+ new_u_area_paved_tertiary.to_numpy() # kgs in stock for lines
infra_materials.loc[idx['stock', :,:],idx[:,'urban_unpaved_roads']]          =  urban_unp_cycle_stock.to_numpy() + urban_unp_informal_stock.to_numpy() + urban_unp_local_stock.to_numpy() + urban_unp_pedestrian_stock.to_numpy() + urban_unp_primary_stock.to_numpy() + urban_unp_secondary_stock.to_numpy() + urban_unp_tertiary_stock.to_numpy()+ new_u_area_unpaved_cycle.to_numpy() + new_u_area_unpaved_informal.to_numpy()+ new_u_area_unpaved_local.to_numpy()+ new_u_area_unpaved_pedestrian.to_numpy()+ new_u_area_unpaved_primary.to_numpy()+ new_u_area_unpaved_secondary.to_numpy()+ new_u_area_unpaved_tertiary.to_numpy()  # kgs in stock for lines
infra_materials.loc[idx['stock', :,:],idx[:,'rural_paved_roads']]          =  rural_cycle_stock.to_numpy() + rural_informal_stock.to_numpy() + rural_local_stock.to_numpy() + rural_motorway_stock.to_numpy() + rural_pedestrian_stock.to_numpy() + rural_primary_stock.to_numpy() + rural_secondary_stock.to_numpy() + rural_tertiary_stock.to_numpy()+ new_r_area_paved_cycle.to_numpy()+ new_r_area_paved_informal.to_numpy()+ new_r_area_paved_local.to_numpy() + new_r_area_paved_motorway.to_numpy() + new_r_area_paved_pedestrian.to_numpy()+ new_r_area_paved_primary.to_numpy()+ new_r_area_paved_secondary.to_numpy()+ new_r_area_paved_tertiary.to_numpy()  # kgs in stock for lines
infra_materials.loc[idx['stock', :,:],idx[:,'rural_unpaved_roads']]          =  rural_unp_cycle_stock.to_numpy() + rural_unp_informal_stock.to_numpy() + rural_unp_local_stock.to_numpy()  + rural_unp_pedestrian_stock.to_numpy() + rural_unp_primary_stock.to_numpy() + rural_unp_secondary_stock.to_numpy() + rural_unp_tertiary_stock.to_numpy()+ new_r_area_unpaved_cycle.to_numpy() + new_r_area_unpaved_informal.to_numpy()+ new_r_area_unpaved_local.to_numpy()+ new_r_area_unpaved_pedestrian.to_numpy()+ new_r_area_unpaved_primary.to_numpy()+ new_r_area_unpaved_secondary.to_numpy()+ new_r_area_unpaved_tertiary.to_numpy()  # kgs in stock for lines
infra_materials.loc[idx['stock', :,:],idx[:,'urban_roads']]          =  urban_cycle_stock.to_numpy() + urban_informal_stock.to_numpy() + urban_local_stock.to_numpy() + urban_motorway_stock.to_numpy() + urban_pedestrian_stock.to_numpy() + urban_primary_stock.to_numpy() + urban_secondary_stock.to_numpy() + urban_tertiary_stock.to_numpy() + urban_unp_cycle_stock.to_numpy() + urban_unp_informal_stock.to_numpy() + urban_unp_local_stock.to_numpy() + urban_unp_pedestrian_stock.to_numpy() + urban_unp_primary_stock.to_numpy() + urban_unp_secondary_stock.to_numpy() + urban_unp_tertiary_stock.to_numpy() # kgs in stock for lines
infra_materials.loc[idx['stock', :,:],idx[:,'rural_roads']]          =  rural_cycle_stock.to_numpy() + rural_informal_stock.to_numpy() + rural_local_stock.to_numpy() + rural_motorway_stock.to_numpy() + rural_pedestrian_stock.to_numpy() + rural_primary_stock.to_numpy() + rural_secondary_stock.to_numpy() + rural_tertiary_stock.to_numpy()  + rural_unp_cycle_stock.to_numpy() + rural_unp_informal_stock.to_numpy() + rural_unp_local_stock.to_numpy()  + rural_unp_pedestrian_stock.to_numpy() + rural_unp_primary_stock.to_numpy() + rural_unp_secondary_stock.to_numpy() + rural_unp_tertiary_stock.to_numpy()# kgs in stock for lines
infra_materials.loc[idx['stock', :,:],idx[:,'urban_bridges']]          =  urban_b_cycle_stock.to_numpy() + urban_b_informal_stock.to_numpy() + urban_b_local_stock.to_numpy() + urban_b_motorway_stock.to_numpy() + urban_b_pedestrian_stock.to_numpy() + urban_b_primary_stock.to_numpy() + urban_b_secondary_stock.to_numpy() + urban_b_tertiary_stock.to_numpy() # kgs in stock for lines
infra_materials.loc[idx['stock', :,:],idx[:,'urban_tunnels']]          =  urban_t_cycle_stock.to_numpy() + urban_t_informal_stock.to_numpy() + urban_t_local_stock.to_numpy() + urban_t_motorway_stock.to_numpy() + urban_t_pedestrian_stock.to_numpy() + urban_t_primary_stock.to_numpy() + urban_t_secondary_stock.to_numpy() + urban_t_tertiary_stock.to_numpy()  # kgs in stock for lines
infra_materials.loc[idx['stock', :,:],idx[:,'rural_bridges']]          =  rural_b_cycle_stock.to_numpy() + rural_b_informal_stock.to_numpy() + rural_b_local_stock.to_numpy() + rural_b_motorway_stock.to_numpy() + rural_b_pedestrian_stock.to_numpy() + rural_b_primary_stock.to_numpy() + rural_b_secondary_stock.to_numpy() + rural_b_tertiary_stock.to_numpy()  # kgs in stock for lines
infra_materials.loc[idx['stock', :,:],idx[:,'rural_tunnels']]          =  rural_t_cycle_stock.to_numpy() + rural_t_informal_stock.to_numpy() + rural_t_local_stock.to_numpy() + rural_t_motorway_stock.to_numpy() + rural_t_pedestrian_stock.to_numpy() + rural_t_primary_stock.to_numpy() + rural_t_secondary_stock.to_numpy() + rural_t_tertiary_stock.to_numpy()  # kgs in stock for lines
infra_materials.loc[idx['stock', :,:],idx[:,'total_bridges']]          =  urban_b_cycle_stock.to_numpy() + urban_b_informal_stock.to_numpy() + urban_b_local_stock.to_numpy() + urban_b_motorway_stock.to_numpy() + urban_b_pedestrian_stock.to_numpy() + urban_b_primary_stock.to_numpy() + urban_b_secondary_stock.to_numpy() + urban_b_tertiary_stock.to_numpy() + rural_b_cycle_stock.to_numpy() + rural_b_informal_stock.to_numpy() + rural_b_local_stock.to_numpy() + rural_b_motorway_stock.to_numpy() + rural_b_pedestrian_stock.to_numpy() + rural_b_primary_stock.to_numpy() + rural_b_secondary_stock.to_numpy() + rural_b_tertiary_stock.to_numpy()# kgs in stock for lines
infra_materials.loc[idx['stock', :,:],idx[:,'total_tunnels']]          =  urban_t_cycle_stock.to_numpy() + urban_t_informal_stock.to_numpy() + urban_t_local_stock.to_numpy() + urban_t_motorway_stock.to_numpy() + urban_t_pedestrian_stock.to_numpy() + urban_t_primary_stock.to_numpy() + urban_t_secondary_stock.to_numpy() + urban_t_tertiary_stock.to_numpy() + rural_t_cycle_stock.to_numpy() + rural_t_informal_stock.to_numpy() + rural_t_local_stock.to_numpy() + rural_t_motorway_stock.to_numpy() + rural_t_pedestrian_stock.to_numpy() + rural_t_primary_stock.to_numpy() + rural_t_secondary_stock.to_numpy() + rural_t_tertiary_stock.to_numpy() # kgs in stock for lines
infra_materials.loc[idx['stock', :,:],idx[:,'total_rail']]   =  rail_stock.to_numpy() # kgs in stock for transformers
infra_materials.loc[idx['stock', :,:],idx[:,'total_urban_rail']]    =  urban_rail_stock.to_numpy() # kgs in stock for substations
infra_materials.loc[idx['stock', :,:],idx[:,'total_HST']]   =  HST_stock.to_numpy() # kgs in stock for transformers

infra_materials.loc[idx['stock', :,:],idx[:,'rail_bridge']]   =  rail_bridge_stock.to_numpy() # kgs in stock for transformers
infra_materials.loc[idx['stock', :,:],idx[:,'rail_tunnel']]    =  rail_tunnel_stock.to_numpy() # kgs in stock for substations

infra_materials.loc[idx['stock', :,:],idx[:,'urban_parking']]   =  urban_parking_stock.to_numpy() + urban_unp_parking_stock  # kgs in stock for transformers
infra_materials.loc[idx['stock', :,:],idx[:,'rural_parking']]    =  rural_parking_stock.to_numpy() + rural_unp_parking_stock # kgs in stock for substations

infra_materials.loc[idx['stock', :,:],idx[:,'hibernating_brid_tun']]   = obsolete_urban_b_cycle_stock.to_numpy() + obsolete_urban_b_informal_stock.to_numpy() + obsolete_urban_b_local_stock.to_numpy() + obsolete_urban_b_motorway_stock.to_numpy() + obsolete_urban_b_pedestrian_stock.to_numpy() + obsolete_urban_b_primary_stock.to_numpy() + obsolete_urban_b_secondary_stock.to_numpy() + obsolete_urban_b_tertiary_stock.to_numpy() +obsolete_urban_t_cycle_stock.to_numpy() + obsolete_urban_t_informal_stock.to_numpy() + obsolete_urban_t_local_stock.to_numpy() + obsolete_urban_t_motorway_stock.to_numpy() + obsolete_urban_t_pedestrian_stock.to_numpy() + obsolete_urban_t_primary_stock.to_numpy() + obsolete_urban_t_secondary_stock.to_numpy() + obsolete_urban_t_tertiary_stock.to_numpy() + obsolete_rural_b_cycle_stock.to_numpy() + obsolete_rural_b_informal_stock.to_numpy() + obsolete_rural_b_local_stock.to_numpy() + obsolete_rural_b_motorway_stock.to_numpy() + obsolete_rural_b_pedestrian_stock.to_numpy() + obsolete_rural_b_primary_stock.to_numpy() + obsolete_rural_b_secondary_stock.to_numpy() + obsolete_rural_b_tertiary_stock.to_numpy()+ obsolete_rural_t_cycle_stock.to_numpy() + obsolete_rural_t_informal_stock.to_numpy() + obsolete_rural_t_local_stock.to_numpy() + obsolete_rural_t_motorway_stock.to_numpy() + obsolete_rural_t_pedestrian_stock.to_numpy() + obsolete_rural_t_primary_stock.to_numpy() + obsolete_rural_t_secondary_stock.to_numpy() + obsolete_rural_t_tertiary_stock.to_numpy()
infra_materials.loc[idx['stock', :,:],idx[:,'hibernating_roads']]   = obsolete_urban_cycle_stock.to_numpy() + obsolete_urban_informal_stock.to_numpy() + obsolete_urban_local_stock.to_numpy() + obsolete_urban_motorway_stock.to_numpy() + obsolete_urban_pedestrian_stock.to_numpy() + obsolete_urban_primary_stock.to_numpy() + obsolete_urban_secondary_stock.to_numpy() + obsolete_urban_tertiary_stock.to_numpy() +obsolete_urban_unp_cycle_stock.to_numpy() + obsolete_urban_unp_informal_stock.to_numpy() + obsolete_urban_unp_local_stock.to_numpy()  + obsolete_urban_unp_pedestrian_stock.to_numpy() + obsolete_urban_unp_primary_stock.to_numpy() + obsolete_urban_unp_secondary_stock.to_numpy() + obsolete_urban_unp_tertiary_stock.to_numpy() + obsolete_rural_cycle_stock.to_numpy() + obsolete_rural_informal_stock.to_numpy() + obsolete_rural_local_stock.to_numpy() + obsolete_rural_motorway_stock.to_numpy() + obsolete_rural_pedestrian_stock.to_numpy() + obsolete_rural_primary_stock.to_numpy() + obsolete_rural_secondary_stock.to_numpy() + obsolete_rural_tertiary_stock.to_numpy()+ obsolete_rural_unp_cycle_stock.to_numpy() + obsolete_rural_unp_informal_stock.to_numpy() + obsolete_rural_unp_local_stock.to_numpy()  + obsolete_rural_unp_pedestrian_stock.to_numpy() + obsolete_rural_unp_primary_stock.to_numpy() + obsolete_rural_unp_secondary_stock.to_numpy() + obsolete_rural_unp_tertiary_stock.to_numpy()

infra_materials.loc[idx['stock', :,:],idx[:,'hibernating_rail']]   = rail_obs_stock.to_numpy() + rail_brid_obs_stock.to_numpy() + rail_tunn_obs_stock.to_numpy()



infra_materials.loc[idx['inflow', :,:],idx[:,'urban_paved_roads']]          =  urban_cycle_in.to_numpy() + urban_informal_in.to_numpy() + urban_local_in.to_numpy() + urban_motorway_in.to_numpy() + urban_pedestrian_in.to_numpy() + urban_primary_in.to_numpy() + urban_secondary_in.to_numpy() + urban_tertiary_in.to_numpy()+ diff_u_area_paved_cycle.to_numpy()+ diff_u_area_paved_informal.to_numpy()+ diff_u_area_paved_local.to_numpy()+ diff_u_area_paved_motorway.to_numpy() + diff_u_area_paved_pedestrian.to_numpy() + diff_u_area_paved_primary.to_numpy() + diff_u_area_paved_secondary.to_numpy() + diff_u_area_paved_tertiary.to_numpy()
  # kgs in stock for lines
infra_materials.loc[idx['inflow', :,:],idx[:,'urban_unpaved_roads']]          =  urban_unp_cycle_in.to_numpy() + urban_unp_informal_in.to_numpy() + urban_unp_local_in.to_numpy() + urban_unp_pedestrian_in.to_numpy() + urban_unp_primary_in.to_numpy() + urban_unp_secondary_in.to_numpy() + urban_unp_tertiary_in.to_numpy()+ diff_u_area_unpaved_cycle.to_numpy() + diff_u_area_unpaved_informal.to_numpy() + diff_u_area_unpaved_local.to_numpy() + diff_u_area_unpaved_pedestrian.to_numpy() + diff_u_area_unpaved_primary.to_numpy() + diff_u_area_unpaved_secondary.to_numpy() + diff_u_area_unpaved_tertiary.to_numpy()
  # kgs in stock for lines
infra_materials.loc[idx['inflow', :,:],idx[:,'rural_paved_roads']]          =  rural_cycle_in.to_numpy() + rural_informal_in.to_numpy() + rural_local_in.to_numpy() + rural_motorway_in.to_numpy() + rural_pedestrian_in.to_numpy() + rural_primary_in.to_numpy() + rural_secondary_in.to_numpy() + rural_tertiary_in.to_numpy() + diff_r_area_paved_cycle.to_numpy()+ diff_r_area_paved_informal.to_numpy()+ diff_r_area_paved_local.to_numpy()+ diff_r_area_paved_motorway.to_numpy()+ diff_r_area_paved_pedestrian.to_numpy()+ diff_r_area_paved_primary.to_numpy() + diff_r_area_paved_secondary.to_numpy()+ diff_r_area_paved_tertiary.to_numpy()
  # kgs in stock for lines
infra_materials.loc[idx['inflow', :,:],idx[:,'rural_unpaved_roads']]          =  rural_unp_cycle_in.to_numpy() + rural_unp_informal_in.to_numpy() + rural_unp_local_in.to_numpy()  + rural_unp_pedestrian_in.to_numpy() + rural_unp_primary_in.to_numpy() + rural_unp_secondary_in.to_numpy() + rural_unp_tertiary_in.to_numpy()+ diff_r_area_unpaved_cycle.to_numpy() + diff_r_area_unpaved_informal.to_numpy()+ diff_r_area_unpaved_local.to_numpy()+ diff_r_area_unpaved_pedestrian.to_numpy()+ diff_r_area_unpaved_primary.to_numpy()+ diff_r_area_unpaved_secondary.to_numpy()+ diff_r_area_unpaved_tertiary.to_numpy()
 # kgs in stock for lines
infra_materials.loc[idx['inflow', :,:],idx[:,'urban_roads']]          =  urban_cycle_in.to_numpy() + urban_informal_in.to_numpy() + urban_local_in.to_numpy() + urban_motorway_in.to_numpy() + urban_pedestrian_in.to_numpy() + urban_primary_in.to_numpy() + urban_secondary_in.to_numpy() + urban_tertiary_in.to_numpy()  + urban_unp_cycle_in.to_numpy() + urban_unp_informal_in.to_numpy() + urban_unp_local_in.to_numpy() + urban_unp_pedestrian_in.to_numpy() + urban_unp_primary_in.to_numpy() + urban_unp_secondary_in.to_numpy() + urban_unp_tertiary_in.to_numpy()  # kgs in stock for lines
infra_materials.loc[idx['inflow', :,:],idx[:,'rural_roads']]          =  rural_cycle_in.to_numpy() + rural_informal_in.to_numpy() + rural_local_in.to_numpy() + rural_motorway_in.to_numpy() + rural_pedestrian_in.to_numpy() + rural_primary_in.to_numpy() + rural_secondary_in.to_numpy() + rural_tertiary_in.to_numpy()  + rural_unp_cycle_in.to_numpy() + rural_unp_informal_in.to_numpy() + rural_unp_local_in.to_numpy()  + rural_unp_pedestrian_in.to_numpy() + rural_unp_primary_in.to_numpy() + rural_unp_secondary_in.to_numpy() + rural_unp_tertiary_in.to_numpy()  # kgs in stock for lines
infra_materials.loc[idx['inflow', :,:],idx[:,'urban_bridges']]          =  urban_b_cycle_in.to_numpy() + urban_b_informal_in.to_numpy() + urban_b_local_in.to_numpy() + urban_b_motorway_in.to_numpy() + urban_b_pedestrian_in.to_numpy() + urban_b_primary_in.to_numpy() + urban_b_secondary_in.to_numpy() + urban_b_tertiary_in.to_numpy() # kgs in stock for lines
infra_materials.loc[idx['inflow', :,:],idx[:,'urban_tunnels']]          =  urban_t_cycle_in.to_numpy() + urban_t_informal_in.to_numpy() + urban_t_local_in.to_numpy() + urban_t_motorway_in.to_numpy() + urban_t_pedestrian_in.to_numpy() + urban_t_primary_in.to_numpy() + urban_t_secondary_in.to_numpy() + urban_t_tertiary_in.to_numpy()  # kgs in stock for lines
infra_materials.loc[idx['inflow', :,:],idx[:,'rural_bridges']]          =  rural_b_cycle_in.to_numpy() + rural_b_informal_in.to_numpy() + rural_b_local_in.to_numpy() + rural_b_motorway_in.to_numpy() + rural_b_pedestrian_in.to_numpy() + rural_b_primary_in.to_numpy() + rural_b_secondary_in.to_numpy() + rural_b_tertiary_in.to_numpy()  # kgs in stock for lines
infra_materials.loc[idx['inflow', :,:],idx[:,'rural_tunnels']]          =  rural_t_cycle_in.to_numpy() + rural_t_informal_in.to_numpy() + rural_t_local_in.to_numpy() + rural_t_motorway_in.to_numpy() + rural_t_pedestrian_in.to_numpy() + rural_t_primary_in.to_numpy() + rural_t_secondary_in.to_numpy() + rural_t_tertiary_in.to_numpy()  # kgs in stock for lines
infra_materials.loc[idx['inflow', :,:],idx[:,'total_bridges']]          =  urban_b_cycle_in.to_numpy() + urban_b_informal_in.to_numpy() + urban_b_local_in.to_numpy() + urban_b_motorway_in.to_numpy() + urban_b_pedestrian_in.to_numpy() + urban_b_primary_in.to_numpy() + urban_b_secondary_in.to_numpy() + urban_b_tertiary_in.to_numpy() + rural_b_cycle_in.to_numpy() + rural_b_informal_in.to_numpy() + rural_b_local_in.to_numpy() + rural_b_motorway_in.to_numpy() + rural_b_pedestrian_in.to_numpy() + rural_b_primary_in.to_numpy() + rural_b_secondary_in.to_numpy() + rural_b_tertiary_in.to_numpy()# kgs in stock for lines
infra_materials.loc[idx['inflow', :,:],idx[:,'total_tunnels']]          =  urban_t_cycle_in.to_numpy() + urban_t_informal_in.to_numpy() + urban_t_local_in.to_numpy() + urban_t_motorway_in.to_numpy() + urban_t_pedestrian_in.to_numpy() + urban_t_primary_in.to_numpy() + urban_t_secondary_in.to_numpy() + urban_t_tertiary_in.to_numpy() + rural_t_cycle_in.to_numpy() + rural_t_informal_in.to_numpy() + rural_t_local_in.to_numpy() + rural_t_motorway_in.to_numpy() + rural_t_pedestrian_in.to_numpy() + rural_t_primary_in.to_numpy() + rural_t_secondary_in.to_numpy() + rural_t_tertiary_in.to_numpy() # kgs in stock for lines
infra_materials.loc[idx['inflow', :,:],idx[:,'total_rail']]   =  rail_in.to_numpy() # kgs in stock for transformers
infra_materials.loc[idx['inflow', :,:],idx[:,'total_urban_rail']]    =  urban_rail_in.to_numpy() # kgs in stock for substations
infra_materials.loc[idx['inflow', :,:],idx[:,'total_HST']]   =  HST_in.to_numpy() # kgs in stock for transformers


infra_materials.loc[idx['inflow', :,:],idx[:,'urban_parking']]   =  urban_parking_in.to_numpy() + urban_unp_parking_in  # kgs in stock for transformers
infra_materials.loc[idx['inflow', :,:],idx[:,'rural_parking']]    =  rural_parking_in.to_numpy() + rural_unp_parking_in # kgs in stock for substations

infra_materials.loc[idx['inflow', :,:],idx[:,'rail_bridge']]   =  rail_bridge_in.to_numpy() # kgs in stock for transformers
infra_materials.loc[idx['inflow', :,:],idx[:,'rail_tunnel']]    =  rail_tunnel_in.to_numpy() # kgs in stock for substations


infra_materials.loc[idx['outflow', :,:],idx[:,'urban_paved_roads']]          =  urban_cycle_out.to_numpy() + urban_informal_out.to_numpy() + urban_local_out.to_numpy() + urban_motorway_out.to_numpy() + urban_pedestrian_out.to_numpy() + urban_primary_out.to_numpy() + urban_secondary_out.to_numpy() + urban_tertiary_out.to_numpy() + urban_parking_out.to_numpy() # kgs in stock for lines
infra_materials.loc[idx['outflow', :,:],idx[:,'urban_unpaved_roads']]          =  urban_unp_cycle_out.to_numpy() + urban_unp_informal_out.to_numpy() + urban_unp_local_out.to_numpy() + urban_unp_pedestrian_out.to_numpy() + urban_unp_primary_out.to_numpy() + urban_unp_secondary_out.to_numpy() + urban_unp_tertiary_out.to_numpy() + urban_unp_parking_out.to_numpy() # kgs in stock for lines
infra_materials.loc[idx['outflow', :,:],idx[:,'rural_paved_roads']]          =  rural_cycle_out.to_numpy() + rural_informal_out.to_numpy() + rural_local_out.to_numpy() + rural_motorway_out.to_numpy() + rural_pedestrian_out.to_numpy() + rural_primary_out.to_numpy() + rural_secondary_out.to_numpy() + rural_tertiary_out.to_numpy() + rural_parking_out.to_numpy() # kgs in stock for lines
infra_materials.loc[idx['outflow', :,:],idx[:,'rural_unpaved_roads']]          =  rural_unp_cycle_out.to_numpy() + rural_unp_informal_out.to_numpy() + rural_unp_local_out.to_numpy()  + rural_unp_pedestrian_out.to_numpy() + rural_unp_primary_out.to_numpy() + rural_unp_secondary_out.to_numpy() + rural_unp_tertiary_out.to_numpy() + rural_unp_parking_out.to_numpy() # kgs in stock for lines
infra_materials.loc[idx['outflow', :,:],idx[:,'urban_roads']]          =  urban_cycle_out.to_numpy() + urban_informal_out.to_numpy() + urban_local_out.to_numpy() + urban_motorway_out.to_numpy() + urban_pedestrian_out.to_numpy() + urban_primary_out.to_numpy() + urban_secondary_out.to_numpy() + urban_tertiary_out.to_numpy() + urban_parking_out.to_numpy() + urban_unp_cycle_out.to_numpy() + urban_unp_informal_out.to_numpy() + urban_unp_local_out.to_numpy() + urban_unp_pedestrian_out.to_numpy() + urban_unp_primary_out.to_numpy() + urban_unp_secondary_out.to_numpy() + urban_unp_tertiary_out.to_numpy() + urban_unp_parking_out.to_numpy() # kgs in stock for lines
infra_materials.loc[idx['outflow', :,:],idx[:,'rural_roads']]          =  rural_cycle_out.to_numpy() + rural_informal_out.to_numpy() + rural_local_out.to_numpy() + rural_motorway_out.to_numpy() + rural_pedestrian_out.to_numpy() + rural_primary_out.to_numpy() + rural_secondary_out.to_numpy() + rural_tertiary_out.to_numpy() + rural_parking_out.to_numpy() + rural_unp_cycle_out.to_numpy() + rural_unp_informal_out.to_numpy() + rural_unp_local_out.to_numpy()  + rural_unp_pedestrian_out.to_numpy() + rural_unp_primary_out.to_numpy() + rural_unp_secondary_out.to_numpy() + rural_unp_tertiary_out.to_numpy() + rural_unp_parking_out.to_numpy() # kgs in stock for lines
infra_materials.loc[idx['outflow', :,:],idx[:,'urban_bridges']]          =  urban_b_cycle_out.to_numpy() + urban_b_informal_out.to_numpy() + urban_b_local_out.to_numpy() + urban_b_motorway_out.to_numpy() + urban_b_pedestrian_out.to_numpy() + urban_b_primary_out.to_numpy() + urban_b_secondary_out.to_numpy() + urban_b_tertiary_out.to_numpy() # kgs in stock for lines
infra_materials.loc[idx['outflow', :,:],idx[:,'urban_tunnels']]          =  urban_t_cycle_out.to_numpy() + urban_t_informal_out.to_numpy() + urban_t_local_out.to_numpy() + urban_t_motorway_out.to_numpy() + urban_t_pedestrian_out.to_numpy() + urban_t_primary_out.to_numpy() + urban_t_secondary_out.to_numpy() + urban_t_tertiary_out.to_numpy()  # kgs in stock for lines
infra_materials.loc[idx['outflow', :,:],idx[:,'rural_bridges']]          =  rural_b_cycle_out.to_numpy() + rural_b_informal_out.to_numpy() + rural_b_local_out.to_numpy() + rural_b_motorway_out.to_numpy() + rural_b_pedestrian_out.to_numpy() + rural_b_primary_out.to_numpy() + rural_b_secondary_out.to_numpy() + rural_b_tertiary_out.to_numpy()  # kgs in stock for lines
infra_materials.loc[idx['outflow', :,:],idx[:,'rural_tunnels']]          =  rural_t_cycle_out.to_numpy() + rural_t_informal_out.to_numpy() + rural_t_local_out.to_numpy() + rural_t_motorway_out.to_numpy() + rural_t_pedestrian_out.to_numpy() + rural_t_primary_out.to_numpy() + rural_t_secondary_out.to_numpy() + rural_t_tertiary_out.to_numpy()  # kgs in stock for lines
infra_materials.loc[idx['outflow', :,:],idx[:,'total_bridges']]          =  urban_b_cycle_out.to_numpy() + urban_b_informal_out.to_numpy() + urban_b_local_out.to_numpy() + urban_b_motorway_out.to_numpy() + urban_b_pedestrian_out.to_numpy() + urban_b_primary_out.to_numpy() + urban_b_secondary_out.to_numpy() + urban_b_tertiary_out.to_numpy() + rural_b_cycle_out.to_numpy() + rural_b_informal_out.to_numpy() + rural_b_local_out.to_numpy() + rural_b_motorway_out.to_numpy() + rural_b_pedestrian_out.to_numpy() + rural_b_primary_out.to_numpy() + rural_b_secondary_out.to_numpy() + rural_b_tertiary_out.to_numpy()# kgs in stock for lines
infra_materials.loc[idx['outflow', :,:],idx[:,'total_tunnels']]          =  urban_t_cycle_out.to_numpy() + urban_t_informal_out.to_numpy() + urban_t_local_out.to_numpy() + urban_t_motorway_out.to_numpy() + urban_t_pedestrian_out.to_numpy() + urban_t_primary_out.to_numpy() + urban_t_secondary_out.to_numpy() + urban_t_tertiary_out.to_numpy() + rural_t_cycle_out.to_numpy() + rural_t_informal_out.to_numpy() + rural_t_local_out.to_numpy() + rural_t_motorway_out.to_numpy() + rural_t_pedestrian_out.to_numpy() + rural_t_primary_out.to_numpy() + rural_t_secondary_out.to_numpy() + rural_t_tertiary_out.to_numpy() # kgs in stock for lines
infra_materials.loc[idx['outflow', :,:],idx[:,'total_rail']]   =  rail_out.to_numpy() # kgs in stock for transformers
infra_materials.loc[idx['outflow', :,:],idx[:,'total_urban_rail']]    =  urban_rail_out.to_numpy() # kgs in stock for substations
infra_materials.loc[idx['outflow', :,:],idx[:,'total_HST']]    =  HST_out.to_numpy() # kgs in stock for substations


infra_materials.loc[idx['outflow', :,:],idx[:,'urban_parking']]   =  urban_parking_out.to_numpy() + urban_unp_parking_out  # kgs in stock for transformers
infra_materials.loc[idx['outflow', :,:],idx[:,'rural_parking']]    =  rural_parking_out.to_numpy() + rural_unp_parking_out # kgs in stock for substations

infra_materials.loc[idx['outflow', :,:],idx[:,'rail_bridge']]   =  rail_bridge_out.to_numpy() # kgs in stock for transformers
infra_materials.loc[idx['outflow', :,:],idx[:,'rail_tunnel']]    =  rail_tunnel_out.to_numpy() # kgs in stock for substations



############OUTPUT FOR SCENARIOS#####################
infra_materials_scenario.loc[idx['stock', :,:],idx[:,'roads']]          =  urban_cycle_stock.to_numpy() + urban_informal_stock.to_numpy() + urban_local_stock.to_numpy() + urban_motorway_stock.to_numpy() + urban_pedestrian_stock.to_numpy() + urban_primary_stock.to_numpy() + urban_secondary_stock.to_numpy() + urban_tertiary_stock.to_numpy() + urban_unp_cycle_stock.to_numpy() + urban_unp_informal_stock.to_numpy() + urban_unp_local_stock.to_numpy() + urban_unp_pedestrian_stock.to_numpy() + urban_unp_primary_stock.to_numpy() + urban_unp_secondary_stock.to_numpy() + urban_unp_tertiary_stock.to_numpy()+ rural_cycle_stock.to_numpy() + rural_informal_stock.to_numpy() + rural_local_stock.to_numpy() + rural_motorway_stock.to_numpy() + rural_pedestrian_stock.to_numpy() + rural_primary_stock.to_numpy() + rural_secondary_stock.to_numpy() + rural_tertiary_stock.to_numpy()  + rural_unp_cycle_stock.to_numpy() + rural_unp_informal_stock.to_numpy() + rural_unp_local_stock.to_numpy()  + rural_unp_pedestrian_stock.to_numpy() + rural_unp_primary_stock.to_numpy() + rural_unp_secondary_stock.to_numpy() + rural_unp_tertiary_stock.to_numpy() + new_total_r.to_numpy() + new_total_u.to_numpy() # kgs in stock for lines
infra_materials_scenario.loc[idx['stock', :,:],idx[:,'bridges']]          =  urban_b_cycle_stock.to_numpy() + urban_b_informal_stock.to_numpy() + urban_b_local_stock.to_numpy() + urban_b_motorway_stock.to_numpy() + urban_b_pedestrian_stock.to_numpy() + urban_b_primary_stock.to_numpy() + urban_b_secondary_stock.to_numpy() + urban_b_tertiary_stock.to_numpy() + rural_b_cycle_stock.to_numpy() + rural_b_informal_stock.to_numpy() + rural_b_local_stock.to_numpy() + rural_b_motorway_stock.to_numpy() + rural_b_pedestrian_stock.to_numpy() + rural_b_primary_stock.to_numpy() + rural_b_secondary_stock.to_numpy() + rural_b_tertiary_stock.to_numpy()# kgs in stock for lines
infra_materials_scenario.loc[idx['stock', :,:],idx[:,'tunnels']]          =  urban_t_cycle_stock.to_numpy() + urban_t_informal_stock.to_numpy() + urban_t_local_stock.to_numpy() + urban_t_motorway_stock.to_numpy() + urban_t_pedestrian_stock.to_numpy() + urban_t_primary_stock.to_numpy() + urban_t_secondary_stock.to_numpy() + urban_t_tertiary_stock.to_numpy() + rural_t_cycle_stock.to_numpy() + rural_t_informal_stock.to_numpy() + rural_t_local_stock.to_numpy() + rural_t_motorway_stock.to_numpy() + rural_t_pedestrian_stock.to_numpy() + rural_t_primary_stock.to_numpy() + rural_t_secondary_stock.to_numpy() + rural_t_tertiary_stock.to_numpy() # kgs in stock for lines
infra_materials_scenario.loc[idx['stock', :,:],idx[:,'rail']]   =  rail_stock.to_numpy() +urban_rail_stock.to_numpy()  + rail_bridge_stock.to_numpy() + rail_tunnel_stock.to_numpy() + HST_stock.to_numpy()   # kgs in stock for transformers
infra_materials_scenario.loc[idx['stock', :,:],idx[:,'parking']]   =  urban_parking_stock.to_numpy() + urban_unp_parking_stock.to_numpy() + rural_parking_stock.to_numpy() + rural_unp_parking_stock.to_numpy()  # kgs in stock for transformers

infra_materials_scenario.loc[idx['stock', :,:],idx[:,'obsolete']]   =  obsolete_urban_b_cycle_stock.to_numpy() + obsolete_urban_b_informal_stock.to_numpy() + obsolete_urban_b_local_stock.to_numpy() + obsolete_urban_b_motorway_stock.to_numpy() + obsolete_urban_b_pedestrian_stock.to_numpy() + obsolete_urban_b_primary_stock.to_numpy() + obsolete_urban_b_secondary_stock.to_numpy() + obsolete_urban_b_tertiary_stock.to_numpy() +obsolete_urban_t_cycle_stock.to_numpy() + obsolete_urban_t_informal_stock.to_numpy() + obsolete_urban_t_local_stock.to_numpy() + obsolete_urban_t_motorway_stock.to_numpy() + obsolete_urban_t_pedestrian_stock.to_numpy() + obsolete_urban_t_primary_stock.to_numpy() + obsolete_urban_t_secondary_stock.to_numpy() + obsolete_urban_t_tertiary_stock.to_numpy() + obsolete_rural_b_cycle_stock.to_numpy() + obsolete_rural_b_informal_stock.to_numpy() + obsolete_rural_b_local_stock.to_numpy() + obsolete_rural_b_motorway_stock.to_numpy() + obsolete_rural_b_pedestrian_stock.to_numpy() + obsolete_rural_b_primary_stock.to_numpy() + obsolete_rural_b_secondary_stock.to_numpy() + obsolete_rural_b_tertiary_stock.to_numpy()+ obsolete_rural_t_cycle_stock.to_numpy() + obsolete_rural_t_informal_stock.to_numpy() + obsolete_rural_t_local_stock.to_numpy() + obsolete_rural_t_motorway_stock.to_numpy() + obsolete_rural_t_pedestrian_stock.to_numpy() + obsolete_rural_t_primary_stock.to_numpy() + obsolete_rural_t_secondary_stock.to_numpy() + obsolete_rural_t_tertiary_stock.to_numpy()
+ obsolete_urban_cycle_stock.to_numpy() + obsolete_urban_informal_stock.to_numpy() + obsolete_urban_local_stock.to_numpy() + obsolete_urban_motorway_stock.to_numpy() + obsolete_urban_pedestrian_stock.to_numpy() + obsolete_urban_primary_stock.to_numpy() + obsolete_urban_secondary_stock.to_numpy() + obsolete_urban_tertiary_stock.to_numpy() +obsolete_urban_unp_cycle_stock.to_numpy() + obsolete_urban_unp_informal_stock.to_numpy() + obsolete_urban_unp_local_stock.to_numpy()  + obsolete_urban_unp_pedestrian_stock.to_numpy() + obsolete_urban_unp_primary_stock.to_numpy() + obsolete_urban_unp_secondary_stock.to_numpy() + obsolete_urban_unp_tertiary_stock.to_numpy() + obsolete_rural_cycle_stock.to_numpy() + obsolete_rural_informal_stock.to_numpy() + obsolete_rural_local_stock.to_numpy() + obsolete_rural_motorway_stock.to_numpy() + obsolete_rural_pedestrian_stock.to_numpy() + obsolete_rural_primary_stock.to_numpy() + obsolete_rural_secondary_stock.to_numpy() + obsolete_rural_tertiary_stock.to_numpy()+ obsolete_rural_unp_cycle_stock.to_numpy() + obsolete_rural_unp_informal_stock.to_numpy() + obsolete_rural_unp_local_stock.to_numpy()  + obsolete_rural_unp_pedestrian_stock.to_numpy() + obsolete_rural_unp_primary_stock.to_numpy() + obsolete_rural_unp_secondary_stock.to_numpy() + obsolete_rural_unp_tertiary_stock.to_numpy()
+rail_obs_stock.to_numpy() + rail_brid_obs_stock.to_numpy() + rail_tunn_obs_stock.to_numpy()


infra_materials_scenario.loc[idx['inflow', :,:],idx[:,'roads']]          =  urban_cycle_in.to_numpy() + urban_informal_in.to_numpy() + urban_local_in.to_numpy() + urban_motorway_in.to_numpy() + urban_pedestrian_in.to_numpy() + urban_primary_in.to_numpy() + urban_secondary_in.to_numpy() + urban_tertiary_in.to_numpy() + urban_unp_cycle_in.to_numpy() + urban_unp_informal_in.to_numpy() + urban_unp_local_in.to_numpy() + urban_unp_pedestrian_in.to_numpy() + urban_unp_primary_in.to_numpy() + urban_unp_secondary_in.to_numpy() + urban_unp_tertiary_in.to_numpy() + rural_cycle_in.to_numpy() + rural_informal_in.to_numpy() + rural_local_in.to_numpy() + rural_motorway_in.to_numpy() + rural_pedestrian_in.to_numpy() + rural_primary_in.to_numpy() + rural_secondary_in.to_numpy() + rural_tertiary_in.to_numpy()  + rural_unp_cycle_in.to_numpy() + rural_unp_informal_in.to_numpy() + rural_unp_local_in.to_numpy()  + rural_unp_pedestrian_in.to_numpy() + rural_unp_primary_in.to_numpy() + rural_unp_secondary_in.to_numpy() + rural_unp_tertiary_in.to_numpy() + diff_total_u.to_numpy() + diff_total_r.to_numpy()  # kgs in stock for lines
infra_materials_scenario.loc[idx['inflow', :,:],idx[:,'bridges']]          =  urban_b_cycle_in.to_numpy() + urban_b_informal_in.to_numpy() + urban_b_local_in.to_numpy() + urban_b_motorway_in.to_numpy() + urban_b_pedestrian_in.to_numpy() + urban_b_primary_in.to_numpy() + urban_b_secondary_in.to_numpy() + urban_b_tertiary_in.to_numpy() + rural_b_cycle_in.to_numpy() + rural_b_informal_in.to_numpy() + rural_b_local_in.to_numpy() + rural_b_motorway_in.to_numpy() + rural_b_pedestrian_in.to_numpy() + rural_b_primary_in.to_numpy() + rural_b_secondary_in.to_numpy() + rural_b_tertiary_in.to_numpy()# kgs in stock for lines
infra_materials_scenario.loc[idx['inflow', :,:],idx[:,'tunnels']]          =  urban_t_cycle_in.to_numpy() + urban_t_informal_in.to_numpy() + urban_t_local_in.to_numpy() + urban_t_motorway_in.to_numpy() + urban_t_pedestrian_in.to_numpy() + urban_t_primary_in.to_numpy() + urban_t_secondary_in.to_numpy() + urban_t_tertiary_in.to_numpy() + rural_t_cycle_in.to_numpy() + rural_t_informal_in.to_numpy() + rural_t_local_in.to_numpy() + rural_t_motorway_in.to_numpy() + rural_t_pedestrian_in.to_numpy() + rural_t_primary_in.to_numpy() + rural_t_secondary_in.to_numpy() + rural_t_tertiary_in.to_numpy() # kgs in stock for lines
infra_materials_scenario.loc[idx['inflow', :,:],idx[:,'rail']]   =  rail_in.to_numpy() +urban_rail_in.to_numpy() + rail_bridge_in.to_numpy() +  rail_tunnel_in.to_numpy() + HST_in.to_numpy() # kgs in stock for transformers
infra_materials_scenario.loc[idx['inflow', :,:],idx[:,'parking']]   =  urban_parking_in.to_numpy() + urban_unp_parking_in.to_numpy() + rural_parking_in.to_numpy() + rural_unp_parking_in.to_numpy()  # kgs in stock for transformers


infra_materials_scenario.loc[idx['outflow', :,:],idx[:,'roads']]          =  urban_cycle_out.to_numpy() + urban_informal_out.to_numpy() + urban_local_out.to_numpy() + urban_motorway_out.to_numpy() + urban_pedestrian_out.to_numpy() + urban_primary_out.to_numpy() + urban_secondary_out.to_numpy() + urban_tertiary_out.to_numpy() + urban_unp_cycle_out.to_numpy() + urban_unp_informal_out.to_numpy() + urban_unp_local_out.to_numpy() + urban_unp_pedestrian_out.to_numpy() + urban_unp_primary_out.to_numpy() + urban_unp_secondary_out.to_numpy() + urban_unp_tertiary_out.to_numpy()+ rural_cycle_out.to_numpy() + rural_informal_out.to_numpy() + rural_local_out.to_numpy() + rural_motorway_out.to_numpy() + rural_pedestrian_out.to_numpy() + rural_primary_out.to_numpy() + rural_secondary_out.to_numpy() + rural_tertiary_out.to_numpy()  + rural_unp_cycle_out.to_numpy() + rural_unp_informal_out.to_numpy() + rural_unp_local_out.to_numpy()  + rural_unp_pedestrian_out.to_numpy() + rural_unp_primary_out.to_numpy() + rural_unp_secondary_out.to_numpy() + rural_unp_tertiary_out.to_numpy()  # kgs in stock for lines
infra_materials_scenario.loc[idx['outflow', :,:],idx[:,'bridges']]          =  urban_b_cycle_out.to_numpy() + urban_b_informal_out.to_numpy() + urban_b_local_out.to_numpy() + urban_b_motorway_out.to_numpy() + urban_b_pedestrian_out.to_numpy() + urban_b_primary_out.to_numpy() + urban_b_secondary_out.to_numpy() + urban_b_tertiary_out.to_numpy() + rural_b_cycle_out.to_numpy() + rural_b_informal_out.to_numpy() + rural_b_local_out.to_numpy() + rural_b_motorway_out.to_numpy() + rural_b_pedestrian_out.to_numpy() + rural_b_primary_out.to_numpy() + rural_b_secondary_out.to_numpy() + rural_b_tertiary_out.to_numpy()# kgs in stock for lines
infra_materials_scenario.loc[idx['outflow', :,:],idx[:,'tunnels']]          =  urban_t_cycle_out.to_numpy() + urban_t_informal_out.to_numpy() + urban_t_local_out.to_numpy() + urban_t_motorway_out.to_numpy() + urban_t_pedestrian_out.to_numpy() + urban_t_primary_out.to_numpy() + urban_t_secondary_out.to_numpy() + urban_t_tertiary_out.to_numpy() + rural_t_cycle_out.to_numpy() + rural_t_informal_out.to_numpy() + rural_t_local_out.to_numpy() + rural_t_motorway_out.to_numpy() + rural_t_pedestrian_out.to_numpy() + rural_t_primary_out.to_numpy() + rural_t_secondary_out.to_numpy() + rural_t_tertiary_out.to_numpy() # kgs in stock for lines
infra_materials_scenario.loc[idx['outflow', :,:],idx[:,'rail']]   =  rail_out.to_numpy() +urban_rail_out.to_numpy() +  rail_bridge_out.to_numpy() + rail_tunnel_out.to_numpy() + HST_in.to_numpy() # kgs in stock for transformers
infra_materials_scenario.loc[idx['outflow', :,:],idx[:,'parking']]   =  urban_parking_out.to_numpy() + urban_unp_parking_out.to_numpy() + rural_parking_out.to_numpy() + rural_unp_parking_out.to_numpy()  # kgs in stock for transformers





# Assuming 'idx' is your indexing function for MultiIndex
# For each infrastructure type, assign values to their own category
# Urban and rural paved roads by category
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'urban_cycle_paved']] = urban_cycle_stock.to_numpy() + new_u_area_paved_cycle.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'urban_informal_paved']] = urban_informal_stock.to_numpy() + new_u_area_paved_informal.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'urban_local_paved']] = urban_local_stock.to_numpy() + new_u_area_paved_local.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'urban_motorway_paved']] = urban_motorway_stock.to_numpy() + new_u_area_paved_motorway.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'urban_pedestrian_paved']] = urban_pedestrian_stock.to_numpy() + new_u_area_paved_pedestrian.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'urban_primary_paved']] = urban_primary_stock.to_numpy() + new_u_area_paved_primary.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'urban_secondary_paved']] = urban_secondary_stock.to_numpy() + new_u_area_paved_secondary.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'urban_tertiary_paved']] = urban_tertiary_stock.to_numpy() + new_u_area_paved_tertiary.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'urban_parking_paved']] = urban_parking_stock.to_numpy() 

# Urban unpaved roads by category
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'urban_cycle_unpaved']] = urban_unp_cycle_stock.to_numpy() + new_u_area_unpaved_cycle.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'urban_informal_unpaved']] = urban_unp_informal_stock.to_numpy()  + new_u_area_unpaved_informal.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'urban_local_unpaved']] = urban_unp_local_stock.to_numpy()  + new_u_area_unpaved_local.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'urban_pedestrian_unpaved']] = urban_unp_pedestrian_stock.to_numpy()  + new_u_area_unpaved_pedestrian.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'urban_primary_unpaved']] = urban_unp_primary_stock.to_numpy()  + new_u_area_unpaved_primary.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'urban_secondary_unpaved']] = urban_unp_secondary_stock.to_numpy() + new_u_area_unpaved_secondary.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'urban_tertiary_unpaved']] = urban_unp_tertiary_stock.to_numpy() + new_u_area_unpaved_tertiary.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'urban_parking_unpaved']] = urban_unp_parking_stock.to_numpy() 

# Rural paved roads by category
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'rural_cycle_paved']] = rural_cycle_stock.to_numpy() + new_r_area_paved_cycle.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'rural_informal_paved']] = rural_informal_stock.to_numpy() + new_r_area_paved_informal.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'rural_local_paved']] = rural_local_stock.to_numpy() + new_r_area_paved_local.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'rural_motorway_paved']] = rural_motorway_stock.to_numpy() + new_r_area_paved_motorway.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'rural_pedestrian_paved']] = rural_pedestrian_stock.to_numpy() + new_r_area_paved_pedestrian.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'rural_primary_paved']] = rural_primary_stock.to_numpy() + new_r_area_paved_primary.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'rural_secondary_paved']] = rural_secondary_stock.to_numpy() + new_r_area_paved_secondary.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'rural_tertiary_paved']] = rural_tertiary_stock.to_numpy() + new_r_area_paved_tertiary.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'rural_parking_paved']] = rural_parking_stock.to_numpy() 

# Rural unpaved roads by category
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'rural_cycle_unpaved']] = rural_unp_cycle_stock.to_numpy() + new_r_area_unpaved_cycle.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'rural_informal_unpaved']] = rural_unp_informal_stock.to_numpy() + new_r_area_unpaved_informal.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'rural_local_unpaved']] = rural_unp_local_stock.to_numpy() + new_r_area_unpaved_local.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'rural_pedestrian_unpaved']] = rural_unp_pedestrian_stock.to_numpy() + new_r_area_unpaved_pedestrian.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'rural_primary_unpaved']] = rural_unp_primary_stock.to_numpy() + new_r_area_unpaved_primary.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'rural_secondary_unpaved']] = rural_unp_secondary_stock.to_numpy() + new_r_area_unpaved_secondary.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'rural_tertiary_unpaved']] = rural_unp_tertiary_stock.to_numpy() + new_r_area_unpaved_tertiary.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'rural_parking_unpaved']] = rural_unp_parking_stock.to_numpy() 

# Urban bridges and tunnels
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'urban_bridge_cycle']] = urban_b_cycle_stock.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'urban_bridge_informal']] = urban_b_informal_stock.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'urban_bridge_local']] = urban_b_local_stock.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'urban_bridge_motorway']] = urban_b_motorway_stock.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'urban_bridge_pedestrian']] = urban_b_pedestrian_stock.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'urban_bridge_primary']] = urban_b_primary_stock.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'urban_bridge_secondary']] = urban_b_secondary_stock.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'urban_bridge_tertiary']] = urban_b_tertiary_stock.to_numpy()


infra_materials_detail.loc[idx['stock', :, :], idx[:, 'obsolete_urban_bridges']] = obsolete_urban_b_cycle_stock.to_numpy() + obsolete_urban_b_informal_stock.to_numpy() + obsolete_urban_b_local_stock.to_numpy() + obsolete_urban_b_motorway_stock.to_numpy() + obsolete_urban_b_pedestrian_stock.to_numpy() + obsolete_urban_b_primary_stock.to_numpy() + obsolete_urban_b_secondary_stock.to_numpy() + obsolete_urban_b_tertiary_stock.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'obsolete_urban_paved_roads']] = obsolete_urban_cycle_stock.to_numpy() + obsolete_urban_informal_stock.to_numpy() + obsolete_urban_local_stock.to_numpy() + obsolete_urban_motorway_stock.to_numpy() + obsolete_urban_pedestrian_stock.to_numpy() + obsolete_urban_primary_stock.to_numpy() + obsolete_urban_secondary_stock.to_numpy()+ obsolete_urban_tertiary_stock.to_numpy()


infra_materials_detail.loc[idx['stock', :, :], idx[:, 'urban_tunnel_cycle']] = urban_t_cycle_stock.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'urban_tunnel_informal']] = urban_t_informal_stock.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'urban_tunnel_local']] = urban_t_local_stock.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'urban_tunnel_motorway']] = urban_t_motorway_stock.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'urban_tunnel_pedestrian']] = urban_t_pedestrian_stock.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'urban_tunnel_primary']] = urban_t_primary_stock.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'urban_tunnel_secondary']] = urban_t_secondary_stock.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'urban_tunnel_tertiary']] = urban_t_tertiary_stock.to_numpy()

infra_materials_detail.loc[idx['stock', :, :], idx[:, 'obsolete_urban_tunnel']] = obsolete_urban_t_cycle_stock.to_numpy() + obsolete_urban_t_informal_stock.to_numpy() + obsolete_urban_t_local_stock.to_numpy() + obsolete_urban_t_motorway_stock.to_numpy() + obsolete_urban_t_pedestrian_stock.to_numpy() + obsolete_urban_t_primary_stock.to_numpy() + obsolete_urban_t_secondary_stock.to_numpy() + obsolete_urban_t_tertiary_stock.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'obsolete_urban_unpaved_roads']] = obsolete_urban_unp_cycle_stock.to_numpy() + obsolete_urban_unp_informal_stock.to_numpy() + obsolete_urban_unp_local_stock.to_numpy()  + obsolete_urban_unp_pedestrian_stock.to_numpy() + obsolete_urban_unp_primary_stock.to_numpy() + obsolete_urban_unp_secondary_stock.to_numpy() + obsolete_urban_unp_tertiary_stock.to_numpy()


infra_materials_detail.loc[idx['stock', :, :], idx[:, 'rural_bridge_cycle']] = rural_b_cycle_stock.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'rural_bridge_informal']] = rural_b_informal_stock.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'rural_bridge_local']] = rural_b_local_stock.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'rural_bridge_motorway']] = rural_b_motorway_stock.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'rural_bridge_pedestrian']] = rural_b_pedestrian_stock.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'rural_bridge_primary']] = rural_b_primary_stock.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'rural_bridge_secondary']] = rural_b_secondary_stock.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'rural_bridge_tertiary']] = rural_b_tertiary_stock.to_numpy()

infra_materials_detail.loc[idx['stock', :, :], idx[:, 'obsolete_rural_bridges']] = obsolete_rural_b_cycle_stock.to_numpy() + obsolete_rural_b_informal_stock.to_numpy() + obsolete_rural_b_local_stock.to_numpy() + obsolete_rural_b_motorway_stock.to_numpy() + obsolete_rural_b_pedestrian_stock.to_numpy() + obsolete_rural_b_primary_stock.to_numpy() + obsolete_rural_b_secondary_stock.to_numpy() + obsolete_rural_b_tertiary_stock.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'obsolete_rural_paved_roads']] = obsolete_rural_cycle_stock.to_numpy() + obsolete_rural_informal_stock.to_numpy() + obsolete_rural_local_stock.to_numpy() + obsolete_rural_motorway_stock.to_numpy() + obsolete_rural_pedestrian_stock.to_numpy() + obsolete_rural_primary_stock.to_numpy() + obsolete_rural_secondary_stock.to_numpy() + obsolete_rural_tertiary_stock.to_numpy()


infra_materials_detail.loc[idx['stock', :, :], idx[:, 'rural_tunnel_cycle']] = rural_t_cycle_stock.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'rural_tunnel_informal']] = rural_t_informal_stock.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'rural_tunnel_local']] = rural_t_local_stock.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'rural_tunnel_motorway']] = rural_t_motorway_stock.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'rural_tunnel_pedestrian']] = rural_t_pedestrian_stock.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'rural_tunnel_primary']] = rural_t_primary_stock.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'rural_tunnel_secondary']] = rural_t_secondary_stock.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'rural_tunnel_tertiary']] = rural_t_tertiary_stock.to_numpy()

infra_materials_detail.loc[idx['stock', :, :], idx[:, 'obsolete_rural_tunnel']] = obsolete_rural_t_cycle_stock.to_numpy() + obsolete_rural_t_informal_stock.to_numpy() + obsolete_rural_t_local_stock.to_numpy() + obsolete_rural_t_motorway_stock.to_numpy() + obsolete_rural_t_pedestrian_stock.to_numpy() + obsolete_rural_t_primary_stock.to_numpy() + obsolete_rural_t_secondary_stock.to_numpy() + obsolete_rural_t_tertiary_stock.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'obsolete_rural_unpaved_roads']] = obsolete_rural_unp_cycle_stock.to_numpy() + obsolete_rural_unp_informal_stock.to_numpy() + obsolete_rural_unp_local_stock.to_numpy()  + obsolete_rural_unp_pedestrian_stock.to_numpy() + obsolete_rural_unp_primary_stock.to_numpy() + obsolete_rural_unp_secondary_stock.to_numpy() + obsolete_rural_unp_tertiary_stock.to_numpy()

# Individual entries for total rail, urban rail, etc., can follow the same pattern
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'total_rail']] = rail_stock.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'total_urban_rail']] = urban_rail_stock.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'rail_bridge']] = rail_bridge_stock.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'rail_tunnel']] = rail_tunnel_stock.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'hibernating_rail']] = rail_obs_stock.to_numpy()

infra_materials_detail.loc[idx['stock', :, :], idx[:, 'hibernating_rail_bridge']] = rail_brid_obs_stock.to_numpy()
infra_materials_detail.loc[idx['stock', :, :], idx[:, 'hibernating_rail_tunnel']] = rail_tunn_obs_stock.to_numpy()

infra_materials_detail.loc[idx['stock', :, :], idx[:, 'total_HST']] = HST_stock.to_numpy()


# Urban and rural paved roads by category
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'urban_cycle_paved']] = urban_cycle_in.to_numpy() + diff_u_area_paved_cycle.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'urban_informal_paved']] = urban_informal_in.to_numpy() + diff_u_area_paved_informal.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'urban_local_paved']] = urban_local_in.to_numpy() + diff_u_area_paved_local.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'urban_motorway_paved']] = urban_motorway_in.to_numpy() + diff_u_area_paved_motorway.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'urban_pedestrian_paved']] = urban_pedestrian_in.to_numpy() + diff_u_area_paved_pedestrian.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'urban_primary_paved']] = urban_primary_in.to_numpy() + diff_u_area_paved_primary.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'urban_secondary_paved']] = urban_secondary_in.to_numpy() + diff_u_area_paved_secondary.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'urban_tertiary_paved']] = urban_tertiary_in.to_numpy() + diff_u_area_paved_tertiary.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'urban_parking_paved']] = urban_parking_in.to_numpy() 

# Urban unpaved roads by category
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'urban_cycle_unpaved']] = urban_unp_cycle_in.to_numpy() + diff_u_area_unpaved_cycle.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'urban_informal_unpaved']] = urban_unp_informal_in.to_numpy() + diff_u_area_unpaved_informal.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'urban_local_unpaved']] = urban_unp_local_in.to_numpy() + diff_u_area_unpaved_local.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'urban_pedestrian_unpaved']] = urban_unp_pedestrian_in.to_numpy() + diff_u_area_unpaved_pedestrian.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'urban_primary_unpaved']] = urban_unp_primary_in.to_numpy() + diff_u_area_unpaved_primary.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'urban_secondary_unpaved']] = urban_unp_secondary_in.to_numpy() + diff_u_area_unpaved_secondary.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'urban_tertiary_unpaved']] = urban_unp_tertiary_in.to_numpy() + diff_u_area_unpaved_tertiary.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'urban_parking_unpaved']] = urban_unp_parking_in.to_numpy() 

# Rural paved roads by category
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'rural_cycle_paved']] = rural_cycle_in.to_numpy() + diff_r_area_paved_cycle.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'rural_informal_paved']] = rural_informal_in.to_numpy() + diff_r_area_paved_informal.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'rural_local_paved']] = rural_local_in.to_numpy() + diff_r_area_paved_local.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'rural_motorway_paved']] = rural_motorway_in.to_numpy() + diff_r_area_paved_motorway.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'rural_pedestrian_paved']] = rural_pedestrian_in.to_numpy() + diff_r_area_paved_pedestrian.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'rural_primary_paved']] = rural_primary_in.to_numpy() + diff_r_area_paved_primary.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'rural_secondary_paved']] = rural_secondary_in.to_numpy() + diff_r_area_paved_secondary.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'rural_tertiary_paved']] = rural_tertiary_in.to_numpy() + diff_r_area_paved_tertiary.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'rural_parking_paved']] = rural_parking_in.to_numpy()

# Rural unpaved roads by category
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'rural_cycle_unpaved']] = rural_unp_cycle_in.to_numpy() + diff_r_area_unpaved_cycle.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'rural_informal_unpaved']] = rural_unp_informal_in.to_numpy() + diff_r_area_unpaved_informal.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'rural_local_unpaved']] = rural_unp_local_in.to_numpy() + diff_r_area_unpaved_local.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'rural_pedestrian_unpaved']] = rural_unp_pedestrian_in.to_numpy() + diff_r_area_unpaved_pedestrian.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'rural_primary_unpaved']] = rural_unp_primary_in.to_numpy() + diff_r_area_unpaved_primary.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'rural_secondary_unpaved']] = rural_unp_secondary_in.to_numpy() + diff_r_area_unpaved_secondary.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'rural_tertiary_unpaved']] = rural_unp_tertiary_in.to_numpy() + diff_r_area_unpaved_tertiary.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'rural_parking_unpaved']] = rural_unp_parking_in.to_numpy() 

# Urban bridges and tunnels
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'urban_bridge_cycle']] = urban_b_cycle_in.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'urban_bridge_informal']] = urban_b_informal_in.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'urban_bridge_local']] = urban_b_local_in.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'urban_bridge_motorway']] = urban_b_motorway_in.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'urban_bridge_pedestrian']] = urban_b_pedestrian_in.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'urban_bridge_primary']] = urban_b_primary_in.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'urban_bridge_secondary']] = urban_b_secondary_in.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'urban_bridge_tertiary']] = urban_b_tertiary_in.to_numpy()

infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'urban_tunnel_cycle']] = urban_t_cycle_in.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'urban_tunnel_informal']] = urban_t_informal_in.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'urban_tunnel_local']] = urban_t_local_in.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'urban_tunnel_motorway']] = urban_t_motorway_in.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'urban_tunnel_pedestrian']] = urban_t_pedestrian_in.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'urban_tunnel_primary']] = urban_t_primary_in.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'urban_tunnel_secondary']] = urban_t_secondary_in.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'urban_tunnel_tertiary']] = urban_t_tertiary_in.to_numpy()


infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'rural_bridge_cycle']] = rural_b_cycle_in.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'rural_bridge_informal']] = rural_b_informal_in.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'rural_bridge_local']] = rural_b_local_in.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'rural_bridge_motorway']] = rural_b_motorway_in.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'rural_bridge_pedestrian']] = rural_b_pedestrian_in.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'rural_bridge_primary']] = rural_b_primary_in.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'rural_bridge_secondary']] = rural_b_secondary_in.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'rural_bridge_tertiary']] = rural_b_tertiary_in.to_numpy()

infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'rural_tunnel_cycle']] = rural_t_cycle_in.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'rural_tunnel_informal']] = rural_t_informal_in.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'rural_tunnel_local']] = rural_t_local_in.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'rural_tunnel_motorway']] = rural_t_motorway_in.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'rural_tunnel_pedestrian']] = rural_t_pedestrian_in.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'rural_tunnel_primary']] = rural_t_primary_in.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'rural_tunnel_secondary']] = rural_t_secondary_in.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'rural_tunnel_tertiary']] = rural_t_tertiary_in.to_numpy()

# Continue similarly for rural bridges, rural tunnels, and rail categories
# Individual entries for total rail, urban rail, etc., can follow the same pattern
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'total_rail']] = rail_in.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'total_urban_rail']] = urban_rail_in.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'rail_bridge']] = rail_bridge_in.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'rail_tunnel']] = rail_tunnel_in.to_numpy()
infra_materials_detail.loc[idx['inflow', :, :], idx[:, 'total_HST']] = HST_in.to_numpy()


# Urban and rural paved roads by category
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'urban_cycle_paved']] = urban_cycle_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'urban_informal_paved']] = urban_informal_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'urban_local_paved']] = urban_local_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'urban_motorway_paved']] = urban_motorway_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'urban_pedestrian_paved']] = urban_pedestrian_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'urban_primary_paved']] = urban_primary_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'urban_secondary_paved']] = urban_secondary_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'urban_tertiary_paved']] = urban_tertiary_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'urban_parking_paved']] = urban_parking_out.to_numpy()

# Urban unpaved roads by category
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'urban_cycle_unpaved']] = urban_unp_cycle_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'urban_informal_unpaved']] = urban_unp_informal_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'urban_local_unpaved']] = urban_unp_local_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'urban_pedestrian_unpaved']] = urban_unp_pedestrian_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'urban_primary_unpaved']] = urban_unp_primary_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'urban_secondary_unpaved']] = urban_unp_secondary_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'urban_tertiary_unpaved']] = urban_unp_tertiary_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'urban_parking_unpaved']] = urban_unp_parking_out.to_numpy()

# Rural paved roads by category
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'rural_cycle_paved']] = rural_cycle_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'rural_informal_paved']] = rural_informal_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'rural_local_paved']] = rural_local_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'rural_motorway_paved']] = rural_motorway_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'rural_pedestrian_paved']] = rural_pedestrian_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'rural_primary_paved']] = rural_primary_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'rural_secondary_paved']] = rural_secondary_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'rural_tertiary_paved']] = rural_tertiary_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'rural_parking_paved']] = rural_parking_out.to_numpy()

# Rural unpaved roads by category
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'rural_cycle_unpaved']] = rural_unp_cycle_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'rural_informal_unpaved']] = rural_unp_informal_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'rural_local_unpaved']] = rural_unp_local_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'rural_pedestrian_unpaved']] = rural_unp_pedestrian_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'rural_primary_unpaved']] = rural_unp_primary_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'rural_secondary_unpaved']] = rural_unp_secondary_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'rural_tertiary_unpaved']] = rural_unp_tertiary_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'rural_parking_unpaved']] = rural_unp_parking_out.to_numpy()

# Urban bridges and tunnels
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'urban_bridge_cycle']] = urban_b_cycle_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'urban_bridge_informal']] = urban_b_informal_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'urban_bridge_local']] = urban_b_local_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'urban_bridge_motorway']] = urban_b_motorway_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'urban_bridge_pedestrian']] = urban_b_pedestrian_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'urban_bridge_primary']] = urban_b_primary_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'urban_bridge_secondary']] = urban_b_secondary_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'urban_bridge_tertiary']] = urban_b_tertiary_out.to_numpy()

infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'urban_tunnel_cycle']] = urban_t_cycle_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'urban_tunnel_informal']] = urban_t_informal_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'urban_tunnel_local']] = urban_t_local_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'urban_tunnel_motorway']] = urban_t_motorway_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'urban_tunnel_pedestrian']] = urban_t_pedestrian_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'urban_tunnel_primary']] = urban_t_primary_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'urban_tunnel_secondary']] = urban_t_secondary_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'urban_tunnel_tertiary']] = urban_t_tertiary_out.to_numpy()


infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'rural_bridge_cycle']] = rural_b_cycle_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'rural_bridge_informal']] = rural_b_informal_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'rural_bridge_local']] = rural_b_local_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'rural_bridge_motorway']] = rural_b_motorway_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'rural_bridge_pedestrian']] = rural_b_pedestrian_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'rural_bridge_primary']] = rural_b_primary_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'rural_bridge_secondary']] = rural_b_secondary_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'rural_bridge_tertiary']] = rural_b_tertiary_out.to_numpy()

infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'rural_tunnel_cycle']] = rural_t_cycle_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'rural_tunnel_informal']] = rural_t_informal_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'rural_tunnel_local']] = rural_t_local_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'rural_tunnel_motorway']] = rural_t_motorway_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'rural_tunnel_pedestrian']] = rural_t_pedestrian_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'rural_tunnel_primary']] = rural_t_primary_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'rural_tunnel_secondary']] = rural_t_secondary_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'rural_tunnel_tertiary']] = rural_t_tertiary_out.to_numpy()

# Continue similarly for rural bridges, rural tunnels, and rail categories
# Individual entries for total rail, urban rail, etc., can follow the same pattern
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'total_rail']] = rail_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'total_urban_rail']] = urban_rail_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'rail_bridge']] = rail_bridge_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'rail_tunnel']] = rail_tunnel_out.to_numpy()
infra_materials_detail.loc[idx['outflow', :, :], idx[:, 'total_HST']] = HST_out.to_numpy()


################# STRUCTURAL SMOOTHING (ROLLING AVERAGE) #################
def apply_rolling_to_multindex_df(df, window=5):
    """
    Applies a rolling average to the 'years' level of the MultiIndex 
    while preserving the shape and index structure of the DataFrame.
    """
    # Ensure data is numeric
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Unstack 'flow' and 'materials' so that only 'years' is in the index
    df_unstacked = df.unstack(level=['flow', 'materials'])
    
    # Apply rolling mean, centered, min_periods=1 to avoid NaNs at edges
    df_rolled = df_unstacked.rolling(window=window, center=True, min_periods=1).mean()
    
    # Stack back to original shape and reorder levels to match original
    df_restacked = df_rolled.stack(level=['flow', 'materials']).reorder_levels(['flow', 'years', 'materials']).sort_index()
    return df_restacked

# Apply the smoothing to the master DataFrames before normalization/export
print("Applying structural 5-year rolling average to mass-balanced final DataFrames...")
infra_materials = apply_rolling_to_multindex_df(infra_materials)
infra_materials_scenario = apply_rolling_to_multindex_df(infra_materials_scenario)
infra_materials_detail = apply_rolling_to_multindex_df(infra_materials_detail)
##########################################################################

# Initialize a new DataFrame with the same structure as infra_materials
infra_materials_normalized = infra_materials.copy()

# Initialize a new DataFrame with the same structure as infra_materials
infra_materials_normalized = infra_materials.copy()

# Identify the items that contain 'urban_'
urban_items = [item for item in items if 'urban' in item]
urban_pop_real = sorted_urban_population * 1000000

for item in urban_items:
    for flow_type in flow:  # Loop through 'stock', 'inflow', 'outflow'
        for year in range(startyear, 2101):  # Loop through the years
            for material in material_list:  # Loop through materials
                for region in sorted_region_list:
                    # Get the corresponding value from the infra_materials DataFrame
                    infra_value = infra_materials.loc[(flow_type, year, material), (region, item)]

                    # Access the population value from urban_pop_real for the given year and region
                    # Since year is in the index and region is in the columns
                    population_value = urban_pop_real.loc[year, region]  # Accessing population for the specific year and region
                    
                    # Perform the division (handling division by zero)
                    if population_value != 0:
                        # Divide the value by the population value
                        infra_materials_normalized.loc[(flow_type, year, material), (region, item)] = infra_value / population_value
                    else:
                        # If population is 0, set the result to 0
                        infra_materials_normalized.loc[(flow_type, year, material), (region, item)] = 0

# Identify the items that contain 'urban_'
rural_items = [item for item in items if 'rural' in item]
rural_pop_real = sorted_rural_population * 1000000


for item in rural_items:
    for flow_type in flow:  # Loop through 'stock', 'inflow', 'outflow'
        for year in range(startyear, 2101):  # Loop through the years
            for material in material_list:  # Loop through materials
                for region in sorted_region_list:
                    # Get the corresponding value from the infra_materials DataFrame
                    infra_value = infra_materials.loc[(flow_type, year, material), (region, item)]

                    # Access the population value from urban_pop_real for the given year and region
                    # Since year is in the index and region is in the columns
                    population_value = rural_pop_real.loc[year, region]  # Accessing population for the specific year and region
                    
                    # Perform the division (handling division by zero)
                    if population_value != 0:
                        # Divide the value by the population value
                        infra_materials_normalized.loc[(flow_type, year, material), (region, item)] = infra_value / population_value
                    else:
                        # If population is 0, set the result to 0
                        infra_materials_normalized.loc[(flow_type, year, material), (region, item)] = 0
                        

# Identify the items that contain 'urban_'
total_items = [item for item in items if 'total_' in item]
total_pop_real = (sorted_rural_population+sorted_urban_population) * 1000000


for item in total_items:
    for flow_type in flow:  # Loop through 'stock', 'inflow', 'outflow'
        for year in range(startyear, 2101):  # Loop through the years
            for material in material_list:  # Loop through materials
                for region in sorted_region_list:
                    # Get the corresponding value from the infra_materials DataFrame
                    infra_value = infra_materials.loc[(flow_type, year, material), (region, item)]

                    # Access the population value from urban_pop_real for the given year and region
                    # Since year is in the index and region is in the columns
                    population_value = total_pop_real.loc[year, region]  # Accessing population for the specific year and region
                    
                    # Perform the division (handling division by zero)
                    if population_value != 0:
                        # Divide the value by the population value
                        infra_materials_normalized.loc[(flow_type, year, material), (region, item)] = infra_value / population_value
                    else:
                        # If population is 0, set the result to 0
                        infra_materials_normalized.loc[(flow_type, year, material), (region, item)] = 0                        


#%% XLSX output files (in kilo-tonnes, kt) - 3 flow types (stock, inflow, outlfow) - 26 Regions - 3 product categories (lines, substations, transformers) - 14 materials (but some are not releveant to the grid, so they are 0)
infra_materials_baseline = pd.concat([infra_materials_scenario], keys=['network'], names=['category']).stack().stack() # add a descriptor column
infra_materials_baseline = pd.concat([infra_materials_baseline], keys=['transport_infra'], names=['sector'])
infra_materials_baseline = infra_materials_baseline.unstack(level=3).reorder_levels([5, 2, 0, 1, 4, 3]) / 1000000000   # to kt
# Remove the first column (1971)
infra_materials_baseline = infra_materials_baseline.drop(columns=1971)

# Sum across all regions
global_sum_baseline = infra_materials_baseline.groupby(level=[1,2,3,4,5]).sum()
# Add a new index value 'Global' for the region (level 5)
global_sum_baseline = global_sum_baseline.assign(region='Global').set_index('region', append=True)
# Step 3: Reorder levels to move 'region' to the first position
global_sum_baseline = global_sum_baseline.reorder_levels(['region', 0, 1, 2, 3, 4])  # Adjust based on actual level names/positions
# Step 4: Apply the same level names as in infra_materials_physical
global_sum_baseline.index.names = infra_materials_baseline.index.names
# Concatenate the original DataFrame with the global sum
infra_materials_baseline_global = pd.concat([infra_materials_baseline, global_sum_baseline])

infra_materials_baseline_global.to_excel('output\\' + variant + '\\' + sa_settings + '\\baseline_global_road_materials_output_mt.xlsx') # in kt


#%% CSV output files (in kilo-tonnes, kt) - 3 flow types (stock, inflow, outlfow) - 26 Regions - 3 product categories (lines, substations, transformers) - 14 materials (but some are not releveant to the grid, so they are 0)

infra_materials_out = pd.concat([infra_materials], keys=['network'], names=['category']).stack().stack() # add a descriptor column
infra_materials_out = pd.concat([infra_materials_out], keys=['transport_infra'], names=['sector'])
infra_materials_out = infra_materials_out.unstack(level=3).reorder_levels([5, 2, 0, 1, 4, 3]) / 1000000   # to kt
# Remove the first column (1971)
infra_materials_out = infra_materials_out.drop(columns=1971)
infra_materials_out.to_excel('output\\' + variant + '\\' + sa_settings + '\\road_materials_output_kt.xlsx') # in kt

# Filter out rows where 'materials' index level contains 'aggregate'
infra_materials_out_no_aggregate = infra_materials_out[infra_materials_out.index.get_level_values('materials') != 'aggregate']

infra_materials_out_pc = pd.concat([infra_materials_normalized], keys=['network'], names=['category']).stack().stack() # add a descriptor column
infra_materials_out_pc = pd.concat([infra_materials_out_pc], keys=['transport_infra'], names=['sector'])
infra_materials_out_pc = infra_materials_out_pc.unstack(level=3).reorder_levels([5, 2, 0, 1, 4, 3]) / 1000000    # to kt
infra_materials_out_pc.to_excel('output\\' + variant + '\\' + sa_settings + '\\road_materials_output_kt_pc.xlsx') # in kt


infra_materials_out_detail = pd.concat([infra_materials_detail], keys=['network'], names=['category']).stack().stack() # add a descriptor column
infra_materials_out_detail = pd.concat([infra_materials_out_detail], keys=['transport_infra'], names=['sector'])
infra_materials_out_detail = infra_materials_out_detail.unstack(level=3).reorder_levels([5, 2, 0, 1, 4, 3]) / 1000000   # to kt
# Remove the first column (1971)
infra_materials_out_detail = infra_materials_out_detail.drop(columns=1971)

# Group by all index levels except 'materials' and 'elements', then sum
# This effectively aggregates (sums) the data across all materials and elements.
infra_materials_aggregated = infra_materials_out_detail.groupby(
    level=['regions', 'flow', 'sector', 'category']
).sum()

# Select only the 'stock' data from the 'flow' index level
infra_materials_stock_only = infra_materials_aggregated.xs('stock', level='flow')

# Save this stock-only DataFrame to a new Excel file
infra_materials_stock_only.to_excel(
    'output\\' + variant + '\\' + sa_settings + '\\detail\\road_materials_output_kt_stock_only.xlsx'
)


# Save the new aggregated DataFrame to a separate Excel file
infra_materials_aggregated.to_excel(
    'output\\' + variant + '\\' + sa_settings + '\\detail\\road_materials_output_kt_aggregated.xlsx'
)


infra_materials_out_detail.to_excel('output\\' + variant + '\\' + sa_settings + '\\detail\\road_materials_output_kt_detail.xlsx') # in kt




