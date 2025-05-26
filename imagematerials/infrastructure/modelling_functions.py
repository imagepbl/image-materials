# -*- coding: utf-8 -*-
"""
Created on Fri May 23 11:20:27 2025

@author: MvEng
"""

from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from imagematerials.vehicles.constants import END_YEAR, FIRST_YEAR, REGIONS

# from read_scripts.dynamic_stock_model_BM import DynamicStockModel as DSM
idx = pd.IndexSlice

column_mapping = {
    'China region': 'China',
    'Indonesia region': 'Indonesia Region',
    'Southeastern Asia': 'South Eastern Asia',
    'Korea region': 'Korea',
    'Russia region': 'Russia Region'
}


def prep_area (area):
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
    years_to_smooth = range(1950, 2071)
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

    # Define the mapping of old column names to new column names
    column_mapping = {
        'China region': 'China',
        'Indonesia region': 'Indonesia Region',
        'Southeastern Asia': 'South Eastern Asia',
        'Korea region': 'Korea',
        'Russia region': 'Russia Region'
    }

    # Rename the columns in IMAGE_urban_area
    IMAGE_urban_area_transposed = IMAGE_urban_area_transposed.rename(columns=column_mapping)
    IMAGE_rural_area_transposed = IMAGE_rural_area_transposed.rename(columns=column_mapping)
    IMAGE_urban_area_transposed = IMAGE_urban_area_transposed.sort_index(axis=1)
    IMAGE_rural_area_transposed = IMAGE_rural_area_transposed.sort_index(axis=1)
    
    return IMAGE_urban_area_transposed, IMAGE_rural_area_transposed

def prep_hdi(hdi):
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
    sorted_IMAGE_hdi = IMAGE_hdi_transposed.sort_index(axis=1)
    sorted_IMAGE_hdi = sorted_IMAGE_hdi.loc[sorted_IMAGE_hdi.index <= 2100]
    return sorted_IMAGE_hdi

def prep_pop_gdp (gdp_df, urban_pop_df, rural_pop_df, gdp_cap):
    # Calculate the denominator: sum of urban and rural populations
    population_sum = urban_pop_df + rural_pop_df

    # Perform the division: GDP / (urban-population + rural-population)
    # Broadcasting is used here to handle division across all columns
    gdp_cap_new = gdp_df * 1000 / population_sum

    # Step 1: Reindex the DataFrame to include all years from 1970 to 2100
    all_years = list(range(1970, 2101))
    urban_pop_df = urban_pop_df.reindex(columns=all_years)
    rural_pop_df = rural_pop_df.reindex(columns=all_years)
    
    # Step 2: Interpolate missing values linearly across rows
    urban_pop_df = urban_pop_df.interpolate(method='linear', axis=1)
    rural_pop_df = rural_pop_df.interpolate(method='linear', axis=1)
    
    pop_sum = urban_pop_df + rural_pop_df
    
    # Transpose the DataFrame
    urban_pop_df = urban_pop_df.T
    rural_pop_df = rural_pop_df.T
      
    all_years = list(range(1990, 2101))
    gdp_cap_new = gdp_cap_new.reindex(columns=all_years)
    gdp_cap_new = gdp_cap_new.interpolate(method='linear', axis=1)
    gdp_cap_new = gdp_cap_new.T
  
    gdp_cap_new = gdp_cap_new.rename(columns=column_mapping)
    gdp_cap_new = gdp_cap_new.reindex(range(1971, 2101))
    
    # Rename the columns in IMAGE_urban_area
    gdp_cap = gdp_cap.rename(columns=column_mapping)
    gdp_cap_sorted = gdp_cap.sort_index(axis=1)
    # Select the years 1971–1989
    gdp_1971_1989 = gdp_cap_sorted.loc[1971:1989]
    # Extract GDP for 1990 (repeated for each row for broadcasting)
    gdp_1990 = gdp_cap_sorted.loc[1990]
    # Compute the fraction
    fractions = gdp_1971_1989.div(gdp_1990)
    # Save the result into a new DataFrame
    fraction_df = fractions.copy()
    # Extract GDP values for 1990 from gdp_cap_new
    gdp_new_1990 = gdp_cap_new.loc[1990]
    # Use the fractions for years 1971–1980
    fractions_1971_1989 = fraction_df.loc[1971:1989]
    # Calculate the new values
    gdp_new_1971_1989 = fractions_1971_1989.mul(gdp_new_1990)
    # Update gdp_cap_new with the calculated values
    gdp_cap_new.update(gdp_new_1971_1989)
        
    rural_pop_df = rural_pop_df.drop(index=1970)
    urban_pop_df = urban_pop_df.drop(index=1970)
    rural_pop_df = rural_pop_df.rename(columns=column_mapping)
    urban_pop_df = urban_pop_df.rename(columns=column_mapping)
    
    urban_population = urban_pop_df.sort_index(axis=1)
    rural_population = rural_pop_df.sort_index(axis=1)
    
    return gdp_cap_new, urban_population, rural_population

def conversion_to_vkm (updated_vkm_pkm, updated_vkm_tkm, region_list):
    
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
    
    # Combine the dataframes by adding the corresponding numerical columns
    #vkm = updated_vkm_tkm.copy()  # Start with updated_vkm_tkm
    
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
    vkm_rail = vkm[['Trains_VKM','regions']]
    vkm_rail_pivoted = vkm_rail.pivot(columns='regions', values='Trains_VKM')
    vkm_rail_pivoted.columns = region_list
    
    vkm['total_vkm'] = vkm['Buses_vkm']  + vkm['Cars_vkm']   + vkm['Trucks_vkm']
    # Keep only 'total_vkm' and 'region' columns
    vkm = vkm[['total_vkm', 'regions']]
    # Pivot the DataFrame to make 'region' values into columns
    vkm_pivoted = vkm.pivot(columns='regions', values='total_vkm')
    vkm_pivoted.columns = region_list

    #Placeholder_smoothing out function needed for vkm data
    vkm_pivoted = pd.read_excel('C:\\Academic\\Chapter2\\2. dMFA\\TRIPI-fut\\\\Update-input\\vkm_road.xlsx', index_col=0)  # Load VKM_road data
    vkm_rail_pivoted = pd.read_excel('C:\\Academic\\Chapter2\\2. dMFA\\TRIPI-fut\\\\Update-input\\vkm_rail.xlsx', index_col=0)  # Load VKM_rail data
    
    # Apply moving average smoothing
    window_avg_size = 10  # You can adjust the window size as needed
    smoothed_data_vkm_road = vkm_pivoted.apply(lambda col: col.rolling(window=window_avg_size, center=True, min_periods=1).mean(), axis=1)
    smoothed_data_vkm_rail = vkm_rail_pivoted.apply(lambda col: col.rolling(window=window_avg_size, center=True, min_periods=1).mean(), axis=1)
    
    # Reinsert the smoothed data back into the original DataFrame
    vkm_pivoted[smoothed_data_vkm_road.columns] = smoothed_data_vkm_road
    vkm_rail_pivoted[smoothed_data_vkm_rail.columns] = smoothed_data_vkm_rail
    
    return vkm_pivoted, vkm_rail_pivoted
    

def urban_vkm(urban_share, rural_share, gdp_total, urban_population, rural_population, vkm_pivoted):
    # Define the regression function for GDP share urban - rural. See supplementary information Table --- Current Excel file C:\Academic\Chapter2\2. dMFA\Formulas\region_pkm_tkm
    def gdp_share_regression(x):
        return 1.0267 * x + 0.0412
    # Apply the regression function to all values in urban_share
    gdp_urban_share = urban_share.applymap(gdp_share_regression)

    # Cap values at a maximum of 0.98
    gdp_urban_share = gdp_urban_share.clip(upper=0.99)

    gdp_rural_share = 1 - gdp_urban_share

    gdp_urban_total = gdp_total * gdp_urban_share
    gdp_rural_total = gdp_total * gdp_rural_share

    gdp_urban_cap = gdp_urban_total / (urban_population * 1000000)
    gdp_rural_cap = gdp_rural_total / (rural_population * 1000000)

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
    
    return gdp_urban_cap, gdp_rural_cap, vkm_urban_IMAGE_cap, vkm_rural_IMAGE_cap
    





