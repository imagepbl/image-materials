# -*- coding: utf-8 -*-
"""
Preprocessing script for the infrastructure module.
Extracts input data, performs regression to determine infrastructure area/length
and converts it to xarray format for use in the dynamic stock model.
"""

import pandas as pd
import numpy as np
import math
from scipy.ndimage import gaussian_filter1d
import xarray as xr
from pathlib import Path

# Import read_mym_df from wherever it is in the repository
import sys
# Assuming read_mym.py is in imagematerials/
from imagematerials.read_mym import read_mym_df
from imagematerials.concepts import KnowledgeGraph, Node
import prism

# v5 line 2192: first constructed highway — start of modern transportation era
FIRST_YEAR_GRID = 1911

def get_preprocessing_data_infrastructure(path_base: Path, scen_folder: str, start_year: int = 1971, end_year: int = 2100):
    """
    Reads data from SSP2 and IMAGE folders, calculates infrastructure stocks
    and returns it as a dictionary ready for the GenericStocks model.
    """
    preprocessing_results = {}
    
    # Define paths
    data_raw_dir = path_base / "data" / "raw"
    image_dir = data_raw_dir / "image" / scen_folder
    infra_dir = path_base / "imagematerials" / "infrastructure" / "dependant_files"
    tripi_dir = infra_dir / "TRIPI-fut"
    grid_dir = infra_dir / "grid_data"
    
    # Define region list mapping
    region_load_in = pd.read_csv(grid_dir / 'region_load.csv', index_col=0, names=None).transpose()
    region_list = list(region_load_in.columns.values)
    
    column_mapping = {
        'China region': 'China',
        'Indonesia region': 'Indonesia Region',
        'Southeastern Asia': 'South Eastern Asia',
        'Korea region': 'Korea',
        'Russia region': 'Russia Region'
    }

    # --- 1. Load Data ---
    data_2024 = pd.read_excel(tripi_dir / '2024-data.xlsx', index_col=0)
    
    # Energy Services (VKM / TKM)
    updated_vkm_pkm = read_mym_df(str(image_dir / 'EnergyServices' / 'trp_trvl_pkm.out'))
    updated_vkm_tkm = read_mym_df(str(image_dir / 'EnergyServices' / 'trp_frgt_Tkm.out'))
    
    # Socioeconomic (GDP, POP)
    gdp_cap = read_mym_df(str(image_dir / 'Socioeconomic' / 'gdp_pc.scn'))
    urban_pop_df = read_mym_df(str(image_dir / 'Socioeconomic' / 'URBPOPTOT.out'))
    rural_pop_df = read_mym_df(str(image_dir / 'Socioeconomic' / 'RURPOPTOT.out'))
    
    # --- 2. Process VKM ---
    updated_vkm_pkm.set_index(['time'], inplace=True)
    updated_vkm_tkm.set_index(['time'], inplace=True)
    
    tkms_label  = ['regions','Inland ships', 'Cargo Trains', 'medium truck', 'heavy truck', 'Cargo Planes', 'Ships', 'empty', 'total']
    pkms_label  = ['regions','walking', 'Bikes', 'Buses', 'Trains', 'Cars', 'HST', 'Planes', 'total']
    
    updated_vkm_tkm.columns = tkms_label
    updated_vkm_pkm.columns = pkms_label
    
    updated_vkm_tkm['Trucks'] = updated_vkm_tkm['medium truck'] + updated_vkm_tkm['heavy truck']
    updated_vkm_tkm = updated_vkm_tkm.drop(columns=['medium truck', 'heavy truck', 'empty', 'total'])
    updated_vkm_pkm = updated_vkm_pkm.drop(columns=['walking', 'total'])
    
    updated_vkm_tkm.iloc[:, 1:] = updated_vkm_tkm.iloc[:, 1:] * 1000000
    updated_vkm_pkm.iloc[:, 1:] = updated_vkm_pkm.iloc[:, 1:] * 1000000000000
    
    vkm = pd.concat([updated_vkm_pkm, updated_vkm_tkm], axis=1)
    vkm = vkm.loc[:, ~vkm.columns.duplicated()]
    vkm = vkm.rename_axis("years")
    vkm = vkm[~vkm["regions"].isin([27, 28])]
    
    # Conversion of pkm and tkm to vkm
    vkm['Buses_vkm'] = vkm['Buses'] * 0.06 / 23 * 0.43 + vkm['Buses'] * 0.94 / 57 * 0.43
    vkm['Cars_vkm'] = vkm['Cars'] / (4 * 0.45)
    vkm['Bicycle_vkm'] = vkm['Bikes']
    vkm['Trucks_vkm'] = vkm['Trucks'] * 0.04 / 0.74 + vkm['Trucks'] * 0.48 / 7.95 + vkm['Trucks'] * 0.48 / 14
    
    vkm['Trains_VKM'] = vkm['Trains'] / 400 + (vkm['Cargo Trains'] / 4165 * 0.45)
    vkm['HST'] = vkm['HST'] / 472
    vkm_rail = vkm[['Trains_VKM','regions']]
    vkm_HST = vkm[['HST','regions']]
    
    vkm_rail_pivoted = vkm_rail.pivot(columns='regions', values='Trains_VKM')
    vkm_rail_HST_pivoted = vkm_HST.pivot(columns='regions', values='HST')
    vkm_rail_pivoted.columns = region_list
    vkm_rail_HST_pivoted.columns = region_list
    
    vkm['total_vkm'] = vkm['Buses_vkm'] + vkm['Cars_vkm'] + vkm['Trucks_vkm']
    vkm = vkm[['total_vkm', 'regions']]
    vkm_pivoted = vkm.pivot(columns='regions', values='total_vkm')
    vkm_pivoted.columns = region_list

    # COVID correction for road VKM (v5 smoothing mechanism, same as rail)
    if 2020 in vkm_pivoted.index and 2019 in vkm_pivoted.index:
        vkm_pivoted.loc[2020] = vkm_pivoted.loc[2019]
    if 2021 in vkm_pivoted.index and 2022 in vkm_pivoted.index:
        vkm_pivoted.loc[2021] = vkm_pivoted.loc[2022]

    # --- 3. Process GDP & Population ---
    gdp_cap = gdp_cap.iloc[:, :-2]
    urban_pop_df = urban_pop_df.iloc[:, :-2]
    rural_pop_df = rural_pop_df.iloc[:, :-2]
    
    gdp_cap.columns = region_list
    urban_pop_df.columns = region_list
    rural_pop_df.columns = region_list
    
    gdp_cap = gdp_cap.rename_axis("years")
    urban_pop_df = urban_pop_df.rename_axis("years")
    rural_pop_df = rural_pop_df.rename_axis("years")

    # COVID correction for GDP (v5: smooth out COVID-related mobility changes)
    # Infrastructure does not decline with GDP during short dips
    if 2020 in gdp_cap.index and 2019 in gdp_cap.index:
        gdp_cap.loc[2020] = gdp_cap.loc[2019]
    if 2021 in gdp_cap.index and 2022 in gdp_cap.index:
        gdp_cap.loc[2021] = gdp_cap.loc[2022]

    gdp_cap_new = gdp_cap
    rural_pop_df = rural_pop_df.rename(columns=column_mapping)
    urban_pop_df = urban_pop_df.rename(columns=column_mapping)
    
    urban_population = urban_pop_df.sort_index(axis=1)
    rural_population = rural_pop_df.sort_index(axis=1)

    # --- 4. Process HDI ---
    hdi = pd.read_excel(tripi_dir / 'HDI-kedi-et-al-2024-SSP2.xlsx', index_col=0)
    hdi = hdi.reset_index()
    if 'Area' in hdi.columns: hdi = hdi.drop(columns=['Area'])
    if 'ISO3' in hdi.columns: hdi = hdi.drop(columns=['ISO3'])
    if 'ISOCode' in hdi.columns: hdi = hdi.drop(columns=['ISOCode'])
    
    IMAGE_hdi = hdi.groupby('IMAGE-region').mean(numeric_only=True)
    all_years = list(range(1970, 2101))
    IMAGE_hdi = IMAGE_hdi.reindex(columns=all_years)
    IMAGE_hdi = IMAGE_hdi.interpolate(method='linear', axis=1)
    
    IMAGE_hdi_transposed = IMAGE_hdi.T
    IMAGE_hdi_transposed = IMAGE_hdi_transposed.rename(columns=column_mapping)
    if 1970 in IMAGE_hdi_transposed.index:
        IMAGE_hdi_transposed = IMAGE_hdi_transposed.drop(index=1970)
    sorted_IMAGE_hdi = IMAGE_hdi_transposed.sort_index(axis=1)
    sorted_IMAGE_hdi = sorted_IMAGE_hdi.loc[sorted_IMAGE_hdi.index <= 2100]

    # --- 5. Process Land and Urban Areas ---
    area = pd.read_excel(tripi_dir / 'urban-area_ssp4_dense.xlsx', index_col=0)
    area = area.reset_index()
    for col in ['Cntry_name', 'eFUA_name', 'GRIP-region','R10']:
        if col in area.columns: area = area.drop(columns=[col])
        
    IMAGE_total_area = area.groupby('IMAGE-region')['FUA_total_area'].sum()
    area_clean = area.dropna()
    IMAGE_urban_share = area_clean.groupby('IMAGE-region').sum(numeric_only=True)
    for col in ['FUA_area', 'FUA_urban', 'FUA_p_2015']:
        if col in IMAGE_urban_share.columns: IMAGE_urban_share = IMAGE_urban_share.drop(columns=[col])
        
    IMAGE_urban_share = IMAGE_urban_share.div(IMAGE_urban_share['FUA_total_area'], axis=0)
    if 'FUA_total_area' in IMAGE_urban_share.columns: IMAGE_urban_share = IMAGE_urban_share.drop(columns=['FUA_total_area'])
    
    all_years_area = list(range(1870, 2101))
    IMAGE_urban_share = IMAGE_urban_share.reindex(columns=all_years_area)
    IMAGE_urban_share = IMAGE_urban_share.interpolate(method='linear', axis=1)

    # Smooth Urban Share
    years_to_smooth = range(1950, 2101)
    IMAGE_urban_share_subset = IMAGE_urban_share[years_to_smooth].copy()
    window_avg_size = 20
    smoothed_data = IMAGE_urban_share_subset.apply(lambda col: col.rolling(window=window_avg_size, center=True, min_periods=1).mean(), axis=1)
    IMAGE_urban_share[smoothed_data.columns] = smoothed_data
    
    IMAGE_rural_share = 1 - IMAGE_urban_share
    IMAGE_urban_area = IMAGE_urban_share.mul(IMAGE_total_area, axis=0)
    IMAGE_rural_area = IMAGE_rural_share.mul(IMAGE_total_area, axis=0)
    
    # Population density
    IMAGE_urban_area_filtered = IMAGE_urban_area.loc[:, 1971:2100].T
    IMAGE_rural_area_filtered = IMAGE_rural_area.loc[:, 1971:2100].T
    IMAGE_urban_area_filtered = IMAGE_urban_area_filtered.rename(columns=column_mapping).sort_index(axis=1)
    IMAGE_rural_area_filtered = IMAGE_rural_area_filtered.rename(columns=column_mapping).sort_index(axis=1)

    urban_pop_density = urban_population * 1000000 / IMAGE_urban_area_filtered
    rural_pop_density = rural_population * 1000000 / IMAGE_rural_area_filtered
    pop_density = ((urban_population + rural_population) * 1000000) / (IMAGE_urban_area_filtered + IMAGE_rural_area_filtered)
    
    sorted_urban_pop_density = urban_pop_density.sort_index(axis=1)
    sorted_rural_pop_density = rural_pop_density.sort_index(axis=1)
    sorted_pop_density = pop_density.sort_index(axis=1)

    # --- 6. Calculate Urban vs Rural GDP ---
    rural_share = rural_population / (rural_population + urban_population)
    urban_share = 1 - rural_share
    gdp_total = gdp_cap_new * ((rural_population + urban_population) * 1000000)
    
    def gdp_share_regression(x): return 1.0267 * x + 0.0412
    gdp_urban_share = urban_share.applymap(gdp_share_regression).clip(upper=0.995)
    gdp_rural_share = 1 - gdp_urban_share

    pop_u_mil = (urban_population * 1000000).replace(0, 1e-9)
    pop_r_mil = (rural_population * 1000000).replace(0, 1e-9)
    
    initial_gdp_urban_total = gdp_total * gdp_urban_share
    initial_gdp_rural_total = gdp_total * gdp_rural_share
    initial_gdp_urban_cap = initial_gdp_urban_total / pop_u_mil
    initial_gdp_rural_cap = initial_gdp_rural_total / pop_r_mil
    
    urban_floor_cap = (initial_gdp_urban_cap.cummax() * 0.75).fillna(0)
    rural_floor_cap = (initial_gdp_rural_cap.cummax() * 0.75).fillna(0)
    
    min_gdp_urban = urban_floor_cap * pop_u_mil
    min_gdp_rural = rural_floor_cap * pop_r_mil
    total_required = min_gdp_urban + min_gdp_rural
    is_unsolvable = total_required > gdp_total
    
    min_allowed_urban_gdp = min_gdp_urban
    max_allowed_urban_gdp = gdp_total - min_gdp_rural
    urban_solvable = initial_gdp_urban_total.clip(lower=min_allowed_urban_gdp, upper=max_allowed_urban_gdp)
    
    urban_share_of_required = min_gdp_urban / (total_required + 1e-9)
    urban_unsolvable = gdp_total * urban_share_of_required
    
    final_gdp_urban_total = np.where(is_unsolvable, urban_unsolvable, urban_solvable)
    final_gdp_urban_total = pd.DataFrame(final_gdp_urban_total, index=gdp_total.index, columns=gdp_total.columns)
    final_gdp_rural_total = gdp_total - final_gdp_urban_total

    gdp_urban_total = final_gdp_urban_total
    gdp_rural_total = final_gdp_rural_total
    gdp_urban_cap = gdp_urban_total / pop_u_mil
    gdp_rural_cap = gdp_rural_total / pop_r_mil
    
    sorted_gdp_urban_cap = gdp_urban_cap.sort_index(axis=1)
    sorted_gdp_rural_cap = gdp_rural_cap.sort_index(axis=1)

    # --- 7. Calculate Urban vs Rural VKM (Cars+Buses+Trucks) ---
    def vkm_urban_share_regression(x): return 0.1153 * x + 1637.4
    vkm_urban_cap = gdp_urban_cap.applymap(vkm_urban_share_regression)
    vkm_urban_tot = vkm_urban_cap * urban_population * 1000000
    
    def vkm_rural_share_regression(x): return 0.5603 * x + 78.578
    vkm_rural_cap = gdp_rural_cap.applymap(vkm_rural_share_regression)
    vkm_rural_tot = vkm_rural_cap * rural_population * 1000000
    
    vkm_urban_share = vkm_urban_tot / (vkm_urban_tot + vkm_rural_tot)
    vkm_rural_share = 1 - vkm_urban_share
    
    vkm_urban_IMAGE_cap = vkm_pivoted * vkm_urban_share / (urban_population * 1000000)
    vkm_rural_IMAGE_cap = vkm_pivoted * vkm_rural_share / (rural_population * 1000000)
    
    sorted_vkm_urban_cap = vkm_urban_IMAGE_cap.sort_index(axis=1)
    sorted_vkm_rural_cap = vkm_rural_IMAGE_cap.sort_index(axis=1)
    sorted_vkm_rail_tot = vkm_rail_pivoted.sort_index(axis=1)
    sorted_vkm_HST_rail_tot = vkm_rail_HST_pivoted.sort_index(axis=1)
    sorted_vkm_rail_cap = sorted_vkm_rail_tot / ((urban_population + rural_population) * 1000000)

    # --- 8. Calculate Infrastructure Areas (Roads and Parking) ---
    def rural_roads(x1, x2): return 0.000317871 + (-6.55403e-5 * np.log(x1)) + (x2 * 5.52978e-9)
    def urban_roads(x1, x2, x3): return 0.000118208 + (-1.37056e-5 * np.log(x1)) + (x2 * 2.56944e-9) + (x3 * 3.30002e-10)
    
    actual_rural_road_area_2024 = data_2024.loc['rural_roads']
    actual_urban_road_area_2024 = data_2024.loc['urban_roads']
    actual_rural_parking_area_2024 = data_2024.loc['rural_parking_area']
    actual_urban_parking_area_2024 = data_2024.loc['urban_parking']

    rural_road_cap = rural_roads(sorted_rural_pop_density, sorted_gdp_rural_cap)
    urban_road_cap = urban_roads(sorted_urban_pop_density, sorted_vkm_urban_cap, sorted_gdp_urban_cap)

    # Smooth Road Data
    std_dev = 5 
    min_positive_values_r = rural_road_cap.apply(lambda row: row[row > 0].nsmallest(3).mean() if (row > 0).sum() >= 3 else row[row > 0].mean(), axis=1)
    min_positive_values_r = pd.Series(gaussian_filter1d(min_positive_values_r, sigma=std_dev), index=min_positive_values_r.index)
    for col in rural_road_cap.columns:
        if (rural_road_cap[col] < 0).any(): rural_road_cap[col] = min_positive_values_r

    min_positive_values_u = urban_road_cap.apply(lambda row: row[row > 0].nsmallest(3).mean() if (row > 0).sum() >= 3 else row[row > 0].mean(), axis=1)
    min_positive_values_u = pd.Series(gaussian_filter1d(min_positive_values_u, sigma=std_dev), index=min_positive_values_u.index)
    for col in urban_road_cap.columns:
        if (urban_road_cap[col] < 0).any(): urban_road_cap[col] = min_positive_values_u

    smoothed_urban_road_cap = urban_road_cap.apply(lambda col: pd.Series(gaussian_filter1d(col, sigma=1), index=col.index), axis=0)
    smoothed_urban_road_cap.index = urban_road_cap.index
    smoothed_urban_road_cap.columns = urban_road_cap.columns
    urban_road_cap = smoothed_urban_road_cap

    # Adjustment logic to 2024
    # Years are in the INDEX, region names are in COLUMNS
    if 2024 in rural_road_cap.index:
        # road_cap at 2024 is a per-capita value; road_area = cap × population
        predicted_rural_road_area_2024 = rural_road_cap.loc[2024] * (rural_population.loc[2024] * 1000000)
        predicted_urban_road_area_2024 = urban_road_cap.loc[2024] * (urban_population.loc[2024] * 1000000)
        rural_adjustment_factor = actual_rural_road_area_2024 / predicted_rural_road_area_2024.replace(0, np.nan)
        urban_adjustment_factor = actual_urban_road_area_2024 / predicted_urban_road_area_2024.replace(0, np.nan)
        rural_adjustment_factor = rural_adjustment_factor.fillna(1.0)
        urban_adjustment_factor = urban_adjustment_factor.fillna(1.0)
    else:
        rural_adjustment_factor = 1.0
        urban_adjustment_factor = 1.0

    rural_road_area = rural_road_cap * (rural_population * 1000000) * rural_adjustment_factor
    urban_road_area = urban_road_cap * (urban_population * 1000000) * urban_adjustment_factor

    # Simplified Decline Prevention (Assuming ROAD_DECLINE_FUNCTION=1 logic)
    adjusted_rural_road_area = rural_road_area.copy()
    adjusted_urban_road_area = urban_road_area.copy()
    for df in [adjusted_rural_road_area, adjusted_urban_road_area]:
        for col in df.columns:
            # We enforce strictly non-decreasing logic over time
            df.loc[:, col] = df[col].cummax(axis=0)

    # Obsolete road area: gap between cummax'd and original (v5 lines 589-673)
    obsolete_road_rural_area = adjusted_rural_road_area - rural_road_area
    obsolete_road_urban_area = adjusted_urban_road_area - urban_road_area

    def _apply_obsolete_rule(series):
        """Zero out values before the last zero that precedes the final non-zero block."""
        processed = series.copy()
        n = len(series)
        last_nz_start = -1
        for i in range(n - 1, -1, -1):
            if series.iloc[i] != 0:
                last_nz_start = i
            else:
                if last_nz_start != -1:
                    break
        if last_nz_start == -1 or (series == 0).all():
            return series
        zero_idx = -1
        for i in range(last_nz_start - 1, -1, -1):
            if series.iloc[i] == 0:
                zero_idx = i
                break
        if zero_idx != -1:
            processed.iloc[:zero_idx + 1] = 0
        return processed

    for obs_df in [obsolete_road_rural_area, obsolete_road_urban_area]:
        for col in obs_df.columns:
            obs_df[col] = _apply_obsolete_rule(obs_df[col])
        # Fix columns that end with zero (v5 fix_obsolete_columns)
        for col in obs_df.columns:
            if obs_df[col].iloc[-1] == 0 and not (obs_df[col] == 0).all():
                obs_df[col] = 0

    # Subtract obsolete from adjusted to get true active stock (v5 line 672-673)
    adjusted_rural_road_area = adjusted_rural_road_area - obsolete_road_rural_area
    adjusted_urban_road_area = adjusted_urban_road_area - obsolete_road_urban_area

    # Parking as ratio of road network (same approach as bridges/tunnels)
    # 2024 data: parking in m², roads in km² → convert parking to km² for ratio
    rur_parking_ratio = (actual_rural_parking_area_2024 / 1000000) / actual_rural_road_area_2024
    urb_parking_ratio = (actual_urban_parking_area_2024 / 1000000) / actual_urban_road_area_2024
    rur_parking_ratio = rur_parking_ratio.fillna(0)
    urb_parking_ratio = urb_parking_ratio.fillna(0)

    rural_parking_area = adjusted_rural_road_area * rur_parking_ratio
    urban_parking_area = adjusted_urban_road_area * urb_parking_ratio

    # Obsolete parking (same ratio applied to obsolete road area)
    obsolete_parking_rural = obsolete_road_rural_area * rur_parking_ratio
    obsolete_parking_urban = obsolete_road_urban_area * urb_parking_ratio

    # --- Parking paving (single GDP regression, v5 lines 710-725) ---
    def rural_paving(x): return 0.1244 * math.log(x) - 0.789
    def urban_paving(x): return 0.151 * math.log(x) - 0.7656

    rural_paving_share = sorted_gdp_rural_cap.applymap(rural_paving).clip(lower=0.0, upper=1.0)
    urban_paving_share = sorted_gdp_urban_cap.applymap(urban_paving).clip(lower=0.0, upper=1.0)

    rural_paved_parking_area = rural_parking_area * rural_paving_share
    rural_unpaved_parking_area = rural_parking_area * (1 - rural_paving_share)
    urban_paved_parking_area = urban_parking_area * urban_paving_share
    urban_unpaved_parking_area = urban_parking_area * (1 - urban_paving_share)

    # Obsolete parking paved/unpaved splits
    rural_paved_obsolete_parking = obsolete_parking_rural * rural_paving_share
    rural_unpaved_obsolete_parking = obsolete_parking_rural * (1 - rural_paving_share)
    urban_paved_obsolete_parking = obsolete_parking_urban * urban_paving_share
    urban_unpaved_obsolete_parking = obsolete_parking_urban * (1 - urban_paving_share)

    # --- 9. Distribute Areas by HDI Road Type + GDP Per-Type Paving ---
    # v5 approach: 1) split by type (HDI), 2) apply per-type paving rate (GDP)

    # Load road type distributions depending on HDI
    u_types = pd.read_excel(tripi_dir / 'urban-type-distribution.xlsx', index_col=0)
    r_types = pd.read_excel(tripi_dir / 'rural-type-distribution.xlsx', index_col=0)
    road_type_cols = u_types.columns  # e.g. ['cycle_share', 'informal_share', ...]

    # Interpolate type distributions to 0.01 HDI resolution
    hdi_bins = np.arange(0, 1.01, 0.01)
    u_types_int = u_types.reindex(hdi_bins).interpolate(method='linear', limit_direction='both')
    r_types_int = r_types.reindex(hdi_bins).interpolate(method='linear', limit_direction='both')
    u_types_int.index = np.round(u_types_int.index, 2)
    r_types_int.index = np.round(r_types_int.index, 2)

    hdi_rounded_u = sorted_IMAGE_hdi.round(2).clip(lower=0, upper=1)
    hdi_rounded_r = sorted_IMAGE_hdi.round(2).clip(lower=0, upper=1)

    # Load per-type paving distributions (GDP-based, v5 lines 840-960)
    u_paving_dist = pd.read_excel(tripi_dir / 'urban-paving-distribution.xlsx', index_col=0)
    r_paving_dist = pd.read_excel(tripi_dir / 'rural-paving-distribution.xlsx', index_col=0)

    # Interpolate paving distributions to 1000 GDP resolution
    u_gdp_bins = np.arange(0, 150001, 1000)
    r_gdp_bins = np.arange(0, 200001, 1000)
    u_paving_int = u_paving_dist.reindex(u_gdp_bins).interpolate(method='linear', limit_direction='both')
    r_paving_int = r_paving_dist.reindex(r_gdp_bins).interpolate(method='linear', limit_direction='both')
    u_paving_int = u_paving_int.clip(lower=0.0, upper=1.0)
    r_paving_int = r_paving_int.clip(lower=0.0, upper=1.0)

    # GDP binned to 1000 resolution for paving lookup
    # Rural: cummax GDP so paving rates only increase over time (v5 ratchet effect)
    u_gdp_binned = (sorted_gdp_urban_cap / 1000).round(0).astype(int).clip(lower=0, upper=150) * 1000
    r_gdp_for_paving = sorted_gdp_rural_cap.cummax()
    r_gdp_binned = (r_gdp_for_paving / 1000).round(0).astype(int).clip(lower=0, upper=200) * 1000

    # Map paving column names to road type labels
    paving_col_map = {
        'cycle': 'Paved share cycle', 'informal': 'Paved share informal',
        'local': 'Paved share local', 'motorway': 'Paved share motorway',
        'pedestrian': 'Paved share pedestrian', 'primary': 'Paved share primary',
        'secondary': 'Paved share secondary', 'tertiary': 'Paved share tertiary'
    }

    # Initialize xarray with structured dimensions
    original_time_coords = list(adjusted_urban_road_area.index)
    region_coords = list(adjusted_urban_road_area.columns)

    # Stock tail: extend time back to FIRST_YEAR_GRID (1911)
    end_year = max(original_time_coords)
    extended_time_coords = list(range(FIRST_YEAR_GRID, end_year + 1))

    road_labels = [c.replace('_share', '') for c in road_type_cols]

    def _build_road_type_das_v2(total_area_df, obs_area_df, loc, type_int, hdi_rnd,
                                 paving_int, gdp_binned, prefix_active, prefix_obs,
                                 time_coords_orig, ext_time_coords, region_cols):
        """Build paved/unpaved road type DataArrays using per-type paving rates.

        v5 approach: type_area = total_area × type_share(HDI)
                     paved = type_area × paving_rate(GDP, type)
                     unpaved = type_area × (1 - paving_rate)
        """
        active_types = []
        active_das = []
        obs_types = []
        obs_das = []

        for col, label in zip(road_type_cols, road_labels):
            # 1. Map HDI → type share for each year/region
            share_mapped = pd.DataFrame(index=hdi_rnd.index, columns=hdi_rnd.columns)
            for y in time_coords_orig:
                if y in hdi_rnd.index:
                    share_mapped.loc[y] = type_int.loc[hdi_rnd.loc[y], col].values
            share_mapped = share_mapped.fillna(0)

            # 2. Map GDP → per-type paving rate for each year/region
            pave_col = paving_col_map[label]
            pave_rate = pd.DataFrame(index=gdp_binned.index, columns=gdp_binned.columns)
            for y in time_coords_orig:
                if y in gdp_binned.index:
                    gdp_vals = gdp_binned.loc[y]
                    pave_rate.loc[y] = paving_int.loc[gdp_vals, pave_col].values
            pave_rate = pave_rate.fillna(0).clip(lower=0, upper=1)

            # 3. Compute type areas
            type_total = total_area_df * share_mapped
            type_obs = obs_area_df * share_mapped

            # 4. Apply per-type paving rate
            for pave_status, rate_mult in [('paved', pave_rate), ('unpaved', 1 - pave_rate)]:
                # Active
                type_name = f"{prefix_active}{loc}_{label}_{pave_status}"
                active_types.append(type_name)
                final_area = type_total * rate_mult
                # Stock_tail
                final_area.index = pd.Index(pd.to_numeric(final_area.index, errors='coerce').astype('Int64'))
                final_area = final_area.apply(pd.to_numeric, errors='coerce')
                final_area.loc[FIRST_YEAR_GRID] = 0.0
                final_area = final_area.reindex(ext_time_coords).interpolate().fillna(0.0)
                active_das.append(xr.DataArray(final_area.values,
                    coords=[ext_time_coords, region_cols], dims=['Time', 'Region']))

                # Obsolete
                obs_name = f"{prefix_obs}{loc}_{label}_{pave_status}"
                obs_types.append(obs_name)
                obs_final = type_obs * rate_mult
                obs_final.index = pd.Index(pd.to_numeric(obs_final.index, errors='coerce').astype('Int64'))
                obs_final = obs_final.apply(pd.to_numeric, errors='coerce')
                obs_final.loc[FIRST_YEAR_GRID] = 0.0
                obs_final = obs_final.reindex(ext_time_coords).interpolate().fillna(0.0)
                obs_das.append(xr.DataArray(obs_final.values,
                    coords=[ext_time_coords, region_cols], dims=['Time', 'Region']))

        return active_types, active_das, obs_types, obs_das

    # Build urban road types
    u_active, u_active_das, u_obs, u_obs_das = _build_road_type_das_v2(
        adjusted_urban_road_area, obsolete_road_urban_area,
        'urban', u_types_int, hdi_rounded_u, u_paving_int, u_gdp_binned,
        "", "obsolete_", original_time_coords, extended_time_coords, region_coords)

    # Build rural road types
    r_active, r_active_das, r_obs, r_obs_das = _build_road_type_das_v2(
        adjusted_rural_road_area, obsolete_road_rural_area,
        'rural', r_types_int, hdi_rounded_r, r_paving_int, r_gdp_binned,
        "", "obsolete_", original_time_coords, extended_time_coords, region_coords)

    all_types = u_active + r_active
    da_list = u_active_das + r_active_das
    obsolete_types = u_obs + r_obs
    obsolete_da_list = u_obs_das + r_obs_das

    # Combine active + obsolete
    combined_types = all_types + obsolete_types
    combined_da_list = da_list + obsolete_da_list

    da_roads = xr.concat(combined_da_list, pd.Index(combined_types, name="Type"))
    da_roads = da_roads.assign_attrs({"units": "km**2"})  # attrs for intermediate; final stocks uses prism.Q_()

    preprocessing_results["stocks"] = da_roads
    preprocessing_results["active_types"] = all_types
    preprocessing_results["obsolete_types"] = obsolete_types

    infra_graph = KnowledgeGraph(Node("Infrastructure"))
    for t in combined_types:
        infra_graph.add(Node(t, inherits_from="Infrastructure"))
    preprocessing_results["knowledge_graph"] = infra_graph
    preprocessing_results["shares"] = None

    # --- 9b. Compute Bridges, Tunnels, Parking stocks ---
    # Bridge & Tunnel ratios from 2024 data (v5 lines 1391-1411)
    actual_rural_bridges_2024 = data_2024.loc['rural_bridges']
    actual_rural_tunnels_2024 = data_2024.loc['rural_tunnels']
    actual_urban_bridges_2024 = data_2024.loc['urban_bridges']
    actual_urban_tunnels_2024 = data_2024.loc['urban_tunnels']
    actual_rural_road_area_2024 = data_2024.loc['rural_roads']
    actual_urban_road_area_2024_bt = data_2024.loc['urban_roads']

    rur_brid_ratio = actual_rural_bridges_2024 / actual_rural_road_area_2024
    rur_tun_ratio = actual_rural_tunnels_2024 / actual_rural_road_area_2024
    urb_brid_ratio = actual_urban_bridges_2024 / actual_urban_road_area_2024_bt
    urb_tun_ratio = actual_urban_tunnels_2024 / actual_urban_road_area_2024_bt

    # Bridge and tunnel areas = ratio * adjusted road area (v5 lines 1403-1411)
    rural_road_bridges = adjusted_rural_road_area * rur_brid_ratio
    urban_road_bridges = adjusted_urban_road_area * urb_brid_ratio
    rural_road_tunnels = adjusted_rural_road_area * rur_tun_ratio
    urban_road_tunnels = adjusted_urban_road_area * urb_tun_ratio

    # Obsolete bridges and tunnels
    obsolete_bridges_rural = obsolete_road_rural_area * rur_brid_ratio
    obsolete_bridges_urban = obsolete_road_urban_area * urb_brid_ratio
    obsolete_tunnels_rural = obsolete_road_rural_area * rur_tun_ratio
    obsolete_tunnels_urban = obsolete_road_urban_area * urb_tun_ratio

    # Load bridge/tunnel type shares (v5 lines 1413-1417)
    bt_sheets = pd.read_excel(tripi_dir / 'material-intensity-bridtun.xlsx',
                              index_col=0, sheet_name=None)
    rur_brid_share = bt_sheets['rur-bridge-share'].rename(index=column_mapping)
    rur_tun_share = bt_sheets['rur-tunnel-share'].rename(index=column_mapping)
    urb_brid_share = bt_sheets['urb-bridge-share'].rename(index=column_mapping)
    urb_tun_share = bt_sheets['urb-tunnel-share'].rename(index=column_mapping)

    # Map bridge/tunnel share column names to road type labels
    bt_col_to_label = {
        'bike_b': 'cycle', 'informal_b': 'informal', 'local_b': 'local',
        'motorway_b': 'motorway', 'pedestrian_b': 'pedestrian',
        'primary_b': 'primary', 'secondary_b': 'secondary', 'tertiary_b': 'tertiary'
    }
    tun_col_to_label = {
        'cycle_t': 'cycle', 'informal_t': 'informal', 'local_t': 'local',
        'motorway_t': 'motorway', 'pedestrian_t': 'pedestrian',
        'primary_t': 'primary', 'secondary_t': 'secondary', 'tertiary_t': 'tertiary'
    }

    def _distribute_by_type_shares(total_df, shares_df, col_map, prefix, ext_tc, reg_c):
        """Distribute total bridge/tunnel area by road type shares, apply stock_tail."""
        types_list = []
        da_items = []
        for col, label in col_map.items():
            type_name = f"{prefix}_{label}"
            types_list.append(type_name)
            # Multiply by regional share (shares_df has regions as index)
            area = total_df.copy()
            for reg in area.columns:
                if reg in shares_df.index:
                    area[reg] = area[reg] * shares_df.loc[reg, col]
                else:
                    area[reg] = 0.0
            # Apply stock_tail
            area.index = pd.Index(pd.to_numeric(area.index, errors='coerce').astype('Int64'))
            area = area.apply(pd.to_numeric, errors='coerce')
            area.loc[FIRST_YEAR_GRID] = 0.0
            area = area.reindex(ext_tc).interpolate().fillna(0.0)
            da = xr.DataArray(area.values, coords=[ext_tc, reg_c], dims=['Time', 'Region'])
            da_items.append(da)
        return types_list, da_items

    # Active bridges/tunnels per type
    bt_settings = [
        (rural_road_bridges, rur_brid_share, bt_col_to_label, 'rural_bridge'),
        (urban_road_bridges, urb_brid_share, bt_col_to_label, 'urban_bridge'),
        (rural_road_tunnels, rur_tun_share, tun_col_to_label, 'rural_tunnel'),
        (urban_road_tunnels, urb_tun_share, tun_col_to_label, 'urban_tunnel'),
    ]
    obs_bt_settings = [
        (obsolete_bridges_rural, rur_brid_share, bt_col_to_label, 'obsolete_rural_bridge'),
        (obsolete_bridges_urban, urb_brid_share, bt_col_to_label, 'obsolete_urban_bridge'),
        (obsolete_tunnels_rural, rur_tun_share, tun_col_to_label, 'obsolete_rural_tunnel'),
        (obsolete_tunnels_urban, urb_tun_share, tun_col_to_label, 'obsolete_urban_tunnel'),
    ]

    bt_active_types = []
    bt_active_das = []
    for total_df, shares_df, col_map, prefix in bt_settings:
        t, d = _distribute_by_type_shares(total_df, shares_df, col_map, prefix,
                                          extended_time_coords, region_coords)
        bt_active_types.extend(t)
        bt_active_das.extend(d)

    bt_obs_types = []
    bt_obs_das = []
    for total_df, shares_df, col_map, prefix in obs_bt_settings:
        t, d = _distribute_by_type_shares(total_df, shares_df, col_map, prefix,
                                          extended_time_coords, region_coords)
        bt_obs_types.extend(t)
        bt_obs_das.extend(d)

    # Parking stocks (already computed: rural/urban paved/unpaved parking areas)
    # v5: parking uses same MI as local roads, lifetime rc3 (urban) / rc2 (rural)
    parking_active_settings = [
        ('urban_parking_paved', urban_paved_parking_area),
        ('urban_parking_unpaved', urban_unpaved_parking_area),
        ('rural_parking_paved', rural_paved_parking_area),
        ('rural_parking_unpaved', rural_unpaved_parking_area),
    ]
    parking_obsolete_settings = [
        ('obsolete_urban_parking_paved', urban_paved_obsolete_parking),
        ('obsolete_urban_parking_unpaved', urban_unpaved_obsolete_parking),
        ('obsolete_rural_parking_paved', rural_paved_obsolete_parking),
        ('obsolete_rural_parking_unpaved', rural_unpaved_obsolete_parking),
    ]
    parking_types = []
    parking_das = []
    parking_obs_types = []
    parking_obs_das = []
    for settings_list, types_out, das_out in [
        (parking_active_settings, parking_types, parking_das),
        (parking_obsolete_settings, parking_obs_types, parking_obs_das),
    ]:
        for pname, parea in settings_list:
            types_out.append(pname)
            parea_c = parea.copy()
            parea_c.index = pd.Index(pd.to_numeric(parea_c.index, errors='coerce').astype('Int64'))
            parea_c = parea_c.apply(pd.to_numeric, errors='coerce')
            parea_c.loc[FIRST_YEAR_GRID] = 0.0
            parea_c = parea_c.reindex(extended_time_coords).interpolate().fillna(0.0)
            da = xr.DataArray(parea_c.values, coords=[extended_time_coords, region_coords],
                              dims=['Time', 'Region'])
            das_out.append(da)

    # --- 9c. Compute Rail stocks ---
    # Rail length via VKM-linked elasticity model (v5 lines 1093-1242)
    MAX_ANNUAL_CHANGE_PERCENT = 0.03
    LOOKAHEAD_YEARS = 10
    MIN_ELASTICITY = 0.1
    MAX_ELASTICITY = 0.6
    GROWTH_MIDPOINT = 0.03
    STEEPNESS = 200

    def _get_dynamic_elasticity(growth_rate):
        return MIN_ELASTICITY + (MAX_ELASTICITY - MIN_ELASTICITY) / (
            1 + np.exp(-STEEPNESS * (growth_rate - GROWTH_MIDPOINT)))

    # Load rail historic length data
    rail_lengths_df = pd.read_excel(tripi_dir / 'rail_historic_length.xlsx', index_col=0)
    rail_lengths_df.columns = rail_lengths_df.columns.astype(str)

    # VKM rail data (already loaded: sorted_vkm_rail_tot, sorted_vkm_HST_rail_tot)
    vkm_data_rail = sorted_vkm_rail_tot.copy()
    vkm_data_rail.index = vkm_data_rail.index.astype(int)
    vkm_data_rail.columns = vkm_data_rail.columns.astype(str) if not isinstance(
        vkm_data_rail.columns[0], str) else vkm_data_rail.columns

    # Align rail_lengths_df columns with vkm regions
    # rail_lengths_df has region names as index, years as columns
    # We need: years as index, regions as columns (like vkm_data_rail)
    # First convert to: index=years, columns=regions
    vkm_data_rail_c = vkm_data_rail.copy()
    vkm_data_rail_c.columns = [str(c) for c in vkm_data_rail_c.columns]

    # Fix VKM data for 2020-2021 (COVID correction, v5 lines 1081-1087)
    if 2020 in vkm_data_rail_c.index and 2019 in vkm_data_rail_c.index:
        vkm_data_rail_c.loc[2020] = vkm_data_rail_c.loc[2019]
    if 2021 in vkm_data_rail_c.index and 2022 in vkm_data_rail_c.index:
        vkm_data_rail_c.loc[2021] = vkm_data_rail_c.loc[2022]
    vkm_data_rail_c = vkm_data_rail_c.sort_index()

    # HST share computation
    vkm_hst = sorted_vkm_HST_rail_tot.copy()
    vkm_hst.index = vkm_hst.index.astype(int)
    if 2020 in vkm_hst.index and 2019 in vkm_hst.index:
        vkm_hst.loc[2020] = vkm_hst.loc[2019]
    if 2021 in vkm_hst.index and 2022 in vkm_hst.index:
        vkm_hst.loc[2021] = vkm_hst.loc[2022]
    vkm_hst = vkm_hst.sort_index()

    HST_share = (vkm_hst / vkm_data_rail_c).fillna(0)

    # Backcast HST shares (v5 lines 1277-1282)
    def _backcast_linear(df, region, anchor_year, start_year_bc=None, min_value=0.01):
        out = df.sort_index().copy()
        if anchor_year not in out.index:
            return out
        if region not in out.columns:
            return out
        if start_year_bc is None:
            start_year_bc = out.index.min()
        y = out[region].astype(float)
        post_mask = (out.index >= anchor_year) & y.notna()
        x_post = out.index[post_mask].values.astype(float)
        y_post = y[post_mask].values
        if len(x_post) < 2:
            return out
        slope, intercept = np.polyfit(x_post[:min(10, len(x_post))], y_post[:min(10, len(y_post))], 1)
        anchor_val = y.loc[anchor_year]
        pre_years = out.index[(out.index < anchor_year) & (out.index >= start_year_bc)].astype(float)
        if pre_years.size:
            backcast_vals = anchor_val - slope * (anchor_year - pre_years)
            backcast_vals = np.clip(backcast_vals, min_value, None)
            out.loc[pre_years.astype(int), region] = backcast_vals
        return out

    HST_share = _backcast_linear(HST_share, 'Japan', 2010)
    HST_share = _backcast_linear(HST_share, 'Western Europe', 2010, start_year_bc=1980)
    HST_share = _backcast_linear(HST_share, 'Korea', 2010, start_year_bc=1990)
    HST_share = _backcast_linear(HST_share, 'China', 2010, start_year_bc=2000)

    # Forecast rail length 2024-2100 (v5 lines 1122-1193)
    backlog = pd.Series(0.0, index=rail_lengths_df.index)
    for year in range(2024, 2101):
        year_str = str(year)
        prev_year_str = str(year - 1)
        if prev_year_str not in rail_lengths_df.columns:
            continue
        L_prev = rail_lengths_df[prev_year_str]
        future_year = min(year + LOOKAHEAD_YEARS, 2100)
        years_ahead = max(1, future_year - year)
        if year not in vkm_data_rail_c.index or future_year not in vkm_data_rail_c.index:
            rail_lengths_df[year_str] = L_prev
            continue
        VKM_curr = vkm_data_rail_c.loc[year]
        VKM_future = vkm_data_rail_c.loc[future_year]
        vkm_growth_rate = ((VKM_future - VKM_curr) / VKM_curr.replace(0, np.nan)) / years_ahead
        vkm_growth_rate = vkm_growth_rate.fillna(0)
        current_elasticity = _get_dynamic_elasticity(vkm_growth_rate)
        active_growth = np.clip(vkm_growth_rate, 0, None)
        backlog = np.where(active_growth > 0, backlog, 0)
        backlog = pd.Series(backlog, index=L_prev.index)
        desired_new_track = (L_prev * active_growth * current_elasticity) + backlog
        max_build_allowed = L_prev * MAX_ANNUAL_CHANGE_PERCENT
        actual_build = np.minimum(desired_new_track, max_build_allowed)
        backlog = desired_new_track - actual_build
        # Decline mechanic
        DECLINE_LAG = 5
        DECLINE_WINDOW = 10
        min_vkm_year = int(vkm_data_rail_c.index.min())
        past_lag_year = max(year - DECLINE_LAG, min_vkm_year)
        past_window_start = max(past_lag_year - DECLINE_WINDOW, min_vkm_year)
        VKM_past_lag = vkm_data_rail_c.loc[past_lag_year]
        VKM_past_start = vkm_data_rail_c.loc[past_window_start]
        is_sustained_decline = VKM_past_lag < VKM_past_start
        years_in_decline = past_lag_year - past_window_start
        if years_in_decline > 0:
            decline_rate = (VKM_past_lag - VKM_past_start) / VKM_past_start.replace(0, np.nan) / years_in_decline
        else:
            decline_rate = pd.Series(0, index=VKM_past_lag.index)
        decline_rate = decline_rate.fillna(0)
        active_decline = np.where((active_growth == 0) & is_sustained_decline, decline_rate, 0)
        actual_decline_amount = L_prev * active_decline * MIN_ELASTICITY
        min_decline_allowed = -L_prev * MAX_ANNUAL_CHANGE_PERCENT
        actual_decline_capped = np.maximum(actual_decline_amount, min_decline_allowed)
        rail_lengths_df[year_str] = L_prev + actual_build + actual_decline_capped

    # Hindcast 1995-1971 (v5 lines 1196-1237)
    for year in range(1995, 1970, -1):
        year_str = str(year)
        prev_year_str = str(year - 1)
        if year_str not in rail_lengths_df.columns:
            continue
        L_t = rail_lengths_df[year_str]
        if year not in vkm_data_rail_c.index or (year - 1) not in vkm_data_rail_c.index:
            rail_lengths_df[prev_year_str] = L_t
            continue
        VKM_t = vkm_data_rail_c.loc[year]
        VKM_prev = vkm_data_rail_c.loc[year - 1]
        vkm_growth_rate = (VKM_t - VKM_prev) / VKM_prev.replace(0, np.nan)
        vkm_growth_rate = vkm_growth_rate.fillna(0)
        historical_elasticity = _get_dynamic_elasticity(vkm_growth_rate)
        combined_rate = vkm_growth_rate * historical_elasticity
        L_prev_calculated = L_t / (1 + combined_rate)
        growth_amount = L_t - L_prev_calculated
        max_change_allowed = L_t * MAX_ANNUAL_CHANGE_PERCENT
        capped_growth = np.clip(growth_amount, -max_change_allowed, max_change_allowed)
        rail_lengths_df[prev_year_str] = L_t - capped_growth

    # Sort columns and transpose to years-as-index format
    final_columns = sorted(rail_lengths_df.columns, key=lambda x: int(x))
    rail_length = rail_lengths_df[final_columns].T
    rail_length = rail_length.iloc[1:] if rail_length.index[0] == rail_length.columns[0] else rail_length
    rail_length.index = pd.Index(pd.to_numeric(rail_length.index, errors='coerce').astype('Int64'))

    # HST length (v5 lines 1284-1286)
    HST_length = rail_length * HST_share.reindex(rail_length.index).fillna(0)
    HST_length = HST_length.rolling(window=10, min_periods=3, center=True).mean().fillna(0)

    # Decline prevention for rail and HST (v5 lines 1290-1370)
    adjusted_rail_length = rail_length.copy()
    for col in adjusted_rail_length.columns:
        adjusted_rail_length[col] = adjusted_rail_length[col].cummax()
    obsolete_rail_length = adjusted_rail_length - rail_length

    # Apply obsolete rule to rail
    for col in obsolete_rail_length.columns:
        obsolete_rail_length[col] = _apply_obsolete_rule(obsolete_rail_length[col])
    for col in obsolete_rail_length.columns:
        if obsolete_rail_length[col].iloc[-1] == 0 and not (obsolete_rail_length[col] == 0).all():
            obsolete_rail_length[col] = 0
    adjusted_rail_length = adjusted_rail_length - obsolete_rail_length

    adjusted_HST_length = HST_length.copy()
    for col in adjusted_HST_length.columns:
        adjusted_HST_length[col] = adjusted_HST_length[col].cummax()

    # Rail bridge/tunnel shares (v5 lines 1377-1389)
    bridge_rail_share = data_2024.loc['rail_bridge_share']
    tunnel_rail_share = data_2024.loc['rail_tunnel_share']
    bridge_rail = adjusted_rail_length * bridge_rail_share
    tunnel_rail = adjusted_rail_length * tunnel_rail_share
    bridge_rail_obsolete = obsolete_rail_length * bridge_rail_share
    tunnel_rail_obsolete = obsolete_rail_length * tunnel_rail_share

    # Urban rail length (v5 lines 1054-1069)
    def _urban_rail_func(hdi, upd):
        return 2.718**-2.048095 * (hdi ** 8.0777) / (upd ** 0.5885)

    urban_rail_cap = _urban_rail_func(sorted_IMAGE_hdi, sorted_urban_pop_density)
    actual_urban_rail_2024 = data_2024.loc['urban_rail']
    if 2024 in urban_rail_cap.index:
        predicted_url_2024 = urban_rail_cap.loc[2024] * (urban_population.loc[2024] * 1000000)
        urban_rail_adj = actual_urban_rail_2024 / predicted_url_2024
    else:
        urban_rail_adj = 1.0
    urban_rail_length = urban_rail_cap * (urban_population * 1000000) * urban_rail_adj

    # Build rail/bridge/tunnel stock DataArrays with stock_tail
    def _stock_tail_da(df, ext_tc, reg_c):
        """Apply stock_tail to a DataFrame and return DataArray."""
        df_c = df.copy()
        df_c.index = pd.Index(pd.to_numeric(df_c.index, errors='coerce').astype('Int64'))
        df_c = df_c.apply(pd.to_numeric, errors='coerce')
        df_c.loc[FIRST_YEAR_GRID] = 0.0
        df_c = df_c.reindex(ext_tc).interpolate().fillna(0.0)
        return xr.DataArray(df_c.values, coords=[ext_tc, reg_c], dims=['Time', 'Region'])

    rail_element_types = []
    rail_element_das = []
    rail_obs_types = []
    rail_obs_das = []

    # Active rail elements
    for name, df in [('total_rail', adjusted_rail_length),
                     ('total_HST', adjusted_HST_length),
                     ('total_urban_rail', urban_rail_length),
                     ('rail_bridge', bridge_rail),
                     ('rail_tunnel', tunnel_rail)]:
        rail_element_types.append(name)
        rail_element_das.append(_stock_tail_da(df, extended_time_coords, region_coords))

    # Obsolete rail elements
    for name, df in [('obsolete_total_rail', obsolete_rail_length),
                     ('obsolete_rail_bridge', bridge_rail_obsolete),
                     ('obsolete_rail_tunnel', tunnel_rail_obsolete)]:
        rail_obs_types.append(name)
        rail_obs_das.append(_stock_tail_da(df, extended_time_coords, region_coords))

    # --- 9d. Combine ALL stocks into single DataArray ---
    # Roads (active + obsolete) + bridges/tunnels + parking (active + obsolete) + rail
    final_all_types = (combined_types + bt_active_types + bt_obs_types +
                       parking_types + parking_obs_types +
                       rail_element_types + rail_obs_types)
    final_all_das = (combined_da_list + bt_active_das + bt_obs_das +
                     parking_das + parking_obs_das +
                     rail_element_das + rail_obs_das)

    da_all = xr.concat(final_all_das, pd.Index(final_all_types, name="Type"))
    da_all = prism.Q_(da_all, "km**2")

    # reorder da_roads to have the order: TIME, REGION, TYPE
    da_all = da_all.transpose("Time", "Region", "Type")

    preprocessing_results["stocks"] = da_all

    # Update type tracking for mass balance
    # Active types now include roads + bridges/tunnels + parking + rail
    all_active_types = all_types + bt_active_types + parking_types + rail_element_types
    all_obsolete_types = obsolete_types + bt_obs_types + parking_obs_types + rail_obs_types
    preprocessing_results["active_types"] = all_active_types
    preprocessing_results["obsolete_types"] = all_obsolete_types

    # Map obsolete types to their active counterparts for mass balance
    # Road obsolete -> road active (already paired by index)
    # Bridge/tunnel obsolete -> bridge/tunnel active (paired by index)
    # Parking obsolete -> parking active (paired by index)
    # Rail obsolete -> rail active (need explicit mapping)
    obs_to_active_map = {}
    for a, o in zip(all_types, obsolete_types):
        obs_to_active_map[o] = a
    for a, o in zip(bt_active_types, bt_obs_types):
        obs_to_active_map[o] = a
    for a, o in zip(parking_types, parking_obs_types):
        obs_to_active_map[o] = a
    obs_to_active_map['obsolete_total_rail'] = 'total_rail'
    obs_to_active_map['obsolete_rail_bridge'] = 'rail_bridge'
    obs_to_active_map['obsolete_rail_tunnel'] = 'rail_tunnel'
    preprocessing_results["obsolete_to_active_map"] = obs_to_active_map

    # Update KnowledgeGraph
    infra_graph = KnowledgeGraph(Node("Infrastructure"))
    for t in final_all_types:
        infra_graph.add(Node(t, inherits_from="Infrastructure"))
    preprocessing_results["knowledge_graph"] = infra_graph

    preprocessing_results["set_unit_flexible"] = prism.U_(da_all)

    # --- 10. Process Material Intensities and Lifetimes ---
    mat_int = pd.read_excel(tripi_dir / 'material-intensity-roads.xlsx', index_col=0)
    weibull = pd.read_excel(grid_dir / 'infra_lifetime_weibull.xlsx', index_col=0)
    materials_paving = pd.read_excel(infra_dir / 'Materials-paving.xlsx', index_col=0)

    # GRIP road class map for material intensity lookup (1=motorway ... 5=local)
    mi_road_class_map = {
        'motorway': 1, 'primary': 2, 'secondary': 3, 'tertiary': 4,
        'cycle': 5, 'informal': 5, 'local': 5, 'pedestrian': 5
    }

    # Lifetime road class map from original v5 script (infra_lifetime_weibull.xlsx)
    # Higher class = more durable. Urban roads get higher classes than rural equivalents.
    lifetime_rc_map = {
        'urban_paved':   {'motorway': 7, 'primary': 6, 'secondary': 5, 'tertiary': 4, 'cycle': 1, 'informal': 1, 'local': 3, 'pedestrian': 1},
        'urban_unpaved': {'motorway': 7, 'primary': 6, 'secondary': 5, 'tertiary': 4, 'cycle': 1, 'informal': 1, 'local': 3, 'pedestrian': 1},
        'rural_paved':   {'motorway': 6, 'primary': 4, 'secondary': 3, 'tertiary': 2, 'cycle': 1, 'informal': 1, 'local': 1, 'pedestrian': 1},
        'rural_unpaved': {'motorway': 6, 'primary': 4, 'secondary': 3, 'tertiary': 2, 'cycle': 1, 'informal': 1, 'local': 1, 'pedestrian': 1},
    }

    # --- Material intensities via Materials-paving.xlsx ---
    # Parse material shares: group by Road_type, Region_type, Paving_type
    paving_shares_dict = {}
    for road_type, road_group in materials_paving.groupby('Road_type'):
        for region_type, region_group in road_group.groupby('Region_type'):
            for paving_type, paving_group in region_group.groupby('Paving_type'):
                # Extract the 3-letter abbreviation as used in original v5
                var_key = f"{road_type[:3]}_{region_type[:3]}_{paving_type[:3]}"
                pg = paving_group.iloc[:, 3:]  # Drop Road_type, Region_type, Paving_type columns
                pg.index = pg.index.str.split('_').str[-1]  # Extract material name from index
                paving_shares_dict[var_key] = pg

    # Pavingstone decomposition factors (from original v5 TRIPI 2024)
    pavingstone_factors = {'brick': 0.0381, 'stone': 0.2705, 'concrete': 0.6913}

    # Base materials that contribute to final material intensities
    base_materials = ['asphalt', 'brick', 'concrete', 'metal', 'stone', 'wood']
    output_materials = ['asphalt', 'brick', 'concrete', 'steel', 'stone', 'wood', 'aggregate']

    # Compute material intensity per road type (kg/km2) per region
    # Build a DataArray with dims (Type, material, Region)
    mi_per_type = {}  # type_name -> dict(material -> Series(region_values))

    for t in all_types:
        parts = t.split('_')
        loc_type = parts[0]     # 'urban' or 'rural'
        road_type = parts[1]    # 'motorway', 'primary', etc.
        pave_type = parts[2]    # 'paved' or 'unpaved'
        grip_rc = mi_road_class_map.get(road_type, 5)

        # Map to Materials-paving key format
        road_abbr_map = {
            'motorway': 'mot', 'primary': 'pri', 'secondary': 'sec', 'tertiary': 'ter',
            'cycle': 'cyc', 'informal': 'inf', 'local': 'loc', 'pedestrian': 'ped'
        }
        region_abbr_map = {'urban': 'urb', 'rural': 'rur'}
        pave_abbr_map = {'paved': 'pav', 'unpaved': 'unp'}

        road_abbr = road_abbr_map.get(road_type, road_type[:3])
        region_abbr = region_abbr_map.get(loc_type, loc_type[:3])
        pave_abbr = pave_abbr_map.get(pave_type, pave_type[:3])

        paving_key = f"{road_abbr}_{region_abbr}_{pave_abbr}"

        # Initialize material dict per region
        mat_result = {m: pd.Series(0.0, index=region_coords) for m in output_materials}

        # Get material intensity rows from mat_int for this road class
        if grip_rc in mat_int.index:
            rc_mats = mat_int.loc[grip_rc]
            if isinstance(rc_mats, pd.Series):
                rc_mats = rc_mats.to_frame().T

            # Get mat_int values per material (per region)
            mat_int_by_material = {}
            for _, row in rc_mats.iterrows():
                mat_name = row['material']
                region_values = row.drop('material')
                # Align region names
                aligned = pd.Series(0.0, index=region_coords)
                for reg in region_coords:
                    if reg in region_values.index:
                        aligned[reg] = float(region_values[reg])
                mat_int_by_material[mat_name] = aligned

            # Look up paving shares if available
            if paving_key in paving_shares_dict:
                paving_shares_df = paving_shares_dict[paving_key]

                # Process base materials
                for mat_name in base_materials:
                    if mat_name in paving_shares_df.index and mat_name in mat_int_by_material:
                        share_row = paving_shares_df.loc[mat_name]
                        aligned_share = pd.Series(0.0, index=region_coords)
                        for reg in region_coords:
                            if reg in share_row.index:
                                aligned_share[reg] = float(share_row[reg])
                        # Material intensity = share × mat_int per region
                        out_mat = mat_name if mat_name != 'metal' else 'steel'
                        mat_result[out_mat] += aligned_share * mat_int_by_material.get(mat_name, 0.0)

                # Process pavingstone → decompose into brick, stone, concrete
                if 'pavingstone' in paving_shares_df.index:
                    ps_share_row = paving_shares_df.loc['pavingstone']
                    aligned_ps_share = pd.Series(0.0, index=region_coords)
                    for reg in region_coords:
                        if reg in ps_share_row.index:
                            aligned_ps_share[reg] = float(ps_share_row[reg])
                    for target_mat, factor in pavingstone_factors.items():
                        ps_intensity = aligned_ps_share * factor
                        if target_mat in mat_int_by_material:
                            mat_result[target_mat] += ps_intensity * mat_int_by_material[target_mat]

                # Process aggregate materials
                for agg_mat in ['asphalt', 'brick', 'concrete', 'pavingstone', 'stone']:
                    if agg_mat in paving_shares_df.index and 'aggregate' in mat_int_by_material:
                        share_row = paving_shares_df.loc[agg_mat]
                        aligned_share = pd.Series(0.0, index=region_coords)
                        for reg in region_coords:
                            if reg in share_row.index:
                                aligned_share[reg] = float(share_row[reg])
                        mat_result['aggregate'] += aligned_share * mat_int_by_material['aggregate']

                # Process unpaved construction material → aggregate
                if 'unpaved - construction' in paving_shares_df.index and 'aggregate' in mat_int_by_material:
                    unp_share_row = paving_shares_df.loc['unpaved - construction']
                    aligned_unp_share = pd.Series(0.0, index=region_coords)
                    for reg in region_coords:
                        if reg in unp_share_row.index:
                            aligned_unp_share[reg] = float(unp_share_row[reg])
                    mat_result['aggregate'] += aligned_unp_share * mat_int_by_material['aggregate']

            else:
                # No paving shares available for this type (e.g. unpaved motorways
                # don't exist in Materials-paving.xlsx) — leave MI at zero.
                pass

        # Scale: original v5 applies x10^6 conversion factor to combined_mi
        for m in output_materials:
            mat_result[m] = mat_result[m] * 1000000

        # V5 line 2096-2099: 80% of aggregate is permanent (non-replaceable subgrade)
        # Store full aggregate MI for permanent tracking before applying 0.2 factor
        mi_per_type_full_agg = mat_result['aggregate'].copy()  # full 100% aggregate MI
        mat_result['aggregate'] = mat_result['aggregate'] * 0.2  # only 20% through DSM

        mi_per_type[t] = mat_result
        mi_per_type[f"_permanent_agg_{t}"] = mi_per_type_full_agg * 0.8  # 80% permanent

    # --- Load additional MI data for bridges, tunnels, parking, rail ---
    mi_bt = pd.read_excel(grid_dir / 'Materials_infra_bt.xlsx', index_col=[0, 1, 2])
    mi_infra = pd.read_excel(grid_dir / 'Materials_infra.xlsx', index_col=[0, 1])
    operational_lt = pd.read_csv(grid_dir / 'operational_lifetime.csv', index_col=0)

    # Extended material list (all materials that appear across all element types)
    extended_materials = ['aggregate', 'asphalt', 'concrete', 'steel', 'wood',
                          'stone', 'brick', 'copper', 'zinc', 'plastics',
                          'aluminium', 'bronze']

    # --- Build MI for bridge/tunnel types (from Materials_infra_bt.xlsx) ---
    # Bridge/tunnel MI: region-specific, per road subtype
    bt_mi_map = {}  # type_name -> {material: value_per_region}
    for bt_type in bt_active_types:
        # Parse type: e.g. "rural_bridge_motorway" -> (rural, bridge, motorway)
        parts = bt_type.split('_')
        loc = parts[0]       # rural/urban
        elem = parts[1]      # bridge/tunnel
        road = parts[2]      # motorway/primary/...

        bt_cat = f"road_{elem}s_{road}"  # e.g. "road_bridges_motorway"
        mat_vals = {m: pd.Series(0.0, index=region_coords) for m in extended_materials}

        # Iterate regions in outer loop, try years in inner loop (use first available year per region)
        for reg in region_coords:
            for yr in [2000, 2050]:
                try:
                    row = mi_bt.loc[(yr, reg, bt_cat)]
                    for m in extended_materials:
                        if m in row.index:
                            mat_vals[m][reg] = float(row[m]) if not pd.isna(row[m]) else 0.0
                    break  # Use first available year for this region
                except KeyError:
                    continue
        bt_mi_map[bt_type] = mat_vals

    # --- Build MI for parking types (same as local roads, v5 approach) ---
    # Parking uses same MI as local roads (v5: parking_mi = local_road_mi)
    parking_mi_map = {}
    parking_local_road_map = {
        'urban_parking_paved': 'urban_local_paved',
        'urban_parking_unpaved': 'urban_local_unpaved',
        'rural_parking_paved': 'rural_local_paved',
        'rural_parking_unpaved': 'rural_local_unpaved',
    }
    for ptype in parking_types:
        local_road_type = parking_local_road_map[ptype]
        if local_road_type in mi_per_type:
            # Copy the local road MI (output_materials → extended_materials)
            # Note: road MI already has 20% aggregate (line 1038) and 80% stored separately
            mat_vals = {m: pd.Series(0.0, index=region_coords) for m in extended_materials}
            for m in output_materials:
                if m in mi_per_type[local_road_type]:
                    mat_vals[m] = mi_per_type[local_road_type][m].copy()
            parking_mi_map[ptype] = mat_vals
            # Also copy the permanent aggregate from the local road type
            perm_key_road = f"_permanent_agg_{local_road_type}"
            perm_key_parking = f"_permanent_agg_{ptype}"
            if perm_key_road in mi_per_type:
                mi_per_type[perm_key_parking] = mi_per_type[perm_key_road].copy()
        else:
            parking_mi_map[ptype] = {m: pd.Series(0.0, index=region_coords) for m in extended_materials}

    # --- Build MI for rail types (from Materials_infra.xlsx) ---
    # Rail MI is NOT region-specific in the data file (same for all regions)
    rail_mi_category = {
        'total_rail': 'standard_rail',
        'total_HST': 'highspeed_rail',
        'total_urban_rail': 'urban_rail',
        'rail_bridge': 'rail_bridges',
        'rail_tunnel': 'rail_tunnels',
    }
    rail_mi_map = {}
    for rtype in rail_element_types:
        cat = rail_mi_category.get(rtype, 'standard_rail')
        mat_vals = {m: pd.Series(0.0, index=region_coords) for m in extended_materials}
        for yr in [2000, 2050]:
            try:
                row = mi_infra.loc[(yr, cat)]
                for m in extended_materials:
                    col_name = m if m != 'brick' else 'bricks'
                    if col_name in row.index:
                        val = float(row[col_name]) if not pd.isna(row[col_name]) else 0.0
                        mat_vals[m] = pd.Series(val, index=region_coords)
                break
            except KeyError:
                continue
        rail_mi_map[rtype] = mat_vals

    # --- Build combined MI DataArray for ALL types ---
    # Use extended_materials for all types (roads get zeros for copper, zinc, etc.)
    mi_data = np.zeros((len(final_all_types), len(extended_materials), len(region_coords)))

    for i, t in enumerate(final_all_types):
        # Determine which MI source to use
        # Strip obsolete_ prefix to find active counterpart
        active_t = t
        if t.startswith("obsolete_"):
            active_t = t.replace("obsolete_", "", 1)

        if active_t in mi_per_type:
            # Road type - use existing road MI (map output_materials to extended_materials)
            for j, m in enumerate(extended_materials):
                if m in mi_per_type[active_t]:
                    mi_data[i, j, :] = mi_per_type[active_t][m].values
        elif active_t in bt_mi_map:
            # Bridge/tunnel type
            for j, m in enumerate(extended_materials):
                if m in bt_mi_map[active_t]:
                    mi_data[i, j, :] = bt_mi_map[active_t][m].values
        elif active_t in parking_mi_map:
            # Parking type
            for j, m in enumerate(extended_materials):
                if m in parking_mi_map[active_t]:
                    mi_data[i, j, :] = parking_mi_map[active_t][m].values
        elif active_t in rail_mi_map:
            # Rail type
            for j, m in enumerate(extended_materials):
                if m in rail_mi_map[active_t]:
                    mi_data[i, j, :] = rail_mi_map[active_t][m].values

    da_mi = xr.DataArray(
        mi_data,
        coords=[final_all_types, extended_materials, region_coords],
        dims=['Type', 'material', 'Region']
    )
    # Cohort range must match the stock time range (from FIRST_YEAR_GRID)
    years = list(range(FIRST_YEAR_GRID, end_year + 1))
    da_mi = da_mi.expand_dims({"Cohort": years})
    # --- Permanent Aggregate MI (80% non-replaceable subgrade) ---
    # v5 lines 2096-2099, 3244, 3352: 80% of aggregate is permanent
    # Apply 80/20 split on aggregate MI for ALL types (active + obsolete)
    agg_idx = extended_materials.index('aggregate')
    perm_agg_data = np.zeros((len(all_active_types), len(region_coords)))

    for i, t in enumerate(all_active_types):
        type_idx = final_all_types.index(t)
        # For road types, mi_per_type already stored 20% aggregate; read the permanent value
        perm_key = f"_permanent_agg_{t}"
        if perm_key in mi_per_type:
            perm_agg_data[i, :] = mi_per_type[perm_key].values
        else:
            # Non-road active types: split the full aggregate MI
            full_agg_mi = mi_data[type_idx, agg_idx, :].copy()
            perm_agg_data[i, :] = full_agg_mi * 0.8
            mi_data[type_idx, agg_idx, :] = full_agg_mi * 0.2

    # Also reduce aggregate MI for obsolete types to match their active counterparts (20%)
    for obs_t in all_obsolete_types:
        if obs_t in final_all_types:
            obs_idx = final_all_types.index(obs_t)
            active_t = obs_to_active_map.get(obs_t, obs_t)
            if active_t in mi_per_type:
                # Road obsolete: already has 20% aggregate from mi_per_type
                pass
            else:
                # Non-road obsolete: reduce to 20%
                full_agg = mi_data[obs_idx, agg_idx, :].copy()
                mi_data[obs_idx, agg_idx, :] = full_agg * 0.2

    # Rebuild da_mi with 20% aggregate through DSM
    da_mi = xr.DataArray(
        mi_data,
        coords=[final_all_types, extended_materials, region_coords],
        dims=['Type', 'material', 'Region']
    )
    da_mi = da_mi.expand_dims({"Cohort": years})
    da_mi = prism.Q_(da_mi, "kg / km**2")
    preprocessing_results["material_intensities"] = da_mi

    # Permanent aggregate MI DataArray for active types (used in infrastructure.py)
    da_perm_agg_mi = xr.DataArray(
        perm_agg_data,
        coords=[all_active_types, region_coords],
        dims=['Type', 'Region']
    )
    preprocessing_results["permanent_aggregate_mi"] = da_perm_agg_mi

    # --- Lifetimes ---
    # Weibull types: roads, road obsolete, bridges, tunnels, bridge/tunnel obsolete, parking
    # FoldedNormal types: rail, urban rail, HSR, rail bridges, rail tunnels, rail obsolete
    rc10_shape = weibull.loc['Road Class 10', 'shape']
    rc10_scale = weibull.loc['Road Class 10', 'scale']
    bc9_shape = weibull.loc['Bridge Class 9', 'shape']
    bc9_scale = weibull.loc['Bridge Class 9', 'scale']

    # Bridge class map (v5)
    bridge_bc_map = {
        'urban_bridge': {'cycle': 1, 'informal': 1, 'local': 3, 'motorway': 6,
                         'pedestrian': 1, 'primary': 5, 'secondary': 4, 'tertiary': 3},
        'rural_bridge': {'cycle': 1, 'informal': 1, 'local': 1, 'motorway': 4,
                         'pedestrian': 1, 'primary': 3, 'secondary': 2, 'tertiary': 2},
    }
    # Tunnels use same mapping as bridges
    tunnel_bc_map = {
        'urban_tunnel': bridge_bc_map['urban_bridge'],
        'rural_tunnel': bridge_bc_map['rural_bridge'],
    }

    # Parking lifetime: urban=rc3, rural=rc2
    parking_rc_map = {
        'urban_paved_parking': 3, 'urban_unpaved_parking': 3,
        'rural_paved_parking': 2, 'rural_unpaved_parking': 2,
    }

    # Collect Weibull lifetime types
    weibull_types = []
    weibull_shapes = []
    weibull_scales = []

    # 1. Road types (active + obsolete)
    for t in combined_types:
        weibull_types.append(t)
        if t.startswith("obsolete_"):
            weibull_shapes.append(rc10_shape)
            weibull_scales.append(rc10_scale)
        else:
            parts = t.split('_')
            category = f"{parts[0]}_{parts[1]}"
            road_type = parts[2]
            rc = lifetime_rc_map.get(category, {}).get(road_type, 1)
            rc_label = f"Road Class {rc}"
            weibull_shapes.append(weibull.loc[rc_label, 'shape'])
            weibull_scales.append(weibull.loc[rc_label, 'scale'])

    # 2. Bridge/tunnel types (active + obsolete)
    for t in bt_active_types:
        weibull_types.append(t)
        parts = t.split('_')  # e.g. urban_bridge_motorway
        loc_elem = f"{parts[0]}_{parts[1]}"
        road_type = parts[2]
        bc_map = bridge_bc_map if 'bridge' in t else tunnel_bc_map
        bc = bc_map.get(loc_elem, {}).get(road_type, 1)
        bc_label = f"Bridge Class {bc}"
        if bc_label in weibull.index:
            weibull_shapes.append(weibull.loc[bc_label, 'shape'])
            weibull_scales.append(weibull.loc[bc_label, 'scale'])
        else:
            weibull_shapes.append(2.0)
            weibull_scales.append(100.0)

    for t in bt_obs_types:
        weibull_types.append(t)
        weibull_shapes.append(bc9_shape)
        weibull_scales.append(bc9_scale)

    # 3. Parking types (active + obsolete)
    for t in parking_types:
        weibull_types.append(t)
        rc = parking_rc_map.get(t, 2)
        rc_label = f"Road Class {rc}"
        weibull_shapes.append(weibull.loc[rc_label, 'shape'])
        weibull_scales.append(weibull.loc[rc_label, 'scale'])

    for t in parking_obs_types:
        weibull_types.append(t)
        weibull_shapes.append(rc10_shape)
        weibull_scales.append(rc10_scale)

    # Build Weibull lifetime DataArray
    c_array = np.array(weibull_shapes, dtype=float)
    scale_array = np.array(weibull_scales, dtype=float)
    param_array = np.stack([c_array, scale_array], axis=0)
    da_lt_weibull = xr.DataArray(
        param_array,
        dims=["ScipyParam", "Type"],
        coords={"ScipyParam": ["c", "scale"], "Type": weibull_types},
        attrs={"loc": 0}
    )
    # Expand Time dimension for ScipySurvival
    da_lt_weibull = da_lt_weibull.expand_dims(Time=extended_time_coords)

    # 4. Rail types use FoldedNormal distribution
    # v5: stdev_mult=5, mean transitions from start_life to target_life between 2025-2050
    STDEV_MULT = 5
    fn_types = rail_element_types + rail_obs_types

    fn_c_data = np.zeros((len(extended_time_coords), len(fn_types)))
    fn_scale_data = np.zeros((len(extended_time_coords), len(fn_types)))

    for j, t in enumerate(fn_types):
        active_t = t.replace("obsolete_", "", 1) if t.startswith("obsolete_") else t
        # Map to operational_lt index
        lt_key_map = {
            'total_rail': 'rail', 'total_HST': 'rail',
            'total_urban_rail': 'urban_rail',
            'rail_bridge': 'rail_bridge', 'rail_tunnel': 'rail_tunnel',
        }
        lt_key = lt_key_map.get(active_t, 'rail')
        if t.startswith("obsolete_"):
            lt_key = 'obs_rail'

        if lt_key in operational_lt.index:
            start_life = float(operational_lt.loc[lt_key, 'start'])
            target_life = float(operational_lt.loc[lt_key, 'end'])
        else:
            start_life, target_life = 30.0, 30.0

        for i, year in enumerate(extended_time_coords):
            if year < 2025:
                mean = start_life
            elif year <= 2050:
                mean = start_life + (target_life - start_life) * ((year - 2025) / 25.0)
            else:
                mean = target_life
            stdev = mean * STDEV_MULT
            # FoldedNormal scipy params: c = mean/stdev, scale = stdev, loc = 0
            fn_c_data[i, j] = mean / stdev if stdev > 0 else 1.0
            fn_scale_data[i, j] = stdev

    fn_param = np.stack([fn_c_data, fn_scale_data], axis=-1)  # (Time, Type, 2)
    fn_param = np.transpose(fn_param, (2, 0, 1))  # (ScipyParam, Time, Type)
    da_lt_foldnorm = xr.DataArray(
        fn_param,
        dims=["ScipyParam", "Time", "Type"],
        coords={"ScipyParam": ["c", "scale"],
                "Time": extended_time_coords,
                "Type": fn_types},
        attrs={"loc": 0}
    )

    # Return lifetimes as dict for multi-distribution support
    preprocessing_results["lifetimes"] = {
        "weibull": da_lt_weibull,
        "folded_norm": da_lt_foldnorm
    }

    return preprocessing_results

if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    _path = Path(r"C:\IMAGE\image-materials").resolve()
    print("Testing get_preprocessing_data_infrastructure...")
    res = get_preprocessing_data_infrastructure(_path, "SSP2")
    print("Test passed! Keys extracted:", list(res.keys()))


