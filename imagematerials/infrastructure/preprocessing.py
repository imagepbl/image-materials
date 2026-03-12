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

def get_preprocessing_data_infrastructure(path_base: Path, scen_folder: str, start_year: int = 1971, end_year: int = 2100):
    """
    Reads data from SSP2 and IMAGE folders, calculates infrastructure stocks
    and returns it as a dictionary ready for the GenericStocks model.
    """
    preprocessing_results = {}
    
    # Define paths
    data_raw_dir = path_base / "data" / "raw"
    image_dir = data_raw_dir / "IMAGE_CircoMod" / scen_folder
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
    if 2024 in rural_road_cap.columns:
        predicted_rural_road_cap_2024 = rural_road_cap[2024]
        predicted_urban_road_cap_2024 = urban_road_cap[2024]
        rural_adjustment_factor = actual_rural_road_area_2024 / predicted_rural_road_cap_2024
        urban_adjustment_factor = actual_urban_road_area_2024 / predicted_urban_road_cap_2024
    else:
        rural_adjustment_factor = 1.0
        urban_adjustment_factor = 1.0

    rural_road_cap_adjusted = rural_road_cap * rural_adjustment_factor
    urban_road_cap_adjusted = urban_road_cap * urban_adjustment_factor

    rural_road_area = rural_road_cap_adjusted * (rural_population * 1000000)
    urban_road_area = urban_road_cap_adjusted * (urban_population * 1000000)

    # Simplified Decline Prevention (Assuming ROAD_DECLINE_FUNCTION=1 logic)
    adjusted_rural_road_area = rural_road_area.copy()
    adjusted_urban_road_area = urban_road_area.copy()
    for df in [adjusted_rural_road_area, adjusted_urban_road_area]:
        for col in df.columns:
            # We enforce strictly non-decreasing logic over time
            df.loc[:, col] = df[col].cummax(axis=0)

    # Parking logic
    def rural_parking(x): return 0.0137 * x - 0.000000008
    def urban_parking(x): return 0.1143 * x - 0.000000002
    
    rural_parking_cap = rural_road_cap.applymap(rural_parking)
    urban_parking_cap = urban_road_cap.applymap(urban_parking)
    
    rural_parking_area = rural_parking_cap * (rural_population * 1000000)
    urban_parking_area = urban_parking_cap * (urban_population * 1000000)

    if 2024 in rural_parking_cap.columns:
        r_parking_adj = actual_rural_parking_area_2024 / rural_parking_area[2024]
        u_parking_adj = actual_urban_parking_area_2024 / urban_parking_area[2024]
    else:
        r_parking_adj = 1.0
        u_parking_adj = 1.0
    
    rural_parking_area = rural_parking_area * r_parking_adj / 1000000
    urban_parking_area = urban_parking_area * u_parking_adj / 1000000

    # Paving Shares
    def rural_paving(x): return 0.1244 * math.log(x) - 0.789
    def urban_paving(x): return 0.151 * math.log(x) - 0.7656
    
    rural_paving_share = sorted_gdp_rural_cap.applymap(rural_paving).clip(lower=0.0, upper=1.0)
    urban_paving_share = sorted_gdp_urban_cap.applymap(urban_paving).clip(lower=0.0, upper=1.0)
    
    rural_unpaved_share = 1 - rural_paving_share
    urban_unpaved_share = 1 - urban_paving_share

    rural_paved_road_area = adjusted_rural_road_area * rural_paving_share
    rural_unpaved_road_area = adjusted_rural_road_area * rural_unpaved_share
    urban_paved_road_area = adjusted_urban_road_area * urban_paving_share
    urban_unpaved_road_area = adjusted_urban_road_area * urban_unpaved_share
    
    rural_paved_parking_area = rural_parking_area * rural_paving_share
    rural_unpaved_parking_area = rural_parking_area * rural_unpaved_share
    urban_paved_parking_area = urban_parking_area * urban_paving_share
    urban_unpaved_parking_area = urban_parking_area * urban_unpaved_share

    # --- 9. Distribute Areas by HDI and Road Type ---
    # Load road type distributions depending on HDI
    u_types = pd.read_excel(tripi_dir / 'urban-type-distribution.xlsx', index_col=0)
    r_types = pd.read_excel(tripi_dir / 'rural-type-distribution.xlsx', index_col=0)
    road_type_cols = u_types.columns # e.g. ['cycle_share', 'informal_share', ...]
    
    # Interpolate to 0.01 resolution
    hdi_bins = np.arange(0, 1.01, 0.01)
    u_types_int = u_types.reindex(hdi_bins).interpolate(method='linear', limit_direction='both')
    r_types_int = r_types.reindex(hdi_bins).interpolate(method='linear', limit_direction='both')
    u_types_int.index = np.round(u_types_int.index, 2)
    r_types_int.index = np.round(r_types_int.index, 2)
    
    hdi_rounded_u = sorted_IMAGE_hdi.round(2).clip(lower=0, upper=1)
    # Using the same HDI logic for rural (assuming Kedi et al. applies universally)
    hdi_rounded_r = sorted_IMAGE_hdi.round(2).clip(lower=0, upper=1)

    # Initialize xarray with structured dimensions
    time_coords = list(urban_paved_road_area.index)
    region_coords = list(urban_paved_road_area.columns)
    
    road_labels = [c.replace('_share', '') for c in road_type_cols]
    settings = [
        ('urban', 'paved', urban_paved_road_area, u_types_int, hdi_rounded_u),
        ('urban', 'unpaved', urban_unpaved_road_area, u_types_int, hdi_rounded_u),
        ('rural', 'paved', rural_paved_road_area, r_types_int, hdi_rounded_r),
        ('rural', 'unpaved', rural_unpaved_road_area, r_types_int, hdi_rounded_r)
    ]
    
    all_types = []
    da_list = []
    for loc, pave, area_df, type_int, hdi_rnd in settings:
        for col, label in zip(road_type_cols, road_labels):
            type_name = f"{loc}_{pave}_{label}"
            all_types.append(type_name)
            
            # Map HDI for each Region/Year to the corresponding share
            share_mapped = pd.DataFrame(index=hdi_rnd.index, columns=hdi_rnd.columns)
            for y in time_coords:
                if y in hdi_rnd.index:
                    shares_for_y = type_int.loc[hdi_rnd.loc[y], col].values
                    share_mapped.loc[y] = shares_for_y
            
            share_mapped = share_mapped.fillna(0)
            final_area = area_df * share_mapped
            
            # Convert to xarray
            da = xr.DataArray(
                final_area.values,
                coords=[time_coords, region_coords],
                dims=['Time', 'Region']
            )
            da_list.append(da)

    # Concatenate all road types into one DataArray
    da_roads = xr.concat(da_list, pd.Index(all_types, name="Type"))
    da_roads = da_roads.assign_attrs({"units": "km**2"})

    preprocessing_results["stocks"] = da_roads
    
    # Initialize basic KnowledgeGraph covering structural mappings
    infra_graph = KnowledgeGraph(Node("Infrastructure"))
    for t in all_types:
        infra_graph.add(Node(t, inherits_from="Infrastructure"))
    preprocessing_results["knowledge_graph"] = infra_graph
    
    preprocessing_results["set_unit_flexible"] = "km**2"
    preprocessing_results["shares"] = None

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
        pave_type = parts[1]    # 'paved' or 'unpaved'
        road_type = parts[2]    # 'motorway', 'primary', etc.
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
                # Fallback: use mat_int directly without paving shares
                for mat_name, intensity_vals in mat_int_by_material.items():
                    if mat_name in output_materials:
                        mat_result[mat_name] += intensity_vals
                    elif mat_name == 'metal':
                        mat_result['steel'] += intensity_vals
                    elif mat_name == 'unpaved - construction':
                        mat_result['aggregate'] += intensity_vals

        # Scale: original v5 applies ×10^6 conversion factor to combined_mi
        for m in output_materials:
            mat_result[m] = mat_result[m] * 1000000

        mi_per_type[t] = mat_result

    # Build material intensities DataArray with dims (Type, material, Region)
    mi_data = np.zeros((len(all_types), len(output_materials), len(region_coords)))
    for i, t in enumerate(all_types):
        for j, m in enumerate(output_materials):
            mi_data[i, j, :] = mi_per_type[t][m].values

    da_mi = xr.DataArray(
        mi_data,
        coords=[all_types, output_materials, region_coords],
        dims=['Type', 'material', 'Region']
    )
    years = list(range(start_year, end_year + 1))
    da_mi = da_mi.expand_dims({"Cohort": years})
    da_mi = da_mi.assign_attrs({"units": "kg / km**2"})
    preprocessing_results["material_intensities"] = da_mi

    # --- Lifetimes using correct road class mapping from original v5 ---
    lifetime_shape = []
    lifetime_scale = []

    for t in all_types:
        parts = t.split('_')
        loc_type = parts[0]     # 'urban' or 'rural'
        pave_type = parts[1]    # 'paved' or 'unpaved'
        road_type = parts[2]    # 'motorway', 'primary', etc.

        category = f"{loc_type}_{pave_type}"
        if category in lifetime_rc_map:
            rc = lifetime_rc_map[category].get(road_type, 1)
        else:
            rc = 1  # Fallback

        rc_label = f"Road Class {rc}"
        if rc_label in weibull.index:
            shape = weibull.loc[rc_label, 'shape']
            scale = weibull.loc[rc_label, 'scale']
        else:
            shape, scale = 2.0, 50.0  # Fallback

        lifetime_shape.append(shape)
        lifetime_scale.append(scale)

    # Form Lifetimes DataArray matching buildings pattern:
    # dims: (ScipyParam, Type) with loc as attribute
    c_array = np.array(lifetime_shape, dtype=float)
    scale_array = np.array(lifetime_scale, dtype=float)
    param_array = np.stack([c_array, scale_array], axis=0)
    da_lifetime = xr.DataArray(
        param_array,
        dims=["ScipyParam", "Type"],
        coords={
            "ScipyParam": ["c", "scale"],
            "Type": all_types
        },
        attrs={"distribution": "weibull", "loc": 0},
        name="InfraLifetime"
    )
    preprocessing_results["lifetimes"] = da_lifetime

    return preprocessing_results

if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    _path = Path(r"C:\IMAGE\image-materials").resolve()
    print("Testing get_preprocessing_data_infrastructure...")
    res = get_preprocessing_data_infrastructure(_path, "SSP2")
    print("Test passed! Keys extracted:", list(res.keys()))


