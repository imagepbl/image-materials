"""Derive area, density, GDP/VKM splits and total road & parking areas (sections 5-8).

This is the second processing stage: starting from the IMAGE drivers
(GDP, population, VKM, HDI) and the FUA-derived urban-area table, it
produces:

- urban / rural / total population density
- urban vs rural GDP per capita
- urban vs rural VKM per capita
- rural and urban road area regressions, monotonic-adjusted and split into
  active stock + obsolete stock
- urban and rural parking area (active + obsolete), further split into
  paved and unpaved subcomponents using the single-regression paving share

The helper ``apply_obsolete_rule`` is also used by the rail processing
(see ``rail.py``), so it lives here.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from imagematerials.infrastructure.constants import COLUMN_MAPPING


def process_urban_area_and_density(scenario_data_dir: Path, urban_population, rural_population):
    """Load urban-area shares and combine with population to get densities.

    Returns
    -------
    sorted_urban_pop_density, sorted_rural_pop_density, sorted_pop_density,
    IMAGE_urban_area_filtered, IMAGE_rural_area_filtered
    """
    area = pd.read_excel(scenario_data_dir / "urban-area_ssp4_dense.xlsx", index_col=0)
    area = area.reset_index()
    for col in ["Cntry_name", "eFUA_name", "GRIP-region", "R10"]:
        if col in area.columns:
            area = area.drop(columns=[col])

    IMAGE_total_area = area.groupby("IMAGE-region")["FUA_total_area"].sum()
    area_clean = area.dropna()
    IMAGE_urban_share = area_clean.groupby("IMAGE-region").sum(numeric_only=True)
    for col in ["FUA_area", "FUA_urban", "FUA_p_2015"]:
        if col in IMAGE_urban_share.columns:
            IMAGE_urban_share = IMAGE_urban_share.drop(columns=[col])

    IMAGE_urban_share = IMAGE_urban_share.div(IMAGE_urban_share["FUA_total_area"], axis=0)
    if "FUA_total_area" in IMAGE_urban_share.columns:
        IMAGE_urban_share = IMAGE_urban_share.drop(columns=["FUA_total_area"])

    all_years_area = list(range(1870, 2101))
    IMAGE_urban_share = IMAGE_urban_share.reindex(columns=all_years_area)
    IMAGE_urban_share = IMAGE_urban_share.interpolate(method="linear", axis=1)

    # Smooth Urban Share
    years_to_smooth = range(1950, 2101)
    IMAGE_urban_share_subset = IMAGE_urban_share[years_to_smooth].copy()
    window_avg_size = 20
    smoothed_data = IMAGE_urban_share_subset.apply(
        lambda col: col.rolling(window=window_avg_size, center=True, min_periods=1).mean(), axis=1)
    IMAGE_urban_share[smoothed_data.columns] = smoothed_data

    IMAGE_rural_share = 1 - IMAGE_urban_share
    IMAGE_urban_area = IMAGE_urban_share.mul(IMAGE_total_area, axis=0)
    IMAGE_rural_area = IMAGE_rural_share.mul(IMAGE_total_area, axis=0)

    # Population density
    IMAGE_urban_area_filtered = IMAGE_urban_area.loc[:, 1971:2100].T
    IMAGE_rural_area_filtered = IMAGE_rural_area.loc[:, 1971:2100].T
    IMAGE_urban_area_filtered = IMAGE_urban_area_filtered.rename(
        columns=COLUMN_MAPPING).sort_index(axis=1)
    IMAGE_rural_area_filtered = IMAGE_rural_area_filtered.rename(
        columns=COLUMN_MAPPING).sort_index(axis=1)

    urban_pop_density = urban_population * 1000000 / IMAGE_urban_area_filtered
    rural_pop_density = rural_population * 1000000 / IMAGE_rural_area_filtered
    pop_density = ((urban_population + rural_population) * 1000000
                   / (IMAGE_urban_area_filtered + IMAGE_rural_area_filtered))

    sorted_urban_pop_density = urban_pop_density.sort_index(axis=1)
    sorted_rural_pop_density = rural_pop_density.sort_index(axis=1)
    sorted_pop_density = pop_density.sort_index(axis=1)

    return (sorted_urban_pop_density, sorted_rural_pop_density, sorted_pop_density,
            IMAGE_urban_area_filtered, IMAGE_rural_area_filtered)


def split_gdp_urban_rural(gdp_cap_new, urban_population, rural_population):
    """Compute urban and rural GDP per capita with feasibility-bounded splits."""
    rural_share = rural_population / (rural_population + urban_population)
    urban_share = 1 - rural_share
    gdp_total = gdp_cap_new * ((rural_population + urban_population) * 1000000)

    def gdp_share_regression(x):
        return 1.0267 * x + 0.0412

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
    urban_solvable = initial_gdp_urban_total.clip(lower=min_allowed_urban_gdp,
                                                   upper=max_allowed_urban_gdp)

    urban_share_of_required = min_gdp_urban / (total_required + 1e-9)
    urban_unsolvable = gdp_total * urban_share_of_required

    final_gdp_urban_total = np.where(is_unsolvable, urban_unsolvable, urban_solvable)
    final_gdp_urban_total = pd.DataFrame(final_gdp_urban_total, index=gdp_total.index,
                                          columns=gdp_total.columns)
    final_gdp_rural_total = gdp_total - final_gdp_urban_total

    gdp_urban_total = final_gdp_urban_total
    gdp_rural_total = final_gdp_rural_total
    gdp_urban_cap = gdp_urban_total / pop_u_mil
    gdp_rural_cap = gdp_rural_total / pop_r_mil

    sorted_gdp_urban_cap = gdp_urban_cap.sort_index(axis=1)
    sorted_gdp_rural_cap = gdp_rural_cap.sort_index(axis=1)
    return sorted_gdp_urban_cap, sorted_gdp_rural_cap


def split_vkm_urban_rural(vkm_pivoted, vkm_rail_pivoted, vkm_rail_HST_pivoted,
                           urban_population, rural_population,
                           sorted_gdp_urban_cap, sorted_gdp_rural_cap):
    """Allocate road VKM into urban and rural per-capita time series; also pass rail through."""
    def vkm_urban_share_regression(x):
        return 0.1153 * x + 1637.4

    vkm_urban_cap = sorted_gdp_urban_cap.applymap(vkm_urban_share_regression)
    vkm_urban_tot = vkm_urban_cap * urban_population * 1000000

    def vkm_rural_share_regression(x):
        return 0.5603 * x + 78.578

    vkm_rural_cap = sorted_gdp_rural_cap.applymap(vkm_rural_share_regression)
    vkm_rural_tot = vkm_rural_cap * rural_population * 1000000

    vkm_urban_share = vkm_urban_tot / (vkm_urban_tot + vkm_rural_tot)
    vkm_rural_share = 1 - vkm_urban_share

    vkm_urban_IMAGE_cap = vkm_pivoted * vkm_urban_share / (urban_population * 1000000)
    vkm_rural_IMAGE_cap = vkm_pivoted * vkm_rural_share / (rural_population * 1000000)

    sorted_vkm_urban_cap = vkm_urban_IMAGE_cap.sort_index(axis=1)
    sorted_vkm_rural_cap = vkm_rural_IMAGE_cap.sort_index(axis=1)
    sorted_vkm_rail_tot = vkm_rail_pivoted.sort_index(axis=1)
    sorted_vkm_HST_rail_tot = vkm_rail_HST_pivoted.sort_index(axis=1)
    sorted_vkm_rail_cap = (sorted_vkm_rail_tot
                           / ((urban_population + rural_population) * 1000000))
    return (sorted_vkm_urban_cap, sorted_vkm_rural_cap, sorted_vkm_rail_tot,
            sorted_vkm_HST_rail_tot, sorted_vkm_rail_cap)


def apply_obsolete_rule(series):
    """Zero out values before the last zero that precedes the final non-zero block.

    Shared between road and rail obsolete-stock construction (v5).
    """
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


def compute_road_areas(data_2024, sorted_rural_pop_density, sorted_urban_pop_density,
                        sorted_gdp_rural_cap, sorted_gdp_urban_cap, sorted_vkm_urban_cap,
                        urban_population, rural_population):
    """Run road-area regressions and return monotonic-adjusted active + obsolete stocks."""
    def rural_roads(x1, x2):
        return 0.000317871 + (-6.55403e-5 * np.log(x1)) + (x2 * 5.52978e-9)

    def urban_roads(x1, x2, x3):
        return (0.000118208 + (-1.37056e-5 * np.log(x1))
                + (x2 * 2.56944e-9) + (x3 * 3.30002e-10))

    actual_rural_road_area_2024 = data_2024.loc["rural_roads"]
    actual_urban_road_area_2024 = data_2024.loc["urban_roads"]

    rural_road_cap = rural_roads(sorted_rural_pop_density, sorted_gdp_rural_cap)
    urban_road_cap = urban_roads(sorted_urban_pop_density, sorted_vkm_urban_cap,
                                  sorted_gdp_urban_cap)

    # Smooth Road Data
    std_dev = 5
    min_positive_values_r = rural_road_cap.apply(
        lambda row: row[row > 0].nsmallest(3).mean() if (row > 0).sum() >= 3
        else row[row > 0].mean(), axis=1)
    min_positive_values_r = pd.Series(
        gaussian_filter1d(min_positive_values_r, sigma=std_dev),
        index=min_positive_values_r.index)
    for col in rural_road_cap.columns:
        if (rural_road_cap[col] < 0).any():
            rural_road_cap[col] = min_positive_values_r

    min_positive_values_u = urban_road_cap.apply(
        lambda row: row[row > 0].nsmallest(3).mean() if (row > 0).sum() >= 3
        else row[row > 0].mean(), axis=1)
    min_positive_values_u = pd.Series(
        gaussian_filter1d(min_positive_values_u, sigma=std_dev),
        index=min_positive_values_u.index)
    for col in urban_road_cap.columns:
        if (urban_road_cap[col] < 0).any():
            urban_road_cap[col] = min_positive_values_u

    smoothed_urban_road_cap = urban_road_cap.apply(
        lambda col: pd.Series(gaussian_filter1d(col, sigma=1), index=col.index), axis=0)
    smoothed_urban_road_cap.index = urban_road_cap.index
    smoothed_urban_road_cap.columns = urban_road_cap.columns
    urban_road_cap = smoothed_urban_road_cap

    # Adjustment logic to 2024 — years are in INDEX, region names in COLUMNS
    if 2024 in rural_road_cap.index:
        predicted_rural_road_area_2024 = (rural_road_cap.loc[2024]
                                           * (rural_population.loc[2024] * 1000000))
        predicted_urban_road_area_2024 = (urban_road_cap.loc[2024]
                                           * (urban_population.loc[2024] * 1000000))
        rural_adjustment_factor = (actual_rural_road_area_2024
                                    / predicted_rural_road_area_2024.replace(0, np.nan))
        urban_adjustment_factor = (actual_urban_road_area_2024
                                    / predicted_urban_road_area_2024.replace(0, np.nan))
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

    for obs_df in [obsolete_road_rural_area, obsolete_road_urban_area]:
        for col in obs_df.columns:
            obs_df[col] = apply_obsolete_rule(obs_df[col])
        # Fix columns that end with zero (v5 fix_obsolete_columns)
        for col in obs_df.columns:
            if obs_df[col].iloc[-1] == 0 and not (obs_df[col] == 0).all():
                obs_df[col] = 0

    # Subtract obsolete from adjusted to get true active stock (v5 line 672-673)
    adjusted_rural_road_area = adjusted_rural_road_area - obsolete_road_rural_area
    adjusted_urban_road_area = adjusted_urban_road_area - obsolete_road_urban_area

    return (adjusted_rural_road_area, adjusted_urban_road_area,
            obsolete_road_rural_area, obsolete_road_urban_area)


def compute_parking_areas(data_2024, adjusted_rural_road_area, adjusted_urban_road_area,
                           obsolete_road_rural_area, obsolete_road_urban_area,
                           sorted_gdp_urban_cap, sorted_gdp_rural_cap):
    """Compute parking area as ratio-of-roads, then split into paved/unpaved.

    Returns active and obsolete paved/unpaved parking areas (8 DataFrames).
    """
    actual_rural_road_area_2024 = data_2024.loc["rural_roads"]
    actual_urban_road_area_2024 = data_2024.loc["urban_roads"]
    actual_rural_parking_area_2024 = data_2024.loc["rural_parking_area"]
    actual_urban_parking_area_2024 = data_2024.loc["urban_parking"]

    # Parking as ratio of road network (same approach as bridges/tunnels)
    # 2024 data: parking in m², roads in km² → convert parking to km² for ratio
    rur_parking_ratio = ((actual_rural_parking_area_2024 / 1000000)
                         / actual_rural_road_area_2024)
    urb_parking_ratio = ((actual_urban_parking_area_2024 / 1000000)
                         / actual_urban_road_area_2024)
    rur_parking_ratio = rur_parking_ratio.fillna(0)
    urb_parking_ratio = urb_parking_ratio.fillna(0)

    rural_parking_area = adjusted_rural_road_area * rur_parking_ratio
    urban_parking_area = adjusted_urban_road_area * urb_parking_ratio

    # Obsolete parking (same ratio applied to obsolete road area)
    obsolete_parking_rural = obsolete_road_rural_area * rur_parking_ratio
    obsolete_parking_urban = obsolete_road_urban_area * urb_parking_ratio

    # --- Parking paving (single GDP regression, v5 lines 710-725) ---
    def rural_paving(x):
        return 0.1244 * math.log(x) - 0.789

    def urban_paving(x):
        return 0.151 * math.log(x) - 0.7656

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

    return {
        "active": {
            "urban_paved": urban_paved_parking_area,
            "urban_unpaved": urban_unpaved_parking_area,
            "rural_paved": rural_paved_parking_area,
            "rural_unpaved": rural_unpaved_parking_area,
        },
        "obsolete": {
            "urban_paved": urban_paved_obsolete_parking,
            "urban_unpaved": urban_unpaved_obsolete_parking,
            "rural_paved": rural_paved_obsolete_parking,
            "rural_unpaved": rural_unpaved_obsolete_parking,
        },
    }
