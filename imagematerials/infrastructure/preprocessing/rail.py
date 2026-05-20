"""Rail, HSR, urban rail, rail bridges/tunnels (section 9c).

Rail length is projected from historic data using a VKM-linked elasticity
model with build-rate capping and a backlog mechanic for unmet demand.
HSR is then derived as a smoothed share of total rail, and urban rail is
computed from HDI and urban-population density via a separate regression.

The output is a set of rail-element DataArrays (active + obsolete),
extended to ``FIRST_YEAR_GRID`` with a stock tail.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from imagematerials.infrastructure.constants import FIRST_YEAR_GRID
from imagematerials.infrastructure.preprocessing.network import apply_obsolete_rule

MAX_ANNUAL_CHANGE_PERCENT = 0.03
LOOKAHEAD_YEARS = 10
MIN_ELASTICITY = 0.1
MAX_ELASTICITY = 0.6
GROWTH_MIDPOINT = 0.03
STEEPNESS = 200


def build_rail_stocks(standard_data_dir: Path, data_2024,
                      sorted_vkm_rail_tot, sorted_vkm_HST_rail_tot,
                      sorted_IMAGE_hdi, sorted_urban_pop_density, urban_population,
                      extended_time_coords, region_coords):
    """Return (rail_element_types, rail_element_das, rail_obs_types, rail_obs_das)."""

    def _get_dynamic_elasticity(growth_rate):
        return MIN_ELASTICITY + (MAX_ELASTICITY - MIN_ELASTICITY) / (
            1 + np.exp(-STEEPNESS * (growth_rate - GROWTH_MIDPOINT)))

    # Load rail historic length data
    rail_lengths_df = pd.read_excel(standard_data_dir / "rail_historic_length.xlsx",
                                      index_col=0)
    rail_lengths_df.columns = rail_lengths_df.columns.astype(str)

    # VKM rail data (already loaded upstream)
    vkm_data_rail = sorted_vkm_rail_tot.copy()
    vkm_data_rail.index = vkm_data_rail.index.astype(int)
    vkm_data_rail.columns = vkm_data_rail.columns.astype(str) if not isinstance(
        vkm_data_rail.columns[0], str) else vkm_data_rail.columns

    # Align rail_lengths_df columns with vkm regions
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
        slope, intercept = np.polyfit(x_post[:min(10, len(x_post))],
                                       y_post[:min(10, len(y_post))], 1)
        anchor_val = y.loc[anchor_year]
        pre_years = out.index[(out.index < anchor_year) & (out.index >= start_year_bc)].astype(float)
        if pre_years.size:
            backcast_vals = anchor_val - slope * (anchor_year - pre_years)
            backcast_vals = np.clip(backcast_vals, min_value, None)
            out.loc[pre_years.astype(int), region] = backcast_vals
        return out

    HST_share = _backcast_linear(HST_share, "Japan", 2010)
    HST_share = _backcast_linear(HST_share, "Western Europe", 2010, start_year_bc=1980)
    HST_share = _backcast_linear(HST_share, "Korea", 2010, start_year_bc=1990)
    HST_share = _backcast_linear(HST_share, "China", 2010, start_year_bc=2000)

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
            decline_rate = ((VKM_past_lag - VKM_past_start)
                            / VKM_past_start.replace(0, np.nan) / years_in_decline)
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
    rail_length.index = pd.Index(pd.to_numeric(rail_length.index, errors="coerce").astype("Int64"))

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
        obsolete_rail_length[col] = apply_obsolete_rule(obsolete_rail_length[col])
    for col in obsolete_rail_length.columns:
        if obsolete_rail_length[col].iloc[-1] == 0 and not (obsolete_rail_length[col] == 0).all():
            obsolete_rail_length[col] = 0
    adjusted_rail_length = adjusted_rail_length - obsolete_rail_length

    adjusted_HST_length = HST_length.copy()
    for col in adjusted_HST_length.columns:
        adjusted_HST_length[col] = adjusted_HST_length[col].cummax()

    # Rail bridge/tunnel shares (v5 lines 1377-1389)
    bridge_rail_share = data_2024.loc["rail_bridge_share"]
    tunnel_rail_share = data_2024.loc["rail_tunnel_share"]
    bridge_rail = adjusted_rail_length * bridge_rail_share
    tunnel_rail = adjusted_rail_length * tunnel_rail_share
    bridge_rail_obsolete = obsolete_rail_length * bridge_rail_share
    tunnel_rail_obsolete = obsolete_rail_length * tunnel_rail_share

    # Urban rail length (v5 lines 1054-1069)
    def _urban_rail_func(hdi, upd):
        return 2.718**-2.048095 * (hdi ** 8.0777) / (upd ** 0.5885)

    urban_rail_cap = _urban_rail_func(sorted_IMAGE_hdi, sorted_urban_pop_density)
    actual_urban_rail_2024 = data_2024.loc["urban_rail"]
    if 2024 in urban_rail_cap.index:
        predicted_url_2024 = urban_rail_cap.loc[2024] * (urban_population.loc[2024] * 1000000)
        urban_rail_adj = actual_urban_rail_2024 / predicted_url_2024
    else:
        urban_rail_adj = 1.0
    urban_rail_length = urban_rail_cap * (urban_population * 1000000) * urban_rail_adj

    # Build rail/bridge/tunnel stock DataArrays with stock_tail
    def _stock_tail_da(df):
        """Apply stock_tail to a DataFrame and return DataArray."""
        df_c = df.copy()
        df_c.index = pd.Index(pd.to_numeric(df_c.index, errors="coerce").astype("Int64"))
        df_c = df_c.apply(pd.to_numeric, errors="coerce")
        df_c.loc[FIRST_YEAR_GRID] = 0.0
        df_c = df_c.reindex(extended_time_coords).interpolate().fillna(0.0)
        return xr.DataArray(df_c.values, coords=[extended_time_coords, region_coords],
                            dims=["Time", "Region"])

    rail_element_types = []
    rail_element_das = []
    rail_obs_types = []
    rail_obs_das = []

    # Active rail elements
    for name, df in [("total_rail", adjusted_rail_length),
                     ("total_HST", adjusted_HST_length),
                     ("total_urban_rail", urban_rail_length),
                     ("rail_bridge", bridge_rail),
                     ("rail_tunnel", tunnel_rail)]:
        rail_element_types.append(name)
        rail_element_das.append(_stock_tail_da(df))

    # Obsolete rail elements
    for name, df in [("obsolete_total_rail", obsolete_rail_length),
                     ("obsolete_rail_bridge", bridge_rail_obsolete),
                     ("obsolete_rail_tunnel", tunnel_rail_obsolete)]:
        rail_obs_types.append(name)
        rail_obs_das.append(_stock_tail_da(df))

    return rail_element_types, rail_element_das, rail_obs_types, rail_obs_das
