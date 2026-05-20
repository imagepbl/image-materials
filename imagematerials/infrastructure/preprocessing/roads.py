"""Distribute total road area by HDI-driven type share and GDP-driven paving share.

This is section 9 in the original v5 script. It takes the urban and rural
adjusted/obsolete road areas (from ``network.py``) and produces 32 typed
road stocks: ``(urban|rural) × (cycle|informal|local|motorway|pedestrian|
primary|secondary|tertiary) × (paved|unpaved)``, each as an
``xr.DataArray(Time, Region)`` extended back to ``FIRST_YEAR_GRID``.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from imagematerials.concepts import KnowledgeGraph, Node
from imagematerials.infrastructure.constants import FIRST_YEAR_GRID

PAVING_COL_MAP = {
    "cycle": "Paved share cycle",
    "informal": "Paved share informal",
    "local": "Paved share local",
    "motorway": "Paved share motorway",
    "pedestrian": "Paved share pedestrian",
    "primary": "Paved share primary",
    "secondary": "Paved share secondary",
    "tertiary": "Paved share tertiary",
}


def build_road_type_stocks(standard_data_dir: Path, adjusted_rural_road_area,
                            adjusted_urban_road_area, obsolete_road_rural_area,
                            obsolete_road_urban_area, sorted_IMAGE_hdi,
                            sorted_gdp_urban_cap, sorted_gdp_rural_cap):
    """Build per-type road stock DataArrays (active and obsolete) and an Infrastructure knowledge graph.

    Returns
    -------
    dict with keys:
        all_types, obsolete_types, combined_types,
        da_list, obsolete_da_list, combined_da_list,
        road_labels, road_type_cols, region_coords,
        original_time_coords, extended_time_coords, end_year,
        knowledge_graph, stocks_da_roads
    """
    # Load road type distributions depending on HDI
    u_types = pd.read_excel(standard_data_dir / "urban-type-distribution.xlsx", index_col=0)
    r_types = pd.read_excel(standard_data_dir / "rural-type-distribution.xlsx", index_col=0)
    road_type_cols = u_types.columns  # e.g. ['cycle_share', 'informal_share', ...]

    # Interpolate type distributions to 0.01 HDI resolution
    hdi_bins = np.arange(0, 1.01, 0.01)
    u_types_int = u_types.reindex(hdi_bins).interpolate(method="linear", limit_direction="both")
    r_types_int = r_types.reindex(hdi_bins).interpolate(method="linear", limit_direction="both")
    u_types_int.index = np.round(u_types_int.index, 2)
    r_types_int.index = np.round(r_types_int.index, 2)

    hdi_rounded_u = sorted_IMAGE_hdi.round(2).clip(lower=0, upper=1)
    hdi_rounded_r = sorted_IMAGE_hdi.round(2).clip(lower=0, upper=1)

    # Load per-type paving distributions (GDP-based, v5 lines 840-960)
    u_paving_dist = pd.read_excel(standard_data_dir / "urban-paving-distribution.xlsx",
                                    index_col=0)
    r_paving_dist = pd.read_excel(standard_data_dir / "rural-paving-distribution.xlsx",
                                    index_col=0)

    # Interpolate paving distributions to 1000 GDP resolution
    u_gdp_bins = np.arange(0, 150001, 1000)
    r_gdp_bins = np.arange(0, 200001, 1000)
    u_paving_int = u_paving_dist.reindex(u_gdp_bins).interpolate(method="linear",
                                                                   limit_direction="both")
    r_paving_int = r_paving_dist.reindex(r_gdp_bins).interpolate(method="linear",
                                                                   limit_direction="both")
    u_paving_int = u_paving_int.clip(lower=0.0, upper=1.0)
    r_paving_int = r_paving_int.clip(lower=0.0, upper=1.0)

    # GDP binned to 1000 resolution for paving lookup
    # Rural: cummax GDP so paving rates only increase over time (v5 ratchet effect)
    u_gdp_binned = (sorted_gdp_urban_cap / 1000).round(0).astype(int).clip(lower=0, upper=150) * 1000
    r_gdp_for_paving = sorted_gdp_rural_cap.cummax()
    r_gdp_binned = (r_gdp_for_paving / 1000).round(0).astype(int).clip(lower=0, upper=200) * 1000

    # Initialize xarray with structured dimensions
    original_time_coords = list(adjusted_urban_road_area.index)
    region_coords = list(adjusted_urban_road_area.columns)

    # Stock tail: extend time back to FIRST_YEAR_GRID (1911)
    end_year = max(original_time_coords)
    extended_time_coords = list(range(FIRST_YEAR_GRID, end_year + 1))

    road_labels = [c.replace("_share", "") for c in road_type_cols]

    def _build_road_type_das_v2(total_area_df, obs_area_df, loc, type_int, hdi_rnd,
                                 paving_int, gdp_binned, prefix_active, prefix_obs):
        """Build paved/unpaved road type DataArrays using per-type paving rates."""
        active_types = []
        active_das = []
        obs_types = []
        obs_das = []

        for col, label in zip(road_type_cols, road_labels):
            # 1. Map HDI → type share for each year/region
            share_mapped = pd.DataFrame(index=hdi_rnd.index, columns=hdi_rnd.columns)
            for y in original_time_coords:
                if y in hdi_rnd.index:
                    share_mapped.loc[y] = type_int.loc[hdi_rnd.loc[y], col].values
            share_mapped = share_mapped.fillna(0)

            # 2. Map GDP → per-type paving rate for each year/region
            pave_col = PAVING_COL_MAP[label]
            pave_rate = pd.DataFrame(index=gdp_binned.index, columns=gdp_binned.columns)
            for y in original_time_coords:
                if y in gdp_binned.index:
                    gdp_vals = gdp_binned.loc[y]
                    pave_rate.loc[y] = paving_int.loc[gdp_vals, pave_col].values
            pave_rate = pave_rate.fillna(0).clip(lower=0, upper=1)

            # 3. Compute type areas
            type_total = total_area_df * share_mapped
            type_obs = obs_area_df * share_mapped

            # 4. Apply per-type paving rate
            for pave_status, rate_mult in [("paved", pave_rate), ("unpaved", 1 - pave_rate)]:
                # Active
                type_name = f"{prefix_active}{loc}_{label}_{pave_status}"
                active_types.append(type_name)
                final_area = type_total * rate_mult
                # Stock_tail
                final_area.index = pd.Index(
                    pd.to_numeric(final_area.index, errors="coerce").astype("Int64"))
                final_area = final_area.apply(pd.to_numeric, errors="coerce")
                final_area.loc[FIRST_YEAR_GRID] = 0.0
                final_area = final_area.reindex(extended_time_coords).interpolate().fillna(0.0)
                active_das.append(xr.DataArray(final_area.values,
                    coords=[extended_time_coords, region_coords], dims=["Time", "Region"]))

                # Obsolete
                obs_name = f"{prefix_obs}{loc}_{label}_{pave_status}"
                obs_types.append(obs_name)
                obs_final = type_obs * rate_mult
                obs_final.index = pd.Index(
                    pd.to_numeric(obs_final.index, errors="coerce").astype("Int64"))
                obs_final = obs_final.apply(pd.to_numeric, errors="coerce")
                obs_final.loc[FIRST_YEAR_GRID] = 0.0
                obs_final = obs_final.reindex(extended_time_coords).interpolate().fillna(0.0)
                obs_das.append(xr.DataArray(obs_final.values,
                    coords=[extended_time_coords, region_coords], dims=["Time", "Region"]))

        return active_types, active_das, obs_types, obs_das

    # Build urban road types
    u_active, u_active_das, u_obs, u_obs_das = _build_road_type_das_v2(
        adjusted_urban_road_area, obsolete_road_urban_area,
        "urban", u_types_int, hdi_rounded_u, u_paving_int, u_gdp_binned,
        "", "obsolete_")

    # Build rural road types
    r_active, r_active_das, r_obs, r_obs_das = _build_road_type_das_v2(
        adjusted_rural_road_area, obsolete_road_rural_area,
        "rural", r_types_int, hdi_rounded_r, r_paving_int, r_gdp_binned,
        "", "obsolete_")

    all_types = u_active + r_active
    da_list = u_active_das + r_active_das
    obsolete_types = u_obs + r_obs
    obsolete_da_list = u_obs_das + r_obs_das

    # Combine active + obsolete
    combined_types = all_types + obsolete_types
    combined_da_list = da_list + obsolete_da_list

    da_roads = xr.concat(combined_da_list, pd.Index(combined_types, name="Type"))
    da_roads = da_roads.assign_attrs({"units": "km**2"})

    infra_graph = KnowledgeGraph(Node("Infrastructure"))
    for t in combined_types:
        infra_graph.add(Node(t, inherits_from="Infrastructure"))

    return {
        "all_types": all_types,
        "obsolete_types": obsolete_types,
        "combined_types": combined_types,
        "da_list": da_list,
        "obsolete_da_list": obsolete_da_list,
        "combined_da_list": combined_da_list,
        "road_labels": road_labels,
        "road_type_cols": road_type_cols,
        "region_coords": region_coords,
        "original_time_coords": original_time_coords,
        "extended_time_coords": extended_time_coords,
        "end_year": end_year,
        "knowledge_graph": infra_graph,
        "stocks_da_roads": da_roads,
    }
