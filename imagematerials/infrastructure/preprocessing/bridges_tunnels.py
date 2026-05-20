"""Bridges, tunnels, and parking stocks (section 9b).

Bridges and tunnels are derived as 2024-anchored ratios of the active road
area, then distributed across road sub-types using region-specific share
tables from ``material-intensity-bridtun.xlsx``.

Parking area was already computed in ``network.py``; this module wraps the
paved/unpaved variants into ``xr.DataArray`` with the same stock-tail
extension that bridges/tunnels use.
"""
from pathlib import Path

import pandas as pd
import xarray as xr

from imagematerials.infrastructure.constants import COLUMN_MAPPING, FIRST_YEAR_GRID

BT_COL_TO_LABEL = {
    "bike_b": "cycle", "informal_b": "informal", "local_b": "local",
    "motorway_b": "motorway", "pedestrian_b": "pedestrian",
    "primary_b": "primary", "secondary_b": "secondary", "tertiary_b": "tertiary",
}
TUN_COL_TO_LABEL = {
    "cycle_t": "cycle", "informal_t": "informal", "local_t": "local",
    "motorway_t": "motorway", "pedestrian_t": "pedestrian",
    "primary_t": "primary", "secondary_t": "secondary", "tertiary_t": "tertiary",
}


def build_bridge_tunnel_stocks(scenario_data_dir: Path, data_2024,
                                adjusted_rural_road_area, adjusted_urban_road_area,
                                obsolete_road_rural_area, obsolete_road_urban_area,
                                extended_time_coords, region_coords):
    """Build active and obsolete bridge/tunnel DataArrays per road sub-type."""
    # Bridge & Tunnel ratios from 2024 data (v5 lines 1391-1411)
    actual_rural_bridges_2024 = data_2024.loc["rural_bridges"]
    actual_rural_tunnels_2024 = data_2024.loc["rural_tunnels"]
    actual_urban_bridges_2024 = data_2024.loc["urban_bridges"]
    actual_urban_tunnels_2024 = data_2024.loc["urban_tunnels"]
    actual_rural_road_area_2024 = data_2024.loc["rural_roads"]
    actual_urban_road_area_2024_bt = data_2024.loc["urban_roads"]

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
    bt_sheets = pd.read_excel(scenario_data_dir / "material-intensity-bridtun.xlsx",
                                index_col=0, sheet_name=None)
    rur_brid_share = bt_sheets["rur-bridge-share"].rename(index=COLUMN_MAPPING)
    rur_tun_share = bt_sheets["rur-tunnel-share"].rename(index=COLUMN_MAPPING)
    urb_brid_share = bt_sheets["urb-bridge-share"].rename(index=COLUMN_MAPPING)
    urb_tun_share = bt_sheets["urb-tunnel-share"].rename(index=COLUMN_MAPPING)

    def _distribute_by_type_shares(total_df, shares_df, col_map, prefix):
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
            area.index = pd.Index(pd.to_numeric(area.index, errors="coerce").astype("Int64"))
            area = area.apply(pd.to_numeric, errors="coerce")
            area.loc[FIRST_YEAR_GRID] = 0.0
            area = area.reindex(extended_time_coords).interpolate().fillna(0.0)
            da = xr.DataArray(area.values, coords=[extended_time_coords, region_coords],
                              dims=["Time", "Region"])
            da_items.append(da)
        return types_list, da_items

    # Active bridges/tunnels per type
    bt_settings = [
        (rural_road_bridges, rur_brid_share, BT_COL_TO_LABEL, "rural_bridge"),
        (urban_road_bridges, urb_brid_share, BT_COL_TO_LABEL, "urban_bridge"),
        (rural_road_tunnels, rur_tun_share, TUN_COL_TO_LABEL, "rural_tunnel"),
        (urban_road_tunnels, urb_tun_share, TUN_COL_TO_LABEL, "urban_tunnel"),
    ]
    obs_bt_settings = [
        (obsolete_bridges_rural, rur_brid_share, BT_COL_TO_LABEL, "obsolete_rural_bridge"),
        (obsolete_bridges_urban, urb_brid_share, BT_COL_TO_LABEL, "obsolete_urban_bridge"),
        (obsolete_tunnels_rural, rur_tun_share, TUN_COL_TO_LABEL, "obsolete_rural_tunnel"),
        (obsolete_tunnels_urban, urb_tun_share, TUN_COL_TO_LABEL, "obsolete_urban_tunnel"),
    ]

    bt_active_types = []
    bt_active_das = []
    for total_df, shares_df, col_map, prefix in bt_settings:
        t, d = _distribute_by_type_shares(total_df, shares_df, col_map, prefix)
        bt_active_types.extend(t)
        bt_active_das.extend(d)

    bt_obs_types = []
    bt_obs_das = []
    for total_df, shares_df, col_map, prefix in obs_bt_settings:
        t, d = _distribute_by_type_shares(total_df, shares_df, col_map, prefix)
        bt_obs_types.extend(t)
        bt_obs_das.extend(d)

    return bt_active_types, bt_active_das, bt_obs_types, bt_obs_das


def build_parking_stocks(parking_areas, extended_time_coords, region_coords):
    """Wrap precomputed parking areas (active + obsolete) into DataArrays.

    ``parking_areas`` is the dict returned by ``network.compute_parking_areas``.
    Parking uses the same MI as local roads (v5) and the same lifetime classes
    (urban=rc3, rural=rc2); see ``materials.py`` / ``lifetimes.py``.
    """
    parking_active_settings = [
        ("urban_parking_paved", parking_areas["active"]["urban_paved"]),
        ("urban_parking_unpaved", parking_areas["active"]["urban_unpaved"]),
        ("rural_parking_paved", parking_areas["active"]["rural_paved"]),
        ("rural_parking_unpaved", parking_areas["active"]["rural_unpaved"]),
    ]
    parking_obsolete_settings = [
        ("obsolete_urban_parking_paved", parking_areas["obsolete"]["urban_paved"]),
        ("obsolete_urban_parking_unpaved", parking_areas["obsolete"]["urban_unpaved"]),
        ("obsolete_rural_parking_paved", parking_areas["obsolete"]["rural_paved"]),
        ("obsolete_rural_parking_unpaved", parking_areas["obsolete"]["rural_unpaved"]),
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
            parea_c.index = pd.Index(
                pd.to_numeric(parea_c.index, errors="coerce").astype("Int64"))
            parea_c = parea_c.apply(pd.to_numeric, errors="coerce")
            parea_c.loc[FIRST_YEAR_GRID] = 0.0
            parea_c = parea_c.reindex(extended_time_coords).interpolate().fillna(0.0)
            da = xr.DataArray(parea_c.values,
                              coords=[extended_time_coords, region_coords],
                              dims=["Time", "Region"])
            das_out.append(da)

    return parking_types, parking_das, parking_obs_types, parking_obs_das
